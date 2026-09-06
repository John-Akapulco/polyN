import re, csv, json, pickle, unicodedata

REPORT_DIR = "/home/gilles/polyN/orca_jobs/report"
BIBLIO_DATA = f"{REPORT_DIR}/scripts/biblio_data"

state = pickle.load(open("/tmp/provenance_state.pkl", "rb"))
name_to_ref, topo = state["name_to_ref"], state["topo"]

# --- text sanitizer: raw OpenAlex titles carry unicode math/punctuation
# that plain pdflatex (utf8 inputenc, no unicode-math) chokes on ---
UNICODE_MAP = {
    "\u2212": "-",   # minus sign
    "\u2013": "--",  # en dash
    "\u2014": "---", # em dash
    "\u22c5": r"$\cdot$",  # dot operator
    "\u00b7": r"$\cdot$",  # middle dot
    "\u2032": "'",   # prime
    "\u2033": "''",  # double prime
    "\u03c0": r"$\pi$",
    "\u03b3": r"$\gamma$",
    "\u03b1": r"$\alpha$",
    "\u03b2": r"$\beta$",
    "\u03b4": r"$\delta$",
    "\u03bc": r"$\mu$",
    "\u2018": "`", "\u2019": "'", "\u201c": "``", "\u201d": "''",
    "\u2026": "...",
    "\u00d7": r"$\times$",
    "\u2264": r"$\le$", "\u2265": r"$\ge$",
    "\u2013": "-",
}

def sanitize_title(s):
    for u, rep in UNICODE_MAP.items():
        s = s.replace(u, rep)
    # A handful of OpenAlex titles carry genuine embedded TeX/MathML-derived
    # markup (e.g. "$$\hbox{A}_{\it n}(\hbox{N}_5)...$$"). Rather than try to
    # re-typeset that faithfully, flatten it to plain text: drop backslash
    # commands and every $ { } ^ token, keep the underlying characters.
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[$^{}]", "", s)
    # normalize remaining non-ASCII to closest ASCII (drops accents on
    # author-name-free titles harmlessly; anything left unencodable is
    # dropped rather than crashing the compile)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    for ch, rep in [("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"),
                    ("~", r"\textasciitilde{}")]:
        s = s.replace(ch, rep)
    return re.sub(r"\s+", " ", s).strip()

# --- Bloc B.1 : les 22 references pour lesquelles polyN-pipeline a deja
# extrait une structure et recalcule son DeltaHf isodesmique ---
FULL_REFS = [
    ("GS", r"Glukhovtsev, Jiao, von Rag\'e Schleyer, \emph{Inorg. Chem.} \textbf{1996}, 35, 7124--7133."),
    ("B", r"Fau, Mobita, Wilson, Perera, Bartlett, \emph{Quantum Theory Project report}, University of Florida."),
]
for m in re.finditer(r"\\item\[\[([A-Z0-9]+)\]\]\s*(.+)", open(f"{BIBLIO_DATA}/table_references.tex").read()):
    code, text = m.group(1), m.group(2).strip()
    if code not in dict(FULL_REFS):
        FULL_REFS.append((code, text))

cited_codes = set()
for r in topo.values():
    if r["matched_origin"] == "biblio_article" and r["all_biblio_names"]:
        for nm in r["all_biblio_names"].split(";"):
            c = name_to_ref.get(nm)
            if c:
                cited_codes.add(c)

lines = []
lines.append(r"\subsubsection*{R\'ef\'erences extraites (structure + $\Delta H_f$ isodesmique dans \texttt{polyN-pipeline})}")
lines.append(r"Ces 22 r\'ef\'erences ont chacune fourni au moins une structure recalcul\'ee "
             r"(GFN2-xTB, isodesmique) dans \texttt{polyN-pipeline}. Parmi elles, "
             + str(len(cited_codes)) + r" correspondent en plus \`a la topologie d'au "
             r"moins un de nos 193 candidats DFT (rep\'er\'ees \emph{cit\'ee dans ce "
             r"rapport} ; num\'ero(s) de citation report\'es en \S\ref{sec:provenance}).")
lines.append(r"\begin{enumerate}[leftmargin=1.6cm,itemsep=3pt]")
num_of_code = {}
for i, (code, text) in enumerate(FULL_REFS, start=1):
    num_of_code[code] = i
    status = r"\textit{structure extraite" + (r", cit\'ee dans ce rapport}" if code in cited_codes else r"}")
    lines.append(f"\\item {text} -- {status}")
lines.append(r"\end{enumerate}")

n1 = len(FULL_REFS)

# --- Bloc B.2 : corpus de recherche par citations (OpenAlex), non encore
# reduit en structures nommees -- titres relus depuis le JSON source (pas
# depuis les .tex deja generes, qui portent des caracteres unicode que
# pdflatex ne sait pas typographier tels quels). ---
gen1 = json.load(open(f"{BIBLIO_DATA}/pure_n_candidates_gen1.json"))
gen2 = json.load(open(f"{BIBLIO_DATA}/curated_gen2.json"))
rows = list(gen1.values()) + list(gen2.values())
rows.sort(key=lambda w: (w.get("year") or 0, w.get("title") or ""))

lines.append(r"\subsubsection*{Corpus de recherche bibliographique (OpenAlex), structures non extraites -- travail restant}")
lines.append(str(len(rows)) + r" articles identifi\'es par recherche de citations "
             r"(g\'en\'erations 1 et 2, articles citant les 38 sources de "
             r"\texttt{polyN-pipeline}, filtr\'es aux compos\'es exclusivement "
             r"azot\'es) -- \textbf{aucun n'a encore de structure extraite ni de "
             r"$\Delta H_f$ recalcul\'e} dans \texttt{polyN-pipeline} ; aucun ne "
             r"peut donc \^etre cit\'e par candidat pour l'instant. Le "
             r"chevauchement \'eventuel avec la liste pr\'ec\'edente n'a "
             r"\textbf{pas} \'et\'e v\'erifi\'e (recoupement titre/DOI \`a faire).")
lines.append(r"{\scriptsize")
lines.append(r"\begin{longtable}{@{}cp{1.0cm}p{8.4cm}p{4.2cm}@{}}")
lines.append(r"\caption{Travail bibliographique restant \`a faire (statut~: aucune structure extraite, cf.\ texte ci-dessus) -- article, ann\'ee, DOI.}")
lines.append(r"\label{tab:biblio-todo}\\")
lines.append(r"\toprule")
lines.append(r"\textbf{N\textdegree} & \textbf{Ann\'ee} & \textbf{Titre} & \textbf{DOI} \\")
lines.append(r"\midrule\endfirsthead")
lines.append(r"\multicolumn{4}{c}{\small (suite)}\\ \toprule")
lines.append(r"\textbf{N\textdegree} & \textbf{Ann\'ee} & \textbf{Titre} & \textbf{DOI} \\")
lines.append(r"\midrule\endhead")
lines.append(r"\bottomrule\endfoot")
lines.append(r"\bottomrule\endlastfoot")
for i, w in enumerate(rows, start=n1 + 1):
    year = w.get("year") or "?"
    title_esc = sanitize_title(w.get("title") or "(titre indisponible)")
    doi = (w.get("doi") or "").replace("https://doi.org/", "")
    doi_disp = doi if doi else "(DOI absent)"
    doi_disp = doi_disp.replace("_", r"\_").replace("#", r"\#")
    # allow the (unbreakable-by-default) DOI to wrap inside its cell
    doi_disp = doi_disp.replace("/", r"/\allowbreak ").replace("-", r"-\allowbreak ")
    lines.append(f"{i} & {year} & {title_esc} & \\texttt{{{doi_disp}}} \\\\")
lines.append(r"\end{longtable}")
lines.append(r"}")

with open(f"{REPORT_DIR}/annexe_references.tex", "w") as fh:
    fh.write("\n".join(lines))

print(f"annexe_references.tex: {n1} extraites (bloc 1) + {len(rows)} a faire (bloc 2) = {n1+len(rows)} references numerotees")
print("codes cites dans ce rapport:", sorted(cited_codes), "-> numeros:", [num_of_code[c] for c in cited_codes])
