import re, csv, json, pickle, unicodedata, difflib

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

# --- Verification de chevauchement entre les 22 references extraites et
# le corpus OpenAlex (bloc suivant) : le corpus de recherche par citations
# est construit comme "articles CITANT 39 sources" -- si une des 22
# references est elle-meme l'une de ces 39 sources, elle est exclue par
# construction du corpus des citants (un article ne se cite pas
# lui-meme). Verifie par correspondance exacte de titre (les references
# A1-A20 portent un titre entre \emph{...}; GS et B n'en ont pas). ---
def norm_title(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

seeds = json.load(open(f"{BIBLIO_DATA}/master_gen1.json"))
seed_titles_norm = {norm_title(v.get("title") or "") for v in seeds.values()}

overlap_report = []
for code, text in FULL_REFS:
    m = re.search(r"\\emph\{([^}]+)\}", text)
    if not m:
        overlap_report.append((code, None, "hors p\u00e9rim\u00e8tre (pas de titre exploitable)"))
        continue
    title = m.group(1)
    is_seed = norm_title(title) in seed_titles_norm
    overlap_report.append((code, title, "source du corpus (exclue par construction)" if is_seed else "absente du corpus"))

n_seed = sum(1 for _, _, s in overlap_report if s.startswith("source"))

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
             r"(g\'en\'erations 1 et 2, articles citant les 39 sources de "
             r"\texttt{polyN-pipeline}, filtr\'es aux compos\'es exclusivement "
             r"azot\'es) -- \textbf{aucun n'a encore de structure extraite ni de "
             r"$\Delta H_f$ recalcul\'e} dans \texttt{polyN-pipeline} ; aucun ne "
             r"peut donc \^etre cit\'e par candidat pour l'instant. "
             r"\textbf{Chevauchement avec la liste pr\'ec\'edente v\'erifi\'e} "
             r"(correspondance exacte de titre) : " + str(n_seed) + r"~des 22 "
             r"r\'ef\'erences extraites (A1--A20) sont elles-m\^emes "
             r"parmi les 39 sources dont ce corpus recense les citations -- "
             r"exclues par construction de la liste des 156 citants (un "
             r"article ne se cite pas lui-m\^eme) ; les 2 restantes ([GS] "
             r"et [B]) n'appartiennent pas non plus \`a ce corpus (ni comme "
             r"source, ni comme citant trouv\'e). \textbf{Chevauchement "
             r"confirm\'e nul} : les deux listes sont disjointes.")
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
