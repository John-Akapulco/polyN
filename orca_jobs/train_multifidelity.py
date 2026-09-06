import csv
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

FEATURES = ["desc_n_atoms", "desc_degree_max", "desc_degree_mean",
            "desc_frac_degree_1", "desc_frac_degree_2", "desc_frac_degree_3",
            "desc_min_cycle_size", "desc_n_cycles_independent", "desc_is_tree",
            "desc_n_fragments", "desc_max_local_cycle_overlap"]


def load(fname, target_col, family_col="family"):
    rows = list(csv.DictReader(open(fname)))
    by_fam = {}
    for fam in set(r[family_col] for r in rows):
        sub = [r for r in rows if r[family_col] == fam]
        X = np.array([[float(r[f].replace("True", "1").replace("False", "0")) for f in FEATURES] for r in sub])
        y = np.array([float(r[target_col]) for r in sub])
        by_fam[fam] = (X, y, [r["name"] for r in sub])
    return by_fam


def evaluate(X, y, n_splits=5):
    n = len(y)
    n_splits = min(n_splits, n)
    if n < 8:
        return None  # trop peu de points pour une CV fiable
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    # baseline : n_atoms seul (regression lineaire)
    X_natoms = X[:, [0]]
    base_pred = cross_val_predict(LinearRegression(), X_natoms, y, cv=kf)
    base_mae = np.mean(np.abs(base_pred - y))

    # modele : gradient boosting sur tous les descripteurs (memes
    # hyperparametres que AdaptiveSurrogate.fit pour rester coherent)
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
    gbr_pred = cross_val_predict(gbr, X, y, cv=kf)
    gbr_mae = np.mean(np.abs(gbr_pred - y))

    return base_mae, gbr_mae, n


print("=" * 70)
print("Delta(DFT - xTB) en eV/atome, par famille")
print("=" * 70)
data = load("/home/gilles/polyN/orca_jobs/multifidelity_xtb_dft.csv", "delta_dft_minus_xtb_ev")
for fam, (X, y, names) in sorted(data.items()):
    res = evaluate(X, y)
    if res is None:
        print(f"  {fam}: n={len(y)} -- trop peu de points pour une CV fiable")
        continue
    base_mae, gbr_mae, n = res
    print(f"  {fam}: n={n}  MAE baseline(n_atoms seul)={base_mae*1000:.1f} meV/atome  "
          f"MAE GBR(11 descripteurs)={gbr_mae*1000:.1f} meV/atome  "
          f"gain={100*(1-gbr_mae/base_mae):.0f}%")

print()
print("=" * 70)
print("Delta(CCSD(T) - DFT) en eV/atome, toutes familles regroupees (echantillon trop petit pour separer)")
print("=" * 70)
data2 = load("/home/gilles/polyN/orca_jobs/multifidelity_dft_ccsdt.csv", "delta_ccsdt_minus_dft_ev")
X_all = np.concatenate([X for X, y, n in data2.values()])
y_all = np.concatenate([y for X, y, n in data2.values()])
res = evaluate(X_all, y_all, n_splits=5)
if res:
    base_mae, gbr_mae, n = res
    print(f"  n={n}  MAE baseline(n_atoms seul)={base_mae*1000:.1f} meV/atome  "
          f"MAE GBR(11 descripteurs)={gbr_mae*1000:.1f} meV/atome  "
          f"gain={100*(1-gbr_mae/base_mae):.0f}%")
    print(f"  (pour reference, std brute du delta = {y_all.std()*1000:.1f} meV/atome)")
