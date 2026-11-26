#!/usr/bin/env python3
"""SatDoctor, dzien 3: detektory ML na cechach OPS-SAT-AD, porownane z baseline'em B2.

Uzycie:  python ml_opssat.py ~/data/opssat
Wymaga: dataset.csv, segments.csv, podzial.csv i dzien2.json (z baseline_opssat.py) w folderze,
oraz baseline_opssat.py obok tego skryptu. Uczy na 'fit', prog dobiera na 'val' pod F0.5.
'przyszlosc' i oficjalny test NIE sa liczone. Kolumny 'label' nie uzywa (przeciek etykiety).

Detektory:
  IF            IsolationForest, uczony tylko na normalnych z 'fit' (bez etykiet)
  LR            regresja logistyczna, class_weight=balanced (z etykietami)
  RF            las losowy, class_weight=balanced (z etykietami)
  RF+B2         jak RF, plus wynik detektora B2 (mediana krocząca, najlepsze okno z dnia 2) jako cecha
Bramka B2 (dzien 5): najlepszy ML ma na val F0.5 >= poprzeczka + 0.03.
Wynik: <folder>/dzien3.json
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("Brak scikit-learn: pip install scikit-learn")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_opssat as B

CECHY = ["mean", "var", "std", "len", "duration", "len_weighted", "gaps_squared", "n_peaks",
         "smooth10_n_peaks", "smooth20_n_peaks", "var_div_duration", "var_div_len",
         "diff_peaks", "diff2_peaks", "diff_var", "diff2_var", "kurtosis", "skew", "sampling"]
# cechy opisujace sposob wyciecia fragmentu, a nie ksztalt sygnalu
METADANE = {"len", "duration", "gaps_squared", "len_weighted", "var_div_duration", "var_div_len", "sampling"}
PROG_SKROTU = 0.6
ZIARNO = 2137
BOOT = 2000
MIN_ROZNICA = 0.03


def oceny(X, y, czesc, cechy_b2):
    fit, val = czesc == "fit", czesc == "val"
    sk = StandardScaler().fit(X[fit & (y == 0)])
    Xs = sk.transform(X)
    w = {}
    m = IsolationForest(n_estimators=300, random_state=ZIARNO).fit(Xs[fit & (y == 0)])
    w["IF"] = -m.decision_function(Xs)
    m = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Xs[fit], y[fit])
    w["LR"] = m.predict_proba(Xs)[:, 1]
    # kontrola skrotu: czy sam sposob wyciecia fragmentu zdradza etykiete?
    for nazwa, kol in (("LR metadane (kontrola)", [c for c in X.columns if c in METADANE]),
                       ("LR bez metadanych", [c for c in X.columns if c not in METADANE])):
        Xm = X[kol]
        sm = StandardScaler().fit(Xm[fit & (y == 0)])
        m = LogisticRegression(class_weight="balanced", max_iter=5000).fit(sm.transform(Xm[fit]), y[fit])
        w[nazwa] = m.predict_proba(sm.transform(Xm))[:, 1]
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", min_samples_leaf=2,
                                random_state=ZIARNO, n_jobs=-1).fit(X[fit], y[fit])
    w["RF"] = rf.predict_proba(X)[:, 1]
    waznosc = sorted(zip(rf.feature_importances_, X.columns), reverse=True)[:8]
    X2 = X.assign(b2=cechy_b2)
    m = RandomForestClassifier(n_estimators=400, class_weight="balanced", min_samples_leaf=2,
                               random_state=ZIARNO, n_jobs=-1).fit(X2[fit], y[fit])
    w["RF+B2"] = m.predict_proba(X2)[:, 1]
    return w, [(c, round(float(v), 3)) for v, c in waznosc]


def f05(y, a):
    return B.metryki(y, a)["F0.5"]


def bootstrap(y, a1, a2):
    """Przedzial 95% roznicy F0.5 (a1 - a2), losowanie fragmentow val ze zwracaniem."""
    rng = np.random.default_rng(ZIARNO)
    n, d = len(y), []
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        d.append(f05(y[i], a1[i]) - f05(y[i], a2[i]))
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = Path(sys.argv[1]).expanduser()
    d2 = json.loads((folder / "dzien2.json").read_text(encoding="utf-8"))
    poprz, okno_b2 = d2["poprzeczka_dla_ML_val_F0.5"], d2["najlepsze_okno_B2"]

    pod = pd.read_csv(folder / "podzial.csv", index_col="segment")
    ds = pd.read_csv(folder / "dataset.csv", index_col="segment")
    ds = ds[ds.index.isin(pod.index[pod.czesc.isin(["fit", "val"])])]
    czesc = pod.loc[ds.index, "czesc"].to_numpy()
    y = ds["anomaly"].to_numpy()
    X = ds[[c for c in CECHY if c in ds.columns]].astype(float)

    seg = pd.read_csv(folder / "segments.csv", parse_dates=["timestamp"])
    g = B.podzial(seg)
    if not g["czesc"].equals(pod.loc[g.index, "czesc"]):
        sys.exit("podzial.csv nie zgadza sie z segments.csv: uruchom ponownie baseline_opssat.py")
    b2 = B.wyniki(seg, g)[okno_b2].reindex(ds.index).to_numpy()

    wyn, waznosc = oceny(X, y, czesc, b2)
    wyn[f"{okno_b2} (baseline)"] = b2
    val = czesc == "val"
    kanaly = ds.loc[val, "channel"].to_numpy()
    tab, alarmy = {}, {}
    for k, s in wyn.items():
        sv = pd.Series(s[val])
        prog = B.najlepszy_prog(sv, pd.Series(y[val]))
        a = s >= prog
        alarmy[k] = a[val]
        kan = {}
        for c in sorted(set(kanaly)):
            mm = kanaly == c
            m = B.metryki(y[val][mm], a[val][mm])
            kan[c] = f"TP {m['TP']} FP {m['FP']} FN {m['FN']}"
        tab[k] = {"prog": round(float(prog), 4), "val": B.metryki(y[val], a[val]),
                  "fit": B.metryki(y[~val], a[~val]), "val_kanaly": kan}

    ml = {k: v for k, v in tab.items() if "baseline" not in k and "kontrola" not in k}
    naj = max(ml, key=lambda k: ml[k]["val"]["F0.5"])
    fn = ml[naj]["val"]["F0.5"]
    przedzial = bootstrap(y[val], alarmy[naj], alarmy[f"{okno_b2} (baseline)"])
    out = {
        "uwaga": "progi dobrane na val, wiec val jest optymistyczny dla wszystkich detektorow jednakowo",
        "cechy": list(X.columns), "waznosc_cech_RF": waznosc, "detektory": tab,
        "najlepszy_ML": naj,
        "roznica_F05_vs_baseline_val": round(fn - tab[f"{okno_b2} (baseline)"]["val"]["F0.5"], 3),
        "przedzial95_roznicy_bootstrap": przedzial,
        "bramka_B2": {"poprzeczka": poprz, "wymagane": round(poprz + MIN_ROZNICA, 3), "najlepszy_ML": fn,
                      "wynik": "pass (robimy CNN w dniach 11-12)" if fn >= poprz + MIN_ROZNICA else "fail (bez CNN, dni 11-12 na UI)"},
    }
    fm = tab["LR metadane (kontrola)"]["val"]["F0.5"]
    out["kontrola_skrotu"] = {
        "LR_metadane_val_F0.5": fm, "LR_bez_metadanych_val_F0.5": tab["LR bez metadanych"]["val"]["F0.5"],
        "prog": PROG_SKROTU,
        "wynik": (f"SKROT: same metadane daja {fm} - zamrazac model bez metadanych" if fm >= PROG_SKROTU
                  else f"brak wyraznego skrotu (metadane {fm} < {PROG_SKROTU})"),
    }
    (folder / "dzien3.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print("Detektor                 prog     | val: F0.5  P     R     TP  FP  FN  FA/100 | fit F0.5")
    for k, v in tab.items():
        a, b = v["val"], v["fit"]
        print(f"{k:24} {v['prog']:<8} | {a['F0.5']:.3f} {a['precyzja']:.3f} {a['czulosc']:.3f} "
              f"{a['TP']:3} {a['FP']:3} {a['FN']:3} {a['FA_na_100_normalnych']:5} | {b['F0.5']:.3f}")
    print(f"\nNa val per kanal ({naj}):")
    for c, s in tab[naj]["val_kanaly"].items():
        print(" ", c, s)
    print("\nNajwazniejsze cechy RF:", waznosc)
    print(f"\n{naj} - baseline: {out['roznica_F05_vs_baseline_val']:+.3f}, 95% bootstrap {przedzial}"
          + ("  (obejmuje 0: nierozstrzygajace)" if przedzial[0] <= 0 <= przedzial[1] else ""))
    print("Bramka B2:", json.dumps(out["bramka_B2"], ensure_ascii=False))
    print("Kontrola skrotu:", out["kontrola_skrotu"]["wynik"])
    print(f"Zapisano {folder / 'dzien3.json'}")


if __name__ == "__main__":
    main()
