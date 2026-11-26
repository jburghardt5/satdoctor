#!/usr/bin/env python3
"""SatDoctor: dodatek do zamrozenia - najlepsze modele z publikacji autorow, zamrozone PRZED testem.

Uzycie:  python dodatek_opssat.py ~/data/opssat
Wymaga: zamrozone.json (z zamroz_opssat.py), pip install pyod xgboost.
  XGBOD  - PyOD, parametry domyslne, jak w publikacji (Ruszczak i in., arXiv 2407.04730: F1 0.918 na tescie)
  MLP    - PRZYBLIZENIE ich FCNN (F1 0.946): architektury FCNN nie opisano ani w artykule, ani w repo,
           wiec to nasza prosta siec (64, 32), a nie ich model.
Oba uczone na 'fit' (cechy skalowane jak w notatniku autorow: StandardScaler na normalnych z treningu),
prog dobrany na 'val' pod F0.5 - ten sam protokol co LR/RF. Test nie jest tu czytany.
Zapis: modele_dodatek.pkl, zamrozone_dodatek.json. Nie nadpisuje i odmawia, gdy test juz uruchomiono.
"""
import hashlib, json, pickle, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

TU = Path(__file__).resolve().parent
sys.path.insert(0, str(TU))
import baseline_opssat as B
import ml_opssat as M

try:
    from pyod.models.xgbod import XGBOD
except ImportError:
    sys.exit("Brak PyOD/XGBoost: pip install pyod xgboost")


def nowy_mlp():
    return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=1000,
                         random_state=M.ZIARNO)


def nowy_xgbod():
    np.random.seed(M.ZIARNO)          # PyOD XGBOD losuje w detektorach skladowych
    return XGBOD(random_state=M.ZIARNO)


def wynik(model, X):
    """Ciagly wynik anomalii (wiekszy = bardziej anomalne)."""
    if isinstance(model, XGBOD):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = Path(sys.argv[1]).expanduser()
    if not (folder / "zamrozone.json").exists():
        sys.exit("Najpierw zamroz_opssat.py")
    if (folder / "wyniki_test.json").exists():
        sys.exit("Test juz uruchomiono - dodawanie modeli teraz byloby dopasowaniem do testu.")
    if (folder / "zamrozone_dodatek.json").exists():
        sys.exit("zamrozone_dodatek.json juz istnieje; nie nadpisuje.")

    pod = pd.read_csv(folder / "podzial.csv", index_col="segment")
    ds = pd.read_csv(folder / "dataset.csv", index_col="segment")
    ds = ds[ds.index.isin(pod.index[pod.czesc.isin(["fit", "val"])])]
    czesc = pod.loc[ds.index, "czesc"].to_numpy()
    fit, val = czesc == "fit", czesc == "val"
    y = ds["anomaly"].to_numpy()
    X = ds[[c for c in M.CECHY if c in ds.columns]].astype(float)

    sk = StandardScaler().fit(X[fit & (y == 0)])
    Xs = sk.transform(X)
    modele = {"MLP": nowy_mlp().fit(Xs[fit], y[fit]), "XGBOD": nowy_xgbod().fit(Xs[fit], y[fit])}
    progi, metr = {}, {}
    for n, m in modele.items():
        s = wynik(m, Xs)
        progi[n] = float(B.najlepszy_prog(pd.Series(s[val]), pd.Series(y[val])))
        metr[n] = {"val": B.metryki(y[val], s[val] >= progi[n]), "fit": B.metryki(y[fit], s[fit] >= progi[n])}

    (folder / "modele_dodatek.pkl").write_bytes(pickle.dumps({"scaler": sk, "modele": modele, "progi": progi,
                                                              "cechy": list(X.columns)}))
    zam = {
        "zamrozono_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_oficjalny_i_przyszlosc": "NIEOTWARTE w chwili zamrozenia",
        "modele": {"XGBOD": "PyOD XGBOD, parametry domyslne (jak w publikacji)",
                   "MLP": "przyblizenie FCNN autorow: MLPClassifier(64,32), alpha=1e-3, early_stopping; architektura FCNN nieopublikowana"},
        "progi": progi, "val_fit": metr,
        "sha256": {"modele_dodatek.pkl": sha(folder / "modele_dodatek.pkl"), "dodatek_opssat.py": sha(TU / "dodatek_opssat.py")},
    }
    (folder / "zamrozone_dodatek.json").write_text(json.dumps(zam, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Zamrozono dodatek. Progi:", {n: round(p, 4) for n, p in progi.items()})
    for n, m in metr.items():
        v = m["val"]
        print(f"  {n:6} val F0.5 {v['F0.5']:.3f}  P {v['precyzja']:.3f}  R {v['czulosc']:.3f}  FA/100 {v['FA_na_100_normalnych']}"
              f"  | fit F0.5 {m['fit']['F0.5']:.3f}")


if __name__ == "__main__":
    main()
