#!/usr/bin/env python3
"""SatDoctor: zamrozenie konfiguracji przed testem i eksport danych walidacyjnych do konsoli.

Uzycie:  python zamroz_opssat.py ~/data/opssat
Wymaga plikow z dni 1-3 (podzial.csv, dzien2.json, dzien3.json) i skryptow baseline_opssat.py, ml_opssat.py obok.
Uczy na 'fit' te same modele co ml_opssat.py, sprawdza, ze progi na 'val' sa identyczne jak w dzien3.json,
zapisuje modele.pkl i zamrozone.json (z sumami sha256) oraz folder konsola/ z danymi WYLACZNIE z 'val'.
Raz zamrozonej konfiguracji nie nadpisuje (chyba ze --nadpisz, co trzeba potem uczciwie opisac w README).
"""
import hashlib, json, pickle, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TU = Path(__file__).resolve().parent
sys.path.insert(0, str(TU))
import baseline_opssat as B
import ml_opssat as M

GLOWNY, CICHY = "LR", "RF"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def b2_staty(seg, g, w):
    """MAD reszt mediany kroczacej na kanal, z normalnych fragmentow 'fit' (jak w baseline_opssat.wyniki)."""
    fit = g[g.czesc == "fit"]
    wart = {k: d.value.to_numpy() for k, d in seg[seg.segment.isin(fit.index)].sort_values(["segment", "timestamp"]).groupby("segment")}
    mr = {}
    for kanal, f in fit.groupby("channel"):
        wzor = f[f.anomaly == 0].index if (f.anomaly == 0).any() else f.index
        mr[kanal] = float(B.mad(np.concatenate([B.reszty(wart[i], w) for i in wzor])))
    return mr


def main():
    arg = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(arg) != 1:
        sys.exit(__doc__)
    folder = Path(arg[0]).expanduser()
    if (folder / "zamrozone.json").exists() and "--nadpisz" not in sys.argv:
        sys.exit("zamrozone.json juz istnieje. Konfiguracja jest zamrozona; nie nadpisuje.")
    d2 = json.loads((folder / "dzien2.json").read_text(encoding="utf-8"))
    d3 = json.loads((folder / "dzien3.json").read_text(encoding="utf-8"))
    okno_nazwa = d2["najlepsze_okno_B2"]
    w = int(okno_nazwa.split("w=")[1])

    pod = pd.read_csv(folder / "podzial.csv", index_col="segment")
    ds = pd.read_csv(folder / "dataset.csv", index_col="segment")
    ds = ds[ds.index.isin(pod.index[pod.czesc.isin(["fit", "val"])])]
    czesc = pod.loc[ds.index, "czesc"].to_numpy()
    fit, val = czesc == "fit", czesc == "val"
    y = ds["anomaly"].to_numpy()
    X = ds[[c for c in M.CECHY if c in ds.columns]].astype(float)
    assert list(X.columns) == d3["cechy"], "inne cechy niz w dniu 3"

    seg = pd.read_csv(folder / "segments.csv", parse_dates=["timestamp"])
    g = B.podzial(seg)
    if not g["czesc"].equals(pod.loc[g.index, "czesc"]):
        sys.exit("podzial.csv nie zgadza sie z segments.csv")

    # te same modele i parametry co w ml_opssat.oceny
    sk = StandardScaler().fit(X[fit & (y == 0)])
    lr = LogisticRegression(class_weight="balanced", max_iter=5000).fit(sk.transform(X[fit]), y[fit])
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", min_samples_leaf=2,
                                random_state=M.ZIARNO, n_jobs=-1).fit(X[fit], y[fit])
    mr = b2_staty(seg, g, w)
    # sortowanie identyczne jak w baseline_opssat.wyniki: znaczniki czasu maja rozdzielczosc 1 s i bywaja
    # zdublowane, a sort po jednej kolumnie (quicksort) przestawia remisy i zmienia mediane krocząca
    wart = dict(list(seg[seg.segment.isin(ds.index)].sort_values(["segment", "timestamp"]).groupby("segment")))
    kan = ds["channel"]
    b2 = np.array([np.max(np.abs(B.reszty(wart[i].value.to_numpy(), w))) / mr[kan[i]] for i in ds.index])
    wyn = {GLOWNY: lr.predict_proba(sk.transform(X))[:, 1], CICHY: rf.predict_proba(X)[:, 1], "B2": b2}

    progi = {}
    for k, s in wyn.items():
        progi[k] = float(B.najlepszy_prog(pd.Series(s[val]), pd.Series(y[val])))
        wzorzec = d3["detektory"][f"{okno_nazwa} (baseline)" if k == "B2" else k]["prog"]
        if abs(round(progi[k], 4) - wzorzec) > 1e-4:
            sys.exit(f"{k}: prog {progi[k]:.4f} rozny od dnia 3 ({wzorzec}); model nie jest odtwarzalny")

    (folder / "modele.pkl").write_bytes(pickle.dumps(
        {"cechy": list(X.columns), "scaler": sk, "lr": lr, "rf": rf, "okno_b2": w, "mad_b2": mr, "progi": progi}))
    metr = {k: B.metryki(y[val], s[val] >= progi[k]) for k, s in wyn.items()}
    zam = {
        "zamrozono_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_oficjalny_i_przyszlosc": "NIEOTWARTE w chwili zamrozenia",
        "glowny": f"{GLOWNY} (regresja logistyczna, 19 cech, class_weight=balanced)",
        "tryb_malo_alarmow": f"{CICHY} (las losowy)",
        "porownawczy": f"B2 mediana krocząca bez probki srodkowej, okno {w}",
        "progi": progi, "cechy": list(X.columns), "mad_b2_na_kanal": mr,
        "val": metr,
        "sha256": {"modele.pkl": sha(folder / "modele.pkl"), "podzial.csv": sha(folder / "podzial.csv"),
                   **{p: sha(TU / p) for p in ("baseline_opssat.py", "ml_opssat.py", "zamroz_opssat.py")}},
    }
    (folder / "zamrozone.json").write_text(json.dumps(zam, indent=1, ensure_ascii=False), encoding="utf-8")

    # dane do konsoli: tylko 'val'
    k = folder / "konsola"
    k.mkdir(exist_ok=True)
    iv = ds.index[val]
    pred = pd.DataFrame({"segment": iv, "kanal": kan[iv].to_numpy(), "start": g.loc[iv, "start"].to_numpy(),
                         "anomalia": y[val], **{f"wynik_{n}": s[val] for n, s in wyn.items()}})
    pred.to_csv(k / "predykcje.csv", index=False)
    syg = []
    for i in iv:
        d = wart[i]
        v = d.value.to_numpy()
        syg.append(pd.DataFrame({"segment": i, "czas": d.timestamp.to_numpy(), "wartosc": v,
                                 "mediana": v - B.reszty(v, w), "b2": np.abs(B.reszty(v, w)) / mr[kan[i]]}))
    pd.concat(syg).to_csv(k / "sygnaly.csv.gz", index=False)
    (k / "konfiguracja.json").write_text(json.dumps(
        {"progi": progi, "okno_b2": w, "val": metr, "zrodlo": "OPS-SAT-AD, Zenodo 12588359, CC BY 4.0 (ESA / KP Labs)"},
        indent=1, ensure_ascii=False), encoding="utf-8")

    print("Zamrozono. Progi:", {n: round(p, 4) for n, p in progi.items()})
    for n, m in metr.items():
        print(f"  {n:3} val F0.5 {m['F0.5']:.3f}  P {m['precyzja']:.3f}  R {m['czulosc']:.3f}  FA/100 {m['FA_na_100_normalnych']}")
    print(f"sha256 modele.pkl: {zam['sha256']['modele.pkl'][:16]}...")
    print(f"Konsola: {len(iv)} fragmentow val w {k}")


if __name__ == "__main__":
    main()
