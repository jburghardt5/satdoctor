#!/usr/bin/env python3
"""SatDoctor, dzien 8: JEDNORAZOWE uruchomienie zamrozonych modeli na tescie.

Uzycie:  python test_opssat.py ~/data/opssat
Sprawdza sumy sha256 zamrozonych modeli, liczy wyniki na:
  - test_oficjalny (529 fragmentow, podzial autorow, przeplata sie w czasie z treningiem),
  - przyszlosc (ostatnie 20% czasu treningu kazdego kanalu - nasz test czasowy),
z progami zamrozonymi na 'val'. Do tego miary bez progu (AUC-ROC, AUC-PR) i bootstrap 95%.
Oficjalny test rozbija na fragmenty z okresu 'fit' i pozniejsze.
Osobno: protokol autorow (trening na calym oficjalnym treningu, domyslne progi) - sprawdzenie, czy
odtwarzamy liczby z publikacji. Drugi raz nie uruchomi sie (wyniki_test.json istnieje).
"""
import json, pickle, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

TU = Path(__file__).resolve().parent
sys.path.insert(0, str(TU))
import baseline_opssat as B
import ml_opssat as M
import dodatek_opssat as D

BOOT = 1000
# arXiv 2407.04730v1, tabela 3 (test, 529 fragmentow): F1 / AUC-PR
PUBLIKACJA = {"XGBOD": (0.918, 0.975), "LR": (0.800, 0.931), "FCNN (u nas MLP)": (0.946, 0.979)}


def pelne(y, s, prog):
    a = s >= prog
    m = B.metryki(y, a)
    jedna = len(set(y)) < 2
    m.update({
        "F1": round(2 * m["precyzja"] * m["czulosc"] / (m["precyzja"] + m["czulosc"]), 3) if m["precyzja"] + m["czulosc"] else 0.0,
        "MCC": round(float(matthews_corrcoef(y, a)), 3),
        "AUC_ROC": None if jedna else round(float(roc_auc_score(y, s)), 3),
        "AUC_PR": None if jedna else round(float(average_precision_score(y, s)), 3),
        "n": int(len(y)), "anomalie": int(y.sum())})
    return m


def przedzial(y, s, prog, rng):
    f1, auc = [], []
    n = len(y)
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        if len(set(y[i])) < 2:
            continue
        a = s[i] >= prog
        tp = (a & (y[i] == 1)).sum(); fp = (a & (y[i] == 0)).sum(); fn = (~a & (y[i] == 1)).sum()
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        auc.append(roc_auc_score(y[i], s[i]))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return {"F1_95": q(f1), "AUC_ROC_95": q(auc)}


def sprawdz_sha(folder, plik_json):
    z = json.loads((folder / plik_json).read_text(encoding="utf-8"))
    for p, h in z["sha256"].items():
        sciezka = folder / p if (folder / p).exists() else TU / p
        if D.sha(sciezka) != h:
            sys.exit(f"{p}: suma sha256 inna niz przy zamrozeniu ({plik_json}). Przerywam.")
    return z


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = Path(sys.argv[1]).expanduser()
    if (folder / "wyniki_test.json").exists():
        sys.exit("Test byl juz uruchomiony (wyniki_test.json). Jednorazowy - nie powtarzam.")
    z = sprawdz_sha(folder, "zamrozone.json")
    sprawdz_sha(folder, "zamrozone_dodatek.json")
    mz = pickle.loads((folder / "modele.pkl").read_bytes())
    md = pickle.loads((folder / "modele_dodatek.pkl").read_bytes())

    pod = pd.read_csv(folder / "podzial.csv", index_col="segment", parse_dates=["start"])
    ds = pd.read_csv(folder / "dataset.csv", index_col="segment")
    seg = pd.read_csv(folder / "segments.csv", parse_dates=["timestamp"])
    X_all = ds[mz["cechy"]].astype(float)
    y_all = ds["anomaly"]

    # wyniki zamrozonych modeli dla test_oficjalny i przyszlosc
    ids = pod.index[pod.czesc.isin(["test_oficjalny", "przyszlosc"])]
    X = X_all.loc[ids]
    w = mz["okno_b2"]
    wart = dict(list(seg[seg.segment.isin(ids)].sort_values(["segment", "timestamp"]).groupby("segment")))
    kan = ds.loc[ids, "channel"]
    Xd = md["scaler"].transform(X[md["cechy"]])
    sc = {"LR": mz["lr"].predict_proba(mz["scaler"].transform(X))[:, 1],
          "RF": mz["rf"].predict_proba(X)[:, 1],
          "B2": np.array([np.max(np.abs(B.reszty(wart[i].value.to_numpy(), w))) / mz["mad_b2"][kan[i]] for i in ids]),
          "MLP": D.wynik(md["modele"]["MLP"], Xd),
          "XGBOD": D.wynik(md["modele"]["XGBOD"], Xd)}
    progi = {**mz["progi"], **md["progi"]}
    sc = pd.DataFrame(sc, index=ids)

    # podzial oficjalnego testu: w okresie 'fit' danego kanalu czy pozniej
    koniec_fit = pod[pod.czesc == "fit"].groupby("channel").start.max()
    czesc = pod.loc[ids, "czesc"].copy()
    te = czesc == "test_oficjalny"
    pozniej = pod.loc[ids, "start"] > pod.loc[ids, "channel"].map(koniec_fit)
    zbiory = {"test_oficjalny": te, "przyszlosc": czesc == "przyszlosc",
              "test_oficjalny_w_okresie_fit": te & ~pozniej, "test_oficjalny_po_okresie_fit": te & pozniej}

    rng = np.random.default_rng(M.ZIARNO)
    wyn = {}
    for zb, maska in zbiory.items():
        m = maska.to_numpy()
        y = y_all.loc[ids].to_numpy()[m]
        wyn[zb] = {}
        for n in sc.columns:
            s = sc[n].to_numpy()[m]
            r = pelne(y, s, progi[n])
            if zb in ("test_oficjalny", "przyszlosc"):
                r.update(przedzial(y, s, progi[n], rng))
            wyn[zb][n] = r

    # protokol autorow: caly oficjalny trening, domyslne progi, ocena na oficjalnym tescie
    tr, ts = ds.train == 1, ds.train == 0
    skA = StandardScaler().fit(X_all[tr & (y_all == 0)])
    XA, XT = skA.transform(X_all[tr]), skA.transform(X_all[ts])
    yA, yT = y_all[tr].to_numpy(), y_all[ts].to_numpy()
    autorzy = {}
    for nazwa, model in (("LR", LogisticRegression(max_iter=5000)), ("XGBOD", D.nowy_xgbod()),
                         ("FCNN (u nas MLP)", D.nowy_mlp())):
        model.fit(XA, yA)
        pred = model.predict(XT).astype(bool)
        s = D.wynik(model, XT)
        mm = B.metryki(yT, pred)
        f1 = round(2 * mm["precyzja"] * mm["czulosc"] / (mm["precyzja"] + mm["czulosc"]), 3) if mm["TP"] else 0.0
        autorzy[nazwa] = {"F1": f1, "precyzja": mm["precyzja"], "czulosc": mm["czulosc"],
                          "AUC_PR": round(float(average_precision_score(yT, s)), 3),
                          "publikacja_F1_AUC_PR": PUBLIKACJA[nazwa]}

    out = {"uruchomiono_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "zamrozono_utc": z["zamrozono_utc"], "progi": progi, "wyniki": wyn, "protokol_autorow": autorzy,
           "uwaga": "F0.5/F1 zalezne od progu z val; AUC-ROC nie zalezy od proporcji anomalii, AUC-PR zalezy"}
    (folder / "wyniki_test.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    for zb, t in wyn.items():
        r0 = next(iter(t.values()))
        print(f"\n=== {zb}: {r0['n']} fragmentow, {r0['anomalie']} anomalii ===")
        print("model   F0.5   F1     P      R      MCC    AUC-ROC AUC-PR FA/100  | 95% F1         95% AUC-ROC")
        for n, r in t.items():
            ci = f"| {r['F1_95']}  {r['AUC_ROC_95']}" if "F1_95" in r else ""
            print(f"{n:7} {r['F0.5']:.3f}  {r['F1']:.3f}  {r['precyzja']:.3f}  {r['czulosc']:.3f}  {r['MCC']:.3f}  "
                  f"{r['AUC_ROC']}   {r['AUC_PR']}  {r['FA_na_100_normalnych']:5}  {ci}")
    print("\n=== Protokol autorow (caly trening, domyslne progi, oficjalny test) ===")
    for n, r in autorzy.items():
        print(f"{n:18} F1 {r['F1']:.3f} (publikacja {r['publikacja_F1_AUC_PR'][0]})   "
              f"AUC-PR {r['AUC_PR']:.3f} (publikacja {r['publikacja_F1_AUC_PR'][1]})")
    print(f"\nZapisano {folder / 'wyniki_test.json'}")


if __name__ == "__main__":
    main()
