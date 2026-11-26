#!/usr/bin/env python3
"""SatDoctor, dzien 2: podzial czasowy i proste detektory na OPS-SAT-AD.

Uzycie:  python baseline_opssat.py ~/data/opssat
Uzywa WYLACZNIE treningu autorow (train==1). Oficjalnego testu nie czyta poza oznaczeniem w podziale.
Trening kazdego kanalu sortuje po czasie i tnie blokami: pierwsze 60% -> 'fit', nastepne 20% -> 'val',
ostatnie 20% -> 'przyszlosc' (wlasny test, NIE liczony tutaj; otwieramy go raz, po zamrozeniu).
Kolumny 'label' nie uzywa: dla anomalii brzmi po prostu 'anomaly', wiec to przeciek etykiety.

Detektory (wynik fragmentu = maksimum po probkach, alarm gdy wynik >= prog):
  B0 kanal        - odsetek anomalii kanalu w 'fit' (sprawdza, czy sam kanal nie zdradza etykiety)
  B1 statyczny    - |x - mediana kanalu| / MAD kanalu (pasmo stale; odpowiednik progu kwantylowego)
  B2 mediana w=.. - |x - mediana krocząca w oknie w| / MAD reszt kanalu
Mediane i MAD licze z fragmentow normalnych 'fit'. Prog i okno dobieram na 'val' pod F0.5,
wiec wynik na 'val' jest optymistyczny - sluzy tylko do bramek, nie do raportu koncowego.
Wynik: <folder>/podzial.csv, <folder>/dzien2.json
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

UDZIALY = (0.6, 0.2)          # fit, val; reszta to 'przyszlosc'
OKNA = (5, 11, 21)
B1_MAX_F05 = 0.90             # bramka B1: jesli baseline MAD ma wiecej na val, +5 pkt jest prawie niemozliwe
B2_MIN_ROZNICA = 0.03


def podzial(seg):
    g = seg.groupby("segment").agg(channel=("channel", "first"), train=("train", "first"),
                                   anomaly=("anomaly", "max"), start=("timestamp", "min"))
    g["czesc"] = "test_oficjalny"
    for _, d in g[g.train == 1].groupby("channel"):
        idx = d.sort_values("start").index
        n = len(idx)
        a, b = round(n * UDZIALY[0]), round(n * (UDZIALY[0] + UDZIALY[1]))
        g.loc[idx[:a], "czesc"] = "fit"
        g.loc[idx[a:b], "czesc"] = "val"
        g.loc[idx[b:], "czesc"] = "przyszlosc"
    return g


def mad(x):
    x = np.asarray(x, float)
    return max(1.4826 * np.median(np.abs(x - np.median(x))), 1e-9)


def reszty(v, w):
    """x minus mediana sasiadow w oknie w, BEZ samej probki (inaczej reszta czesto = 0 i MAD -> 0)."""
    h = w // 2
    p = np.pad(np.asarray(v, float), h, constant_values=np.nan)
    okna = np.lib.stride_tricks.sliding_window_view(p, w)
    sasiedzi = np.delete(okna, h, axis=1)
    return v - np.nanmedian(sasiedzi, axis=1)


def wyniki(seg, g):
    """Wynik kazdego detektora dla fragmentow fit i val."""
    uzyte = g[g.czesc.isin(["fit", "val"])]
    s = seg[seg.segment.isin(uzyte.index)].sort_values(["segment", "timestamp"])
    wart = {k: d.value.to_numpy() for k, d in s.groupby("segment")}
    out = pd.DataFrame(index=uzyte.index)
    fit = uzyte[uzyte.czesc == "fit"]
    for kanal, d in uzyte.groupby("channel"):
        f = fit[fit.channel == kanal]
        wzor = f[f.anomaly == 0].index if (f.anomaly == 0).any() else f.index   # zapas, gdy brak normalnych
        if len(wzor) == 0:
            continue
        x0 = np.concatenate([wart[i] for i in wzor])
        med, m = np.median(x0), mad(x0)
        out.loc[d.index, "B0 kanal"] = f.anomaly.mean() if len(f) else np.nan
        out.loc[d.index, "B1 statyczny"] = [np.max(np.abs(wart[i] - med)) / m for i in d.index]
        for w in OKNA:
            mr = mad(np.concatenate([reszty(wart[i], w) for i in wzor]))
            out.loc[d.index, f"B2 mediana w={w}"] = [np.max(np.abs(reszty(wart[i], w))) / mr for i in d.index]
    return out


def metryki(y, alarm):
    tp = int((alarm & (y == 1)).sum()); fp = int((alarm & (y == 0)).sum()); fn = int((~alarm & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 1.25 * p * r / (0.25 * p + r) if p + r else 0.0
    return {"F0.5": round(f, 3), "precyzja": round(p, 3), "czulosc": round(r, 3), "TP": tp, "FP": fp, "FN": fn,
            "FA_na_100_normalnych": round(100 * fp / max(int((y == 0).sum()), 1), 1)}


def najlepszy_prog(score, y):
    ok = score.notna()
    sc, yy = score[ok].to_numpy(), y[ok].to_numpy()
    naj = (-1, None)
    for t in np.unique(sc):
        f = metryki(yy, sc >= t)["F0.5"]
        if f > naj[0]:
            naj = (f, float(t))
    return naj[1]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = Path(sys.argv[1]).expanduser()
    seg = pd.read_csv(folder / "segments.csv", parse_dates=["timestamp"])
    g = podzial(seg)
    g.drop(columns="anomaly").to_csv(folder / "podzial.csv")
    tr = g[g.train == 1]
    print("Podzial treningu (fragmenty / anomalie):")
    print(tr.groupby("czesc").anomaly.agg(["size", "sum"]).to_string())

    sc = wyniki(seg, g)
    y = g.loc[sc.index, "anomaly"]
    val = g.loc[sc.index, "czesc"] == "val"
    tab = {}
    for det in sc.columns:
        prog = najlepszy_prog(sc.loc[val, det], y[val])
        alarm = (sc[det] >= prog).to_numpy() if prog is not None else np.zeros(len(sc), bool)
        wiersz = {"prog": None if prog is None else round(prog, 4)}
        wiersz["val"] = metryki(y[val].to_numpy(), alarm[val.to_numpy()])
        wiersz["fit"] = metryki(y[~val].to_numpy(), alarm[~val.to_numpy()])
        kan = {}
        for kanal in sorted(g.channel.unique()):
            m = (val & (g.loc[sc.index, "channel"] == kanal)).to_numpy()
            if m.any():
                k = metryki(y.to_numpy()[m], alarm[m])
                kan[kanal] = f"TP {k['TP']} FP {k['FP']} FN {k['FN']}"
        wiersz["val_kanaly"] = kan
        tab[det] = wiersz

    b2 = {k: v for k, v in tab.items() if k.startswith("B2")}
    naj_b2 = max(b2, key=lambda k: b2[k]["val"]["F0.5"])
    poprz = max(tab["B1 statyczny"]["val"]["F0.5"], b2[naj_b2]["val"]["F0.5"])
    f_b2 = b2[naj_b2]["val"]["F0.5"]
    out = {
        "uwaga": "prog i okno dobrane na val pod F0.5, wiec wynik val jest optymistyczny; 'przyszlosc' i test oficjalny nietkniete",
        "podzial": {c: {"fragmenty": int(n), "anomalie": int(a)} for c, (n, a) in tr.groupby("czesc").anomaly.agg(["size", "sum"]).iterrows()},
        "detektory": tab,
        "najlepsze_okno_B2": naj_b2,
        "poprzeczka_dla_ML_val_F0.5": poprz,
        "bramka_B1_warunek3": {"baseline_MAD_val_F0.5": f_b2, "limit": B1_MAX_F05,
                               "wynik": "pass" if f_b2 <= B1_MAX_F05 else "proste progi wystarczaja (zmiana kierunku)"},
        "bramka_B2_nota": f"dzien 5: ML musi pobic {poprz} na val o >= {B2_MIN_ROZNICA}",
    }
    (folder / "dzien2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\nDetektor                 prog     | val: F0.5  P     R     TP  FP  FN  FA/100 | fit F0.5")
    for k, v in tab.items():
        a, b = v["val"], v["fit"]
        print(f"{k:24} {str(v['prog']):8} | {a['F0.5']:.3f} {a['precyzja']:.3f} {a['czulosc']:.3f} "
              f"{a['TP']:3} {a['FP']:3} {a['FN']:3} {a['FA_na_100_normalnych']:5} | {b['F0.5']:.3f}")
    print(f"\nNa val per kanal ({naj_b2}):")
    for kanal, s in tab[naj_b2]["val_kanaly"].items():
        print(" ", kanal, s)
    print("\nB1 warunek 3:", json.dumps(out["bramka_B1_warunek3"], ensure_ascii=False))
    print(f"Zapisano {folder / 'dzien2.json'} i {folder / 'podzial.csv'}")


if __name__ == "__main__":
    main()
