#!/usr/bin/env python3
"""SatDoctor, dzien 1: pobranie i przeglad OPS-SAT-AD (Zenodo 12588359, CC BY 4.0).

Uzycie:  python -I inspect_opssat.py ~/data/opssat
Pobiera segments.csv i dataset.csv (jesli ich brak), sprawdza md5 z Zenodo, liczy fragmenty
normalne/anomalne w podziale autorow (kolumna train), sprawdza grupowanie po kanale i czasie
i rysuje po 3 fragmenty normalne i anomalne WYLACZNIE z treningu. Z testu bierze tylko licznosci.
Wynik: <folder>/inspect.json i <folder>/wykresy/*.png
"""
import hashlib, json, sys, urllib.request
from pathlib import Path

import pandas as pd

REKORD = "https://zenodo.org/api/records/12588359"
PLIKI = ("segments.csv", "dataset.csv")
MIN_ANOMALII_TEST = 20
ZIARNO = 2137


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pobierz(folder):
    with urllib.request.urlopen(REKORD, timeout=60) as r:
        pliki = {f["key"]: f for f in json.load(r)["files"]}
    for nazwa in PLIKI:
        p, f = folder / nazwa, pliki[nazwa]
        oczek = f["checksum"].split(":", 1)[1]
        if p.exists() and md5(p) == oczek:
            print(f"{nazwa}: jest, md5 OK")
            continue
        print(f"{nazwa}: pobieram {f['size'] / 1e6:.1f} MB ...")
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(f["links"]["self"], tmp)
        if md5(tmp) != oczek:
            tmp.unlink()
            sys.exit(f"{nazwa}: zla suma md5, przerwano")
        tmp.replace(p)
        print(f"{nazwa}: pobrany, md5 OK")


def zakresy(seg):
    """Dla kazdego fragmentu: kanal, podzial, etykieta, poczatek, koniec, liczba probek."""
    g = seg.groupby("segment")
    return pd.DataFrame({
        "channel": g["channel"].first(), "train": g["train"].first(), "anomaly": g["anomaly"].max(),
        "sampling": g["sampling"].first(), "start": g["timestamp"].min(), "koniec": g["timestamp"].max(),
        "probki": g.size()})


def przeplot(z):
    """Czy test lezy w czasie miedzy fragmentami treningu tego samego kanalu (losowy podzial)?"""
    w = {}
    for kanal, d in z.groupby("channel"):
        tr, te = d[d.train == 1], d[d.train == 0]
        if tr.empty or te.empty:
            w[kanal] = "tylko jeden podzial"
            continue
        wewnatrz = ((te.start > tr.start.min()) & (te.koniec < tr.koniec.max())).mean()
        w[kanal] = {"test_od": str(te.start.min()), "test_do": str(te.koniec.max()),
                    "trening_od": str(tr.start.min()), "trening_do": str(tr.koniec.max()),
                    "udzial_testu_wewnatrz_zakresu_treningu": round(float(wewnatrz), 3)}
    return w


def wykresy(seg, z, folder):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Brak matplotlib: pomijam wykresy (pip install matplotlib)")
        return []
    out, zapisane = folder / "wykresy", []
    out.mkdir(exist_ok=True)
    tr = z[z.train == 1]
    for etyk, nazwa in ((0, "normalny"), (1, "anomalia")):
        for i, s in enumerate(tr[tr.anomaly == etyk].sample(3, random_state=ZIARNO).index):
            d = seg[seg.segment == s]
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(d.timestamp, d.value, lw=0.8, color="tab:red" if etyk else "tab:blue")
            ax.set_title(f"trening, {nazwa}, fragment {s}, kanal {z.loc[s, 'channel']}")
            fig.autofmt_xdate()
            fig.tight_layout()
            p = out / f"{nazwa}_{i + 1}_seg{s}.png"
            fig.savefig(p, dpi=90)
            plt.close(fig)
            zapisane.append(p.name)
    return zapisane


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = Path(sys.argv[1]).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    pobierz(folder)

    seg = pd.read_csv(folder / "segments.csv", parse_dates=["timestamp"])
    ds = pd.read_csv(folder / "dataset.csv")
    print(f"\nsegments.csv: {len(seg)} wierszy, kolumny: {list(seg.columns)}")
    print(f"dataset.csv: {len(ds)} wierszy, kolumny: {list(ds.columns)}")
    print(seg.dtypes.to_string())

    z = zakresy(seg)
    niespojne = int((seg.groupby("segment")["anomaly"].nunique() > 1).sum())
    licz = z.groupby(["train", "anomaly"]).size().rename({0: "test", 1: "trening"}, level=0)
    kanaly = z.groupby(["channel", "train", "anomaly"]).size().unstack([1, 2], fill_value=0)
    a_test = int(((z.train == 0) & (z.anomaly == 1)).sum())
    zgodne = sorted(set(ds["segment"])) == sorted(z.index)

    print("\nFragmenty (podzial autorow):\n", licz.to_string())
    print("\nWedlug kanalu (train, anomaly):\n", kanaly.to_string())
    print("\nProbki na fragment:\n", z.probki.describe().round(1).to_string())
    print("\nCzestotliwosc probkowania (kolumna sampling):\n", z.sampling.value_counts().to_string())
    prz = przeplot(z)
    print("\nPrzeplot czasowy test/trening na kanal:")
    for k, v in prz.items():
        print(" ", k, v if isinstance(v, str) else f"{v['udzial_testu_wewnatrz_zakresu_treningu']:.0%} testu w zakresie treningu")

    b1 = {
        "podzial_autorow_odtworzony": bool(set(z.train) == {0, 1}),
        "anomalie_w_tescie": a_test,
        "warunek_min_anomalii": f">= {MIN_ANOMALII_TEST}",
        "ok_licznosc": a_test >= MIN_ANOMALII_TEST,
        "uwaga": "trzeci warunek B1 (baseline MAD F0.5 > 0.90 na walidacji) sprawdza dzien 2",
    }
    out = {
        "zrodlo": "Zenodo 12588359, OPS-SAT-AD, CC BY 4.0 (Ruszczak i in., Scientific Data 2025)",
        "wiersze_segments": len(seg), "fragmenty": len(z),
        "fragmenty_z_niespojna_etykieta": niespojne,
        "dataset_csv_zgodny_z_segments": zgodne,
        "licznosci": {f"{a}_{'anomalia' if b else 'normalny'}": int(v) for (a, b), v in licz.items()},
        "kanaly": {str(k): {f"{'trening' if t else 'test'}_{'anomalia' if a else 'normalny'}": int(v)
                            for (t, a), v in r.items()} for k, r in kanaly.iterrows()},
        "probki_na_fragment": z.probki.describe().round(1).to_dict(),
        "sampling": {str(k): int(v) for k, v in z.sampling.value_counts().items()},
        "przeplot_czasowy": prz,
        "bramka_B1_czesc1": b1,
    }
    out["wykresy"] = wykresy(seg, z, folder)
    (folder / "inspect.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("\nB1 (czesc 1):", json.dumps(b1, ensure_ascii=False))
    print(f"Zapisano {folder / 'inspect.json'} i {len(out['wykresy'])} wykresow w {folder / 'wykresy'}")


if __name__ == "__main__":
    main()
