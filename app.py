"""SatDoctor: konsola operatora - anomalie w telemetrii satelity ESA OPS-SAT.

Uruchom:  streamlit run app.py
Dane: folder konsola/ z zamroz_opssat.py (domyslnie ~/data/opssat/konsola, albo zmienna SATDOCTOR_DANE).
Aplikacja nie trenuje modeli; czyta gotowe wyniki. Pokazuje WYLACZNIE fragmenty walidacyjne.
"""
import json, os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_obok = Path(__file__).resolve().parent / "konsola"   # w repo / na Streamlit Cloud
DANE = Path(os.environ.get("SATDOCTOR_DANE", _obok if _obok.exists() else "~/data/opssat/konsola")).expanduser()
DETEKTORY = {
    "LR": "Regresja logistyczna (główny)",
    "RF": "Las losowy (mało fałszywych alarmów)",
    "B2": "Mediana krocząca + MAD (prosty próg)",
}

st.set_page_config(page_title="SatDoctor", layout="wide")


@st.cache_data
def wczytaj(folder):
    p = pd.read_csv(folder / "predykcje.csv", parse_dates=["start"])
    s = pd.read_csv(folder / "sygnaly.csv.gz", parse_dates=["czas"])
    k = json.loads((folder / "konfiguracja.json").read_text(encoding="utf-8"))
    return p, s, k


def metryki(y, a):
    tp = int((a & (y == 1)).sum()); fp = int((a & (y == 0)).sum()); fn = int((~a & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 1.25 * p * r / (0.25 * p + r) if p + r else 0.0
    return dict(F05=f, P=p, R=r, TP=tp, FP=fp, FN=fn, FA100=100 * fp / max(int((y == 0).sum()), 1))


def krzywa(y, s):
    pk = []
    for t in np.unique(s):
        m = metryki(y, s >= t)
        pk.append((m["R"], m["P"], m["FA100"], t))
    return pd.DataFrame(pk, columns=["czulosc", "precyzja", "FA100", "prog"])


if not (DANE / "predykcje.csv").exists():
    st.error(f"Brak danych w {DANE}. Uruchom najpierw: python zamroz_opssat.py ~/data/opssat")
    st.stop()
pred, syg, konf = wczytaj(DANE)
progi0 = konf["progi"]

# ---------- panel boczny ----------
st.sidebar.title("SatDoctor")
det = st.sidebar.selectbox("Detektor", list(DETEKTORY), format_func=DETEKTORY.get)
kol = f"wynik_{det}"
# suwak po kolejnych wartosciach wynikow (rangach), nie liniowo: prawdopodobienstwa LR/RF tloczy sie przy 1
wartosci = sorted(set(pred[kol].tolist()))
klucz_progu = f"prog_{det}"
i_zam = min(range(len(wartosci)), key=lambda i: abs(wartosci[i] - progi0[det]))
zamrozony = wartosci[i_zam]


def przywroc():
    st.session_state[klucz_progu] = i_zam


if klucz_progu not in st.session_state:
    przywroc()
# opcje to indeksy (rangi); etykieta pokazuje prog i ile fragmentow alarmuje
i_prog = st.sidebar.select_slider("Próg alarmu (w prawo = mniej alarmów)", list(range(len(wartosci))), key=klucz_progu,
                                  format_func=lambda i: f"{wartosci[i]:.6g}  ({int((pred[kol] >= wartosci[i]).sum())} alarmów)",
                                  help="Każdy krok to kolejny wynik fragmentu. Domyślnie próg zamrożony przed testem.")
prog = wartosci[i_prog]
st.sidebar.button("Przywróć zamrożony próg", on_click=przywroc, disabled=prog == zamrozony)
kanaly = st.sidebar.multiselect("Kanały", sorted(pred.kanal.unique()), default=sorted(pred.kanal.unique()))
widok = st.sidebar.radio("Pokaż fragmenty", ["wszystkie", "przeoczenia", "fałszywe alarmy", "trafienia", "poprawne normalne"],
                         help="Filtry inne niż „wszystkie” zdradzają etykietę.")
st.sidebar.caption("Dane: OPS-SAT-AD (ESA / KP Labs), Zenodo 12588359, CC BY 4.0. "
                   "Pokazane są tylko fragmenty walidacyjne; test nie był użyty do strojenia.")

p = pred[pred.kanal.isin(kanaly)].copy()
p["alarm"] = p[kol] >= prog
y = p.anomalia.to_numpy()

# ---------- metryki i krzywa ----------
st.title("SatDoctor: anomalie w telemetrii OPS-SAT")
m = metryki(y, p.alarm.to_numpy())
m0 = metryki(y, (p[kol] >= progi0[det]).to_numpy())
c = st.columns(5)
c[0].metric("F0.5", f"{m['F05']:.3f}", f"{m['F05'] - m0['F05']:+.3f} vs zamrożony" if prog != zamrozony else None)
c[1].metric("Precyzja", f"{m['P']:.2f}")
c[2].metric("Czułość", f"{m['R']:.2f}")
c[3].metric("Fałszywe alarmy / 100 normalnych", f"{m['FA100']:.1f}")
c[4].metric("TP / FP / FN", f"{m['TP']} / {m['FP']} / {m['FN']}")

with st.expander("Kompromis: wykryte anomalie a fałszywe alarmy", expanded=False):
    fig = go.Figure()
    for d, nazwa in DETEKTORY.items():
        k = krzywa(y, p[f"wynik_{d}"].to_numpy())
        fig.add_scatter(x=k.FA100, y=k.czulosc, mode="lines", name=nazwa, line_shape="hv")
    fig.add_scatter(x=[m["FA100"]], y=[m["R"]], mode="markers", marker=dict(size=14, color="#f5a623", symbol="x"),
                    name="Twój próg")
    fig.update_layout(xaxis_title="Fałszywe alarmy na 100 normalnych fragmentów", yaxis_title="Czułość (wykryte anomalie)",
                      height=400, margin=dict(t=10, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Im wyżej i bardziej w lewo, tym lepiej. Krzyżyk to obecny próg wybranego detektora.")

# ---------- wybor fragmentu ----------
filtr = {"wszystkie": np.ones(len(p), bool),
         "przeoczenia": (~p.alarm & (p.anomalia == 1)).to_numpy(),
         "fałszywe alarmy": (p.alarm & (p.anomalia == 0)).to_numpy(),
         "trafienia": (p.alarm & (p.anomalia == 1)).to_numpy(),
         "poprawne normalne": (~p.alarm & (p.anomalia == 0)).to_numpy()}[widok]
lista = p[filtr].sort_values("start")
if lista.empty:
    st.info("Brak fragmentów dla tego filtra.")
    st.stop()
etyk = {r.segment: f"{r.start:%Y-%m-%d %H:%M}  |  {r.kanal}  |  fragment {r.segment}" for r in lista.itertuples()}
seg_id = st.selectbox(f"Fragment ({len(lista)})", list(etyk), format_func=etyk.get)
r = p.set_index("segment").loc[seg_id]
s = syg[syg.segment == seg_id]

# ---------- wykres sygnalu ----------
prog_b2 = prog if det == "B2" else progi0["B2"]
alarmy = s[s.b2 >= prog_b2]
fig = go.Figure()
fig.add_scatter(x=s.czas, y=s.wartosc, mode="lines+markers", marker=dict(size=4), name="sygnal")
fig.add_scatter(x=s.czas, y=s.mediana, mode="lines", line=dict(dash="dot", color="gray"), name=f"mediana sąsiadów (okno {konf['okno_b2']})")
if len(alarmy):
    fig.add_scatter(x=alarmy.czas, y=alarmy.wartosc, mode="markers", marker=dict(size=11, color="red", symbol="circle-open"),
                    name="próbki ponad progiem B2")
    fig.add_vline(x=alarmy.czas.iloc[0], line=dict(color="red", dash="dash"))
    st.caption(f"Pierwszy alarm detektora B2: {alarmy.czas.iloc[0]:%H:%M:%S} UTC")
fig.update_layout(height=400, margin=dict(t=10, b=10), xaxis_title="czas (UTC)", yaxis_title=f"wartość kanału {r.kanal}",
                  legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
st.plotly_chart(fig, width="stretch")

# ---------- werdykty ----------
c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Werdykty detektorów")
    w = []
    for d, nazwa in DETEKTORY.items():
        pr = prog if d == det else progi0[d]
        w.append({"detektor": nazwa, "wynik": round(float(r[f"wynik_{d}"]), 4), "próg": round(pr, 4),
                  "werdykt": "ALARM" if r[f"wynik_{d}"] >= pr else "ok"})
    st.dataframe(pd.DataFrame(w), hide_index=True, width="stretch")
    st.caption("Konsola pokazuje, który sygnał i kiedy przekroczył próg. Nie diagnozuje fizycznej przyczyny; "
               "kanały OPS-SAT są opisane tylko kodem.")
with c2:
    st.subheader("Twój werdykt")
    klucz = f"odsloniete_{seg_id}"
    twoj = st.radio("Ten fragment to:", ["normalny", "anomalia"], horizontal=True, key=f"werdykt_{seg_id}")
    if st.button("Odsłoń etykietę", key=f"btn_{seg_id}"):
        st.session_state[klucz] = True
        wynik = st.session_state.setdefault("gra", {"ty": 0, "model": 0, "n": 0})
        if not st.session_state.get(f"liczone_{seg_id}"):
            st.session_state[f"liczone_{seg_id}"] = True
            prawda = int(r.anomalia)
            wynik["n"] += 1
            wynik["ty"] += int((twoj == "anomalia") == bool(prawda))
            wynik["model"] += int(bool(r[kol] >= prog) == bool(prawda))
    if st.session_state.get(klucz):
        if r.anomalia == 1:
            st.error("Etykieta operatora: ANOMALIA")
        else:
            st.success("Etykieta operatora: normalny")
    g = st.session_state.get("gra")
    if g and g["n"]:
        st.caption(f"Ty kontra {det}: {g['ty']} : {g['model']} trafień na {g['n']} odsłoniętych fragmentów")
