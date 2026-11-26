# SatDoctor: anomalie w telemetrii satelity ESA OPS-SAT

Konsola operatora i uczciwa ocena detektorów anomalii na publicznym zbiorze
**OPS-SAT-AD** (ESA / KP Labs, [Zenodo 12588359](https://zenodo.org/records/12588359), CC BY 4.0).

> Wersja robocza. Pełny opis metody i wyników w przygotowaniu.

## Najważniejsze wyniki (test uruchomiony jednorazowo, po zamrożeniu konfiguracji)

- **Odtworzenie publikacji:** XGBOD na protokole autorów: F1 0.913 (publikacja 0.918), AUC-PR 0.978 (0.975).
- **Losowy podział zawyża wyniki ML.** Te same modele, oficjalny test podzielony na fragmenty z okresu treningu
  i późniejsze (AUC-ROC): RF 0.999 → 0.906, XGBOD 0.995 → 0.909, LR 0.994 → 0.933,
  a prosty próg (mediana krocząca + MAD) bez zmian: 0.902 → 0.898. Wskazówka, nie dowód
  (39 anomalii w pierwszej części, bez przedziałów ufności dla tego podziału).
- **Na danych z przyszłości ML nadal wygrywa z prostym progiem:** LR AUC-ROC 0.950 [0.92–0.975]
  vs próg 0.905 [0.873–0.935].
- **Tryb „mało alarmów”:** RF i XGBOD dały 0 fałszywych alarmów na 624 normalnych fragmentach testowych,
  wykrywając 57–66% anomalii.

Liczby: [`wyniki/wyniki_test.json`](wyniki/wyniki_test.json).

## Konsola

`streamlit run app.py` — pokazuje tylko fragmenty walidacyjne (test nie był używany do strojenia):
suwak progu, kompromis wykryte anomalie / fałszywe alarmy, przebieg sygnału z momentem alarmu,
gra „Ty kontra model”.

## Odtworzenie

```bash
pip install -r requirements.txt scikit-learn pyod xgboost matplotlib
python inspect_opssat.py ~/data/opssat     # pobranie danych z Zenodo + przegląd
python baseline_opssat.py ~/data/opssat    # podział czasowy, proste detektory
python ml_opssat.py ~/data/opssat          # modele ML na walidacji
python zamroz_opssat.py ~/data/opssat      # zamrożenie + dane konsoli
python dodatek_opssat.py ~/data/opssat     # modele z publikacji, zamrożone przed testem
python test_opssat.py ~/data/opssat        # jednorazowy test
```

## Źródło danych

Ruszczak B., Kotowski K., Evans D., Nalepa J., *The OPS-SAT benchmark for detecting anomalies in
satellite telemetry*, Scientific Data (2025), DOI 10.1038/s41597-025-05035-3. Dane: CC BY 4.0.
