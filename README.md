# TGV Punctuality Predictor

Machine learning model predicting on-time performance of French high-speed trains.

**Live app:** https://j-sncf-delay-predictor.streamlit.app/

## Context

TGV (Train à Grande Vitesse) is France's high-speed rail network operated by SNCF. Trains reach 320 km/h connecting major French and European cities. This project predicts monthly punctuality rates for 130+ routes based on historical performance data.

## How it works

A Random Forest model trained on 7 years of SNCF open data (2018-2025) predicts the percentage of trains arriving on time for a given route, month, and service type.

**Model accuracy:**
- MAE: 3.58 percentage points
- R²: 0.52

## Features

- Punctuality prediction by route, month, year
- Historical trend visualization
- Route performance comparison
- Bilingual interface (EN/FR)

## Tech stack

Python, Pandas, Scikit-learn, Plotly, Streamlit

## Run locally
```bash
git clone https://github.com/jaliss9/sncf-delay-predictor.git
cd sncf-delay-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Data

Source: [SNCF Open Data](https://data.sncf.com/) - Régularité mensuelle TGV

## Author

Jaliss Ch.

## License MIT