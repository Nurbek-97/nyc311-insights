# NYC 311 Insights Challenge — NYC Edition

This repository is my submission for the **NYC 311 Insights Challenge**.  
The solution forecasts **daily 311 call volume**, detects **anomalies** (spikes & dips), and provides an **interactive Streamlit dashboard** for city officials.  

---

## 🚀 Deliverables

- `submission.ipynb` → End-to-end notebook (CPU-only, ≤10 min)  
- `submission.csv` → Forecasted total calls per day (scoring window)  
- `anomalies.csv` → Top spikes/dips with anomaly scores  
- `app.py` → Interactive dashboard (Streamlit)  
- `report.pdf` → 2-page insights report  
- `dashboard_url.txt` → Live deployed dashboard link  
- `.gitignore` → Ensures large data (`train.csv`, `test.csv`) not pushed  

---

## ⚙️ Environment Setup

Copy & paste the following:

```bash
# Clone repo
git clone https://github.com/Nurbek-97/nyc311-insights.git
cd nyc311-insights

# Setup Python environment
python -m venv .venv
.venv\Scripts\activate        # (Windows PowerShell)
# source .venv/bin/activate   # (Linux/Mac)

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt



📊 Methodology

Data window:

Training: Aug 1, 2024 – Apr 30, 2025

Forecast: May 1, 2025 – Aug 1, 2025

Features: Calendar (day-of-week, month, cyclic encodings), trend features

Model: XGBoost regressor + weekday baseline blend

🔎 Anomaly Detection Method

Anomalies are defined as significant spikes or dips compared to model expectations.

Formula:

Anomaly Score
=
Actual
−
Expected
Expected
Anomaly Score=
Expected
Actual−Expected
	​


Steps:

Forecast daily calls using the trained model.

Compare with actual observed calls (from test data).

Compute Anomaly Score as percent deviation.

Rank days by absolute deviation.

Select Top-5 largest spikes and Top-5 deepest dips.

Add optional brief notes (e.g., Holiday, Storm, Data glitch).
