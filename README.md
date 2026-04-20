# DataLens - AI Powered EDA Dashboard

DataLens is a Streamlit-based Exploratory Data Analysis dashboard that helps students, analysts, and researchers upload a dataset and get instant, interactive insights without writing code. It supports automated EDA, data cleaning actions, and optional AI-generated observations powered by Google Gemini.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-636efa)
![Gemini](https://img.shields.io/badge/AI-Gemini%201.5%20Flash-00d4ff)

## Screenshot

Add your app screenshot here after launch.

## Features

- Upload CSV, Excel (`.xlsx`), and JSON files
- Preview first 10 rows with core dataset metrics
- Clean data from sidebar:
  - Drop duplicates
  - Fill missing values (mean for numeric, mode for text)
  - Drop columns with >50% missing values
- EDA tabs:
  - Overview (shape, dtypes, summary, missing values heatmap)
  - Univariate analysis (hist/box for numeric, bar/pie for categorical)
  - Bivariate analysis (scatter, line, bar comparison)
  - Correlation analysis (interactive heatmap, top correlated pairs, scatter matrix)
  - Outlier detection (box plots + outlier count table)
  - AI insights from Gemini (`gemini-1.5-flash`)
  - Export cleaned CSV + text summary report
- Dark themed UI with custom CSS and Plotly dark template

## Installation

1. Clone the repository.
2. Move into the project directory:
   ```bash
   cd eda_dashboard
   ```
3. Create virtual environment:
   ```bash
   python -m venv .venv
   ```
4. Activate virtual environment:
   - Windows (PowerShell):
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Gemini API Key (Free Tier)

1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Sign in and create an API key.
3. Open `.env` and set:
   ```env
   GEMINI_API_KEY=your_actual_api_key
   ```

If the key is missing, the app still runs and all non-AI features remain available.

## Run Locally

```bash
streamlit run app.py
```

## Deploy on Streamlit Cloud (Free)

1. Push this project to a GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**.
4. Select your GitHub repo, branch, and set the main file path to:
   - `app.py`
5. Open **Advanced settings** and add secret:
   - `GEMINI_API_KEY = your_actual_api_key`
6. Click **Deploy**.
7. Wait for build completion and copy your public app URL.

## Project Structure

```text
eda_dashboard/
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```
