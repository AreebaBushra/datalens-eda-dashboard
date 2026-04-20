import io
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from scipy.stats import zscore


load_dotenv()
st.set_page_config(page_title="DataLens - AI Powered EDA Dashboard", layout="wide")


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #121212;
                color: #f8fafc;
            }
            section[data-testid="stSidebar"] {
                background-color: #1a1a1a !important;
                border-right: 1px solid rgba(0, 212, 255, 0.22);
            }
            section[data-testid="stSidebar"] * {
                color: #f8fafc !important;
            }
            .main-card {
                background: #1b1b1b;
                border: 1px solid rgba(0, 212, 255, 0.28);
                border-radius: 14px;
                padding: 14px;
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.45);
                margin-bottom: 12px;
            }
            .insight-card {
                background: linear-gradient(145deg, #202020, #2a2a2a);
                border-left: 4px solid #00d4ff;
                border-radius: 12px;
                padding: 12px;
                margin-bottom: 10px;
            }
            h1, h2, h3 {
                color: #00d4ff;
            }
            .stApp p, .stApp li, .stApp label, .stApp span {
                color: #f8fafc !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_data(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    if file_name.endswith(".json"):
        content = uploaded_file.read()
        try:
            json_obj = json.loads(content)
            return pd.json_normalize(json_obj)
        except json.JSONDecodeError:
            uploaded_file.seek(0)
            return pd.read_json(io.BytesIO(content))
    raise ValueError("Unsupported file type. Please upload CSV, XLSX, or JSON.")


def basic_profile(df: pd.DataFrame, file_size: int) -> Dict[str, int]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing": int(df.isna().sum().sum()),
        "file_size_kb": round(file_size / 1024, 2),
    }


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    updated = df.copy()
    numeric_cols = updated.select_dtypes(include=[np.number]).columns
    categorical_cols = updated.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        if updated[col].isna().any():
            updated[col] = updated[col].fillna(updated[col].mean())

    for col in categorical_cols:
        if updated[col].isna().any():
            mode_values = updated[col].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            updated[col] = updated[col].fillna(fill_value)

    return updated


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    min_non_null = int((1 - threshold) * len(df))
    return df.dropna(axis=1, thresh=min_non_null)


def get_top_correlated_pairs(df: pd.DataFrame, n: int = 5) -> List[Tuple[str, str, float]]:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return []

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
        .values.tolist()
    )
    return [(str(a), str(b), float(c)) for a, b, c in pairs]


def create_dataset_summary(df: pd.DataFrame) -> str:
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = get_top_correlated_pairs(df, n=5)

    summary_lines = [
        f"Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns.",
        f"Columns: {', '.join(df.columns.astype(str).tolist())}",
        "Data types:",
        df.dtypes.astype(str).to_string(),
        "Missing values by column:",
        df.isna().sum().to_string(),
        "Basic stats (numeric describe):",
        numeric_df.describe(include="all").to_string() if not numeric_df.empty else "No numeric columns found.",
        "Top correlations:",
    ]

    if correlations:
        summary_lines.extend([f"{a} vs {b}: {c:.3f}" for a, b, c in correlations])
    else:
        summary_lines.append("No strong correlations available.")

    return "\n".join(summary_lines)


def fetch_gemini_insights(summary_text: str, api_key: str) -> List[str]:
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash:generateContent"
    )
    prompt = (
        "You are a senior data analyst. Based on the dataset summary below, produce 5-10 concise "
        "insights in plain English. Focus on missing values, correlation, outliers, skewness, and "
        "recommended next steps. Return one bullet per line.\n\n"
        f"{summary_text}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(
            f"{endpoint}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )

        if response.status_code == 429:
            raise RuntimeError("Rate limit reached for Gemini. Please wait a minute and try again.")
        if response.status_code in (401, 403):
            raise RuntimeError("Gemini API key is invalid or missing permission.")
        if response.status_code >= 400:
            raise RuntimeError("Gemini service is currently unavailable. Please try again later.")

        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        insights = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
        return insights[:10]
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("Gemini request timed out. Please try again.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError("Could not connect to Gemini API. Please check your internet connection.") from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("Received an unexpected response from Gemini.") from exc


def generate_report_text(df: pd.DataFrame) -> str:
    top_corr = get_top_correlated_pairs(df, n=5)
    lines = [
        "DataLens EDA Summary Report",
        "==========================",
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        "",
        "Data Types:",
        df.dtypes.astype(str).to_string(),
        "",
        "Missing Values:",
        df.isna().sum().to_string(),
        "",
        "Top Correlated Pairs:",
    ]
    if top_corr:
        lines.extend([f"- {a} vs {b}: {c:.3f}" for a, b, c in top_corr])
    else:
        lines.append("- Not available (need at least two numeric columns)")
    return "\n".join(lines)


def main() -> None:
    apply_custom_css()
    st.title("DataLens - AI Powered EDA Dashboard")
    st.caption("Upload a dataset, clean it, analyze it, and get AI-generated insights.")

    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "clean_df" not in st.session_state:
        st.session_state.clean_df = None
    if "profile_before" not in st.session_state:
        st.session_state.profile_before = None

    with st.sidebar:
        st.header("DataLens")
        st.markdown("### 📊 Smart EDA Assistant")
        uploaded_file = st.file_uploader(
            "Upload CSV / XLSX / JSON",
            type=["csv", "xlsx", "json"],
            help="Max recommended size: 50MB",
        )

    if uploaded_file is not None:
        try:
            if st.session_state.raw_df is None or uploaded_file.name != st.session_state.get("loaded_name"):
                df = load_data(uploaded_file)
                st.session_state.raw_df = df.copy()
                st.session_state.clean_df = df.copy()
                st.session_state.profile_before = basic_profile(df, uploaded_file.size)
                st.session_state.loaded_name = uploaded_file.name
        except Exception as exc:
            st.error(f"Could not read file. Please verify format/content. Details: {exc}")
            return
    else:
        st.info("Upload a file from the sidebar to start the analysis.")
        return

    df_raw = st.session_state.raw_df
    df_clean = st.session_state.clean_df

    with st.sidebar:
        st.subheader("Data Cleaning")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Drop Duplicates"):
                try:
                    st.session_state.clean_df = st.session_state.clean_df.drop_duplicates()
                    st.success("Duplicate rows removed.")
                except Exception as exc:
                    st.warning(f"Could not drop duplicates: {exc}")
        with col_b:
            if st.button("Fill Missing"):
                try:
                    st.session_state.clean_df = fill_missing_values(st.session_state.clean_df)
                    st.success("Missing values filled (mean/mode).")
                except Exception as exc:
                    st.warning(f"Could not fill missing values: {exc}")

        if st.button("Drop >50% Missing Columns"):
            try:
                st.session_state.clean_df = drop_high_missing_columns(st.session_state.clean_df, threshold=0.5)
                st.success("Columns with more than 50% missing values dropped.")
            except Exception as exc:
                st.warning(f"Could not drop high-missing columns: {exc}")

        st.subheader("Analysis Selector")
        selected_column = st.selectbox(
            "Select a column",
            options=st.session_state.clean_df.columns.tolist(),
            index=0,
        )

    df_clean = st.session_state.clean_df
    profile_after = basic_profile(df_clean, uploaded_file.size)

    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df_clean.head(10), use_container_width=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", profile_after["rows"], profile_after["rows"] - st.session_state.profile_before["rows"])
    m2.metric(
        "Columns",
        profile_after["columns"],
        profile_after["columns"] - st.session_state.profile_before["columns"],
    )
    m3.metric(
        "Duplicates",
        profile_after["duplicates"],
        profile_after["duplicates"] - st.session_state.profile_before["duplicates"],
    )
    m4.metric(
        "Missing Values",
        profile_after["missing"],
        profile_after["missing"] - st.session_state.profile_before["missing"],
    )
    m5.metric("File Size (KB)", profile_after["file_size_kb"])
    st.markdown("</div>", unsafe_allow_html=True)

    tabs = st.tabs(
        [
            "Overview",
            "Univariate Analysis",
            "Bivariate Analysis",
            "Correlation Analysis",
            "Outlier Detection",
            "AI Insights",
            "Export",
        ]
    )

    with tabs[0]:
        try:
            st.subheader("Dataset Overview")
            st.write(f"Shape: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")
            st.dataframe(pd.DataFrame({"Column": df_clean.columns, "Data Type": df_clean.dtypes.astype(str)}))
            st.subheader("Statistical Summary")
            st.dataframe(df_clean.describe(include="all").transpose(), use_container_width=True)

            st.subheader("Missing Values Heatmap")
            missing_matrix = df_clean.isna().astype(int).T
            fig_missing = px.imshow(
                missing_matrix,
                color_continuous_scale=[[0, "#1f2937"], [1, "#00d4ff"]],
                labels={"x": "Row Index", "y": "Column", "color": "Missing"},
                aspect="auto",
            )
            fig_missing.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_missing, use_container_width=True)
        except Exception as exc:
            st.warning(f"Overview analysis skipped due to: {exc}")

    with tabs[1]:
        try:
            st.subheader(f"Univariate Analysis: {selected_column}")
            col_data = df_clean[selected_column]
            if pd.api.types.is_numeric_dtype(col_data):
                c1, c2 = st.columns(2)
                with c1:
                    fig_hist = px.histogram(df_clean, x=selected_column, template="plotly_dark", nbins=30)
                    st.plotly_chart(fig_hist, use_container_width=True)
                with c2:
                    fig_box = px.box(df_clean, y=selected_column, points="outliers", template="plotly_dark")
                    fig_box.update_traces(marker_color="#00d4ff")
                    st.plotly_chart(fig_box, use_container_width=True)
            else:
                c1, c2 = st.columns(2)
                counts = col_data.astype(str).value_counts().reset_index()
                counts.columns = [selected_column, "count"]
                with c1:
                    fig_bar = px.bar(counts, x=selected_column, y="count", template="plotly_dark")
                    st.plotly_chart(fig_bar, use_container_width=True)
                with c2:
                    fig_pie = px.pie(counts, names=selected_column, values="count", template="plotly_dark")
                    st.plotly_chart(fig_pie, use_container_width=True)
        except Exception as exc:
            st.warning(f"Univariate analysis skipped due to: {exc}")

    with tabs[2]:
        try:
            st.subheader("Bivariate Analysis")
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df_clean.columns.tolist()

            if len(all_cols) >= 2:
                x_col = st.selectbox("X-axis", all_cols, key="bivariate_x")
                y_col = st.selectbox("Y-axis", all_cols, key="bivariate_y", index=min(1, len(all_cols) - 1))
                c1, c2, c3 = st.columns(3)
                with c1:
                    fig_scatter = px.scatter(df_clean, x=x_col, y=y_col, template="plotly_dark", opacity=0.7)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                with c2:
                    fig_line = px.line(df_clean.reset_index(), x="index", y=y_col, template="plotly_dark")
                    st.plotly_chart(fig_line, use_container_width=True)
                with c3:
                    if not numeric_cols:
                        st.info("No numeric columns available for bar comparison.")
                    else:
                        agg = (
                            df_clean.groupby(x_col, dropna=False)[numeric_cols[0]]
                            .mean()
                            .reset_index()
                            .head(20)
                        )
                        fig_bar_cmp = px.bar(agg, x=x_col, y=numeric_cols[0], template="plotly_dark")
                        st.plotly_chart(fig_bar_cmp, use_container_width=True)
            else:
                st.info("Need at least two columns for bivariate analysis.")
        except Exception as exc:
            st.warning(f"Bivariate analysis skipped due to: {exc}")

    with tabs[3]:
        try:
            st.subheader("Correlation Analysis")
            numeric_df = df_clean.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 2:
                st.info("Correlation analysis requires at least two numeric columns.")
            else:
                corr = numeric_df.corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    template="plotly_dark",
                    aspect="auto",
                )
                st.plotly_chart(fig_corr, use_container_width=True)

                st.markdown("**Top 5 Correlated Pairs**")
                top_pairs = get_top_correlated_pairs(df_clean, n=5)
                st.dataframe(
                    pd.DataFrame(top_pairs, columns=["Column A", "Column B", "Correlation (abs)"]),
                    use_container_width=True,
                )

                pairplot_cols = numeric_df.columns[: min(5, numeric_df.shape[1])]
                fig_matrix = px.scatter_matrix(
                    numeric_df[pairplot_cols],
                    template="plotly_dark",
                    height=700,
                )
                st.plotly_chart(fig_matrix, use_container_width=True)
        except Exception as exc:
            st.warning(f"Correlation analysis skipped due to: {exc}")

    with tabs[4]:
        try:
            st.subheader("Outlier Detection")
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.info("No numeric columns available for outlier detection.")
            else:
                outlier_rows = []
                for col in numeric_cols:
                    series = df_clean[col].dropna()
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_count = int(((series < lower) | (series > upper)).sum())
                    outlier_rows.append({"Column": col, "Outlier Count": outlier_count})

                for col in numeric_cols:
                    fig_out = go.Figure()
                    fig_out.add_trace(
                        go.Box(
                            y=df_clean[col],
                            name=col,
                            boxpoints="outliers",
                            marker_color="#00d4ff",
                            line_color="#00d4ff",
                            fillcolor="rgba(0, 212, 255, 0.35)",
                            marker=dict(outliercolor="red", size=6),
                        )
                    )
                    fig_out.update_layout(template="plotly_dark", title=f"Outliers in {col}")
                    st.plotly_chart(fig_out, use_container_width=True)

                st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)
        except Exception as exc:
            st.warning(f"Outlier detection skipped due to: {exc}")

    with tabs[5]:
        try:
            st.subheader("AI Insights (Gemini)")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.info(
                    "Gemini API key not found. Add `GEMINI_API_KEY` to `.env` to enable AI insights. "
                    "All other dashboard features remain available."
                )
            else:
                if st.button("Generate AI Insights"):
                    with st.spinner("Generating insights from Gemini..."):
                        summary_text = create_dataset_summary(df_clean)
                        insights = fetch_gemini_insights(summary_text, api_key)
                    if insights:
                        for insight in insights:
                            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
                    else:
                        st.warning("AI insights unavailable.")
        except Exception:
            st.warning("AI insights unavailable at the moment. Please try again later.")

    with tabs[6]:
        try:
            st.subheader("Export")
            csv_data = df_clean.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Cleaned Data (CSV)",
                data=csv_data,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
            )

            report_text = generate_report_text(df_clean)
            st.download_button(
                label="Download Summary Report (TXT)",
                data=report_text,
                file_name="eda_summary_report.txt",
                mime="text/plain",
            )
            st.text_area("Report Preview", report_text, height=250)
        except Exception as exc:
            st.warning(f"Export failed: {exc}")


if __name__ == "__main__":
    main()
