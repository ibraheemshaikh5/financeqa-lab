import os
import pandas as pd
import streamlit as st
import altair as alt

# configuation
st.set_page_config(
    page_title="FinanceQA Failure Analysis",
    page_icon="📈",
    layout="wide",
)

CSV_PATH = "data/financeqa_labeled.csv"

# ****CHATGPT GENERATED CSS UI FEATURES TO MATCH AFTERQUERY WEBSITE****
ERROR_COLORS = {
    "ARITHMETIC_ERROR": "#FFE5E5",
    "WRONG_METRIC_OR_CONCEPT": "#FFF4CC",
    "CONTEXT_MISUSE_OR_HALLUCINATION": "#EFE8FF",
    "MISSING_OR_WRONG_ASSUMPTION": "#FFECCC",
    "NON_ANSWER_OR_GENERIC": "#F0F0F0",
    "CORRECT": "#D8F5D0",
}

ERROR_TEXT_COLORS = {
    "ARITHMETIC_ERROR": "#B42318",
    "WRONG_METRIC_OR_CONCEPT": "#92400E",
    "CONTEXT_MISUSE_OR_HALLUCINATION": "#4C1D95",
    "MISSING_OR_WRONG_ASSUMPTION": "#92400E",
    "NON_ANSWER_OR_GENERIC": "#374151",
    "CORRECT": "#166534",
}

def inject_css():
    st.markdown(
        """
<style>

html, body, .main {
    margin: 0 !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-height: 100vh;
    overflow-y: auto;
}

section[data-testid="stSidebar"] {
    padding-top: 1rem !important;
}

.main {
    background: radial-gradient(circle at top left, #f6f5ff 0, #f9fafb 45%, #ffffff 100%);
}

/* Tighten vertical spacing */
.element-container {
    margin-bottom: 0.25rem !important;
}

h1, h2, h3 {
    font-family: "Georgia", "Times New Roman", serif;
}

/* Chips */
.chip {
    display: inline-flex;
    align-items: center;
    padding: 0.28rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 0.35rem;
}
.chip-blue { background: #E8F1FF; color: #1D4ED8; }
.chip-yellow { background: #FFF1C2; color: #92400E; }
.chip-coral { background: #FFD2D2; color: #B42318; }
.chip-purple { background: #E9E6FF; color: #4C1D95; }

/* KPI Cards */
.kpi-card {
    border-radius: 16px;
    padding: 1rem 1.2rem;
    background: white;
    box-shadow: 0 8px 20px rgba(15,23,42,0.04);
    border: 1px solid #f1f5f9;
    margin-bottom: 0.5rem;
}

.kpi-label {
    font-size: 0.8rem;
    color: #6b7280;
    letter-spacing: 0.06em;
}
.kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #111827;
    margin-top: 0.2rem;
}

/* Error pill */
.error-pill, .label-large {
    display: inline-flex;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid rgba(15,23,42,0.08);
}

/* Panels */
.panel {
    border-radius: 16px;
    padding: 1.2rem;
    background: #ffffff;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
    border: 1px solid #e5e7eb;
    margin-bottom: 0.5rem;
}

/* Metadata */
.meta-label {
    font-size: 0.7rem;
    color: #6b7280;
    text-transform: uppercase;
}
.meta-value {
    font-size: 0.9rem;
    margin-bottom: 0.2rem;
}

/* Context box */
.context-box {
    max-height: 200px;
    overflow-y: auto;
    padding: 0.8rem;
    background: #f9fafb;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}

/* Question box */
.question-box {
    font-size: 0.95rem;
    padding: 0.75rem 0.9rem;
    background: #f9fafb;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    max-height: 100vh;
    overflow-y: auto;
}

.step-panel {
    display: flex;
    gap: 1rem;
    max-height: 70vh; 
    overflow-y: auto; /* don't let outer page scroll */
}

.rationale-box {
    max-height: 150px;
    overflow-y: auto;
    padding: 0.75rem;
    background: #f9fafb;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    font-size: 0.9rem;
    line-height: 1.5;
}

</style>
        """,
        unsafe_allow_html=True,
    )


# load the data from the csv
@st.cache_data
def load_data(path):
    # error catching
    if not os.path.exists(path):
        st.error(f"CSV not found at {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

# function to calculate important data/trends that will be visually displayed
def compute_metrics(df):
    total = len(df)
    correct = (df["error_label"] == "CORRECT").sum()
    accuracy = correct / total if total else 0

    error_counts = df["error_label"].value_counts()
    tmp = error_counts.drop(labels=["CORRECT"], errors="ignore")
    most_common = tmp.idxmax() if len(tmp) else "CORRECT"

    return total, accuracy, error_counts, most_common


# UI:
# organize and load all information
def main():
    # use GPT css
    inject_css()

    # load csv file
    df = load_data(CSV_PATH)

    # error checking
    if df.empty:
        return

    # title/header
    st.markdown(
        """
        <h1 style="margin-bottom:0rem;">FinanceQA: Model Failure Analysis</h1>
        <p style="color:#4b5563;max-width:750px;margin-top:0.2rem;">
        Comprehensive Analysis of <code>gpt-4o-mini</code> on SEC-style FinanceQA test questions.
        \nCreated by <code>Ibraheem Shaikh -- 11/16/2025</code>
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin-top:0.8rem;margin-bottom:1.2rem;">
            <span class="chip chip-blue">Weak-to-Strong Evaluation</span>
            <span class="chip chip-yellow">Error Classification</span>
            <span class="chip chip-coral">Financial Modeling</span>
            <span class="chip chip-purple">SEC Filings</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # performance indicator cards (call function to populate)
    total, accuracy, error_counts, most_common = compute_metrics(df)

    # organize 4 columns of metrics
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Samples</div>
                <div class="kpi-value">{total}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Accuracy</div>
                <div class="kpi-value">{accuracy*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        color = ERROR_COLORS.get(most_common, "#EEE")
        fg = ERROR_TEXT_COLORS.get(most_common, "#111")
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Most Common Error</div>
                <div class="label-large" style="background:{color};color:{fg};margin-top:0.4rem;">{most_common}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        failures = total - (df["error_label"] == "CORRECT").sum()
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Failures</div>
                <div class="kpi-value">{failures}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # visual representation of the error distribution
    # using a horiontal chart
    st.subheader("Error Breakdown")

    chart_df = (
        error_counts.rename_axis("error")
        .reset_index(name="count")
        .sort_values("count")
    )

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadius=6)
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("error:N", sort=None),
            color=alt.Color(
                "error:N",
                scale=alt.Scale(
                    domain=list(ERROR_COLORS.keys()),
                    range=list(ERROR_COLORS.values()),
                ),
                legend=None,
            ),
        )
        .properties(height=250)
    )

    st.altair_chart(chart, width="stretch") 
    st.markdown("---")

    # step-through example viewer
    st.subheader("Step-Through Example Viewer")

    if "deep_index" not in st.session_state:
        st.session_state.deep_index = 0

    nav1, nav2 = st.columns([1, 1])

    # previous button
    with nav1:
        if st.button("⬅ Previous", disabled=st.session_state.deep_index <= 0):
            # use session_state to prevent errors
            st.session_state.deep_index -= 1

    # next button
    with nav2:
        if st.button("Next ➡", disabled=st.session_state.deep_index >= len(df) - 1):
            # use session_state to prevent errors
            st.session_state.deep_index += 1

    idx = st.session_state.deep_index
    row = df.iloc[idx]

    st.caption(f"Showing {idx+1} of {len(df)}")

    left, right = st.columns([1.05, 1.3])

    # LEFT PANEL
    with left:
        st.markdown("###### Context & Metadata", unsafe_allow_html=True)

        st.markdown("<span class='meta-label'>Company</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='meta-value'>{row.get('company','')}</div>", unsafe_allow_html=True)

        st.markdown("<span class='meta-label'>Question Type</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='meta-value'>{row.get('question_type','')}</div>", unsafe_allow_html=True)

        fname = row.get("file_name", "")
        link = row.get("file_link", "")

        st.markdown("<span class='meta-label'>Source File</span>", unsafe_allow_html=True)
        if str(link).strip():
            st.markdown(f"<div class='meta-value'><a target='_blank' href='{link}'>{fname}</a></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='meta-value'>{fname}</div>", unsafe_allow_html=True)

        context = str(row.get("context","")).strip()
        if context:
            st.markdown("<span class='meta-label'>Context</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='context-box'>{context}</div>", unsafe_allow_html=True)
        else:
            st.caption("No context provided.")

    # RIGHT PANEL
    with right:
        label = row["error_label"]
        bg = ERROR_COLORS.get(label, "#EEE")
        fg = ERROR_TEXT_COLORS.get(label, "#111")

        st.markdown(
            f"<span class='label-large' style='background:{bg};color:{fg};'>Error: {label}</span>",
            unsafe_allow_html=True,
        )

        st.markdown("###### Question")
        st.markdown(f"<div class='question-box'>{row['question']}</div>", unsafe_allow_html=True)

        st.markdown("###### Ground Truth")
        st.code(str(row["answer"]), language="text")

        st.markdown("###### Model Answer")
        st.code(str(row["model_answer"]), language="text")

        st.markdown("###### Rationale")
        rationale = str(row.get("error_rationale",""))
        st.markdown(f"<div class='rationale-box'>{rationale}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
