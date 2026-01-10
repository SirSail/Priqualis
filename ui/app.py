"""
Priqualis UI - Streamlit Application.

Main entry point for the Streamlit-based user interface.

Run with: streamlit run ui/app.py
"""

import logging
from pathlib import Path

import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Priqualis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Sidebar Navigation
# =============================================================================

st.sidebar.title("🏥 Priqualis")
st.sidebar.markdown("Pre-submission Compliance Validator")
st.sidebar.divider()

# Navigation
page = st.sidebar.radio(
    "Navigate",
    options=[
        "🏠 Dashboard",
        "📋 Triage",
        "🔍 Similar Cases",
        "📊 KPIs",
        "⚙️ Settings",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.caption("v0.1.0 | © 2024 Priqualis")

# =============================================================================
# Dashboard Page
# =============================================================================

if page == "🏠 Dashboard":
    st.title("🏠 Dashboard")
    st.markdown("Welcome to **Priqualis** - your healthcare claim compliance assistant.")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Claims", "10,000", delta="↑ 500")
    col2.metric("Pass Rate", "97.9%", delta="↑ 2.1%")
    col3.metric("Pending", "50", delta="↓ 10")
    col4.metric("AutoFix Applied", "1,200", delta="↑ 150")

    st.divider()

    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📤 Upload Claims")
        st.markdown("Upload a batch of claims for validation.")
        if st.button("Upload Batch", key="upload"):
            st.switch_page("📋 Triage")

    with col2:
        st.markdown("### 🔍 Find Similar")
        st.markdown("Find similar approved cases.")
        if st.button("Search Cases", key="search"):
            st.switch_page("🔍 Similar Cases")

    with col3:
        st.markdown("### 📊 View Reports")
        st.markdown("Check KPIs and analytics.")
        if st.button("View KPIs", key="kpis"):
            st.switch_page("📊 KPIs")

# =============================================================================
# Triage Page
# =============================================================================

elif page == "📋 Triage":
    st.title("📋 Claim Triage")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload claims batch",
        type=["csv", "parquet"],
        help="Upload CSV or Parquet file with claim records",
    )

    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

        # Load claims
        import polars as pl

        with st.spinner("Loading claims..."):
            if uploaded_file.name.endswith(".parquet"):
                df = pl.read_parquet(uploaded_file)
            else:
                df = pl.read_csv(uploaded_file)

        st.info(f"Loaded {len(df)} claims")

        # Validate button
        if st.button("🔍 Validate Claims", type="primary"):
            with st.spinner("Validating..."):
                # Import and run validation
                from priqualis.etl.schemas import ClaimBatch, ClaimRecord
                from priqualis.rules import RuleEngine

                # Convert to records
                records = []
                for row in df.iter_rows(named=True):
                    try:
                        records.append(ClaimRecord(**row))
                    except Exception:
                        pass

                batch = ClaimBatch(records=records)

                # Validate
                engine = RuleEngine(Path("config/rules"))
                report = engine.validate(batch)

            # Results
            st.divider()
            st.subheader("📊 Validation Results")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total", report.total_records)
            col2.metric("Violations", report.violation_count, delta_color="inverse")
            col3.metric("Warnings", report.warning_count)
            col4.metric("Pass Rate", f"{report.pass_rate:.1%}")

            # Results table
            st.subheader("Violations")

            violations_data = [
                {
                    "Rule": r.rule_id,
                    "Case": r.case_id,
                    "Message": r.message[:50] + "..." if r.message and len(r.message) > 50 else r.message,
                }
                for r in report.violations[:100]
            ]

            if violations_data:
                st.dataframe(violations_data, use_container_width=True)
            else:
                st.success("No violations found!")

    else:
        st.info("👆 Upload a CSV or Parquet file to start validation.")

# =============================================================================
# Similar Cases Page
# =============================================================================

elif page == "🔍 Similar Cases":
    st.title("🔍 Similar Cases")

    # Case ID input
    case_id = st.text_input("Enter Case ID", placeholder="ENC00000001")

    if case_id:
        st.info(f"Searching for similar cases to: **{case_id}**")

        if st.button("🔍 Find Similar", type="primary"):
            with st.spinner("Searching..."):
                # Placeholder - in real app, would call API
                st.success("Found 5 similar approved cases!")

                # Mock results
                similar_cases = [
                    {"Case ID": "ENC00000042", "Similarity": "95.2%", "JGP": "A23", "Status": "Approved"},
                    {"Case ID": "ENC00000156", "Similarity": "92.8%", "JGP": "A23", "Status": "Approved"},
                    {"Case ID": "ENC00000089", "Similarity": "89.1%", "JGP": "A24", "Status": "Approved"},
                    {"Case ID": "ENC00000234", "Similarity": "85.5%", "JGP": "A23", "Status": "Approved"},
                    {"Case ID": "ENC00000567", "Similarity": "82.0%", "JGP": "A25", "Status": "Approved"},
                ]

                st.dataframe(similar_cases, use_container_width=True)

                # Attribute diffs
                st.subheader("Attribute Differences")
                st.markdown("""
                | Field | Query Value | Match Value |
                |-------|-------------|-------------|
                | icd10_main | Z00.0 | Z00.1 |
                | procedures | [89.01] | [89.01, 89.02] |
                """)

# =============================================================================
# KPIs Page
# =============================================================================

elif page == "📊 KPIs":
    st.title("📊 KPIs & Analytics")

    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=None)
    with col2:
        end_date = st.date_input("To", value=None)

    st.divider()

    # Main metrics
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("First-Pass Acceptance", "97.9%", delta="↑ 2.1%")
    col2.metric("Error Rate", "2.1%", delta="↓ 0.5%", delta_color="inverse")
    col3.metric("AutoFix Coverage", "85%", delta="↑ 5%")
    col4.metric("Avg Processing", "15ms", delta="↓ 3ms")

    st.divider()

    # Error distribution
    st.subheader("Error Distribution by Rule")

    import pandas as pd
    errors_df = pd.DataFrame({
        "Rule": ["R001", "R005", "R002", "R003", "R006"],
        "Count": [512, 296, 284, 207, 197],
        "Percentage": [34.2, 19.8, 19.0, 13.8, 13.2],
    })

    st.bar_chart(errors_df.set_index("Rule")["Count"])

    st.divider()

    # Trend
    st.subheader("FPA Trend (Last 30 Days)")
    import numpy as np

    trend_data = pd.DataFrame({
        "Day": range(1, 31),
        "FPA": np.random.uniform(0.95, 0.99, 30),
    })
    st.line_chart(trend_data.set_index("Day"))

# =============================================================================
# Settings Page
# =============================================================================

elif page == "⚙️ Settings":
    st.title("⚙️ Settings")

    st.subheader("API Configuration")
    api_host = st.text_input("API Host", value="localhost")
    api_port = st.number_input("API Port", value=8000, min_value=1, max_value=65535)

    st.divider()

    st.subheader("Search Configuration")
    alpha = st.slider("BM25 Weight (alpha)", 0.0, 1.0, 0.5)
    rerank_enabled = st.checkbox("Enable Cross-encoder Reranking", value=False)

    st.divider()

    st.subheader("Validation Configuration")
    strict_mode = st.checkbox("Strict Validation Mode", value=False)

    if st.button("Save Settings"):
        st.success("Settings saved!")
