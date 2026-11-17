
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from src.rag_pipeline import RAGPipeline
from src.analytics import calculate_metrics, answer_analytics_question


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Aavya AI | Knowledge & Analytics",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. SESSION STATE
# ==================================================

if "sales_data" not in st.session_state:
    st.session_state.sales_data = {}

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()


# ==================================================
# 3. CUSTOM UI STYLING
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(80, 50, 180, 0.19), transparent 35%),
        radial-gradient(ellipse at 90% 10%, rgba(0, 180, 220, 0.12), transparent 35%),
        #080b16;
    color: #eef2ff;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f4f6ff !important;
}

.hero {
    padding: 30px;
    border: 1px solid rgba(135, 120, 255, 0.30);
    border-radius: 22px;
    background: linear-gradient(
        125deg,
        rgba(80, 50, 160, 0.30),
        rgba(20, 120, 160, 0.12),
        rgba(20, 25, 50, 0.55)
    );
    margin-bottom: 25px;
    box-shadow: 0 12px 45px rgba(0, 0, 0, 0.25);
}

.hero-tag {
    color: #a9a1ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 34px;
    font-weight: 700;
    margin: 10px 0;
    color: #ffffff;
}

.hero-description {
    color: #b8c1d9;
    font-size: 14px;
}

.section-label {
    color: #9d9aff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.metric-card {
    padding: 20px;
    border-radius: 17px;
    border: 1px solid rgba(140, 150, 220, 0.19);
    background: linear-gradient(
        145deg,
        rgba(30, 39, 72, 0.88),
        rgba(15, 22, 43, 0.88)
    );
    min-height: 125px;
}

.metric-label {
    color: #aab5d2;
    font-size: 13px;
}

.metric-value {
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 27px;
    font-weight: 700;
    margin-top: 9px;
}

.metric-icon {
    color: #a59aff;
    font-size: 20px;
}

div[data-testid="stButton"] button {
    background: linear-gradient(100deg, #7457ff, #398cfa);
    color: white;
    border: none;
    border-radius: 11px;
    font-weight: 700;
    padding: 0.6rem 1.2rem;
    transition: all 0.2s ease;
}

div[data-testid="stButton"] button:hover {
    box-shadow: 0 0 22px rgba(105, 100, 255, 0.45);
    border: none;
    transform: translateY(-1px);
}

div[data-testid="stFileUploader"] {
    border: 1px dashed rgba(130, 130, 255, 0.45);
    border-radius: 15px;
    padding: 12px;
    background: rgba(35, 40, 75, 0.30);
}

div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background: rgba(18, 25, 48, 0.85);
    color: #ffffff;
    border: 1px solid rgba(130, 140, 220, 0.3);
    border-radius: 12px;
}

div[data-testid="stSelectbox"] > div > div {
    background: rgba(18, 25, 48, 0.85);
    border-radius: 10px;
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(130, 140, 220, 0.22);
    border-radius: 12px;
    background: rgba(20, 27, 50, 0.55);
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #11152b, #0a1020);
    border-right: 1px solid rgba(140, 130, 255, 0.20);
}

.sidebar-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 700;
    color: #ffffff;
    padding: 10px 0 3px 0;
}

.sidebar-caption {
    font-size: 11px;
    color: #a5aed0;
    letter-spacing: 1.4px;
    margin-bottom: 25px;
}

.status-pill {
    border: 1px solid rgba(60, 220, 170, 0.25);
    background: rgba(30, 180, 130, 0.10);
    color: #63e5b7;
    padding: 9px 12px;
    border-radius: 10px;
    font-size: 12px;
    margin-bottom: 8px;
}

.footer {
    color: #8490ae;
    font-size: 11px;
    text-align: center;
    padding: 25px 0 5px 0;
}

hr {
    border-color: rgba(140, 150, 220, 0.15);
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. INITIALIZE RAG PIPELINE
# ==================================================

@st.cache_resource
def get_pipeline():
    return RAGPipeline()


try:
    pipeline = get_pipeline()
    pipeline_ready = True

except Exception as error:
    pipeline_ready = False
    pipeline_error = str(error)


# ==================================================
# 5. FILE UTILITIES
# ==================================================

def save_uploaded_file(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]

    safe_name = Path(uploaded_file.name).name

    destination = UPLOAD_DIR / f"{file_hash}_{safe_name}"

    destination.write_bytes(file_bytes)

    return destination


def format_currency(value):

    if value is None:
        return "—"

    return f"₹{value:,.0f}"


# ==================================================
# 6. SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">✦ Aavya AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-caption">INTELLIGENT KNOWLEDGE PLATFORM</div>',
        unsafe_allow_html=True
    )

    st.markdown("### SYSTEM STATUS")

    if pipeline_ready:

        st.markdown(
            '<div class="status-pill">● OpenSearch connected</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="status-pill">● Hugging Face embeddings ready</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="status-pill">● RAG pipeline operational</div>',
            unsafe_allow_html=True
        )

    else:

        st.error("Pipeline initialization failed")
        st.caption(pipeline_error)

    st.divider()

    st.markdown("### DATA SOURCES")

    st.write("📊 CSV Analytics")
    st.write("📄 TXT Documents")
    st.write("📝 Markdown Files")
    st.write("🗄️ SQLite Databases")

    st.divider()

    st.caption("POWERED BY")
    st.caption("Python · Streamlit · OpenSearch")
    st.caption("Hugging Face · Ollama")


# ==================================================
# 7. HERO SECTION
# ==================================================

# Important: HTML starts at the beginning of each line.
# This prevents Streamlit from displaying it as a code block.

st.markdown(
    """
<div class="hero">
<div class="hero-tag">✦ Aavya AI | Intelligent Knowledge Platform</div>
<div class="hero-title">Your Data. Your Intelligence.</div>
<div class="hero-description">
Explore multiple data sources, discover business insights,
and ask questions using retrieval-augmented generation.
</div>
</div>
""",
    unsafe_allow_html=True
)


if not pipeline_ready:

    st.error(
        "The application could not connect to its backend. "
        "Check Docker and Ollama before continuing."
    )

    st.stop()


# ==================================================
# 8. BUSINESS ANALYTICS DASHBOARD
# ==================================================

st.markdown(
    '<div class="section-label">01 / BUSINESS OVERVIEW</div>',
    unsafe_allow_html=True
)

metrics = calculate_metrics(st.session_state.sales_data)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">◈</div>
            <div class="metric-label">Total Revenue</div>
            <div class="metric-value">
                {format_currency(metrics.get("revenue"))}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    units = metrics.get("units")

    units_display = (
        f"{units:,.0f}" if units is not None else "—"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">▤</div>
            <div class="metric-label">Units Sold</div>
            <div class="metric-value">{units_display}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    products = metrics.get("products")

    products_display = (
        str(products) if products is not None else "—"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">◇</div>
            <div class="metric-label">Unique Products</div>
            <div class="metric-value">{products_display}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    top_product = metrics.get("top_product") or "—"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">✧</div>
            <div class="metric-label">Top-Selling Product</div>
            <div class="metric-value">{top_product}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# 9. ANALYTICS CHARTS
# ==================================================

revenue_by_product = metrics.get("revenue_by_product")

if revenue_by_product is not None and not revenue_by_product.empty:

    st.write("")

    chart_col1, chart_col2 = st.columns([1.5, 1])

    with chart_col1:

        st.markdown("### Revenue by Product")

        chart_data = revenue_by_product.rename(
            "Revenue"
        ).to_frame()

        st.bar_chart(chart_data, y="Revenue")

    with chart_col2:

        st.markdown("### Product Revenue Breakdown")

        for product, revenue in revenue_by_product.items():

            st.write(f"**{product}**")

            maximum = revenue_by_product.max()

            progress_value = (
                float(revenue / maximum) if maximum > 0 else 0
            )

            st.progress(progress_value)

            st.caption(format_currency(revenue))


st.divider()


# ==================================================
# 10. DOCUMENT INGESTION
# ==================================================

st.markdown(
    '<div class="section-label">02 / DATA INGESTION</div>',
    unsafe_allow_html=True
)

st.header("Connect Your Data")

st.write(
    "Upload files from different sources and index them "
    "for semantic search."
)

uploaded_files = st.file_uploader(
    "Drop your files here",
    type=[
        "csv",
        "txt",
        "md",
        "markdown",
        "db",
        "sqlite",
        "sqlite3"
    ],
    accept_multiple_files=True
)


if uploaded_files:

    st.caption(f"{len(uploaded_files)} file(s) selected")

    if st.button(
        "✦ Process and Index Data",
        use_container_width=False
    ):

        progress = st.progress(0)

        status = st.empty()

        total_chunks = 0

        success_count = 0

        for index, uploaded_file in enumerate(uploaded_files):

            try:

                status.info(
                    f"Processing {uploaded_file.name}"
                )

                saved_path = save_uploaded_file(uploaded_file)

                extension = saved_path.suffix.lower()

                if extension == ".csv":

                    df = pd.read_csv(saved_path)

                    st.session_state.sales_data[
                        uploaded_file.name
                    ] = df

                    count = pipeline.ingest_file(
                        str(saved_path)
                    )

                    total_chunks += count

                    success_count += 1

                    st.success(
                        f"CSV indexed: {uploaded_file.name} "
                        f"({count} chunks)"
                    )

                elif extension in [".db", ".sqlite", ".sqlite3"]:

                    tables = pipeline.loader.list_sqlite_tables(
                        str(saved_path)
                    )

                    for table in tables:

                        count = pipeline.ingest_sqlite(
                            str(saved_path),
                            table
                        )

                        total_chunks += count

                    success_count += 1

                    st.success(
                        f"SQLite database processed: "
                        f"{uploaded_file.name}"
                    )

                else:

                    count = pipeline.ingest_file(
                        str(saved_path)
                    )

                    total_chunks += count

                    success_count += 1

                    st.success(
                        f"Document indexed: {uploaded_file.name} "
                        f"({count} chunks)"
                    )

            except Exception as error:

                st.error(
                    f"Error processing {uploaded_file.name}: {error}"
                )

            progress.progress(
                (index + 1) / len(uploaded_files)
            )

        status.empty()

        st.success(
            f"Processing finished: {success_count} file(s) "
            f"processed, {total_chunks} chunks indexed."
        )

        st.rerun()


st.divider()


# ==================================================
# 11. AI QUESTION ANSWERING
# ==================================================

st.markdown(
    '<div class="section-label">03 / AI KNOWLEDGE ASSISTANT</div>',
    unsafe_allow_html=True
)

st.header("Ask Your Data Anything")

st.write(
    "Ask questions about your documents and business data. "
    "The assistant retrieves relevant context before answering."
)

source_options = {
    "All Sources": None,
    "CSV": ["csv"],
    "Text": ["txt"],
    "Markdown": ["md", "markdown"],
    "SQLite": ["sqlite"]
}


question_col, settings_col = st.columns([3, 1])


with settings_col:

    selected_source = st.selectbox(
        "Search source",
        list(source_options.keys())
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=5
    )


with question_col:

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: What is the total revenue? "
            "Or: Where is the company headquartered?"
        ),
        height=120
    )


if st.button(
    "✦ Generate Intelligence",
    type="primary"
):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Analysing data and retrieving knowledge..."
        ):

            try:

                # Numerical questions use Pandas calculations.

                analytics_answer = answer_analytics_question(
                    question,
                    st.session_state.sales_data
                )

                if analytics_answer:

                    st.subheader(
                        "📊 Business Analytics Answer"
                    )

                    st.success(analytics_answer)

                    st.caption(
                        "Calculated directly from uploaded CSV data "
                        "using Pandas."
                    )

                else:

                    # Other questions use the RAG pipeline.

                    result = pipeline.ask(
                        question=question,
                        top_k=top_k,
                        source_types=source_options[selected_source]
                    )

                    st.subheader(
                        "✦ AI Generated Answer"
                    )

                    st.markdown(result["answer"])

                    st.divider()

                    st.subheader(
                        "Retrieved Context and Sources"
                    )

                    if result["sources"]:

                        for index, source in enumerate(
                            result["sources"],
                            start=1
                        ):

                            with st.expander(
                                f"Source {index} · "
                                f"{source['source']} · "
                                f"Score {source['score']:.4f}"
                            ):

                                st.caption(
                                    f"Source type: "
                                    f"{source['source_type']}"
                                )

                                st.write(source["content"])

                    else:

                        st.info(
                            "No relevant documents were retrieved."
                        )

            except Exception as error:

                st.error(
                    f"Unable to answer the question: {error}"
                )


# ==================================================
# 12. FOOTER
# ==================================================

st.markdown(
    """
<div class="footer">
AAVYA AI · MULTI-SOURCE RAG ANALYTICS ASSISTANT<br>
Built with Python, Streamlit, Hugging Face,
OpenSearch and Ollama
</div>
""",
    unsafe_allow_html=True
)
