"""
app.py — Optional Streamlit UI for the Local AI Document Processing project.

Launch with:
    streamlit run app.py

The CLI remains the primary deliverable. This UI is a bonus layer on top.
"""

import json
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_input_files, ensure_dir
from src.ingest import ingest_file
from src.classify import classify, classify_by_filename
from src.extract import extract_fields
from src.retrieval import build_index, search as semantic_search

INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_FILE = PROJECT_ROOT / "output.json"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Document Processor",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2d3147; }
    .main-title { font-size: 2rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0; }
    .sub-title  { font-size: 0.95rem; color: #64748b; margin-top: 0.1rem; margin-bottom: 1.5rem; }
    .card {
        background: #1e2130; border: 1px solid #2d3147;
        border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.75rem;
    }
    .class-badge {
        display: inline-block; padding: 0.2rem 0.7rem;
        border-radius: 999px; font-size: 0.78rem; font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .badge-Invoice      { background: #1e3a5f; color: #60a5fa; }
    .badge-Resume       { background: #1a3a2a; color: #4ade80; }
    .badge-Utility      { background: #3a2a1a; color: #fb923c; }
    .badge-Other        { background: #2d2040; color: #a78bfa; }
    .badge-Unclassifiable { background: #2a2020; color: #94a3b8; }
    .field-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.4rem; }
    .field-chip {
        background: #252a3a; border: 1px solid #3d4460;
        border-radius: 6px; padding: 0.15rem 0.5rem;
        font-size: 0.78rem; color: #94a3b8;
    }
    .field-chip span { color: #e2e8f0; font-weight: 500; }
    .score-bar { height: 4px; border-radius: 2px; background: #4f46e5; }
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_output() -> dict:
    if OUTPUT_FILE.exists():
        return json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    return {}


def badge_html(doc_class: str) -> str:
    key = doc_class.replace(" ", "")
    return f'<span class="class-badge badge-{key}">{doc_class}</span>'


def run_full_pipeline(use_ocr: bool = False) -> dict:
    """Run the full ingest → classify → extract → index pipeline."""
    ensure_dir(str(INPUT_DIR))
    files = get_input_files(str(INPUT_DIR), include_images=use_ocr)

    if not files:
        st.error("No documents found in data/input/. Run prepare_sample_dataset.py first.")
        return {}

    output = {}
    docs_for_index = []
    progress = st.progress(0, text="Starting pipeline…")

    for i, f in enumerate(files):
        progress.progress((i + 1) / len(files), text=f"Processing: {f.name}")
        doc = ingest_file(f, use_ocr=use_ocr)
        docs_for_index.append(doc)

        text = doc["text"]
        doc_class = classify_by_filename(f.name) or classify(text)
        fields = extract_fields(doc_class, text)
        record = {"class": doc_class}
        record.update({k: v for k, v in fields.items() if v is not None})
        output[f.name] = record

    progress.progress(1.0, text="Building search index…")
    build_index(docs_for_index)

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    progress.empty()
    return output


# ---------------------------------------------------------------------------
# Sidebar — navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🗂️ AI Doc Processor")
st.sidebar.markdown("*Local-only · Open-source*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["📄 Documents", "⚙️ Process", "🔍 Search", "📊 Results"],
    label_visibility="collapsed"
)
st.sidebar.divider()

# Sidebar stats
output_data = load_output()
if output_data:
    classes = [v.get("class", "?") for v in output_data.values()]
    st.sidebar.markdown("**Processed documents**")
    for cls in ["Invoice", "Resume", "Utility Bill", "Other", "Unclassifiable"]:
        count = classes.count(cls)
        if count:
            st.sidebar.markdown(f"- {cls}: **{count}**")

# ---------------------------------------------------------------------------
# Page: Documents
# ---------------------------------------------------------------------------
if page == "📄 Documents":
    st.markdown('<div class="main-title">📄 Input Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Files currently in data/input/</div>', unsafe_allow_html=True)

    ensure_dir(str(INPUT_DIR))
    files = list(sorted(INPUT_DIR.iterdir()))
    files = [f for f in files if f.is_file()]

    if not files:
        st.info("No documents found. Run `prepare_sample_dataset.py` first, or add files to `data/input/`.")
    else:
        st.success(f"Found **{len(files)}** document(s)")
        cols = st.columns(3)
        for i, f in enumerate(files):
            size_kb = f.stat().st_size / 1024
            ext = f.suffix.upper().lstrip(".")
            result = output_data.get(f.name, {})
            doc_class = result.get("class", "—")

            with cols[i % 3]:
                cls_badge = badge_html(doc_class) if doc_class != "—" else ""
                st.markdown(
                    f"""<div class="card">
                        {cls_badge}
                        <div style="font-weight:600;color:#e2e8f0;margin-bottom:0.25rem">{f.name}</div>
                        <div style="font-size:0.78rem;color:#64748b">{ext} · {size_kb:.1f} KB</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# Page: Process
# ---------------------------------------------------------------------------
elif page == "⚙️ Process":
    st.markdown('<div class="main-title">⚙️ Run Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Ingest → Classify → Extract → Index</div>', unsafe_allow_html=True)

    use_ocr = st.toggle("Enable OCR for JPG/PNG images", value=False,
                         help="Requires easyocr. Install with: pip install easyocr")
    st.caption("OCR is optional. The pipeline works fully on PDF and TXT files without it.")

    col1, col2 = st.columns([1, 3])
    with col1:
        run_btn = st.button("▶ Run Processing", use_container_width=True)

    if run_btn:
        with st.spinner("Running pipeline…"):
            result = run_full_pipeline(use_ocr=use_ocr)
        if result:
            st.success(f"✅ Pipeline complete! {len(result)} document(s) processed → `output.json`")
            st.json(result)

    if OUTPUT_FILE.exists():
        st.divider()
        st.markdown("**Current output.json**")
        with st.expander("View raw JSON", expanded=False):
            st.code(OUTPUT_FILE.read_text(encoding="utf-8"), language="json")

        st.download_button(
            "⬇ Download output.json",
            data=OUTPUT_FILE.read_bytes(),
            file_name="output.json",
            mime="application/json",
        )

# ---------------------------------------------------------------------------
# Page: Search
# ---------------------------------------------------------------------------
elif page == "🔍 Search":
    st.markdown('<div class="main-title">🔍 Semantic Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Find documents by meaning, not just keywords</div>', unsafe_allow_html=True)

    query = st.text_input(
        "Search query",
        placeholder="e.g. Find all documents mentioning payments due in January",
    )
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if st.button("🔍 Search", use_container_width=False) and query:
        try:
            results = semantic_search(query, top_k=top_k)
            if not results:
                st.warning("No results found. Run the pipeline first.")
            else:
                st.markdown(f"**Top {len(results)} result(s) for:** *{query}*")
                for rank, r in enumerate(results, 1):
                    doc_info = output_data.get(r["filename"], {})
                    doc_class = doc_info.get("class", "Unknown")
                    score_pct = int(r["score"] * 100)

                    st.markdown(
                        f"""<div class="card">
                            <div style="display:flex;justify-content:space-between;align-items:center">
                                <div>
                                    {badge_html(doc_class)}
                                    <span style="font-weight:600;color:#e2e8f0;margin-left:0.5rem">
                                        #{rank} · {r["filename"]}
                                    </span>
                                </div>
                                <div style="color:#64748b;font-size:0.82rem">
                                    similarity: <strong style="color:#a5b4fc">{r["score"]:.4f}</strong>
                                </div>
                            </div>
                            <div class="score-bar" style="width:{score_pct}%;margin-top:0.5rem"></div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    # Show extracted fields for matched document
                    fields = {k: v for k, v in doc_info.items() if k != "class"}
                    if fields:
                        with st.expander(f"Extracted fields — {r['filename']}"):
                            st.json(fields)

        except FileNotFoundError:
            st.error("No search index found. Run the pipeline first (⚙️ Process page).")

# ---------------------------------------------------------------------------
# Page: Results
# ---------------------------------------------------------------------------
elif page == "📊 Results":
    st.markdown('<div class="main-title">📊 Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">output.json — all classified and extracted documents</div>', unsafe_allow_html=True)

    if not output_data:
        st.info("No results yet. Go to ⚙️ Process to run the pipeline.")
    else:
        # Summary metrics
        classes = [v.get("class", "?") for v in output_data.values()]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total", len(output_data))
        col2.metric("Invoices", classes.count("Invoice"))
        col3.metric("Resumes", classes.count("Resume"))
        col4.metric("Utility Bills", classes.count("Utility Bill"))
        col5.metric("Other / ??", classes.count("Other") + classes.count("Unclassifiable"))

        st.divider()

        # Filter by class
        filter_class = st.selectbox(
            "Filter by class",
            ["All"] + ["Invoice", "Resume", "Utility Bill", "Other", "Unclassifiable"],
        )

        # Select doc to inspect
        filtered = {
            k: v for k, v in output_data.items()
            if filter_class == "All" or v.get("class") == filter_class
        }

        selected_doc = st.selectbox("Select document to inspect", list(filtered.keys()))

        if selected_doc:
            doc_data = filtered[selected_doc]
            doc_class = doc_data.get("class", "Unknown")

            st.markdown(
                f"""<div class="card">
                    {badge_html(doc_class)}
                    <div style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-top:0.4rem">
                        {selected_doc}
                    </div>
                    <div class="field-row">""" +
                "".join(
                    f'<div class="field-chip">{k}: <span>{v}</span></div>'
                    for k, v in doc_data.items() if k != "class"
                ) +
                "</div></div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("**All results table**")

        # Build table rows
        rows = []
        for fname, data in filtered.items():
            row = {"filename": fname, "class": data.get("class", "?")}
            row.update({k: v for k, v in data.items() if k != "class"})
            rows.append(row)

        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
