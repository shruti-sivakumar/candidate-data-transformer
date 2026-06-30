"""Thin Streamlit demo for the candidate data transformer."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.transformer.pipeline import PipelineInputs, read_text, run_pipeline

SAMPLES = Path("samples")

_SAMPLE_BUNDLES = {
    "Kelsey Hightower (all available sources)": {
        "csv": SAMPLES / "recruiter_csv/kelsey_hightower.csv",
        "ats": SAMPLES / "ats_json/kelsey_hightower.json",
        "github": SAMPLES / "github/kelsey_hightower.json",
        "notes": SAMPLES / "recruiter_notes/kelsey_hightower.txt",
    },
    "Andrej Karpathy (structured + GitHub)": {
        "csv": SAMPLES / "recruiter_csv/andrej_karpathy.csv",
        "ats": SAMPLES / "ats_json/andrej_karpathy.json",
        "github": SAMPLES / "github/andrej_karpathy.json",
        "notes": None,
    },
    "Custom uploads": {
        "csv": None,
        "ats": None,
        "github": None,
        "notes": None,
    },
}


def _read_uploaded_file(upload) -> str | None:
    """Decode one uploaded file into UTF-8 text."""
    if upload is None:
        return None
    return upload.getvalue().decode("utf-8")


def _bundle_payload(bundle_name: str, key: str, upload) -> str | None:
    """Resolve a payload from either a built-in sample bundle or a user upload."""
    if bundle_name == "Custom uploads":
        return _read_uploaded_file(upload)
    sample_path = _SAMPLE_BUNDLES[bundle_name][key]
    if sample_path is None:
        return None
    return read_text(sample_path)


def _render_summary(output: dict[str, object]) -> None:
    """Render a tiny, reviewer-friendly summary above the raw JSON."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Candidate ID", str(output.get("candidate_id", "")))
    col2.metric("Name", str(output.get("full_name", "")))
    overall = output.get("overall_confidence")
    col3.metric(
        "Overall confidence",
        f"{overall:.2f}" if isinstance(overall, (int, float)) else "n/a",
    )


st.set_page_config(
    page_title="Candidate Data Transformer Demo",
    layout="wide",
)

st.title("Candidate Data Transformer")
st.caption(
    "Thin demo wrapper over the existing extract -> normalize -> merge -> score -> project pipeline."
)

with st.sidebar:
    st.header("Inputs")
    bundle_name = st.selectbox("Choose a bundle", list(_SAMPLE_BUNDLES.keys()))
    include_audit = st.checkbox("Include audit log", value=True)
    config_upload = st.file_uploader(
        "Optional projection config (.json)",
        type=["json"],
        help="Upload a custom projection config, or leave empty to use the default assignment-aligned output.",
    )
    st.caption(
        "Built-in bundles are fastest for demo. Choose 'Custom uploads' to bring your own files."
    )

st.subheader("Sources")
col1, col2 = st.columns(2)

with col1:
    csv_upload = st.file_uploader("Recruiter CSV", type=["csv"])
    ats_upload = st.file_uploader("ATS JSON", type=["json"])

with col2:
    github_upload = st.file_uploader("GitHub JSON", type=["json"])
    notes_upload = st.file_uploader("Recruiter notes", type=["txt"])

if st.button("Transform", type="primary", use_container_width=True):
    try:
        result = run_pipeline(
            PipelineInputs(
                csv_payload=_bundle_payload(bundle_name, "csv", csv_upload),
                ats_payload=_bundle_payload(bundle_name, "ats", ats_upload),
                github_payload=_bundle_payload(bundle_name, "github", github_upload),
                notes_payload=_bundle_payload(bundle_name, "notes", notes_upload),
                config_payload=_read_uploaded_file(config_upload),
            ),
            include_audit=include_audit,
        )
    except Exception as exc:  # noqa: BLE001 - UI should surface the real failure
        st.error(str(exc))
    else:
        st.success("Transformation complete.")
        output = result["output"] if include_audit else result
        if isinstance(output, dict):
            _render_summary(output)

        data_col, audit_col = st.columns([3, 2])
        with data_col:
            st.markdown("### Output JSON")
            st.json(output, expanded=2)
        with audit_col:
            if include_audit:
                st.markdown("### Audit Log")
                st.json(result["audit_log"], expanded=False)
            else:
                st.markdown("### Audit Log")
                st.info("Enable 'Include audit log' in the sidebar to inspect pipeline decisions.")

        with st.expander("Raw JSON payload"):
            st.code(json.dumps(result, indent=2, sort_keys=True), language="json")
