from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="Demo Launcher", layout="wide")
st.title("Demo Launcher")
st.write(
    "Primary UIs are now split by purpose: Chainlit chat for consultant simulation, "
    "and Streamlit testing/admin pages for evaluation and data inspection."
)

st.code("./scripts/start_demo.sh", language="bash")
st.markdown("- Chainlit Chat: [http://127.0.0.1:8501](http://127.0.0.1:8501)")
st.markdown("- Streamlit Testing UI: [http://127.0.0.1:8502](http://127.0.0.1:8502)")
st.markdown("- Backend API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)")
st.markdown("- Langfuse: [http://127.0.0.1:3000](http://127.0.0.1:3000)")
st.markdown("- Promptfoo Viewer: [http://127.0.0.1:15500](http://127.0.0.1:15500)")

st.info(
    "This launcher is optional convenience only. "
    "Use the standalone Chainlit + Streamlit apps for the demo."
)
