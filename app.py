# app.py — Streamlit UI for the full simulation→analysis pipeline

import json
from typing import Dict, Tuple

import streamlit as st
from pathlib import Path

from core.parser import parse_nl_input
from core.codegen import generate_code
from mcp_sim_tool.core.param_utils import (
    extract_param_settings,
    generate_param_grid,
)
from mcp_sim_tool.core.runner import run_batch
from core.agent_loop import ask
from db.schema import init_db
from db.store import get_simulation_script
from streamlit_chat import message

# ─────────────────────────────────────────────────────────────────────
# Initialize DB once
# ─────────────────────────────────────────────────────────────────────
init_db()
st.set_page_config(page_title="SimExplorer", layout="wide")

# ─────────────────────────────────────────────────────────────────────
# Persist state across reruns
# ─────────────────────────────────────────────────────────────────────
state = st.session_state
state.setdefault("metadata",        None)
state.setdefault("raw_metadata_json","")
state.setdefault("model_id",         None)
state.setdefault("script_code",      "")
state.setdefault("code_approved",   False)
state.setdefault("param_ranges",    {})
state.setdefault("ranges_set",      False)
state.setdefault("grid_size",       1000)
state.setdefault("last_df",         None)
state.setdefault("analysis_result", None)
state.setdefault("stop_requested",  False)
state.setdefault("analysis_history", [])
state.setdefault("chat_history", [])

# ─────────────────────────────────────────────────────────────────────
# Persist state across reruns
# ─────────────────────────────────────────────────────────────────────
state = st.session_state
for key, default in {
    "metadata": None,
    "raw_metadata_json": "",
    "model_id": None,
    "script_code": "",
    "code_approved": False,
    "param_ranges": {},
    "ranges_set": False,
    "grid_size": 1000,
    "last_df": None,
    "chat_history": [],
    "stop_requested": False
}.items():
    state.setdefault(key, default)

# ─────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────
page = st.sidebar.radio("Select page", ["Configure & Run", "Results", "Analysis"])

# ─────────────────────────────────────────────────────────────────────
# Page: Configure & Run
# ─────────────────────────────────────────────────────────────────────
if page == "Configure & Run":
    st.title("🔬 SimExplorer: Configure & Run")

    # 1) Describe & parse
    st.header("1. Describe Your Experiment")
    sim_query = st.text_area(
        "Natural-language description",
        height=150,
        placeholder="e.g. simple pendulum, vary L from 0.1 to 1.0 m, measure period…",
    )
    if st.button("🔍 Parse Metadata"):
        if not sim_query.strip():
            st.warning("Please enter a description first.")
        else:
            with st.spinner("Parsing…"):
                meta = parse_nl_input(sim_query, retries=3, temperature=0.0)
            state.metadata = meta
            state.raw_metadata_json = json.dumps(meta, indent=2)
            # reset downstream
            for k in ("model_id","script_code","code_approved",
                      "param_ranges","ranges_set","last_df","analysis_result"):
                state[k] = None
            st.success("Metadata parsed! Review below.")

    # 1b) review / edit
    if state.metadata:
        st.subheader("Parsed Metadata (editable)")
        st.json(state.metadata)
        edited = st.text_area(
            "Edit metadata JSON",
            value=state.raw_metadata_json,
            height=200,
            key="meta_edit_area"
        )
        if st.button("✏️ Update Metadata"):
            try:
                state.metadata = json.loads(edited)
                state.raw_metadata_json = edited
                for k in ("model_id","script_code","code_approved",
                          "param_ranges","ranges_set","last_df","analysis_result"):
                    state[k] = None
                st.success("Metadata updated.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    # 2) code generation
    if state.metadata and state.model_id is None:
        if st.button("▶️ Generate Code & Model"):
            with st.spinner("Generating code…"):
                state.model_id = generate_code(state.metadata)
            path_or_code = get_simulation_script(state.model_id)
            code_text    = Path(path_or_code).read_text() if Path(path_or_code).exists() else path_or_code
            state.script_code   = code_text
            state.code_approved = False
            st.success(f"Model `{state.model_id}` generated; review code below.")

    # 2b) review / approve code
    if state.script_code and not state.code_approved:
        st.header("Review & Edit Generated Code")
        edited_code = st.text_area(
            "simulate.py",
            value=state.script_code,
            height=300,
            key="script_edit_area"
        )
        c1, c2 = st.columns(2)
        if c1.button("✅ Approve Code"):
            state.script_code   = edited_code
            state.code_approved = True
            st.success("Code approved!")
        if c2.button("🔄 Regenerate Code"):
            with st.spinner("Regenerating…"):
                state.model_id = generate_code(state.metadata)
            p = Path(get_simulation_script(state.model_id))
            state.script_code = p.read_text() if p.exists() else "<error>"
            state.code_approved = False
            st.success(f"Model `{state.model_id}` regenerated.")
        st.code(edited_code, language="python")

    # 3) parameter ranges
    # ─────────────────────────────────────────────────────────────────────
    # Page: Configure & Run → Specify Parameter Ranges (updated)
    # ─────────────────────────────────────────────────────────────────────
    # 3) Specify Parameter Ranges
    if state.code_approved and not state.ranges_set:
        st.header("3. Specify Parameter Ranges")

        # 1) Pull out float settings and tuple‐ranges
        settings = extract_param_settings(state.metadata)

        # 2) Build the sweep list:
        #    start with metadata["vary_variable"], then auto‑add
        #    any param having an explicit start/end in metadata["parameters"]
        vary_list = list(state.metadata.get("vary_variable", []))
        for name, desc in state.metadata.get("parameters", {}).items():
            if isinstance(desc, dict) and "start" in desc and "end" in desc:
                if name not in vary_list:
                    vary_list.append(name)
                    settings[name] = (float(desc["start"]), float(desc["end"]))

        ranges: Dict[str, Tuple[float, float]] = {}

        # 3) Render sliders for each sweep param
        for param in vary_list:
            raw = settings.get(param, ())
            if isinstance(raw, tuple) and len(raw) == 2:
                init_lo, init_hi = raw
            else:
                init_lo, init_hi = 0.0, 1.0

            key = f"{param}_sld"
            default = st.session_state.get(key, (init_lo, init_hi))
            sel = st.slider(
                f"{param} range",
                min_value=-100.0,
                max_value=100.0,
                value=default,
                key=key
            )
            ranges[param] = sel

        # 4) Show all other parameters as fixed
        for param, val in settings.items():
            if param in vary_list:
                continue

            try:
                num = float(val)
            except:
                num = 0.0
            st.number_input(
                f"{param} (fixed)",
                value=num,
                disabled=True,
                key=f"{param}_fixed"
            )
            ranges[param] = (num, num)

        # 5) Commit
        if st.button("✅ Set Ranges"):
            state.param_ranges = ranges
            state.ranges_set = True
            st.success("Ranges saved—ready to run!")

    # 4) run
    if state.ranges_set and state.last_df is None:
        # print("------------", state.param_ranges)
        st.header("Run Simulations")
        state.grid_size = st.number_input("Grid size", 10, 100_000, state.grid_size, 10)
        if st.button("▶️ Execute Batch"):
            with st.spinner("Running batch…"):
                grid = generate_param_grid(state.param_ranges, total=state.grid_size)
                out_csv = Path(f"experiments/{state.model_id}_results.csv")
                out_csv.parent.mkdir(exist_ok=True)
                run_batch(model_id=state.model_id, param_grid=grid, output_csv=str(out_csv))
                import pandas as pd
                state.last_df = pd.read_csv(out_csv)
            st.success("Simulation complete!")

# ─────────────────────────────────────────────────────────────────────
# Page: Results
# ─────────────────────────────────────────────────────────────────────
elif page == "Results":
    st.title("📊 Results")
    if state.last_df is None:
        st.warning("No results yet. Run a simulation first.")
    else:
        st.write(f"Model `{state.model_id}` – {len(state.last_df)} rows")
        st.dataframe(state.last_df)
        csv_path = Path(f"experiments/{state.model_id}_results.csv")
        if csv_path.exists():
            st.download_button("Download CSV", csv_path.read_bytes(), csv_path.name)

# ─────────────────────────────────────────────────────────────────────
# Page: Analysis with streamlit-chat
# ─────────────────────────────────────────────────────────────────────
elif page == "Analysis":
    st.title("💬 Analysis Chat")

    from core.utils import extract_code_map
    from streamlit_chat import message

    # Stop button
    if st.button("🛑 Stop Reasoning", key="stop_btn"):
        state.stop_requested = True

    # No‑data guard
    if state.last_df is None:
        st.warning("No data to analyze. Run a simulation first.")
    else:
        # 1) Render the full history so far
        for i, msg in enumerate(state.chat_history):
            is_user = (msg["role"] == "user")
            message(msg["content"], is_user=is_user, key=f"hist_{i}")
            # If this was an assistant message and it had images, show them
            if not is_user and msg.get("images"):
                with st.expander("📈 Plot(s)", expanded=False):
                    for img in msg["images"]:
                        st.image(img, use_column_width=True)

        # 2) User input + retry
        user_q = st.chat_input("Ask a question…", key="analysis_q")
        last_q = next((m["content"] for m in reversed(state.chat_history)
                       if m["role"] == "user"), None)
        if last_q and st.button("↻ Retry Last Query", key="retry_btn"):
            user_q = last_q

        # 3) Only proceed if user typed something
        if user_q:
            # 3a) Echo & persist user question
            idx = len(state.chat_history)
            message(user_q, is_user=True, key=f"user_{idx}")
            state.chat_history.append({"role": "user", "content": user_q})

            # 3b) Show “thinking…”
            spinner = st.empty()
            spinner.markdown("*🧠 Thinking…*")

            # 3c) Call the agent
            result = ask(
                state.model_id,
                user_q,
                backend="openai",
                stop_flag=lambda: state.stop_requested
            )
            spinner.empty()
            state.stop_requested = False

            # 3d) Show only **this** query’s code expanders
            code_map = result.get("code_map", {}) or extract_code_map(result.get("history", []))
            for step_idx, code in code_map.items():
                with st.expander(f"▶️ Step {step_idx+1} Code", expanded=False):
                    st.code(code, language="python")

            # 3e) Parse out the JSON wrapper if present, then display the clean answer
            raw = result.get("answer", "(no answer)")
            try:
                payload = json.loads(raw)
                answer_text = payload.get("answer", raw)
            except json.JSONDecodeError:
                answer_text = raw

            message(answer_text, is_user=False, key=f"assist_{idx}")

            # 3f) Show plots under an expander and persist them
            images = result.get("images", [])
            if images:
                with st.expander("📈 Plot(s)", expanded=False):
                    for img in images:
                        st.image(img, use_column_width=True)

            # 3g) **Persist only** the clean answer text (no braces) + images
            state.chat_history.append({
                "role":   "assistant",
                "content": answer_text,
                "images":  images
            })
