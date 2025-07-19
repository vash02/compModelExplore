# app.py — Streamlit UI for the full simulation→analysis pipeline

import json
import streamlit as st
from pathlib import Path

from core.parser import parse_nl_input
from core.codegen import generate_code
from mcp_sim_tool.core.param_utils import (
    extract_ranges_from_prompt,
    generate_param_grid,
)
from mcp_sim_tool.core.runner import run_batch
from core.agent_loop import ask
from db.schema import init_db
from db.store import get_simulation_script

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
                meta = parse_nl_input(sim_query, retries=3, temperature=0.3)
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
    if state.code_approved and not state.ranges_set:
        st.header("Specify Parameter Ranges")
        auto = extract_ranges_from_prompt(state.metadata)
        st.write("Auto‑extracted:", auto or "None")
        ranges = {}
        for p in state.metadata["parameters"]:
            lo, hi = st.slider(f"{p}", -100.0, 100.0, value=(0.0,1.0), key=f"{p}_sld")
            ranges[p] = (lo, hi)
        if st.button("✅ Set Ranges"):
            state.param_ranges = ranges
            state.ranges_set = True
            st.success("Ranges set!")

    # 4) run
    if state.ranges_set and state.last_df is None:
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
# Page: Analysis
# ─────────────────────────────────────────────────────────────────────
elif page == "Analysis":
    st.title("💬 Analysis Chat")

    if state.last_df is None:
        st.warning("No data to analyze. Run a simulation first.")
        st.stop()

    # ───────────────────────────────────────────────────
    # 1) Render the existing chat
    # ───────────────────────────────────────────────────
    for msg in state.analysis_history:
        role = msg["role"]
        if msg.get("is_code"):
            st.chat_message(role).code(msg["content"], language="python")
        elif msg.get("is_tool"):
            payload = msg["payload"]
            # show stdout if any
            if payload.get("stdout"):
                st.chat_message("tool").write(payload["stdout"])
            # show returned value
            if payload.get("value") is not None:
                st.chat_message("tool").json({"value": payload["value"]})
            # show any new images
            for img in payload.get("images", []):
                st.chat_message("tool").image(img, use_column_width=True)
        elif msg.get("is_image"):
            st.chat_message(role).image(msg["content"], use_column_width=True)
        else:
            st.chat_message(role).markdown(msg["content"])

    # ───────────────────────────────────────────────────
    # 2) Accept a new user question
    # ───────────────────────────────────────────────────
    user_q = st.chat_input("Type your question here…")
    if user_q:
        # 2a) append the user
        state.analysis_history.append({"role": "user", "content": user_q})
        # 2b) call the agent
        with st.spinner("Thinking…"):
            result = ask(state.model_id, user_q, backend="openai")
        # 2c) render every step in result["history"]
        for step in result.get("history", []):
            if step["role"] == "assistant" and step.get("function_call"):
                # code snippet requested
                args = json.loads(step["function_call"]["arguments"])
                code = args.get("code", "")
                state.analysis_history.append({
                    "role": "assistant",
                    "content": code,
                    "is_code": True
                })
            elif step["role"] == "function":
                # tool output
                payload = json.loads(step["content"])
                state.analysis_history.append({
                    "role": "tool",
                    "payload": payload,
                    "is_tool": True
                })
            else:
                # plain assistant/user text
                state.analysis_history.append({
                    "role": step["role"],
                    "content": step["content"]
                })
        # 2d) final answer
        answer = result.get("answer", "")
        state.analysis_history.append({
            "role": "assistant", "content": answer
        })
        # 2e) any final images
        for img in result.get("images", []):
            state.analysis_history.append({
                "role": "assistant",
                "content": img,
                "is_image": True
            })
        # 2f) re-run so the new history shows up
        st.rerun()
