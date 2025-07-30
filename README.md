# SimExR

A lightweight **framework** for turning natural‑language experiment descriptions into:  
1. A fully runnable simulation script  
2. Automated batch runs over parameter sweeps  
3. Interactive analysis via an LLM‑powered reasoning agent  

> **Why “Framework”?**  
> While you can already use SimExR as a standalone tool, its modular structure (parser → codegen → runner → reasoning) is designed for extension and customization. 
> We anticipate third‑party plugins, new backends, MCP servers and richer UIs—all hallmarks of a framework.

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/vash02/compModelExplore.git
cd compModelExplore
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure your OpenAI key

Create a file `config.yaml` (and add it to `.gitignore`):

<pre><code class="language-yaml">
openai:
  api_key: "sk-…"
</code></pre>

---

## 3. Run the App

<pre><code class="language-bash">
streamlit run app.py
</code></pre>

This will launch a three‑page UI:

---

### **Configure & Run**

- Write a natural‑language description of your system.
- Parse it into JSON metadata.
- Generate or edit the simulation code (`simulate.py`).
- Specify parameter sweeps with sliders.
- Run the batch and produce `results.csv` (and store in `mcp.db`).

---

### **Results**

- Browse the raw DataFrame.
- Download CSV of all parameter combinations and outcomes.

---

### **Analysis**

- Chat with the reasoning agent (GPT‑4.1) over your results.
- It will issue Python snippets under the hood, generate plots, and return JSON‑wrapped answers.
- View intermediate code steps (collapsed), final interpretations, and plots.

## **Reproducing an Experiment**

### Describe your experiment

<pre><code class="language-text">
Simulate a simple pendulum with equations d2θ/dt2 + (g/L) sin θ = 0.  
Vary L from 0.1m to 1.0m in 0.1m steps; measure period. Use g=9.81.  
</code></pre>

---

### Parse & generate

Click **Parse Metadata** → edit if needed → **Generate Code & Model** → **Approve Code**.

---

### Set ranges & run

- Sliders appear for any sweep variables.
- Click **Execute Batch** to run 100–100000 simulations.

---

### Inspect results

- Download `results.csv`.
- Or use the **Results** page to view in‑browser.

---

### Analyze

- Ask “Plot θ vs t for L = 0.5” or “What are the fixed points?”
- See code steps, saved plots, and natural‑language interpretations.

---

## Extending the Framework

- **Custom backends**: swap out the LLM engine (e.g. local Llama).
- **New parsers**: integrate a specialized metadata extractor.
- **Alternative UIs**: hook into Gradio, Dash, or a CLI.
- **Plugin system**: drop in new analysis tools or simulation cores.


## License & Acknowledgements

- **MIT License**
- Built on **Streamlit**, **OpenAI GPT‑4**, **Pandas**, **Matplotlib**
- Inspired by interactive computational notebooks and AI‑powered analysis assistants
