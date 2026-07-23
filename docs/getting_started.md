# Getting Started (From Zero)

This is a complete, beginner-friendly walkthrough that assumes you have **nothing
installed** yet. By the end you will be able to:

- install Python and the framework,
- turn a CSV / JSON / text file into a knowledge graph with the core (offline) install,
- optionally set up **Ollama** (local LLM extraction), **Neo4j** (graph database),
  **Kuzu** (embedded graph), and **GLiNER** (model-based entity extraction), and
- verify everything with the built-in `doctor` command.

You do **not** need any of the optional services to try the framework. The core
install runs fully offline. Add optional pieces only when you need them.

> Want to understand how the framework forms graphs? See
> [`knowledge_graph_modelling.md`](knowledge_graph_modelling.md).

---

## Table of contents

1. [Step 1 — Install the basics (Python, git)](#step-1--install-the-basics-python-git)
2. [Step 2 — Get the code and create a virtual environment](#step-2--get-the-code-and-create-a-virtual-environment)
3. [Step 3 — Install the framework (core, offline)](#step-3--install-the-framework-core-offline)
4. [Step 4 — Run your first knowledge graph](#step-4--run-your-first-knowledge-graph)
5. [Step 5 — Check your setup with `doctor`](#step-5--check-your-setup-with-doctor)
6. [Which install do I need? (extras map)](#which-install-do-i-need-extras-map)
7. [Optional service — Ollama (local LLM extraction)](#optional-service--ollama-local-llm-extraction)
8. [Optional service — Neo4j (graph database ingestion)](#optional-service--neo4j-graph-database-ingestion)
9. [Optional service — Kuzu (embedded graph, no server)](#optional-service--kuzu-embedded-graph-no-server)
10. [Optional — GLiNER and transformer models](#optional--gliner-and-transformer-models)
11. [Environment variables and the `.env` file](#environment-variables-and-the-env-file)
12. [Running everything (CLI, Python, notebooks)](#running-everything-cli-python-notebooks)
13. [Troubleshooting](#troubleshooting)

---

## Step 1 — Install the basics (Python, git)

You need **Python 3.11 or 3.12** and **git**. Pick your operating system.

### macOS

Install [Homebrew](https://brew.sh) first if you do not have it, then:

```bash
brew install python@3.12 git
python3.12 --version   # should print Python 3.12.x
```

### Ubuntu / Debian Linux

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip git
# If your distro does not ship 3.12, python3.11 works too:
# sudo apt install -y python3.11 python3.11-venv python3-pip git
python3.12 --version
```

### Windows

1. Install **Python 3.12** from <https://www.python.org/downloads/windows/>.
   On the first installer screen, tick **“Add python.exe to PATH.”**
2. Install **Git** from <https://git-scm.com/download/win>.
3. Use **PowerShell** for the commands below. Where this guide shows
   `source .venv/bin/activate`, use `.\.venv\Scripts\Activate.ps1` instead.

> The `make` targets in this guide are a convenience for macOS/Linux. On Windows,
> either use **WSL** (Windows Subsystem for Linux) or run the underlying `python`
> commands directly — nothing in the core workflow requires `make`.

---

## Step 2 — Get the code and set it up

```bash
git clone https://github.com/swaroop-kaza/knowledge_graphs.git
cd knowledge_graphs
```

### The one-command path (recommended)

```bash
make setup
```

`make setup` creates an isolated virtual environment in `.venv`, installs the
package into it (core, fully offline), and runs a quick health check. You do **not**
need to create or activate a virtual environment yourself.

> **Why a virtual environment?** On modern macOS (Homebrew) and Debian/Ubuntu, the
> system Python is “externally managed”
> ([PEP 668](https://peps.python.org/pep-0668/)), so a bare `pip install` outside a
> virtual environment fails on purpose with an `externally-managed-environment`
> error. `make setup` avoids this entirely by always installing into `.venv`.

### Manual path (or Windows, where `make` may be unavailable)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### Optional: the fast path with `uv`

If you use [uv](https://docs.astral.sh/uv/) (a fast Python package/venv manager):

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

The core install pulls only lightweight, pure-Python dependencies (pydantic,
PyYAML, jsonpath-ng, openpyxl, SQLAlchemy, pypdf, rapidfuzz, rdflib, pyshacl).
**No database, model, or internet service is required.**

### Just the CLI, without cloning (advanced)

If you only want the `kg-construction` command to run on **your own** files — and do
not need the bundled examples, configs, or notebooks — you can install straight from
GitHub without cloning, using [pipx](https://pipx.pypa.io) or
[uv](https://docs.astral.sh/uv/):

```bash
# Install the CLI into its own isolated environment:
pipx install "git+https://github.com/swaroop-kaza/knowledge_graphs.git"

# ...or run it once without installing anything permanently:
uvx --from "git+https://github.com/swaroop-kaza/knowledge_graphs.git" kg-construction doctor
```

Then point it at your data:

```bash
kg-construction --input /path/to/your.csv --output graph.json
```

Caveats:

- This installs the **core** framework only. To include an optional extra, use a
  PEP 508 direct reference, e.g.
  `pipx install "kg-construction[graph] @ git+https://github.com/swaroop-kaza/knowledge_graphs.git"`.
- The `examples/`, `configs/`, and `notebooks/` referenced throughout this guide ship
  only in a clone, so the walkthrough commands that use them need `git clone`.
- The repository must be publicly reachable (or your git credentials configured).
- The package is **not yet published to PyPI**, so `pip install kg-construction` by
  name does not work yet — use the git URL above.

---

## Step 3 — What you just installed

The install puts four command-line tools onto your `PATH` (available once `.venv`
is active — `source .venv/bin/activate`):

| Command | What it does |
| --- | --- |
| `kg-construction` | Build a knowledge graph from a source file. |
| `kg-construction doctor` | Diagnose your install and optional services. |

Handy `make` shortcuts wrap the common tasks so they work whether or not the venv is
activated (`make` uses `.venv` automatically): `make quickstart`, `make doctor`,
`make test`, `make verify`, `make quality`. Run `make help` for the full list.

---

## Step 4 — Run your first knowledge graph

The 60-second quickstart builds graphs from the bundled sample files and prints a
summary. It runs entirely offline:

```bash
make quickstart
# equivalent to: python3 examples/quickstart_60_second.py
```

Or from the command line, turn a text file into property-graph, RDF, and evidence
outputs:

```bash
kg-construction \
  --input examples/data_to_kg/sample.txt \
  --source-type text \
  --output data/processed/data_to_kg_text_property_graph.json \
  --rdf-output data/processed/data_to_kg_text_rdf.json \
  --evidence-output data/processed/data_to_kg_text_evidence_graph.json
```

Or from Python:

```python
from kg_construction import construct_kg

result = construct_kg("examples/data_to_kg/sample.csv")
print(len(result.property_graph.nodes), len(result.property_graph.edges))
```

> **Do I need `PYTHONPATH=src`?** No. After `pip install -e .`, the
> `kg-construction` command, `import kg_construction`, and
> `python3 -m kg_construction.interfaces.cli.construct` all work directly. You may
> still see `PYTHONPATH=src python3 -m ...` in some older docs — that form runs the
> code straight from the `src/` tree and also works, but it is not required.

---

## Step 5 — Check your setup with `doctor`

`doctor` is a read-only health check. It never changes anything and reports missing
optional services as **skipped** rather than errors.

```bash
make doctor                            # activation-free (uses .venv automatically)

# ...or, with the virtual environment activated (source .venv/bin/activate):
kg-construction doctor                 # human-readable report
kg-construction doctor --json          # machine-readable
kg-construction doctor --skip-services # skip Ollama/Neo4j/Kuzu probes
```

It checks: Python version, the installed package, core imports, the public API,
which optional extras are installed, an RDF/SHACL smoke validation, notebook
dependencies, every YAML config under `configs/`, and — unless `--skip-services` —
whether Ollama, Neo4j, and Kuzu are reachable.

Run it now: a fresh core install should show core checks **OK** and optional
services **SKIP**. That is expected and healthy.

---

## Which install do I need? (extras map)

Install optional dependency groups (“extras”) only when you need the capability.
Syntax: `python -m pip install -e ".[name1,name2]"`.

| Extra | Install when you want to… | Notable packages | External service? |
| --- | --- | --- | --- |
| *(core)* | Load CSV/JSON/text/PDF/Excel/SQL, map, extract with rules, build property/RDF/evidence graphs, export Cypher/RDF | pydantic, rdflib, pyshacl | none |
| `dev` | Run tests, linting, type checks, build the package | pytest, ruff, mypy, build | none |
| `rdf` | Extra RDF/JSON-LD features | rdflib, pyshacl, pyld | none |
| `graph` | Ingest into **Neo4j**/**Kuzu**, use NetworkX, numpy | neo4j, kuzu, networkx, numpy | Neo4j (Kuzu is embedded) |
| `notebook` | Run the Jupyter notebooks | jupyter, ipykernel, nbformat | none |
| `gliner` | Model-based zero-shot entity extraction | gliner (pulls torch, transformers) | downloads a model once |
| `llm` | LangChain-based Ollama / Neo4j integrations | langchain-ollama, langchain-neo4j | Ollama |
| `bert` | Transformer NER / REBEL relation extraction | transformers, torch | downloads models |
| `linking` | Embedding-based entity linking | sentence-transformers | downloads models |
| `dedup` | Probabilistic / active-learning entity resolution | splink, dedupe, datasketch | none |
| `ml` | Graph ML: embeddings, link prediction, communities | torch, torch-geometric, pykeen, igraph, leidenalg | none |

You can combine them, e.g. the full developer + notebook environment:

```bash
python -m pip install -e ".[dev,rdf,graph,notebook]"
```

> **Note on the built-in Ollama extractor:** it talks to Ollama over plain HTTP and
> needs **no** extra Python packages — just the Ollama server and a pulled model
> (see the next section). The `[llm]` extra is only for the LangChain-based
> integrations.

---

## Optional service — Ollama (local LLM extraction)

[Ollama](https://ollama.com) runs open LLMs locally so the framework can do
LLM-assisted entity/relation extraction and label proposal — no cloud, no API key.
The default model is **`llama3.1`**.

### 1. Install Ollama

**macOS:**
```bash
brew install ollama
```
…or download the app from <https://ollama.com/download>.

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** download and run the installer from <https://ollama.com/download>.
It runs as a background service.

### 2. Start the server

The macOS app and the Windows/Linux service usually start Ollama automatically. If
not, start it manually (leave it running in its own terminal):

```bash
ollama serve
```

By default it listens on `http://localhost:11434`.

### 3. Pull the model

```bash
ollama pull llama3.1
```

This downloads several gigabytes once. You can use a different model
(`ollama pull mistral`, etc.) and point the framework at it via the config YAML
(`llm.model`) — see `configs/ollama_gliner_label_proposal.example.yaml`.

### 4. Tell the framework where Ollama is (optional)

The default is already `http://localhost:11434`. Override only if you run Ollama
elsewhere:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
```

### 5. Verify

```bash
kg-construction doctor            # the "ollama" service check should now be OK
# or directly:
curl http://localhost:11434/api/tags
```

### 6. Use it

Build a graph with an Ollama-backed extraction technique using the example config:

```bash
kg-construction \
  --input examples/data_to_kg/sample.txt \
  --source-type text \
  --config configs/ollama_gliner_label_proposal.example.yaml \
  --output data/processed/ollama_text_property_graph.json
```

If Ollama is not running, LLM-backed extraction is reported as skipped while the
deterministic techniques still run — the framework never silently fabricates output.

---

## Optional service — Neo4j (graph database ingestion)

You only need Neo4j to **load graphs into a live database**. The framework can
already **export** Neo4j Cypher with the core install (no database needed). Live
ingestion additionally requires the `[graph]` extra (which installs the `neo4j`
Python driver) plus a running database.

```bash
python -m pip install -e ".[graph]"
```

Then pick one way to run Neo4j:

### Option A — Docker (fastest)

```bash
docker run --name kg-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/please-change-me \
  neo4j:5
```

- Browser UI: <http://localhost:7474>
- Bolt endpoint (used by the driver): `bolt://localhost:7687`

### Option B — Neo4j Desktop

Install [Neo4j Desktop](https://neo4j.com/download/), create a local database, set a
password, and start it. Note its Bolt URI (usually `bolt://localhost:7687`).

### Option C — Neo4j Aura (cloud, no local install)

Create a free instance at <https://neo4j.com/cloud/aura/>. Use the connection URI it
gives you (starts with `neo4j+s://`).

### Configure the connection

The framework reads these from your environment (see the `.env` note below):

```bash
export NEO4J_URI=bolt://localhost:7687      # or your Aura neo4j+s://... URI
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=please-change-me
export NEO4J_DATABASE=neo4j
```

### Verify

```bash
kg-construction doctor   # the "neo4j" service check should now be OK
```

`doctor` reports **SKIP** if `NEO4J_URI` is unset, **WARN** if the URI/user are set
but the password or connection is missing, and **OK** once connectivity is verified.

---

## Optional service — Kuzu (embedded graph, no server)

[Kuzu](https://kuzudb.com) is an **embedded** graph database — it is just a Python
package with an on-disk file, so there is no server to run.

```bash
python -m pip install -e ".[graph]"
```

Kuzu is used as an embedded graph store: point ingestion at a database directory with
`--kuzu-database data/processed/kuzu_graph`. `kg-construction doctor` runs a Kuzu smoke
query automatically when the package is installed.

---

## Optional — GLiNER and transformer models

**GLiNER** does zero-shot, model-based entity extraction (no rules or gazetteers
needed). The default model is **`gliner-community/gliner_small-v2.5`**.

```bash
python -m pip install -e ".[gliner]"
```

Notes:

- This pulls **PyTorch and Transformers**, which are large downloads (hundreds of
  MB to a few GB). Installing may take several minutes.
- The **first run downloads the model** from Hugging Face, so you need internet
  access once. After that it is cached locally (`~/.cache/huggingface`).
- It runs on **CPU** out of the box (slower); a CUDA GPU speeds it up but is not
  required.
- The `[bert]` and `[linking]` extras behave the same way (transformer + first-run
  model download). Install them only for REBEL/transformer relation extraction or
  embedding-based linking respectively.

Verify GLiNER is importable:

```bash
kg-construction doctor   # "installed_extras" should list gliner
```

A common high-quality combination is **Ollama label proposal → GLiNER extraction**,
wired up in `configs/ollama_gliner_label_proposal.example.yaml`.

---

## Environment variables and the `.env` file

A template lives at [`.env.example`](../.env.example):

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j
KG_CONSTRUCTION_LOG_LEVEL=INFO
```

Important details:

- **The framework does not auto-load `.env`.** It reads variables from your actual
  shell environment. Copy the template and export the values:

  ```bash
  cp .env.example .env
  # edit .env to add your real Neo4j password, then load it into the shell:
  set -a; source .env; set +a
  ```

  (On Windows PowerShell, set each variable with `$Env:NEO4J_PASSWORD = "..."`.)

- `OLLAMA_BASE_URL`, `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` are read by the
  framework/`doctor`. `OLLAMA_MODEL` is a convenience placeholder for your own
  scripts — the model actually used comes from the config YAML (`llm.model`) or
  `OllamaConfig`, defaulting to `llama3.1`.
- **Never commit real secrets.** `.env` is git-ignored; only `.env.example` (with
  empty/placeholder values) is committed.

---

## Running everything (CLI, Python, notebooks)

### Build a graph (CLI)

```bash
kg-construction \
  --input examples/data_to_kg/sample.txt \
  --source-type text \
  --output data/processed/data_to_kg_text_property_graph.json \
  --rdf-output data/processed/data_to_kg_text_rdf.json \
  --evidence-output data/processed/data_to_kg_text_evidence_graph.json \
  --neo4j-output data/processed/data_to_kg_text_neo4j.cypher
```

### Python API

```python
from kg_construction import construct_kg, load_sources, profile_dataset

sources = load_sources("examples/data_to_kg/sample.csv")
profile = profile_dataset(sources)

result = construct_kg("examples/data_to_kg/sample.csv")
print(len(result.property_graph.nodes), len(result.property_graph.edges))
```

### Notebooks

```bash
python -m pip install -e ".[dev,rdf,graph,notebook]"
jupyter lab
```

Start with `notebooks/01_data_loading_and_profiling.ipynb` and work upward.

### Verify the whole install

```bash
kg-construction doctor
make verify          # runs unit tests + a construction smoke flow (macOS/Linux)
```

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `command not found: kg-construction` | The virtual environment is not active, or `pip install -e .` was not run. Run `source .venv/bin/activate` then `python -m pip install -e .`. |
| `ModuleNotFoundError: kg_construction` when running a script | Run from the repo root and prefix with `PYTHONPATH=src`, or use the `kg-construction` console command. |
| `doctor` shows `ollama … not reachable` | Ollama is not running. Start it (`ollama serve`) and `ollama pull llama3.1`. Optional unless you use LLM techniques. |
| `doctor` shows Neo4j **WARN** | `NEO4J_URI`/`NEO4J_USER` are set but `NEO4J_PASSWORD` is missing or the DB is down. Export the password and confirm the database is running. |
| `pip install ".[gliner]"` / `".[ml]"` is slow or fails | These pull PyTorch. Ensure a recent `pip` (`python -m pip install --upgrade pip`) and enough disk space; on some platforms install `torch` first per the [PyTorch instructions](https://pytorch.org/get-started/locally/). |
| An LLM/GLiNER extraction step is reported as skipped | Its optional dependency or service is not configured. Configure it (this guide) or ignore it — the deterministic techniques still run. |
| `make: command not found` (Windows) | `make` is optional. Use WSL, or run the `python`/`kg-construction` commands directly. |

---

## Where to go next

- [`knowledge_graph_modelling.md`](knowledge_graph_modelling.md) — how the framework forms graphs: the three modelling pillars and their code entry points.
- [`../README.md`](../README.md) — project overview and quickstart.
