# ARIA — Automated Routing and Intelligent Assignment

An agentic AI pipeline that reads an IT support ticket description and predicts
the correct assignment group using semantic search over historical tickets and a
locally running LLM.

**No ServiceNow connection needed. Everything runs locally on your machine.**

---

## What It Does

When a support ticket comes in, someone has to manually read it and decide which
team handles it. With 125,000+ historical tickets across 23 assignment groups,
this is slow and inconsistent. ARIA automates this entirely — type a ticket
description, get an instant prediction with a confidence score.

---

## Architecture

```
User types short description
        │
        ▼
PreprocessingAgent
  Clean text, remove ticket IDs, normalise whitespace
        │
        ▼
EmbeddingAgent  (all-MiniLM-L6-v2 — runs locally, no API key)
  Convert text to 384-dimensional vector
        │
        ▼
KnowledgeBaseAgent  (ChromaDB)
  Search 125,000 historical tickets stored as vectors
  Return top 5 most semantically similar tickets
        │
        ▼
LLMAgent  (Gemma 2B via Ollama — runs locally)
  Read the 5 similar tickets and their assignment groups
  Reason about which group best fits the new ticket
  If LLM unavailable → weighted similarity vote fallback
        │
        ▼
Predicted Assignment Group
  + Confidence score (1–10)
  + Confidence label (HIGH / MEDIUM / LOW)
  + Which similar tickets influenced the decision
  + Similarity scores (1–10)
```

---

## Project Structure

```
HaviAgenticServiceAssignment/
│
├── predict.py                    ← Run this for interactive predictions
├── build_knowledge_base.py       ← Run once to load CSV into ChromaDB
├── install.py                    ← Install all dependencies
├── requirements.txt
│
├── agents/
│   ├── preprocessing_agent.py    ← Text cleaning
│   ├── embedding_agent.py        ← Sentence transformer embeddings
│   ├── knowledge_base_agent.py   ← ChromaDB build and search
│   └── llm_agent.py              ← Ollama LLM reasoning + fallback voting
│
├── config/
│   └── config.yaml               ← Model settings, assignment groups
│
└── data/
    ├── training_tickets.csv      ← Your ServiceNow CSV export (you provide this)
    └── chroma_db/                ← ChromaDB vector storage (auto-created)
```

---

## Requirements

- Python 3.10 or above
- Windows / macOS / Linux
- 8 GB RAM minimum (16 GB recommended)
- ~3 GB free disk space (model + vector DB)

---

## Installation

### Step 1 — Install Python dependencies

```
python install.py
```

Or manually:

```
pip install -r requirements.txt
```

### Step 2 — Set HuggingFace token (required for model download)

The embedding model downloads from HuggingFace on first run. A free token
is required to avoid rate limiting.

1. Sign up free at https://huggingface.co
2. Go to **Settings → Access Tokens → New token** (Read access)
3. Copy the token (starts with `hf_...`)

**Windows:**
```
setx HF_TOKEN "hf_your_token_here"
```
Then close and reopen your command prompt.

**macOS / Linux:**
```
export HF_TOKEN="hf_your_token_here"
```

### Step 3 — Install Ollama (local LLM runtime)

Download and install from https://ollama.com

Then pull the model:

```
ollama pull gemma:2b
```

> If you have more RAM available, `mistral` gives better accuracy:
> `ollama pull mistral`

---

## Setup Steps

### Step 1 — Prepare your CSV

Place your ServiceNow export at: `data/training_tickets.csv`

The CSV must have exactly these column names:

| Column | Required | Description |
|---|---|---|
| `Short Description` | Yes | Ticket subject — main signal |
| `Description` | Yes | Ticket body — additional context |
| `Assignment Team` | Yes | Which team resolved it — the label |

**ServiceNow export query:**
```
Table   : incident
Filter  : resolved_at in the last 12 months
          AND assignment_group IS NOT EMPTY
          AND state IN (Resolved, Closed)
Fields  : short_description, description, assignment_group
Format  : CSV
```

Rename columns after export:
```
short_description  →  Short Description
description        →  Description
assignment_group   →  Assignment Team
```

### Step 2 — Update assignment groups in config

Edit `config/config.yaml` and update the `assignment_groups` list to match
your exact ServiceNow group names:

```yaml
assignment_groups:
  - "IT-Service Desk"
  - "IT-Portal-Central"
  - "IT-Wintel Support"
  - "IT-SC GBS App Support"
  # ... add all your groups here
```

> **Note:** Adding a new group later? Just add it to this list and run
> `predict.py` — no rebuild needed, as long as tickets for that group are
> already in the CSV that was used for the knowledge base build.

### Step 3 — Build the knowledge base

The knowledge base is built **once** and saved to disk. This is the most
time-consuming step (~25–35 minutes for 125,000 tickets on an 8GB machine).

**On machines with limited RAM, load in batches of 10,000:**

```
python build_knowledge_base.py --start 0     --end 10000
python build_knowledge_base.py --start 10000 --end 20000
python build_knowledge_base.py --start 20000 --end 30000
```

Each run appends to the existing knowledge base — nothing is overwritten.
The script will print the exact next command to run at the end of each batch.

**On machines with plenty of RAM, load everything at once:**

```
python build_knowledge_base.py --start 0 --end 125000
```

**To wipe and rebuild from scratch:**

```
python build_knowledge_base.py --rebuild
```

### Step 4 — Start predicting

```
python predict.py
```

---

## Usage

### Interactive mode

```
python predict.py
```

Example output:

```
============================================================
  ARIA -- Automated Routing and Intelligent Assignment
============================================================

  Embedding model  : all-MiniLM-L6-v2
  LLM model        : gemma:2b via Ollama
  Assignment groups: 23
  Knowledge base   : 125000 tickets loaded
  LLM              : gemma:2b (Ollama running)

  Type a ticket short description to get the assignment group.
  Type 'exit' or press Ctrl+C to quit.

  --------------------------------------------------------
  Ticket description: Issue with HaviConnect website

  ========================================================
  PREDICTION RESULT
  ========================================================

  Ticket       : Issue with HaviConnect website
  Assignment   : IT-Portal-Central
  Confidence   : HIGH  |  Score: 8/10  (4 of 5 similar tickets matched)

  Similar historical tickets used:

  Rank  Short Description                    Assignment Group       Similarity (1-10)
  ----- ------------------------------------ ---------------------- -----------------
  1.    Issue with HaviConnect Website       IT-Portal-Central  <-- 8.9
  2.    I can't access the havi connect...   IT-Service Desk        8.1
  3.    HAVI connect issue                   IT-Portal-Central  <-- 7.6
  4.    Problem with haviconnect app...      IT-Portal-Central  <-- 7.5
  5.    HaviConnect Be                       IT-Portal-Central  <-- 7.2

  ========================================================
```

### Single prediction mode (for scripting or testing)

```
python predict.py --once "SAP workflow approval stuck"
```

---

## Confidence Score

Every prediction includes a score from **1 to 10** and a label:

| Score | Label | Meaning |
|---|---|---|
| 7 – 10 | HIGH | Strong agreement — high confidence in prediction |
| 4 – 6 | MEDIUM | Some agreement — prediction likely correct |
| 1 – 3 | LOW | Split evidence — recommend human review |

The score is calculated by **weighted similarity voting** — each of the top 5
similar tickets votes for its assignment group, weighted by its similarity score.
The winning group's share of total weighted votes maps to the 1–10 scale.

---

## Models & Tools

| Component | Model / Tool | Purpose |
|---|---|---|
| Embedding | `all-MiniLM-L6-v2` | Converts ticket text to 384-dim vectors |
| Vector DB | `ChromaDB` | Stores and searches 125k ticket vectors |
| LLM | `Gemma 2B via Ollama` | Reasons over similar tickets to predict group |

### Embedding model — `all-MiniLM-L6-v2`
- Made by Microsoft, free on HuggingFace
- Downloads once (~90 MB), cached locally after that
- Runs fully locally — no internet needed after first download

### LLM — `Gemma 2B via Ollama`
- Made by Google, runs locally via Ollama
- No API calls, no cloud costs, no data leaving the machine
- Requires ~2 GB RAM

### Available LLM options

| Model | RAM Required | Notes |
|---|---|---|
| `gemma:2b` | ~2 GB | Default — good balance |
| `mistral` | ~4 GB | Best accuracy |
| `llama3.2` | ~2 GB | Good alternative |
| `phi3` | ~1.5 GB | Lightest option |

Change the model in `config/config.yaml` then run `ollama pull <model-name>`.

---

## Configuration Reference

`config/config.yaml`:

```yaml
embedding:
  model: "all-MiniLM-L6-v2"      # downloads once (~90MB), cached locally

vector_db:
  path: "data/chroma_db"          # where ChromaDB stores its files
  collection: "snow_tickets"      # collection name inside ChromaDB
  top_k: 5                        # how many similar tickets to retrieve

llm:
  provider: "ollama"
  model: "gemma:2b"               # change to "mistral" for better accuracy
  temperature: 0.1                # low = more consistent, deterministic output
  max_tokens: 512

data:
  csv_path: "data/training_tickets.csv"

assignment_groups:
  - "IT-SC-EPAM-SAP-AMS-Support"
  - "IT-SC-EPAM-SAP Basis Support"
  - "IT-SC-EPAM-SAP Workflow"
  - "IT-SC-EPAM-SAP Hybris CRM Support"
  - "IT-SC GBS App Support"
  - "IT-Unix/Linux Support"
  - "IT-SC Operations Application Support"
  - "IT-INFOR-WMS"
  - "IT-Azure RBAC Team"
  - "IT-ERP JDE Technical"
  - "IT-Service Desk"
  - "IT-Reporting Services"
  - "IT-Wintel Support"
  - "IT-HMDP Support"
  - "IT-Portal-Central"
  - "IT-Digital-Delivery"
  - "IT-SCT Integrations COE team"
  - "IT-Messaging Support"
  - "HAVI Digital Workplace"
  - "IT-tms-I&O-Identity-Collab-End Point Support"
  - "IT-Intune Support"
  - "IT-AVD Support"
  - "IT-OB Operations"
```

---

## Adding New Assignment Groups

If the new group's tickets are **already in the CSV that was used to build
the knowledge base:**

1. Add the group name to `config/config.yaml` under `assignment_groups`
2. Run `predict.py` — done. No rebuild needed.

If the new group's tickets are **not yet in the knowledge base:**

1. Add the new tickets to `data/training_tickets.csv`
2. Add the group name to `config/config.yaml`
3. Rebuild: `python build_knowledge_base.py --rebuild`

---

## Fallback Behaviour

If Ollama is not running or the LLM is unavailable, the system automatically
falls back to **weighted similarity voting**:

- Each of the top 5 similar tickets votes for its assignment group
- Each vote is weighted by that ticket's similarity score
- The group with the highest total weighted score wins
- The output shows `Note: LLM unavailable - used weighted similarity vote`

Predictions still work correctly in fallback mode.

---

## Troubleshooting

**`memory allocation failed` during build**

Use the incremental batch approach to load 10,000 rows at a time:
```
python build_knowledge_base.py --start 0 --end 10000
```

**`401 Unauthorized` when downloading embedding model**

Set your HuggingFace token (see Installation Step 2), then close and reopen
your command prompt before running again. Also delete the corrupt cache:
```
# Windows
rmdir /s /q C:\Users\<YourUsername>\.cache\huggingface

# macOS / Linux
rm -rf ~/.cache/huggingface
```

**Knowledge base is empty after build**

Check CSV column names match exactly (case-sensitive):
- `Short Description`
- `Description`
- `Assignment Team`

**LLM not found or Ollama not running**

```
ollama serve
ollama list
ollama pull gemma:2b
```

**Prediction returns wrong group**

Check that the group name in `config/config.yaml` exactly matches the
`Assignment Team` value in your CSV — spelling, spacing, and capitalisation
must be identical.

**Build is slow**

Increase batch size if RAM allows. Edit `build_knowledge_base.py` and change:
```python
parser.add_argument("--end", type=int, default=10000, ...)
```
Use `--end 20000` or `--end 50000` per run if the machine has enough free RAM.

---

## Key Benefits

- **Runs entirely locally** — no cloud costs, no data leaving the network
- **Instant predictions** — under 1 second per ticket after setup
- **One-time knowledge base build** — saved to disk, reused forever
- **Explainable** — every prediction shows which historical tickets influenced it
- **Fallback safe** — works correctly even without the LLM
- **Incremental loading** — build the knowledge base in batches on low-RAM machines
- **Scalable** — more historical data means more accurate predictions
