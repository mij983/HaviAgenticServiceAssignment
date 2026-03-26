# ARIA - Automated Routing and Intelligent Assignment

An agentic AI pipeline that takes a ticket short description as input
and predicts the correct assignment group using semantic search over
historical tickets **and KB articles / documents**, combined with a
locally running LLM.

No ServiceNow connection needed. Runs entirely on your machine.

---

## Architecture

```
User types short description
        |
        v
Preprocessing Agent
  Clean text, remove ticket IDs, normalise whitespace
        |
        v
Embedding Agent  (sentence-transformers all-MiniLM-L6-v2)
  Convert text to 384-dimensional vector. Runs locally. No API key.
        |
        v
ChromaDB Vector Search
  Search historical tickets AND KB articles stored as embeddings
  Return top 5 most semantically similar results (tickets + docs mixed)
        |
        v
LLM Agent  (Mistral or Gemma via Ollama - runs locally)
  Read similar tickets AND relevant KB article chunks
  Reason about which group best fits the new ticket
        |
        v
Predicted Assignment Group
  + confidence level
  + which similar tickets / KB articles matched
  + similarity scores
  + source type ([csv] ticket or [doc] article)
```

---

## Project Structure

```
aria/
|
|-- predict.py                          <- Run this for the interactive prompt
|-- build_knowledge_base.py             <- Load CSV tickets into ChromaDB
|-- build_knowledge_base_docs.py        <- Load KB articles into ChromaDB  (NEW)
|
|-- agents/
|   |-- preprocessing_agent.py         <- Text cleaning
|   |-- embedding_agent.py             <- Sentence transformer embeddings
|   |-- knowledge_base_agent.py        <- ChromaDB build and search
|   |-- llm_agent.py                   <- Ollama LLM reasoning
|   |-- document_ingestion_agent.py    <- KB document reader & chunker  (NEW)
|
|-- config/
|   |-- config.yaml                    <- Model settings, assignment groups
|
|-- data/
|   |-- training_tickets.csv           <- Your 1-year ServiceNow CSV export
|   |-- kb_docs/                       <- KB articles folder  (NEW)
|   |   |-- vpn_guide.md
|   |   |-- sap_access_faq.txt
|   |   |-- network_troubleshooting.pdf
|   |   |-- IT-Service Desk/           <- Sub-folder = auto team tag
|   |       |-- password_reset.md
|   |-- chroma_db/                     <- ChromaDB storage (auto-created)
|
|-- requirements.txt
```

---

## Requirements

### Python Version

Python 3.10 or above required.

### Install Ollama (local LLM runtime)

Ollama runs the LLM entirely on your machine. No API key needed.

Download from https://ollama.com and install for your OS.

Then pull the model:

```
ollama pull mistral
```

Or for a lighter model (less RAM):

```
ollama pull gemma:2b
```

### Install Python packages

```
pip install -r requirements.txt
```

Full list:
```
sentence-transformers   sentence embedding model
faiss-cpu               vector similarity (used internally)
chromadb                persistent vector database
ollama                  Python client for local Ollama LLM
pandas                  CSV reading
numpy                   numerical operations
pyyaml                  config file reading
pypdf                   PDF document support  (NEW, optional)
beautifulsoup4          HTML document support (NEW, optional)
```

---

## Setup Steps

### Step 1 - Prepare your CSV

Your CSV must have these columns:

| Column            | Required | Description                       |
|-------------------|----------|-----------------------------------|
| Short Description | Yes      | Ticket subject - main signal      |
| Description       | Yes      | Ticket body - additional context  |
| Assignment Team   | Yes      | Which team resolved it - label    |

Place it at:  data/training_tickets.csv

For production, export 1 year of resolved tickets from ServiceNow:

```
Table   : incident
Filter  : resolved_at in the last 12 months
          AND assignment_group IS NOT EMPTY
          AND state IN (Resolved, Closed)
Fields  : short_description, description, assignment_group
Format  : CSV
```

Rename the columns:
```
short_description -> Short Description
description       -> Description
assignment_group  -> Assignment Team
```

### Step 2 - Update assignment groups in config

Edit config/config.yaml and update the assignment_groups list to match
your exact ServiceNow group names.

### Step 3 - Build the knowledge base from CSV

Run once. This embeds all CSV tickets into ChromaDB.

```
python build_knowledge_base.py
```

Supports incremental loading in chunks of 10,000 rows:

```
python build_knowledge_base.py --start 0     --end 10000
python build_knowledge_base.py --start 10000 --end 20000
```

To wipe and rebuild from scratch:

```
python build_knowledge_base.py --rebuild
```

### Step 4 - Add KB Articles / Documents  (NEW)

Place your knowledge-base articles and runbooks in data/kb_docs/.
Supported formats: .txt  .md  .pdf  .html

```
data/kb_docs/
  vpn_troubleshooting.md
  sap_password_reset.txt
  network_guide.pdf
  IT-Service Desk/          <- folder name = auto team tag
    onboarding_checklist.md
```

#### Optional: Tag documents with the target team

Add a team tag on the first line of any document:

```
TEAM: IT-Network Support
VPN Troubleshooting Guide
...
```

Recognised tag formats:
```
TEAM: IT-Service Desk
Assignment Group: IT-Network Support
Group: IT-Wintel Support
```

Or place the file in a sub-folder named after the team — the folder
name is used as the team tag automatically.

#### Ingest documents into ChromaDB:

```
python build_knowledge_base_docs.py
```

With a custom folder:
```
python build_knowledge_base_docs.py --docs-path path/to/articles
```

Wipe existing document chunks and re-ingest:
```
python build_knowledge_base_docs.py --rebuild-docs
```

List what is currently stored:
```
python build_knowledge_base_docs.py --list
```

Documents and CSV tickets share the same ChromaDB collection. Both
are returned in search results, ranked by semantic similarity. The
result table shows [csv] for ticket matches and [doc] for article
matches so you can see the source at a glance.

### Step 5 - Start the interactive prompt

```
python predict.py
```

---

## Usage

### Interactive mode

```
python predict.py
```

Example session:

```
============================================================
  ARIA -- Automated Routing and Intelligent Assignment
============================================================

  Embedding model  : all-MiniLM-L6-v2
  LLM model        : mistral via Ollama
  Assignment groups: 23
  Knowledge base   : 1247 tickets loaded
  LLM              : mistral (Ollama running)

  ----------------------------------------------------------
  Ticket description: VPN not connecting from home office

  ==========================================================
  PREDICTION RESULT
  ==========================================================

  Ticket       : VPN not connecting from home office
  Assignment   : IT-Network Support
  Confidence   : HIGH (3 of 5 similar tickets matched)

  Similar historical tickets and KB articles used:

  Rank  Short Description                                 Assignment Group                    Similarity Source
  ----- ------------------------------------------------ ----------------------------------- ---------- --------
  1.    VPN connection dropping after 10 minutes          IT-Network Support                  9.2        [csv]  <--
  2.    VPN Troubleshooting Guide (part 1)                IT-Network Support                  8.9        [doc]  <--
  3.    SSL VPN certificate error on login page           IT-Network Support                  8.4        [csv]  <--
  4.    Cannot reach internal file share from remote      IT-Wintel Support                   8.1        [csv]
  5.    Firewall rule blocking new SaaS application       IT-Network Support                  7.9        [csv]  <--
```

### Single prediction mode (for scripting)

```
python predict.py --once "SAP workflow approval stuck"
```

---

## Configuration Reference

config/config.yaml:

```yaml
embedding:
  model: "all-MiniLM-L6-v2"     # runs locally, downloads once (~90MB)

vector_db:
  path: "data/chroma_db"         # where ChromaDB stores its files
  collection: "snow_tickets"     # collection name
  top_k: 5                       # how many similar results to retrieve

llm:
  provider: "ollama"
  model: "mistral"               # or "gemma:2b" for lighter model
  temperature: 0.1               # lower = more consistent output
  max_tokens: 512

data:
  csv_path: "data/training_tickets.csv"
  docs_path: "data/kb_docs"      # folder with KB articles (NEW)

docs:                            # NEW - document chunking settings
  chunk_size: 400                # words per chunk
  chunk_overlap: 80              # overlapping words between chunks

assignment_groups:
  - "IT-SC-EPAM-SAP-AMS-Support"
  ...
```

---

## KB Document Ingestion - How It Works

### Supported file types

| Type     | Extension     | Extra library needed         |
|----------|---------------|------------------------------|
| Text     | .txt          | none                         |
| Markdown | .md           | none                         |
| PDF      | .pdf          | pip install pypdf             |
| HTML     | .html / .htm  | pip install beautifulsoup4    |

### Chunking

Long articles are automatically split into overlapping chunks so that
the sentence-transformer model can embed each piece meaningfully.

Default: 400-word chunks with 80-word overlap between consecutive chunks.
Adjust in config.yaml under the docs: section or via CLI flags:

```
python build_knowledge_base_docs.py --chunk-size 300 --chunk-overlap 60
```

### Team tagging

Three ways to associate a document with an assignment group:

1. Front-matter tag on the first line of the file:
   TEAM: IT-Network Support

2. Sub-folder name:
   data/kb_docs/IT-Network Support/vpn_guide.md

3. No tag - document is ingested without a team tag.
   The LLM will still use it as context but cannot use it
   as a direct team hint.

### Upsert behaviour

Documents use stable IDs derived from filename + chunk index.
Re-running build_knowledge_base_docs.py without --rebuild-docs
will upsert - updating existing chunks if content changed, without
duplicating them.

---

## Adding New Assignment Groups

Step 1 - Add the group name to config/config.yaml under assignment_groups

Step 2 - Add resolved tickets for that group to data/training_tickets.csv
         AND/OR add KB articles for that group to data/kb_docs/

Step 3 - Rebuild

```
python build_knowledge_base.py --rebuild
python build_knowledge_base_docs.py --rebuild-docs
```

No model retraining needed.

---

## How It Works in Detail

### Why sentence-transformers instead of keyword matching

Keyword matching fails on tickets like "user cannot get into the
system" which means the same as "login failing". Sentence transformers
understand semantic meaning so similar intent tickets are retrieved
even when the exact words are different. KB articles written in prose
are found even if the user's ticket uses different terminology.

### Why ChromaDB

ChromaDB stores the embeddings on disk so the expensive embedding step
only happens once during build. Queries are fast because ChromaDB uses
HNSW approximate nearest neighbour search.

### Why a local LLM instead of OpenAI

Your ticket data stays on your machine. No data leaves your network.
Ollama runs Mistral or Gemma locally. No API key, no cost, no
data privacy concern.

### How the LLM uses KB articles

The LLM receives both historical ticket matches AND document chunk
matches. Document chunks give the LLM extra prose context - for
example, a runbook saying "VPN issues should be routed to
IT-Network Support" provides a strong signal even if historical
ticket matches are sparse.

### Fallback if Ollama is not running

If Ollama is not available, the system falls back to a weighted
similarity vote from the top-K results (tickets + documents combined).
Confidence is shown as LOW.

---

## Troubleshooting

**Knowledge base is empty after build_knowledge_base.py**

Check the CSV column names match exactly:
- Short Description (capital S and D)
- Description
- Assignment Team

**No documents found after build_knowledge_base_docs.py**

Check the docs folder exists and contains supported files:
```
python build_knowledge_base_docs.py --list
```

**PDF ingestion fails**

```
pip install pypdf
```

**HTML tags appearing in document chunks**

```
pip install beautifulsoup4
```

**LLM returning wrong group names**

Check that config.yaml group names exactly match the Assignment Team
column values in your CSV.

**Ollama not found or model not available**

```
ollama serve
ollama list
ollama pull mistral
```

**Out of memory when running Mistral**

Switch to a lighter model in config/config.yaml:
```yaml
llm:
  model: "gemma:2b"
```

Then: ollama pull gemma:2b
