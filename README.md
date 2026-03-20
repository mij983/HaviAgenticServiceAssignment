# ARIA - Automated Routing and Intelligent Assignment

An agentic AI pipeline that takes a ticket short description as input
and predicts the correct assignment group using semantic search over
historical tickets and a locally running LLM.

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
  Search 1 year of historical tickets stored as embeddings
  Return top 5 most semantically similar tickets
        |
        v
LLM Agent  (Mistral or Gemma via Ollama - runs locally)
  Read the 5 similar tickets and their assignment groups
  Reason about which group best fits the new ticket
        |
        v
Predicted Assignment Group
  + confidence level
  + which similar tickets matched
  + similarity scores
```

---

## Project Structure

```
aria/
|
|-- predict.py                    <- Run this for the interactive prompt
|-- build_knowledge_base.py       <- Run once to load CSV into ChromaDB
|
|-- agents/
|   |-- preprocessing_agent.py   <- Text cleaning
|   |-- embedding_agent.py       <- Sentence transformer embeddings
|   |-- knowledge_base_agent.py  <- ChromaDB build and search
|   |-- llm_agent.py             <- Ollama LLM reasoning
|
|-- config/
|   |-- config.yaml              <- Model settings, assignment groups
|
|-- data/
|   |-- training_tickets.csv     <- Your 1-year ServiceNow CSV export
|   |-- chroma_db/               <- ChromaDB storage (auto-created)
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
rich                    terminal output formatting
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
your exact ServiceNow group names:

```yaml
assignment_groups:
  - "IT-SC-EPAM-SAP-AMS-Support"
  - "IT-SC-EPAM-SAP Basis Support"
  - "IT-SC-EPAM-SAP Workflow"
  ...
```

### Step 3 - Build the knowledge base

Run once. This embeds all CSV tickets into ChromaDB.

```
python build_knowledge_base.py
```

If you get a new CSV export later, rebuild:

```
python build_knowledge_base.py --rebuild
```

### Step 4 - Start the interactive prompt

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

  Type a ticket short description to get the assignment group.
  Type 'exit' to quit.

  ----------------------------------------------------------
  Ticket description: VPN not connecting from home office

  ==========================================================
  PREDICTION RESULT
  ==========================================================

  Ticket       : VPN not connecting from home office
  Assignment   : IT-tms-I&O-Identity-Collab-End Point Support
  Confidence   : HIGH (4 of 5 similar tickets matched)

  Similar historical tickets used:

  Rank  Short Description                                   Assignment Group                    Similarity
  ----- -------------------------------------------------- ----------------------------------- ----------
  1.    VPN connection dropping after 10 minutes            IT-tms-I&O-Identity-Collab-End P... 0.923  <--
  2.    SSL VPN certificate error on login page             IT-tms-I&O-Identity-Collab-End P... 0.887  <--
  3.    IPSec tunnel to partner company dropped             IT-tms-I&O-Identity-Collab-End P... 0.841  <--
  4.    Cannot reach internal file share from remote office IT-Wintel Support                   0.812
  5.    Firewall rule blocking new SaaS application         IT-tms-I&O-Identity-Collab-End P... 0.798  <--
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
  top_k: 5                       # how many similar tickets to retrieve

llm:
  provider: "ollama"
  model: "mistral"               # or "gemma:2b" for lighter model
  temperature: 0.1               # lower = more consistent output
  max_tokens: 512

data:
  csv_path: "data/training_tickets.csv"

assignment_groups:
  - "IT-SC-EPAM-SAP-AMS-Support"
  - "IT-SC-EPAM-SAP Basis Support"
  ...all 23 groups...
```

### Changing the LLM model

Edit config/config.yaml and change the model field.

Then pull the model with Ollama:

```
ollama pull gemma:2b
```

Available options:
```
mistral       Best accuracy, requires ~4GB RAM
gemma:2b      Lighter option, requires ~2GB RAM
llama3.2      Good balance, requires ~2GB RAM
phi3          Very lightweight, requires ~1.5GB RAM
```

---

## Adding New Assignment Groups

Step 1 - Add the group name to config/config.yaml under assignment_groups

Step 2 - Add resolved tickets for that group to data/training_tickets.csv

Step 3 - Rebuild the knowledge base

```
python build_knowledge_base.py --rebuild
```

No model retraining needed. The LLM reasons from the retrieved
tickets, so new groups are available immediately after rebuild.

---

## How It Works in Detail

### Why sentence-transformers instead of keyword matching

Keyword matching fails on tickets like "user cannot get into the
system" which means the same as "login failing". Sentence transformers
understand semantic meaning so similar intent tickets are retrieved
even when the exact words are different.

### Why ChromaDB

ChromaDB stores the embeddings on disk so the expensive embedding step
only happens once during build. Queries are fast because ChromaDB uses
HNSW approximate nearest neighbour search.

### Why a local LLM instead of OpenAI

Your ticket data stays on your machine. No data leaves your network.
Ollama runs Mistral or Gemma locally. No API key, no cost, no
data privacy concern.

### How the LLM decides

The LLM receives:
- The new ticket description
- The 5 most similar historical tickets with their assignment groups
- The full list of 23 valid groups

It must return exactly one group name from the valid list. The
temperature is set to 0.1 so responses are consistent and deterministic.

### Fallback if Ollama is not running

If Ollama is not available, the system falls back to a majority vote
from the 5 similar tickets. The most common assignment group among
them is returned. Confidence is shown as LOW to indicate LLM was
not used.

---

## Troubleshooting

**Knowledge base is empty after build_knowledge_base.py**

Check the CSV column names match exactly:
- Short Description (capital S and D)
- Description
- Assignment Team

**LLM returning wrong group names**

Ollama may be returning a group name that does not match the list
exactly. The system does fuzzy matching as a fallback. If this
happens consistently, check that config.yaml group names exactly
match what is in your CSV Assignment Team column.

**Ollama not found or model not available**

Make sure Ollama is running:
```
ollama serve
```

Check the model is downloaded:
```
ollama list
```

Pull it if missing:
```
ollama pull mistral
```

**Embedding model download slow on first run**

The all-MiniLM-L6-v2 model (~90MB) downloads automatically from
HuggingFace on the first run and is cached locally. This only
happens once.

**Out of memory when running Mistral**

Switch to a lighter model in config/config.yaml:
```yaml
llm:
  model: "gemma:2b"
```

Then:
```
ollama pull gemma:2b
```
