# ARIA - Automated Routing and Intelligent Assignment

An agentic AI pipeline that predicts the correct ServiceNow assignment group
for any IT support ticket using semantic search over historical tickets,
KB articles, and user-confirmed feedback — powered by **Azure OpenAI** or
a local Ollama LLM.

No ServiceNow connection needed. Runs entirely on your machine.

---

## What's New in This Version

| Feature | Details |
|---|---|
| **Azure OpenAI** | Use GPT-4o / GPT-4 / GPT-35-turbo as the LLM instead of local Ollama |
| **Feedback Loop** | After each prediction, user confirms correct/wrong — saved to feedback.jsonl |
| **KB from feedback** | Confirmed entries are promoted into ChromaDB as new training examples |
| **Corrections** | Wrong predictions + user-provided correct group also go into KB |
| **Feedback report** | Per-group accuracy report, export to CSV |
| **KB documents** | Ingest .txt .md .pdf .html articles (from previous version, unchanged) |

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
  Search: CSV tickets + KB articles + confirmed feedback entries
  Return top-K most semantically similar results
        |
        v
LLM Agent  (Azure OpenAI GPT-4o  OR  local Ollama)
  Read similar tickets, KB article chunks, valid groups
  Reason -> output exactly one assignment group name
        |
        v
Predicted Assignment Group
  + confidence score (1-10)
  + source labels: [csv] [doc] [fb]
        |
        v
Feedback Prompt
  "Was this correct? (y/n/skip)"
  Saved to data/feedback.jsonl
        |
        v
  python feedback_loop.py --apply
  Confirmed entries promoted into ChromaDB -> improves future predictions
```

---

## Project Structure

```
aria/
|
|-- predict.py                          <- Interactive prediction prompt
|-- build_knowledge_base.py             <- Load CSV tickets into ChromaDB
|-- build_knowledge_base_docs.py        <- Load KB articles into ChromaDB
|-- feedback_loop.py                    <- Manage feedback: apply/report/export  (NEW)
|
|-- agents/
|   |-- preprocessing_agent.py         <- Text cleaning
|   |-- embedding_agent.py             <- Sentence transformer embeddings
|   |-- knowledge_base_agent.py        <- ChromaDB build and search
|   |-- llm_agent.py                   <- Azure OpenAI + Ollama support  (UPDATED)
|   |-- document_ingestion_agent.py    <- KB document reader and chunker
|   |-- feedback_agent.py              <- Feedback recording and KB promotion  (NEW)
|
|-- config/
|   |-- config.yaml                    <- Provider, model, feedback settings
|
|-- data/
|   |-- training_tickets.csv           <- ServiceNow CSV export
|   |-- kb_docs/                       <- KB articles folder
|   |-- feedback.jsonl                 <- Feedback log (auto-created)  (NEW)
|   |-- chroma_db/                     <- ChromaDB storage (auto-created)
|
|-- .env                               <- Azure credentials (YOU CREATE THIS)
|-- .env.example                       <- Template for .env
|-- requirements.txt
```

---

## Requirements

Python 3.10 or above.

### Install Python packages

```
pip install -r requirements.txt
```

---

## Setup: Azure OpenAI

### Step 1 - Create an Azure OpenAI resource

1. Go to https://portal.azure.com
2. Search for "Azure OpenAI" and create a new resource
3. Choose a region (e.g. East US, Sweden Central)
4. Wait for deployment to complete (~2 minutes)

### Step 2 - Deploy a model

1. Go to your Azure OpenAI resource
2. Click "Go to Azure OpenAI Studio" (or https://oai.azure.com)
3. Click "Deployments" -> "Create new deployment"
4. Choose model: gpt-4o (recommended) or gpt-35-turbo (cheaper)
5. Set a deployment name (e.g. "gpt-4o" — this goes in your .env)
6. Click Deploy

### Step 3 - Get your credentials

From your Azure OpenAI resource page:

1. Click "Keys and Endpoint" in the left menu
2. Copy KEY 1 -> this is your AZURE_OPENAI_API_KEY
3. Copy Endpoint URL -> this is your AZURE_OPENAI_ENDPOINT
   Example: https://my-openai-resource.openai.azure.com/

API version to use: 2024-02-01 (stable, works with all GPT-4 models)

### Step 4 - Create your .env file

Copy .env.example to .env:

```
cp .env.example .env
```

Fill in your values:

```
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=abc123yourkeyhere
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

IMPORTANT: Never commit .env to Git. It is already in .gitignore.

### Step 5 - Set provider in config.yaml

Edit config/config.yaml:

```yaml
llm:
  provider: "azure_openai"
  model: "gpt-4o"        # must match your deployment name exactly
  temperature: 0.1
```

### Step 6 - Test the connection

```
python predict.py --once "Cannot log in to SAP"
```

You should see:  LLM : gpt-4o (Azure OpenAI)  [OK]

---

## Setup: Ollama (local, no API key)

To keep using Ollama instead of Azure OpenAI:

```yaml
# config/config.yaml
llm:
  provider: "ollama"
  model: "mistral"
  temperature: 0.1
```

Install Ollama from https://ollama.com and pull the model:

```
ollama pull mistral
```

No .env file needed for Ollama.

---

## Setup: Knowledge Base (CSV tickets)

### Step 1 - Prepare your CSV

Place your ServiceNow CSV export at:  data/training_tickets.csv

Required columns:
  Short Description   (ticket subject)
  Description         (ticket body)
  Assignment Team     (which team resolved it)

### Step 2 - Build the knowledge base

```
python build_knowledge_base.py
```

Incremental loading (for large CSV files):

```
python build_knowledge_base.py --start 0     --end 10000
python build_knowledge_base.py --start 10000 --end 20000
```

Rebuild from scratch:

```
python build_knowledge_base.py --rebuild
```

---

## Setup: KB Articles (optional)

Place .txt / .md / .pdf / .html articles in data/kb_docs/

Optional: add a team tag on the first line of the file:
  TEAM: IT-Network Support

Or organise files in sub-folders named after the team:
  data/kb_docs/IT-Network Support/vpn_guide.md

Ingest into ChromaDB:

```
python build_knowledge_base_docs.py
```

List ingested documents:

```
python build_knowledge_base_docs.py --list
```

---

## Running Predictions

### Interactive mode

```
python predict.py
```

After each prediction you will be asked:
  "Was this correct? (y/n/skip)"

Answering builds the feedback log in data/feedback.jsonl.

### Disable feedback prompts

```
python predict.py --no-feedback
```

### Single prediction mode

```
python predict.py --once "SAP workflow approval stuck"
```

### Source labels in results table

  [csv]  — match came from historical CSV ticket
  [doc]  — match came from a KB article / document
  [fb]   — match came from a previously confirmed feedback entry

---

## Feedback Loop

### View accuracy report

```
python feedback_loop.py --report
```

Shows:
  - Total predictions, confirmed correct/wrong, accuracy %
  - Per-group accuracy breakdown
  - List of wrong predictions with user-provided correct groups

### Promote confirmed feedback into the knowledge base

After collecting enough feedback (e.g. daily or weekly):

```
python feedback_loop.py --apply-all
```

This runs both steps in one command:
  1. Confirmed-correct entries -> added as [fb] training examples
  2. User corrections (confirmed-wrong + correct group provided) -> added as [fb] training examples

Run individually if needed:

```
python feedback_loop.py --apply              # confirmed correct only
python feedback_loop.py --apply-corrections  # corrections only
```

Each entry is upserted — safe to run multiple times, no duplicates.

### Export feedback to CSV

```
python feedback_loop.py --export data/feedback_export.csv
```

Useful for review in Excel or importing into ServiceNow reporting.

### Clear unrated entries

```
python feedback_loop.py --clear-pending
```

Removes entries where no feedback was given (status: pending).

---

## Configuration Reference

config/config.yaml:

```yaml
embedding:
  model: "all-MiniLM-L6-v2"

vector_db:
  path: "data/chroma_db"
  collection: "snow_tickets"
  top_k: 5

llm:
  provider: "azure_openai"      # "azure_openai" | "ollama"
  model: "gpt-4o"               # Azure deployment name OR Ollama model
  temperature: 0.1
  max_tokens: 512

data:
  csv_path: "data/training_tickets.csv"
  docs_path: "data/kb_docs"

docs:
  chunk_size: 400
  chunk_overlap: 80

feedback:
  path: "data/feedback.jsonl"
  enabled: true

assignment_groups:
  - "IT-Service Desk"
  - "IT-Network Support"
  ...
```

---

## How the Feedback Loop Improves Accuracy

Every time a user confirms a prediction is correct, that short description
and its assignment group are added as a new entry in ChromaDB with
source_type = "feedback".

When a similar ticket comes in next time, this confirmed example appears
in the top-K results alongside CSV tickets and KB articles, giving the LLM
a direct precedent from real confirmed cases.

When a user corrects a wrong prediction, the corrected example (with the
right group) is also promoted. This directly counteracts the case that
was getting misrouted.

Over time, frequently-seen ticket types accumulate confirmed examples and
become increasingly accurate without any retraining or model updates.

---

## Azure OpenAI vs Ollama Comparison

| | Azure OpenAI | Ollama (local) |
|---|---|---|
| Model quality | GPT-4o — best accuracy | Good (Mistral, LLaMA) |
| Setup | API key + endpoint required | Install Ollama, pull model |
| Cost | Pay per token (Azure pricing) | Free, runs on your hardware |
| Data privacy | Data sent to Azure | Data stays on machine |
| Internet required | Yes | No |
| RAM required | None (cloud) | 2-4GB depending on model |
| Speed | Fast (cloud inference) | Depends on hardware |

---

## Troubleshooting

**Azure OpenAI: AuthenticationError**

Check your AZURE_OPENAI_API_KEY in .env. Make sure you copied KEY 1
from "Keys and Endpoint" in the Azure portal.

**Azure OpenAI: DeploymentNotFound**

The AZURE_OPENAI_DEPLOYMENT in .env must exactly match the deployment
name in Azure OpenAI Studio -> Deployments. It is case-sensitive.

**Azure OpenAI: ResourceNotFound or 404**

Check AZURE_OPENAI_ENDPOINT ends with a trailing slash:
  https://your-resource.openai.azure.com/   (correct)
  https://your-resource.openai.azure.com    (may fail)

**Azure OpenAI: RateLimitError**

Your Azure OpenAI quota is exhausted. Go to Azure portal -> your
OpenAI resource -> "Quotas" to check limits and request increases.

**Feedback not saving**

Check that data/ folder exists (it is created automatically on first run).
Check that you are not running with --no-feedback flag.

**Feedback report shows 0 entries**

Run a few predictions first and answer the feedback prompts.
Then run: python feedback_loop.py --report

**Knowledge base empty after build**

Check CSV column names match exactly:
  Short Description   Description   Assignment Team

**Ollama not found**

Make sure Ollama is running:
  ollama serve
  ollama list
  ollama pull mistral
