# Agentic AI — ServiceNow Ticket Assignment

An agentic AI system that automatically monitors incoming ServiceNow Incidents, predicts the correct assignment group using a machine-learning model trained on historical ticket data, validates predictions against a Knowledge Article (stored locally or in Azure Blob Storage), and auto-assigns tickets when confidence exceeds a configurable threshold. Low-confidence tickets are sent to manual triage, and every human decision is fed back into a continuous learning loop that retrains the model automatically.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [How Each Agent Works](#how-each-agent-works)
- [Confidence Scoring](#confidence-scoring)
- [Continuous Learning Loop](#continuous-learning-loop)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Step 1 — Clone and Install](#step-1--clone-and-install)
- [Step 2 — Configure ServiceNow Credentials](#step-2--configure-servicenow-credentials)
- [Step 3 — Configure Azure Blob (Optional)](#step-3--configure-azure-blob-optional)
- [Step 4 — Set Up ServiceNow Sandbox](#step-4--set-up-servicenow-sandbox)
- [Step 5 — Add Training Data](#step-5--add-training-data)
- [Step 6 — Train the Model](#step-6--train-the-model)
- [Step 7 — Run the Agent](#step-7--run-the-agent)
- [Training New Data — Full Workflow](#training-new-data--full-workflow)
- [Commands Reference](#commands-reference)
- [Configuration File Reference](#configuration-file-reference)
- [Environment Variables](#environment-variables)
- [Feature Engineering](#feature-engineering)
- [SQLite Database Schema](#sqlite-database-schema)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [Next Steps (Production Enhancements)](#next-steps-production-enhancements)

---

## Architecture Overview

```
ServiceNow (Incidents)
        │
        │ REST API polling  (every 60s)
        ▼
┌──────────────────────────┐
│  Ticket Ingestion Agent  │  ← polls & normalises raw ticket payloads
└────────────┬─────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│        Knowledge & History Layer        │
│  KnowledgeAgent  (local JSON / Azure)   │  ← active groups + deprecated map
│  HistoricalDataAgent                    │  ← keyword-based feature engineering
└────────────┬────────────────────────────┘
             │
             ▼
┌──────────────────────────┐
│    Prediction Agent      │  ← scikit-learn LogisticRegression / RandomForest
│  + ConfidenceScoringEngine│  ← 3-signal composite score (1–10)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│      Decision Agent      │  ← confidence > threshold → auto-assign
└────────────┬─────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
 Auto-assign    Manual triage
 (PATCH API)  (work note added)
      │             │
      └──────┬──────┘
             ▼
┌──────────────────────────┐
│      Learning Agent      │  ← stores outcomes, auto-retrains model
│   (SQLite feedback DB)   │  ← hot-swaps model without restart
└──────────────────────────┘
```

The loop repeats indefinitely. Each iteration:

1. Fetches all unassigned open incidents from ServiceNow.
2. Predicts the best assignment group for each ticket.
3. Scores the prediction on a 1–10 confidence scale.
4. Auto-assigns high-confidence tickets via a PATCH API call.
5. Adds a work note on low-confidence tickets, listing the AI suggestion and asking a human to assign.
6. Checks whether any previously triaged tickets have since been assigned by a human, captures that as labelled training data, retrains the model, and hot-reloads it.

---

## How Each Agent Works

### TicketIngestionAgent (`agents/ingestion_agent.py`)

Polls the ServiceNow Table API for incidents with an empty `assignment_group` and `state=1` (New). Returns up to 20 tickets per poll cycle, normalising the REST response into a flat Python dict with the fields the downstream agents expect: `sys_id`, `number`, `short_description`, `description`, `category`, `subcategory`, `business_service`, `priority`.

### KnowledgeAgent (`agents/knowledge_agent.py`)

Loads the list of currently active assignment groups and any deprecated-group mappings. Resolution order:

1. Azure Blob Storage (if `connection_string` is configured).
2. Local file `data/assignment_groups.json`.
3. Hardcoded fallback list of the eight groups from the training CSV.

The agent is refreshed on every poll cycle so group changes in Blob take effect within one minute.

### HistoricalDataAgent (`agents/historical_data_agent.py`)

Converts a normalised ticket dict into an 11-element numeric feature vector that the ML model consumes:

| Index | Feature | How computed |
|-------|---------|--------------|
| 0 | `short_desc_len` | `len(short_description)` |
| 1 | `desc_len` | `len(description)` |
| 2 | `combined_len` | sum of both lengths |
| 3 | `category_hash` | MD5 hash mod 100 |
| 4 | `subcategory_hash` | MD5 hash mod 100 |
| 5 | `business_service_hash` | MD5 hash mod 100 |
| 6 | `short_desc_first5_hash` | first-five-words hash mod 100 |
| 7 | `desc_first5_hash` | first-five-words hash mod 100 |
| 8 | `priority_numeric` | integer 1–4 |
| 9 | `short_desc_word_count` | word count |
| 10 | `desc_word_count` | word count |

The agent also exposes `TEAM_KEYWORDS` — a per-group set of domain-specific words (e.g. `"infor"`, `"kisoft"`, `"vim"`, `"dims"`) used by the ConfidenceScoringEngine to calculate the text-similarity signal.

### AssignmentPredictionAgent (`agents/prediction_agent.py`)

Wraps a `joblib`-serialised scikit-learn model (`models/assignment_model.pkl`). Exposes:

- `predict(features)` → `(predicted_group, raw_probability)`
- `predict_top_n(features, n=3)` → top-N `[(group, probability)]`
- `reload()` → hot-swaps the model file without restarting the service

Falls back to a dummy three-class LogisticRegression model if the `.pkl` file is absent, so the service always starts.

### ConfidenceScoringEngine (`agents/confidence_engine.py`)

Combines three independent signals into a single 1–10 score:

| Signal | Weight | Description |
|--------|--------|-------------|
| ML model probability | 60% | Raw `predict_proba` value from the classifier |
| Text keyword similarity | 30% | SD keywords counted twice (2×), description once (1×); capped at 3 hits per field |
| Knowledge article validation | 10% | 1.0 if predicted group is in the active list, 0.0 otherwise |

```
composite = 0.60 × ml + 0.30 × text + 0.10 × knowledge
score     = round(1.0 + composite × 9.0, 1)
```

Auto-assignment requires score > 7 (configurable).

### DecisionAgent (`agents/decision_agent.py`)

Applies the threshold rule. Also supports a per-group dynamic threshold: once a group has accumulated enough feedback and demonstrates high accuracy, its threshold is automatically lowered so similar tickets auto-assign sooner without waiting for the global threshold.

### ServiceNowUpdateAgent (`agents/servicenow_agent.py`)

- `assign_ticket(sys_id, group)` — PATCH `assignment_group` on the incident record and adds an automated work note.
- `add_work_note(sys_id, note)` — adds a human-readable work note with the AI suggestion and top-3 predictions for manual triage tickets.
- `fetch_resolved_tickets(sys_ids)` — checks whether triaged tickets have since been assigned by a human; used by the learning loop.

### LearningAgent (`agents/learning_agent.py`)

Maintains a SQLite database (`data/feedback.db`) with three tables:

- `audit_log` — every routing decision, predicted group, confidence, and outcome.
- `manual_triage` — tickets that went to human triage, pending outcome collection.
- `feedback` — confirmed ground-truth records (AI prediction vs. human decision).

Key methods:

- `store_manual_triage(...)` — saves a triage ticket for later outcome polling.
- `poll_manual_triage_outcomes(sn_agent)` — checks ServiceNow for pending triage tickets that now have an assignment group set; converts them to feedback records and triggers retraining if enough new records have accumulated.
- `retrain_model()` — merges the original CSV training data with all human feedback, fits a new RandomForestClassifier, and saves the updated `.pkl`.
- `get_group_threshold(group)` — returns a dynamically lowered threshold for groups that have proved their accuracy.
- `get_learning_stats()` — returns a stats dict for dashboard display.

---

## Confidence Scoring

```
Score 1–4   → Very low confidence. Manual triage, work note added.
Score 5–6   → Moderate. Manual triage unless per-group threshold earned.
Score 7     → Boundary. Just below auto-assign threshold (>7 required).
Score 8–10  → High confidence. Auto-assign via PATCH API.
```

The confidence bar shown in the terminal uses colour coding:

- Green  — score > 7 (auto-assign)
- Yellow — score 5–7 (borderline)
- Red    — score < 5 (low confidence)

---

## Continuous Learning Loop

```
New ticket → low confidence → work note: "AI suggests X, please assign"
                                │
                    Human assigns in ServiceNow
                                │
                 poll_manual_triage_outcomes() detects it
                                │
               Stores (features, human_group) as feedback
                                │
              retrain_model() merges CSV + all feedback
                                │
              prediction_agent.reload() hot-swaps the model
                                │
        Next similar ticket → auto-assigns without human help ✅
```

Retraining also lowers the per-group confidence threshold automatically when a group reaches 10 or more feedback records with 80%+ accuracy, gradually expanding the set of tickets that the AI handles autonomously.

---

## Repository Structure

```
agentic-servicenow-assignment/
│
├── config/
│   └── config.yaml                   ← all configuration lives here
│
├── agents/
│   ├── __init__.py
│   ├── ingestion_agent.py            ← polls ServiceNow REST API for unassigned tickets
│   ├── knowledge_agent.py            ← loads active groups from JSON / Azure Blob
│   ├── historical_data_agent.py      ← feature engineering + keyword dictionaries
│   ├── prediction_agent.py           ← ML model wrapper (load, predict, reload)
│   ├── confidence_engine.py          ← 3-signal composite confidence score (1–10)
│   ├── decision_agent.py             ← threshold gating + per-group dynamic thresholds
│   ├── servicenow_agent.py           ← PATCH assign + work notes + fetch resolved
│   └── learning_agent.py             ← SQLite feedback store + auto-retraining
│
├── models/
│   └── assignment_model.pkl          ← trained model (generated by training scripts)
│
├── data/
│   ├── assignment_groups.json        ← active group list (synced to Azure Blob)
│   ├── training_tickets.csv          ← labelled ticket CSV used for initial training
│   └── feedback.db                   ← SQLite: audit log + triage + feedback (auto-created)
│
├── scripts/
│   ├── train_model.py                ← trains / retrains from a CSV export
│   └── accuracy_report.py            ← prints accuracy stats from feedback.db
│
├── tests/
│   └── test_agents.py                ← 22 unit tests (pytest)
│
├── main.py                           ← primary orchestrator (continuous polling loop)
├── run_agent.py                      ← rich terminal UI version of main.py
├── retrain_now.py                    ← manually trigger outcome collection + retrain
├── add_training_tickets.py           ← creates 60 resolved training tickets in ServiceNow
├── export_servicenow_data.py         ← exports your resolved tickets to CSV
├── create_sample_tickets.py          ← creates 50 unassigned test tickets in ServiceNow
├── train_from_tickets.py             ← trains from the bundled training_tickets.csv
├── train_own_model.py                ← trains from any exported CSV with --analyze flag
├── requirements.txt
├── .env.example                      ← copy to .env and fill credentials
└── README.md
```

---

## Prerequisites

- Python 3.10 or later
- A ServiceNow Developer Instance (register free at developer.servicenow.com)
- Azure Storage Account (optional — local JSON fallback works without it)
- `pip` and `venv` available in your Python installation

---

## Step 1 — Clone and Install

```bash
git clone https://github.com/your-org/agentic-servicenow-assignment.git
cd agentic-servicenow-assignment

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Step 2 — Configure ServiceNow Credentials

All configuration lives in `config/config.yaml`. Open it and fill in the values for your instance:

```yaml
servicenow:
  instance_url: "https://YOUR-INSTANCE.service-now.com"   # e.g. https://dev123456.service-now.com
  username: "ai_assignment_bot"                            # the integration user you create below
  password: "your-secure-password"                         # or set via SN_API_PASSWORD env var

azure_blob:
  connection_string: "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
  container_name: "knowledge-container"
  blob_name: "assignment_groups.json"

confidence_threshold: 7           # auto-assign only when score > this value (1–10 scale)

polling:
  interval_seconds: 60            # how often to poll ServiceNow for new tickets

model:
  path: "models/assignment_model.pkl"

database:
  feedback_db: "data/feedback.db"

logging:
  level: "INFO"
  log_file: "data/audit.log"
```

### Creating the ServiceNow Integration User

1. In your ServiceNow instance navigate to **System Security → Users and Groups → Users → New**.
2. Fill in the following fields:

   | Field    | Value               |
   |----------|---------------------|
   | User ID  | `ai_assignment_bot` |
   | Password | (choose a strong password and record it) |
   | Roles    | `rest_api_explorer`, `itil` |

3. Save the user.
4. Put the credentials into `config/config.yaml` (or into the `.env` file — see [Environment Variables](#environment-variables)).

---

## Step 3 — Configure Azure Blob (Optional)

If you want the Knowledge Agent to pull the active-groups list from Azure Blob Storage (useful for centralised, hot-updatable group management without redeploying):

1. Create a Storage Account in the Azure Portal.
2. Create a container called `knowledge-container` (or any name — update `config.yaml`).
3. Upload `data/assignment_groups.json` to the container.
4. Copy the connection string from **Access Keys** in the Storage Account blade.
5. Paste it into `config.yaml` under `azure_blob.connection_string`.

If you skip Azure, the agent reads `data/assignment_groups.json` from disk and falls back to a hardcoded list. The service works fine without Azure.

### Updating the Assignment Groups File

Edit `data/assignment_groups.json` to match your real ServiceNow assignment groups:

```json
{
  "active_assignment_groups": [
    "IT-SC Operations Application Support",
    "IT-SC GBS App Support",
    "IT-Asia-SOA-Central-App-Support",
    "IT-SC-EPAM-SAP Workflow",
    "IT-SC-EPAM-SAP-AMS-Support",
    "IT-SC-EPAM-SAP Hybris CRM Support",
    "IT-Portal-Central",
    "IT-SCM-EPAM L2"
  ],
  "deprecated_mapping": {
    "Old Team Name": "New Team Name"
  }
}
```

If using Azure Blob, re-upload this file after any edits. The agent refreshes it on every poll cycle.

---

## Step 4 — Set Up ServiceNow Sandbox

### Create Assignment Groups

Navigate to **User Administration → Groups → New** and create a group for each team in your assignment list. The names must match exactly what is in `data/assignment_groups.json` — they are case-sensitive.

### Create Test Incidents (Optional)

The `create_sample_tickets.py` script pushes 50 unassigned incidents directly to your ServiceNow instance. Run it after the model is trained to have a ready-made batch for testing:

```bash
python create_sample_tickets.py
```

All 50 tickets are in state "New" with no assignment group set. When you then run `run_agent.py`, the AI will route each one.

---

## Step 5 — Add Training Data

You have three options, in increasing quality order.

### Option A — Use the Bundled CSV (Fastest)

A pre-labelled CSV is already at `data/training_tickets.csv` with 50 tickets across the eight assignment groups. Use it to get started immediately:

```bash
python train_from_tickets.py
```

This reads `data/training_tickets.csv`, applies six augmentations per row (original + 5 variants), and trains a RandomForestClassifier. Cross-validation accuracy is printed before the model is saved to `models/assignment_model.pkl`.

### Option B — Add Resolved Tickets to ServiceNow Then Export

This gives you a larger, more realistic dataset:

1. Push 60 pre-resolved training tickets (20 per group) into your ServiceNow instance:

   ```bash
   python add_training_tickets.py --preview   # inspect what will be created
   python add_training_tickets.py             # create them in ServiceNow
   ```

2. Export them back as a CSV (the script pulls only resolved/closed tickets with an assignment group set):

   ```bash
   python export_servicenow_data.py
   # output: data/my_servicenow_tickets.csv
   ```

3. Train on the export:

   ```bash
   python train_own_model.py --csv data/my_servicenow_tickets.csv
   ```

### Option C — Export Your Own Production Data

If you have a production ServiceNow instance with thousands of resolved tickets:

1. Export via the script (fetches up to 5,000 records by default):

   ```bash
   python export_servicenow_data.py --limit 5000 --output data/production_tickets.csv
   ```

2. Inspect the group distribution and train:

   ```bash
   python train_own_model.py --csv data/production_tickets.csv --analyze
   ```

   The `--analyze` flag shows a per-group bar chart and category breakdown before training. Groups with fewer than 5 records (configurable with `--min-samples`) are excluded automatically.

---

## Step 6 — Train the Model

### Quick train from the bundled CSV

```bash
python train_from_tickets.py
```

Output: `models/assignment_model.pkl`

### Train from any CSV export

```bash
python train_own_model.py --csv data/my_servicenow_tickets.csv
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | (required) | Path to your labelled CSV file |
| `--output` | `models/assignment_model.pkl` | Where to save the trained model |
| `--analyze` | off | Show data analysis charts before training |
| `--min-samples` | `5` | Exclude groups with fewer samples than this |

The script prints cross-validation accuracy (5-fold stratified), per-group precision/recall/F1 on the held-out test set, and guidance on whether more data is needed.

### What the Training Script Does Internally

1. Reads the CSV, builds the 11-feature vector for every row using `HistoricalDataAgent`.
2. Filters out groups with fewer than `--min-samples` tickets.
3. Stratified train/test split (80/20).
4. Trains a `LogisticRegression(max_iter=2000)`.
5. Runs 5-fold cross-validation and prints per-fold scores.
6. Evaluates on the test set and prints a full classification report.
7. Saves the model as `models/assignment_model.pkl` and a `models/assignment_model_meta.json` sidecar with accuracy stats, class names, and feature descriptions.

---

## Step 7 — Run the Agent

### Standard run (one poll cycle, then exit)

```bash
python run_agent.py
```

### Watch mode (continuous polling until Ctrl+C)

```bash
python run_agent.py --watch
```

### Run via the main orchestrator

```bash
python main.py           # continuous loop
python main.py --once    # single poll cycle then exit
```

### Check the current state of all tickets

```bash
python run_agent.py --status
```

### Reset all assigned tickets for re-testing

```bash
python run_agent.py --reset
```

---

## Training New Data — Full Workflow

This section covers the complete end-to-end flow for retraining the model with new data after the system has been running in production.

### Step 1 — Let the Agent Run and Collect Triage Decisions

Run the agent in watch mode. For any ticket the AI is not confident about (score ≤ 7), it will:

- Store the ticket in the `manual_triage` table in `data/feedback.db`.
- Add a work note to the ServiceNow ticket: `"AI Suggestion: '<group>' (confidence X/10 — below threshold). Top predictions: ... Please assign to the correct group. Your assignment will train the AI to auto-route similar tickets."`

### Step 2 — Assign Triage Tickets in ServiceNow

Open each triaged ticket in your ServiceNow instance and set the `Assignment Group` field to the correct team. This human decision becomes the ground-truth label the learning loop needs.

### Step 3 — Collect Outcomes and Retrain

```bash
python retrain_now.py
```

This script:

1. Lists all manual triage tickets and their current status (pending or outcome collected).
2. Queries ServiceNow for each pending ticket to check whether a human has assigned it.
3. Saves each human decision as a feedback record in `data/feedback.db`.
4. Merges the original CSV training data with all feedback records.
5. Retrains the model with `RandomForestClassifier(n_estimators=300)`.
6. Saves the new `models/assignment_model.pkl`.
7. Prints a full learning loop summary including AI accuracy on triage tickets and per-group confidence thresholds, highlighting any groups whose threshold has been automatically lowered.

If no outcomes have been collected yet (humans have not assigned the tickets), the script prints which tickets still need assignment and exits cleanly, telling you exactly what to do.

### Step 4 — Reload and Verify

The next time `run_agent.py` or `main.py` poll for outcomes, they call `prediction_agent.reload()` automatically to hot-swap the new model. If you ran `retrain_now.py` manually while the agent is not running, just restart it.

### Step 5 — Monitor Accuracy Over Time

```bash
python scripts/accuracy_report.py
```

Prints a table of per-group accuracy from the SQLite feedback database. As the feedback table grows, accuracy improves and per-group thresholds lower automatically.

### Adding More Training Tickets at Any Time

To bulk-add more resolved tickets with pre-labelled groups:

```bash
python add_training_tickets.py --preview   # see what will be created
python add_training_tickets.py             # create 60 resolved tickets in ServiceNow
python export_servicenow_data.py           # pull them back as CSV
python train_own_model.py --csv data/my_servicenow_tickets.csv
```

Then restart the agent (or wait for the next poll cycle to hot-reload).

---

## Commands Reference

| Command | Purpose |
|---------|---------|
| `python main.py` | Start continuous polling loop |
| `python main.py --once` | One poll cycle then exit |
| `python run_agent.py` | Same as main.py with richer terminal UI |
| `python run_agent.py --watch` | Keep polling (Ctrl+C to stop) |
| `python run_agent.py --status` | Show current ticket states in ServiceNow |
| `python run_agent.py --reset` | Clear all assignments for re-testing |
| `python retrain_now.py` | Collect human outcomes + retrain model |
| `python train_from_tickets.py` | Train from bundled `data/training_tickets.csv` |
| `python train_own_model.py --csv <file>` | Train from any exported CSV |
| `python train_own_model.py --csv <file> --analyze` | Show data analysis then train |
| `python add_training_tickets.py` | Create 60 resolved training tickets in ServiceNow |
| `python add_training_tickets.py --preview` | Show what would be created |
| `python export_servicenow_data.py` | Export resolved tickets to CSV for training |
| `python export_servicenow_data.py --limit 2000` | Export up to 2,000 records |
| `python create_sample_tickets.py` | Create 50 unassigned test tickets in ServiceNow |
| `python scripts/accuracy_report.py` | Print accuracy stats from feedback DB |
| `pytest tests/ -v` | Run all 22 unit tests |

---

## Configuration File Reference

`config/config.yaml` is the single source of truth for all runtime settings.

```yaml
# ── ServiceNow connection ──────────────────────────────────────────────────────
servicenow:
  instance_url: "https://YOUR-INSTANCE.service-now.com"
  # Integration user — needs roles: rest_api_explorer, itil
  username: "ai_assignment_bot"
  # Never commit real passwords to source control.
  # Use an environment variable or a secrets manager in production.
  password: "your-secure-password"

# ── Azure Blob Storage (optional) ─────────────────────────────────────────────
# If connection_string is empty or omitted, the agent reads data/assignment_groups.json
# from disk and falls back to a hardcoded list. Azure is not required.
azure_blob:
  connection_string: "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
  container_name: "knowledge-container"
  blob_name: "assignment_groups.json"

# ── Auto-assignment threshold ──────────────────────────────────────────────────
# Tickets with a confidence score STRICTLY GREATER THAN this value are auto-assigned.
# Score is on a 1–10 scale.  Default 7 means the AI needs > 70% composite confidence.
# Lower this value if the AI is under-assigning.
# Raise it if the AI is making assignment mistakes.
confidence_threshold: 7

# ── Polling ───────────────────────────────────────────────────────────────────
polling:
  interval_seconds: 60            # seconds between ServiceNow polls

# ── Model ─────────────────────────────────────────────────────────────────────
model:
  path: "models/assignment_model.pkl"

# ── SQLite feedback database ───────────────────────────────────────────────────
database:
  feedback_db: "data/feedback.db"

# ── Logging ───────────────────────────────────────────────────────────────────
logging:
  level: "INFO"         # DEBUG | INFO | WARNING | ERROR
  log_file: "data/audit.log"
```

---

## Environment Variables

As an alternative to storing credentials directly in `config/config.yaml`, use environment variables. Copy `.env.example` to `.env` and fill in real values:

```bash
cp .env.example .env
# edit .env with your actual credentials — never commit this file
```

`.env.example` shows all available variables:

```env
# ServiceNow
SN_INSTANCE_URL=https://your-instance.service-now.com
SN_USERNAME=ai_assignment_bot
SN_PASSWORD=your-secure-password

# Azure Blob Storage
AZURE_BLOB_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net
AZURE_CONTAINER_NAME=knowledge-container
AZURE_BLOB_NAME=assignment_groups.json

# App settings
CONFIDENCE_THRESHOLD=7
POLLING_INTERVAL_SECONDS=60
```

Export them before running the agent:

```bash
export SN_INSTANCE_URL="https://dev123456.service-now.com"
export SN_PASSWORD="my-secret-password"
python run_agent.py
```

For production environments use Azure Key Vault, AWS Secrets Manager, or HashiCorp Vault instead of plain env vars. The `.gitignore` already excludes `.env` from source control.

---

## Feature Engineering

The `HistoricalDataAgent` builds an 11-element feature vector for every ticket. This table explains each feature and its purpose.

| # | Name | Description | Why it matters |
|---|------|-------------|----------------|
| 0 | `short_desc_len` | Character count of short description | Longer SDs tend to come from specific technical teams |
| 1 | `desc_len` | Character count of description | Empty descriptions point to certain ticket types |
| 2 | `combined_len` | Sum of both lengths | Overall ticket verbosity signal |
| 3 | `category_hash` | MD5 of category string mod 100 | Maps categorical field to numeric without one-hot explosion |
| 4 | `subcategory_hash` | MD5 of subcategory string mod 100 | Same |
| 5 | `business_service_hash` | MD5 of business_service field mod 100 | Same |
| 6 | `short_desc_first5_hash` | MD5 of first 5 words of SD mod 100 | Opening phrase is often the strongest routing signal |
| 7 | `desc_first5_hash` | MD5 of first 5 words of description mod 100 | Same for description |
| 8 | `priority_numeric` | Integer 1–4 (default 3) | Priority correlates with certain team types |
| 9 | `short_desc_word_count` | Word count of short description | Brief SDs like "DIMS" or "VIM" are distinct from verbose ones |
| 10 | `desc_word_count` | Word count of description | Same |

The text-similarity signal (used in the ConfidenceScoringEngine, separate from the ML features) uses a keyword dictionary. To add routing keywords for a new team, add an entry to the `TEAM_KEYWORDS` dict in `agents/historical_data_agent.py`:

```python
"New Team Name": {
    "keyword1", "keyword2", "application_name", "system_name",
},
```

Short description keywords carry double weight (2×) because the SD is always filled and contains the primary routing signal for most tickets.

---

## SQLite Database Schema

The feedback database at `data/feedback.db` is created automatically on first run and contains three tables.

```sql
-- Every routing decision ever made (full audit trail)
CREATE TABLE audit_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_number    TEXT NOT NULL,
    sys_id           TEXT NOT NULL,
    predicted_group  TEXT,
    confidence       REAL,
    auto_assigned    INTEGER,      -- 1 = auto-assigned, 0 = manual triage
    reason           TEXT,
    top_predictions  TEXT,         -- JSON: [(group, probability), ...]
    created_at       REAL          -- Unix timestamp
);

-- Tickets sent to manual triage, waiting for a human outcome
CREATE TABLE manual_triage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_number     TEXT NOT NULL,
    sys_id            TEXT NOT NULL,
    short_description TEXT,
    description       TEXT,
    features          TEXT,        -- JSON: 11-element feature vector
    ai_predicted      TEXT,
    ai_confidence     REAL,
    human_assigned    TEXT,        -- filled by poll_manual_triage_outcomes()
    outcome_checked   INTEGER DEFAULT 0,
    created_at        REAL
);

-- Ground-truth training feedback (AI prediction vs. human decision)
CREATE TABLE feedback (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_number     TEXT NOT NULL,
    sys_id            TEXT NOT NULL,
    short_description TEXT,
    description       TEXT,
    features          TEXT,        -- JSON feature vector used for retraining
    ai_predicted      TEXT,
    human_assigned    TEXT,
    was_correct       INTEGER,     -- 1 if AI matched human, 0 otherwise
    created_at        REAL
);
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 22 tests should pass in under 10 seconds. The tests mock the ServiceNow API and Azure Blob calls so no real credentials are needed.

To run a specific test class:

```bash
pytest tests/test_agents.py::TestConfidenceEngine -v
```

---

## Troubleshooting

**"Model file not found — using dummy model"**
Run `python train_from_tickets.py` to create `models/assignment_model.pkl` before starting the agent.

**"Cannot connect to https://your-instance.service-now.com"**
Verify that `instance_url` in `config.yaml` is correct, the integration user exists, and the password is right. Run `python run_agent.py --status` to test the connection in isolation.

**All tickets going to manual triage (confidence always ≤ 7)**
The model may not have enough training data for the groups in your instance. Run `python scripts/accuracy_report.py` to see per-group accuracy. If accuracy is below 60%, add more resolved training tickets (Option B or C in Step 5) and retrain.

**"Assignment group not found" when auto-assigning**
The predicted group name must exactly match a group that exists in ServiceNow. Check `data/assignment_groups.json` and ensure names are identical to ServiceNow group names, including case and spacing.

**Azure Blob connection failing**
The agent falls back to the local JSON file gracefully — no data is lost. Check the connection string in `config.yaml` and that the container and blob names match. Verify the storage account firewall allows your IP address.

**Database locked errors**
Only run one instance of `main.py` or `run_agent.py` at a time. SQLite supports only one concurrent writer.

**Retrain accuracy is low after adding feedback**
Aim for at least 20 resolved tickets per assignment group. If you have fewer, the model does not have enough signal. Use `python add_training_tickets.py` to add more resolved examples, then export and retrain.

---

## Next Steps (Production Enhancements)

- **Replace polling with ServiceNow webhooks** — use Business Rules and a REST Message outbound call to push new tickets to the agent instead of polling every 60 seconds, reducing latency to near-real-time.
- **Deploy on Azure App Service or AKS** — containerise with Docker using the provided `.dockerignore`.
- **Add Azure Key Vault** for secrets management instead of plain environment variables.
- **Connect Azure Monitor / Application Insights** for uptime dashboards and alerting on agent errors.
- **Scheduled retraining** — run `retrain_now.py` via an Azure Functions cron trigger nightly to keep the model fresh without manual intervention.
- **Add sentence-transformer embeddings** to the feature vector for significantly better NLP routing on free-text fields, especially for multilingual tickets.
- **Introduce CMDB data** as an additional feature signal — Configuration Item to owning team mapping tends to be a very strong routing signal.
- **Multi-language support** — the keyword dictionaries already include German and Portuguese terms; extend `TEAM_KEYWORDS` for other locales used in your organisation.
- **Slack / Teams notifications** — post a message when a batch is processed, when the model is retrained, or when accuracy drops below a threshold.
- **A/B testing** — run two models simultaneously and compare routing accuracy before promoting a new model to production.
