# Havi Agentic Service Assignment – AI Ticket Routing System

> **AI-powered intelligent ticket routing system** that automatically predicts the correct assignment group for IT incidents using semantic search, vector databases, and local LLMs.

---

## 📋 Technology Stack

| Technology | Purpose |
|------------|---------|
| Python | Core backend development |
| Flask/FastAPI | API and web application |
| ChromaDB | Vector database storage |
| Sentence Transformers | Embedding generation |
| Transformers | NLP processing |
| Torch | Deep learning runtime |
| Ollama | Local LLM serving |
| Gemma/Mistral | AI reasoning models |
| Pandas | CSV and dataset handling |
| NumPy | Numerical operations |
| Scikit-learn | Similarity utilities |
| LangChain (optional) | AI orchestration |
| HTML/CSS/JS | Frontend UI |

---

## 🎯 Project Goal

**"When a user creates a ticket, the system intelligently identifies which support team or assignment group should handle that issue."**

---

## 🏗️ High-Level Architecture Flow

```
User submits incident/ticket
        ↓
    Frontend/UI Layer
        ↓
    API/Application Layer
        ↓
    Preprocessing Agent
        ↓
    Embedding Agent
        ↓
Vector Database Search (ChromaDB)
        ↓
Retrieve Similar Historical Tickets + KB Articles
        ↓
    Candidate Group Agent
        ↓
    LLM Reasoning Agent
        ↓
    Confidence Calculation
        ↓
    Decision Agent
        ↓
    Explanation Agent
        ↓
Final Assignment Group Prediction
```

**This is a Retrieval-Augmented Generation (RAG) based AI architecture.**

---

## 🔄 Step-by-Step Workflow

### Step 1 – User Ticket Input
User enters ticket description.
> **Example:** *"Unable to connect to VPN after password reset"*

### Step 2 – Preprocessing Agent
Cleans and normalizes the ticket text.

| Before | After |
|--------|-------|
| `INC000123 – VPN ISSUE!!!` | `vpn issue` |

**Activities:** Remove special characters, ticket IDs, extra spaces → lowercase conversion → normalize sentence

### Step 3 – Embedding Generation
Converts text into vector embeddings using **Sentence Transformers**.

```
"vpn access issue" → [0.234, -0.781, 0.991, ...]
```

**Models used:** `all-mpnet-base-v2` or `all-MiniLM-L6-v2`

**Why all-mpnet-base-v2?**
- Excellent semantic similarity accuracy
- Better contextual understanding
- Strong sentence embedding performance
- Good balance between speed and quality

### Step 4 – ChromaDB Semantic Search
Searches embeddings against stored historical tickets and KB articles using **cosine similarity**.

**Stored data includes:**
- Historical incidents
- KB articles
- SOP documents
- Troubleshooting guides

**Example retrieved results:**

| Ticket | Assignment Group | Similarity |
|--------|-----------------|------------|
| VPN login failure | Network Team | 0.95 |
| VPN authentication issue | IAM Team | 0.91 |
| Remote access problem | Network Team | 0.89 |

### Step 5 – LLM Reasoning Layer
The LLM receives and analyzes:
- Current ticket
- Similar incidents
- KB documents
- Candidate assignment groups

**LLM Models (via Ollama):** Gemma | Mistral | Llama

> **Example reasoning:** *"Similar VPN incidents were historically assigned to Network Support."*

### Step 6 – Final Prediction

| Output | Value |
|--------|-------|
| Assignment Group | Network Support |
| Confidence | 92% |
| Reason | Similar VPN incidents were previously resolved by Network Support |

---

## 🤖 Agent-by-Agent Breakdown

| Agent | Responsibility |
|-------|----------------|
| **preprocessing_agent.py** | Clean and normalize text, remove noise, prepare AI-friendly input |
| **embedding_agent.py** | Load transformer model, generate embeddings, convert text to vectors |
| **knowledge_base_agent.py** | ChromaDB interactions, vector insertion, similarity search (semantic memory engine) |
| **document_ingestion_agent.py** | Read KB files, extract text, chunk documents, prepare embeddings |
| **llm_agent.py** | Prompt engineering, send context to LLM, get predictions, generate explanations (reasoning engine) |
| **candidate_group_agent.py** | Identify likely assignment groups, narrow down candidate teams |
| **confidence_agent.py** | Calculate prediction confidence, measure reliability |
| **decision_agent.py** | Combine similarity scores + LLM reasoning, select final assignment |
| **explanation_agent.py** | Generate human-readable explanations |

---

## 📁 Project Structure Walkthrough

| File | Purpose |
|------|---------|
| **app.py** | Main web application (frontend entry + backend API handler) |
| **predict.py** | Core prediction workflow (brain controller of the application) |
| **build_knowledge_base.py** | Builds vector embeddings from historical tickets (CSV → ChromaDB) |
| **build_knowledge_base_docs.py** | Loads KB articles (TXT, MD, PDF, HTML) into ChromaDB |
| **rlhf_train.py** | Reinforcement learning for improvement training |

---

## 🧠 Training & Knowledge Building

### Historical Ticket Training Flow
```
CSV Historical Tickets → Load Dataset → Clean Ticket Text → Generate Embeddings → Store in ChromaDB
```

### Knowledge Base Training Flow
```
KB Documents → Document Loader → Chunking → Embedding Generation → Store in ChromaDB
```

**Supported files:** PDF, TXT, Markdown, HTML

**Why chunking is needed:** Large documents exceed token limits; smaller chunks improve retrieval quality.

---

## 💡 Why Local LLMs?

| Benefit | Description |
|---------|-------------|
| No external API dependency | Self-contained system |
| Better data privacy | Enterprise IT requirement |
| Lower operational cost | No API fees |
| Offline capability | Works without internet |
| Faster internal processing | Reduced latency |

---

## 🔬 RAG Architecture

**RAG = Retriever + Generator**

| Component | Technology |
|-----------|------------|
| Retriever | ChromaDB semantic search |
| Generator | Gemma / Mistral LLM |

**Benefits:**
- Reduces hallucinations
- Uses organizational knowledge
- Improves prediction quality
- Provides explainability

---

## ✅ Why This Architecture Is Powerful

| Advantage | Description |
|-----------|-------------|
| **Semantic Understanding** | Understands meaning instead of keywords |
| **Modular Agent Design** | Easy to extend, debug, and replace components |
| **Explainable AI** | Provides reasons for routing decisions |
| **Local AI Processing** | Data privacy, no cloud dependency, reduced cost |
| **Hybrid Knowledge System** | Combines historical incidents + KB documentation |

---

## 📊 One-Line Summary

> *"The system predicts the correct IT support assignment group by comparing new incidents with historical tickets and KB articles using embeddings, vector search, and a local LLM."*

---

## 🚀 Quick Start Commands

```bash
# Build knowledge base from historical tickets
python build_knowledge_base.py

# Build knowledge base from documents
python build_knowledge_base_docs.py

# Run the main application
python app.py

# Optional: Run reinforcement learning training
python rlhf_train.py
```

---

## 📝 Example Input/Output

**Input:**
> `Unable to connect to VPN after password reset`

**Output:**
```
Assignment Group: Network Support
Confidence: 92%
Explanation: Similar VPN incidents were previously resolved by Network Support.
Matched with previous VPN access incidents handled by Network Team.
```

---

*For questions or clarifications during the meeting, please refer to the agent-specific documentation in the `/agents` folder.*
```

This README is structured for a meeting presentation with:
- Clear section headers for easy navigation
- Tables for technology stack and agent responsibilities
- Visual flow diagrams using ASCII art
- Before/after examples
- Bullet points for key takeaways
- A one-line summary for quick recall
