# Havi Agentic Service Assignment – AI Ticket Routing System
## Complete File-by-File Technical Documentation

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

## 📁 COMPLETE FILE STRUCTURE & DETAILS

### Root Directory Files

---

#### **1. app.py** - Main Web Application

**What it does:**
- Entry point for the entire application
- Starts the Flask/FastAPI web server
- Creates the web UI interface
- Handles HTTP requests from users
- Accepts ticket input from forms/API
- Calls the prediction pipeline
- Displays prediction results back to user

**Key Responsibilities:**
| Responsibility | Description |
|----------------|-------------|
| Web server initialization | Starts the application on a port (e.g., 5000) |
| Route handling | Defines URLs like `/`, `/predict`, `/health` |
| Request processing | Parses incoming ticket data |
| Response rendering | Shows results in HTML format |
| API endpoint | Provides REST API for external integrations |

**Typical Code Structure:**
```python
from flask import Flask, request, render_template
from predict import predict_assignment_group

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticket_text = request.form['ticket']
    result = predict_assignment_group(ticket_text)
    return render_template('result.html', prediction=result)
```

**Think of this as:** Frontend entry point + backend API handler.

---

#### **2. predict.py** - Core Prediction Workflow

**What it does:**
- Main orchestration file for prediction
- Coordinates all agents in sequence
- Manages the end-to-end prediction pipeline
- Ties together preprocessing, embedding, search, and LLM

**Key Responsibilities:**
| Responsibility | Description |
|----------------|-------------|
| Pipeline orchestration | Calls agents in correct order |
| Data flow management | Passes data between agents |
| Error handling | Manages failures gracefully |
| Logging | Records prediction attempts |
| Result aggregation | Collects output from all agents |

**Typical Flow Inside predict.py:**
```python
def predict_assignment_group(ticket_text):
    # Step 1: Preprocess
    cleaned_text = preprocessing_agent.clean(ticket_text)
    
    # Step 2: Generate embedding
    embedding = embedding_agent.encode(cleaned_text)
    
    # Step 3: Search similar tickets
    similar_tickets = knowledge_base_agent.search(embedding)
    
    # Step 4: Get LLM prediction
    prediction = llm_agent.predict(cleaned_text, similar_tickets)
    
    # Step 5: Calculate confidence
    confidence = confidence_agent.calculate(prediction)
    
    # Step 6: Generate explanation
    explanation = explanation_agent.explain(prediction)
    
    return {
        'assignment_group': prediction,
        'confidence': confidence,
        'explanation': explanation
    }
```

**Think of this as:** The brain controller of the application.

---

#### **3. build_knowledge_base.py** - Historical Ticket Training

**What it does:**
- Builds vector embeddings from historical CSV tickets
- Reads training data from CSV files
- Converts ticket text into vector embeddings
- Stores embeddings in ChromaDB
- Creates searchable ticket memory

**Key Responsibilities:**
| Responsibility | Description |
|----------------|-------------|
| CSV loading | Reads historical tickets using Pandas |
| Text extraction | Extracts ticket descriptions and assignment groups |
| Batch processing | Processes tickets in batches for efficiency |
| Embedding generation | Converts text to vectors |
| ChromaDB storage | Saves vectors with metadata |

**Typical Code Structure:**
```python
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

def build_knowledge_base():
    # Load CSV
    df = pd.read_csv('historical_tickets.csv')
    
    # Initialize model
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # Initialize ChromaDB
    client = chromadb.Client()
    collection = client.create_collection('tickets')
    
    # Process each ticket
    for idx, row in df.iterrows():
        text = row['description']
        assignment = row['assignment_group']
        
        # Generate embedding
        embedding = model.encode(text)
        
        # Store in ChromaDB
        collection.add(
            embeddings=[embedding],
            metadatas=[{'assignment_group': assignment, 'text': text}],
            ids=[f'ticket_{idx}']
        )
```

**Think of this as:** Building the system's memory from past tickets.

---

#### **4. build_knowledge_base_docs.py** - KB Document Ingestion

**What it does:**
- Loads knowledge base articles and documents
- Supports multiple file formats (TXT, MD, PDF, HTML)
- Splits large documents into smaller chunks
- Generates embeddings for each chunk
- Stores document chunks in ChromaDB

**Key Responsibilities:**
| Responsibility | Description |
|----------------|-------------|
| Multi-format reading | Parses PDF, TXT, MD, HTML files |
| Document chunking | Splits large docs into smaller pieces |
| Text extraction | Extracts raw text from documents |
| Embedding generation | Creates vectors for each chunk |
| Source tracking | Stores document source information |

**Why Chunking is Important:**
| Issue | Solution |
|-------|----------|
| Large documents exceed token limits | Split into 500-1000 char chunks |
| Retrieval accuracy decreases | Smaller chunks = better semantic search |
| LLM context window limitations | Chunks fit within token limits |

**Typical Code Structure:**
```python
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_doc_knowledge_base():
    # Load documents
    pdf_loader = PyPDFLoader('docs/manual.pdf')
    documents = pdf_loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # Generate embeddings and store
    for chunk in chunks:
        embedding = model.encode(chunk.page_content)
        collection.add(embeddings=[embedding], documents=[chunk.page_content])
```

**Think of this as:** Teaching the system company knowledge from documentation.

---

#### **5. rlhf_train.py** - Reinforcement Learning Training

**What it does:**
- Improves prediction quality over time
- Learns from user feedback
- Fine-tunes routing logic
- Updates model weights based on corrections
- Implements reinforcement learning from human feedback

**Key Responsibilities:**
| Responsibility | Description |
|----------------|-------------|
| Feedback collection | Captures user corrections to predictions |
| Reward calculation | Determines if prediction was correct |
| Model updating | Adjusts weights based on feedback |
| Performance tracking | Monitors improvement over time |
| A/B testing | Compares old vs new models |

**Typical Code Structure:**
```python
def rlhf_training_loop():
    # Collect feedback data
    feedback_data = load_user_feedback()
    
    for feedback in feedback_data:
        ticket = feedback['ticket']
        correct_group = feedback['correct_group']
        predicted_group = feedback['predicted_group']
        
        # Calculate reward (1 for correct, -1 for incorrect)
        reward = 1 if correct_group == predicted_group else -1
        
        # Update model based on reward
        if reward < 0:
            # Adjust weights to improve prediction
            update_model_weights(ticket, correct_group)
    
    # Evaluate improved model
    new_accuracy = evaluate_model()
    save_improved_model()
```

**Think of this as:** Continuous learning from mistakes to get better over time.

---

### 📂 Agents Folder (agents/)

---

#### **6. agents/preprocessing_agent.py** - Text Cleaner

**What it does:**
- Cleans raw incident text before embedding
- Removes noise and special characters
- Normalizes ticket content
- Standardizes formatting
- Prepares AI-friendly text

**Key Responsibilities:**
| Responsibility | Description |
|----------------|-------------|
| Remove special chars | Strips `!@#$%^&*()` etc. |
| Remove ticket IDs | Eliminates `INC000123` patterns |
| Lowercase conversion | Converts all text to lowercase |
| Remove extra spaces | Collapses multiple spaces to one |
| Normalize words | Standardizes common variations |

**Before/After Examples:**
| Before | After |
|--------|-------|
| `INC000123 - VPN ISSUE!!!` | `vpn issue` |
| `EMAIL  NOT  WORKING!!!!` | `email not working` |
| `PASSWORD*&^% RESET` | `password reset` |

**Typical Code Structure:**
```python
import re

class PreprocessingAgent:
    def clean_text(self, text):
        # Remove ticket IDs
        text = re.sub(r'INC\d+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
```

**Think of this as:** The text cleaner that removes garbage before important processing.

---

#### **7. agents/embedding_agent.py** - Vector Generator

**What it does:**
- Converts text into vector embeddings
- Loads transformer models
- Handles batch embedding generation
- Manages model caching
- Returns numerical vectors

**Key Responsibilities:**
| Responsibility | Description |
|----------------|-------------|
| Model loading | Loads Sentence Transformer models |
| Tokenization | Converts text to tokens |
| Encoding | Runs tokens through transformer |
| Vector generation | Produces embedding vectors |
| Batch processing | Handles multiple texts efficiently |

**Models Used:**
| Model | Dimension | Best For |
|-------|-----------|----------|
| all-mpnet-base-v2 | 768 | High accuracy |
| all-MiniLM-L6-v2 | 384 | Speed & balance |

**Typical Code Structure:**
```python
from sentence_transformers import SentenceTransformer

class EmbeddingAgent:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
    
    def encode(self, text):
        # Generate embedding
        embedding = self.model.encode(text)
        # Returns list of floats like [0.234, -0.781, 0.991, ...]
        return embedding
    
    def encode_batch(self, texts):
        # Process multiple texts at once
        return self.model.encode(texts)
```

**Think of this as:** The translator that converts human language to math.

---

#### **8. agents/knowledge_base_agent.py** - ChromaDB Handler

**What it does:**
- Handles all ChromaDB vector database operations
- Inserts embeddings with metadata
- Performs similarity search queries
- Returns top matching results
- Manages database collections

**Key Responsibilities:**
| Responsibility | Description |
|----------------|-------------|
| Collection management | Creates and manages DB collections |
| Vector insertion | Stores embeddings with metadata |
| Similarity search | Performs cosine similarity queries |
| Result ranking | Orders results by similarity score |
| Metadata filtering | Filters searches by metadata |

**Typical Code Structure:**
```python
import chromadb

class KnowledgeBaseAgent:
    def __init__(self):
        self.client = chromadb.PersistentClient('db/')
        self.collection = self.client.get_or_create_collection('tickets')
    
    def search(self, embedding, top_k=5):
        # Search similar vectors
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        return results
    
    def add_ticket(self, text, embedding, metadata):
        # Store ticket with metadata
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
            ids=[generate_id()]
        )
```

**Think of this as:** The librarian that stores and retrieves semantic memories.

---

#### **9. agents/document_ingestion_agent.py** - KB File Processor

**What it does:**
- Processes knowledge base documents
- Reads various file formats
- Extracts text from documents
- Chunks documents into manageable pieces
- Prepares documents for embedding

**Key Responsibilities:**
| Responsibility | Description |
|----------------|-------------|
| File format detection | Identifies PDF, TXT, MD, HTML |
| Text extraction | Pulls raw text from documents |
| Chunking strategy | Splits text intelligently |
| Metadata preservation | Keeps source info with chunks |
| Error handling | Manages corrupted or unreadable files |

**Supported Formats:**
| Format | Library Used |
|--------|---------------|
| PDF | PyPDF2 / pdfplumber |
| TXT | Built-in open() |
| Markdown | markdown library |
| HTML | BeautifulSoup |

**Typical Code Structure:**
```python
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentIngestionAgent:
    def process_document(self, filepath):
        # Load based on extension
        if filepath.endswith('.pdf'):
            loader = PyPDFLoader(filepath)
        elif filepath.endswith('.txt'):
            loader = TextLoader(filepath)
        
        documents = loader.load()
        
        # Chunk documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        chunks = splitter.split_documents(documents)
        
        return chunks
```

**Think of this as:** The document processor that reads and prepares company knowledge.

---

#### **10. agents/llm_agent.py** - AI Reasoning Engine

**What it does:**
- Main AI reasoning engine
- Creates prompts for the LLM
- Sends context to local Ollama models
- Receives predictions from LLM
- Generates explanations for decisions

**Key Responsibilities:**
| Responsibility | Description |
|----------------|-------------|
| Prompt engineering | Constructs effective prompts |
| Ollama integration | Connects to local LLM server |
| Response parsing | Extracts prediction from LLM output |
| Context formatting | Formats retrieved data for LLM |
| Model management | Switches between Mistral/Gemma |

**Prompt Template Example:**
```
You are an IT ticket routing expert. 

NEW TICKET: {ticket_text}

SIMILAR HISTORICAL TICKETS:
{tickets_context}

KB ARTICLES FOUND:
{kb_context}

Based on the above, predict the correct assignment group.
Respond with only the assignment group name.
```

**Typical Code Structure:**
```python
import requests

class LLMAgent:
    def __init__(self, model="mistral"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
    
    def predict(self, ticket_text, similar_tickets, kb_docs):
        prompt = self.build_prompt(ticket_text, similar_tickets, kb_docs)
        
        response = requests.post(self.ollama_url, json={
            "model": self.model,
            "prompt": prompt
        })
        
        return response.json()['response']
```

**Think of this as:** The brain that makes intelligent decisions using AI.

---

#### **11. agents/candidate_group_agent.py** - Team Filter

**What it does:**
- Filters possible assignment groups
- Identifies candidate teams based on similarity
- Narrows down routing options
- Removes impossible or invalid groups

**Key Responsibilities:**
| Responsibility | Description |
|----------------|-------------|
| Group filtering | Removes irrelevant teams |
| Pattern matching | Finds matching team patterns |
| Priority scoring | Ranks candidate teams |
| Constraint validation | Ensures groups exist in system |

**Candidate Groups Example:**
| Ticket Type | Possible Groups |
|-------------|-----------------|
| VPN issues | Network Team, IAM Team, Security Team |
| Email problems | Exchange Team, Infrastructure Team |
| Password reset | IAM Team, Service Desk |

**Typical Code Structure:**
```python
class CandidateGroupAgent:
    def __init__(self):
        self.valid_groups = [
            'Network Team', 'IAM Team', 'Security Team',
            'Service Desk', 'Database Team', 'Application Team'
        ]
    
    def filter_candidates(self, similar_tickets):
        # Extract groups from similar tickets
        candidate_groups = set()
        for ticket in similar_tickets:
            candidate_groups.add(ticket['assignment_group'])
        
        # Return only valid groups
        return [g for g in candidate_groups if g in self.valid_groups]
```

**Think of this as:** The filter that narrows down which teams could handle this ticket.

---

#### **12. agents/confidence_agent.py** - Reliability Calculator

**What it does:**
- Calculates prediction confidence score
- Measures how reliable the prediction is
- Considers similarity scores and LLM certainty
- Returns percentage confidence

**Key Responsibilities:**
| Responsibility | Description |
|----------------|-------------|
| Score aggregation | Combines multiple confidence factors |
| Similarity analysis | Uses ChromaDB distance scores |
| LLM certainty | Analyzes model confidence |
| Threshold classification | Determines high/medium/low confidence |

**Confidence Levels:**
| Level | Score Range | Meaning |
|-------|-------------|---------|
| High | 80-100% | Very reliable prediction |
| Medium | 50-79% | Somewhat reliable, needs review |
| Low | 0-49% | Unreliable, manual routing needed |

**Factors Considered:**
| Factor | Weight |
|--------|--------|
| Top similarity score | 40% |
| LLM confidence | 30% |
| Number of similar matches | 20% |
| Candidate group agreement | 10% |

**Typical Code Structure:**
```python
class ConfidenceAgent:
    def calculate(self, similarity_scores, llm_output):
        # Calculate based on top similarity
        top_score = max(similarity_scores) * 100
        
        # Adjust based on LLM certainty
        if "unsure" in llm_output.lower():
            top_score *= 0.7
        
        # Return final confidence
        return min(100, top_score)
```

**Think of this as:** The quality checker that tells you if you can trust the prediction.

---

#### **13. agents/decision_agent.py** - Final Decision Maker

**What it does:**
- Makes final routing decision
- Combines all agent outputs
- Resolves conflicts between agents
- Selects final assignment group

**Key Responsibilities:**
| Responsibility | Description |
|----------------|-------------|
| Input aggregation | Collects outputs from all agents |
| Conflict resolution | Decides when agents disagree |
| Priority handling | Uses confidence to weight decisions |
| Final selection | Picks one assignment group |

**Decision Logic:**
```
IF confidence > 80%:
    USE LLM prediction directly
ELSE IF confidence > 50%:
    USE candidate group with highest similarity
ELSE:
    FALLBACK to default assignment group + flag for review
```

**Typical Code Structure:**
```python
class DecisionAgent:
    def decide(self, llm_prediction, candidates, confidence):
        if confidence > 80:
            return llm_prediction
        
        elif confidence > 50:
            # Use most common candidate
            return max(set(candidates), key=candidates.count)
        
        else:
            # Flag for human review
            return {
                'assignment_group': 'Service Desk',
                'requires_review': True
            }
```

**Think of this as:** The boss that makes the final call based on all inputs.

---

#### **14. agents/explanation_agent.py** - Reason Generator

**What it does:**
- Generates human-readable explanations
- Explains WHY a team was selected
- Provides transparency for decisions
- Builds trust in AI predictions

**Key Responsibilities:**
| Responsibility | Description |
|----------------|-------------|
| Reason synthesis | Creates explanation from decision factors |
| Template filling | Uses predefined explanation templates |
| Evidence citation | References similar tickets used |
| User-friendly language | Avoids technical jargon |

**Explanation Templates:**
| Scenario | Explanation Template |
|----------|---------------------|
| Similar ticket match | "Matched with previous {ticket_type} incidents handled by {team}" |
| KB article match | "KB article {article_id} suggests this team handles {issue_type}" |
| High confidence | "Strong match with {count} similar historical tickets" |
| Low confidence | "Limited historical data, consider manual review" |

**Typical Code Structure:**
```python
class ExplanationAgent:
    def explain(self, decision, similar_tickets, confidence):
        if confidence > 80:
            top_match = similar_tickets[0]
            return f"Predicted {decision} because similar ticket '{top_match['text']}' was assigned to this team with {top_match['score']}% similarity."
        
        elif confidence > 50:
            return f"Recommended {decision} based on {len(similar_tickets)} similar historical patterns."
        
        else:
            return f"Low confidence prediction. Manual review recommended due to insufficient similar incidents."
```

**Think of this as:** The translator that turns AI decisions into plain English explanations.

---

## 🔄 COMPLETE END-TO-END WORKFLOW

### When a User Submits a Ticket:

```
1. USER INPUT
   "Unable to connect to VPN after password reset"
        ↓
2. app.py RECEIVES REQUEST
   Validates input, calls predict.py
        ↓
3. predict.py ORCHESTRATES
   Calls each agent in sequence
        ↓
4. preprocessing_agent.clean()
   "unable to connect to vpn after password reset"
        ↓
5. embedding_agent.encode()
   [0.234, -0.781, 0.991, -0.442, ...] (768 numbers)
        ↓
6. knowledge_base_agent.search()
   Finds top 5 similar tickets from ChromaDB
        ↓
7. document_ingestion_agent.retrieve()
   Finds related KB articles about VPN
        ↓
8. candidate_group_agent.filter()
   [Network Team, IAM Team]
        ↓
9. llm_agent.predict()
   Analyzes context, returns "Network Team"
        ↓
10. confidence_agent.calculate()
    Calculates 92% confidence
        ↓
11. decision_agent.decide()
    Confirms "Network Team" as final decision
        ↓
12. explanation_agent.explain()
    "Similar VPN incidents were historically assigned to Network Support"
        ↓
13. app.py RETURNS RESULT
    Displays prediction to user
```

---

## 🧠 TRAINING PIPELINES

### Historical Ticket Training (build_knowledge_base.py)
```
CSV File (historical_tickets.csv)
    ↓ Pandas loads data
    ↓ Extract description + assignment_group
    ↓ preprocessing_agent cleans text
    ↓ embedding_agent generates vectors
    ↓ knowledge_base_agent stores in ChromaDB
    
Result: Searchable memory of past tickets
```

### Knowledge Base Training (build_knowledge_base_docs.py)
```
Documents (PDF/TXT/MD/HTML)
    ↓ document_ingestion_agent loads files
    ↓ Text split into 500-character chunks
    ↓ embedding_agent generates vectors per chunk
    ↓ knowledge_base_agent stores in ChromaDB
    
Result: Searchable company knowledge base
```

### Reinforcement Learning (rlhf_train.py)
```
User Feedback (correct/incorrect)
    ↓ Feedback collected
    ↓ Reward calculated (+1/-1)
    ↓ Model weights adjusted
    ↓ Performance evaluated
    
Result: Continuous improvement over time
```

---

## 📊 DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER SUBMITS TICKET                       │
│                  "VPN not working after reset"                   │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                           app.py                                 │
│                    Receives HTTP Request                         │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                         predict.py                               │
│                    Main Orchestration                            │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
        ┌───────────────────────────────────────────┐
        │              AGENT PIPELINE                │
        ├───────────────────────────────────────────┤
        │ 1. preprocessing_agent ──→ cleaned text   │
        │ 2. embedding_agent ──────→ vector [0.2...]│
        │ 3. knowledge_base_agent ─→ similar tickets│
        │ 4. document_ingestion ───→ KB articles    │
        │ 5. candidate_group ──────→ possible teams │
        │ 6. llm_agent ────────────→ prediction     │
        │ 7. confidence_agent ─────→ 92% sure       │
        │ 8. decision_agent ───────→ final choice   │
        │ 9. explanation_agent ────→ reason         │
        └───────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                      FINAL OUTPUT                                │
│  Group: Network Support                                         │
│  Confidence: 92%                                                │
│  Reason: Similar VPN incidents handled by Network Team          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💡 WHY THIS ARCHITECTURE WORKS

| Component | Why It's Needed |
|-----------|-----------------|
| **Preprocessing Agent** | Dirty text = bad embeddings = wrong predictions |
| **Embedding Agent** | Converts meaning to math for similarity search |
| **ChromaDB** | Fast semantic search across thousands of tickets |
| **KB Documents** | Provides company-specific knowledge LLM doesn't have |
| **LLM Agent** | Adds reasoning beyond simple pattern matching |
| **Confidence Agent** | Tells you when to trust vs when to review |
| **Explanation Agent** | Builds trust through transparency |

---

## 🚀 QUICK START COMMANDS

```bash
# Step 1: Build knowledge base from historical tickets
python build_knowledge_base.py

# Step 2: Build knowledge base from documents
python build_knowledge_base_docs.py

# Step 3: Start Ollama (in separate terminal)
ollama serve
ollama pull mistral  # or gemma

# Step 4: Run the application
python app.py

# Step 5: (Optional) Train with feedback
python rlhf_train.py
```

---

## 📝 COMPLETE EXAMPLE

### Input:
```
Ticket: "Unable to connect to VPN after password reset"
```

### What Happens Internally:
| Step | Agent | Input → Output |
|------|-------|----------------|
| 1 | preprocessing | `INC123 - VPN!!!` → `vpn issue` |
| 2 | embedding | `vpn issue` → `[0.23, -0.78, 0.99]` |
| 3 | knowledge_base | Finds 5 similar tickets (95%, 91%, 89%...) |
| 4 | document_ingestion | Finds VPN troubleshooting KB article |
| 5 | candidate_group | `[Network, IAM, Security]` → `[Network, IAM]` |
| 6 | llm | Analyzes → `Network Team` |
| 7 | confidence | 92% confidence |
| 8 | decision | Final: `Network Team` |
| 9 | explanation | "Similar VPN incidents were resolved by Network Team" |

### Output:
```json
{
    "assignment_group": "Network Team",
    "confidence": 92,
    "explanation": "Matched with previous VPN access incidents handled by Network Team. Similarity score 95% with ticket INC004567.",
    "requires_review": false
}
```

---

## 📂 COMPLETE FILE LIST FOR REFERENCE

| # | File | Purpose |
|---|------|---------|
| 1 | `app.py` | Main web application |
| 2 | `predict.py` | Core prediction orchestration |
| 3 | `build_knowledge_base.py` | Train on historical tickets |
| 4 | `build_knowledge_base_docs.py` | Ingest KB documents |
| 5 | `rlhf_train.py` | Reinforcement learning training |
| 6 | `agents/preprocessing_agent.py` | Text cleaning |
| 7 | `agents/embedding_agent.py` | Vector generation |
| 8 | `agents/knowledge_base_agent.py` | ChromaDB operations |
| 9 | `agents/document_ingestion_agent.py` | KB file processing |
| 10 | `agents/llm_agent.py` | AI reasoning |
| 11 | `agents/candidate_group_agent.py` | Team filtering |
| 12 | `agents/confidence_agent.py` | Confidence scoring |
| 13 | `agents/decision_agent.py` | Final decision |
| 14 | `agents/explanation_agent.py` | Reason generation |

---

## 🎯 ONE-LINE SUMMARY

> *"The system predicts the correct IT support assignment group by comparing new incidents with historical tickets and KB articles using embeddings, vector search, and a local LLM - with every component working together through a modular agent architecture."*

---
