# The Complete AI Engineer Roadmap — Beginner to Job-Ready (2026)

---

## 0. How to read this document — and your personalized starting point

You are **not** starting from zero. Most roadmaps assume a true beginner who has never coded. You already have:

- Strong full-stack/backend engineering (Node.js, Express, MongoDB, REST APIs, JWT auth) from two internships
- Git/GitHub, CLI, JSON, HTTP/APIs, debugging, testing (Jest) — all already practiced in a real project (your Candidate Data Transformer pipeline)
- Active DSA practice in C++ (DP, graphs, heaps)

This means **Stage 1 is 80% revision, not new learning** — the only new thing is **Python**, because Python is the working language of the entire ML/AI ecosystem (NumPy, PyTorch, Hugging Face, LangChain). Keep doing DSA in C++ — you do **not** need to redo it in Python; interviewers don't care which language you used for DSA.

Everywhere in this roadmap, I'll flag **[You can compress this]** vs **[This is genuinely new for you]** so you don't waste time.

---

## 1. The Master Prerequisite Chain

```
Python (fast, since you already code)
   → Math for ML (Linear Algebra, Calculus, Probability — applied only)
      → Data Handling (NumPy, Pandas, SQL, EDA)
         → Machine Learning (Scikit-learn)
            → Deep Learning (PyTorch: NN → CNN → RNN → Transformer)
               → Generative AI & LLMs (APIs, prompting, fine-tuning, Hugging Face)
                  → RAG & LLM Applications (embeddings, vector DBs, LangChain/LlamaIndex)
                     → AI Agents (LangGraph, tool calling, MCP)
                        → MLOps & Deployment (FastAPI, Docker, CI/CD, monitoring)
                           → AI System Design (interview-level system design)
```

Do not skip forward. Each stage's tools are only as strong as the fundamentals below them — a person who can call an LLM API but doesn't understand embeddings, tokens, or overfitting will plateau fast and be exposed in interviews.

---

## 2. Milestone Table (Level 0 → Level 9)

| Level | Focus | Est. Time (studying ~15-20 hrs/week) | Ready-for-next-level test |
|---|---|---|---|
| 0 | Setup & orientation | 2-3 days | Python, Git, VS Code, Jupyter all installed and working |
| 1 | Python foundations | 2-3 weeks (compressed — see note) | Can write a class-based CLI tool with error handling and tests, unaided |
| 2 | Math + Data handling | 4-5 weeks | Can explain gradient descent on paper; can clean a messy CSV and produce EDA plots without a tutorial |
| 3 | Classical ML | 5-6 weeks | Can take a raw tabular dataset to a tuned, evaluated Scikit-learn model, unaided |
| 4 | Deep Learning | 6-8 weeks | Can build and train a CNN and a small Transformer from scratch in PyTorch, explain backprop |
| 5 | GenAI & LLMs | 4-5 weeks | Can explain tokenization/embeddings/attention; can fine-tune a small model with LoRA |
| 6 | RAG & LLM apps | 4-5 weeks | Can build a RAG app over your own PDFs with proper chunking + evaluation, unaided |
| 7 | AI Agents | 3-4 weeks | Can build a multi-step tool-using agent with LangGraph and explain failure modes |
| 8 | MLOps & Deployment | 4-5 weeks | Can containerize, deploy, and monitor a model/LLM API end-to-end |
| 9 | System Design & Job Prep | Ongoing (3-4 weeks focused) | Can whiteboard "Design a RAG customer support bot" in 30-45 minutes |

**Total: roughly 8-10 months of consistent, focused work** to go from where you are now to genuinely job-ready for AI Engineer / GenAI Engineer roles — faster than a true beginner because Stage 1 and parts of Stage 3 (APIs, backend, testing) are already yours.

---

## 3. Stage-by-Stage Detail

### Stage 1: Programming Foundations — **[Mostly compression for you]**

**Why it exists:** Every later stage assumes fluent programming. You already have this in JS; Python is the one gap.

**What to learn (Must Learn):**
- Python syntax, data types, control flow, functions — 3-4 days (you'll recognize everything from JS)
- OOP in Python (classes, inheritance, dunder methods) — you already know OOP from Node/Express class-based patterns
- Python-specific idioms: list/dict comprehensions, generators, decorators, context managers (`with`), `*args`/`**kwargs`
- Virtual environments (`venv`) and `pip`/`poetry` — direct equivalent of `npm`/`package.json`
- File I/O, exception handling, logging module
- Writing tests with `pytest` (you already know Jest — same mental model)

**What to skip initially:** Async Python (`asyncio`) — revisit only when you get to FastAPI in Stage 9. Metaclasses, deep C-extension internals — not needed.

**Prerequisites:** None — your existing programming background is the prerequisite, and you've already met it.

**Depth needed:** You should be able to read and write idiomatic Python without translating from JS in your head. That's it — you don't need to be a Python wizard, just fluent.

**Connects to next stage:** NumPy/Pandas are just Python libraries — once Python syntax is second nature, Stage 3 (data handling) is fast.

**Resources:**
- [Must] **"Python for Everybody" (Dr. Chuck, freeCodeCamp/YouTube)** — fast, practical, skip the very basic parts you already know from JS
- [Should] **Official Python docs tutorial** (docs.python.org) — best reference for idioms
- [Optional] *Fluent Python* (book) — only if you want deep mastery later, not needed now

**Project (Beginner):**
1. **Name:** CLI Personal Finance Tracker
2. **Problem:** Track expenses/income from CSV, categorize, summarize
3. **Tech:** Python, argparse, pytest, JSON/CSV
4. **Concepts practiced:** OOP, file I/O, exceptions, testing
5. **Features:** add/list/filter transactions, category summaries, monthly report
6. **Advanced improvement:** Add a SQLite backend instead of CSV
7. **What you'll learn:** Python fluency without following a tutorial line-by-line (build this from your own memory of the finance-tracker pattern you'd build in Node)

---

### Stage 2: Mathematics for AI — **[Genuinely new, but keep it applied, not academic]**

**Why it exists:** ML/DL algorithms are literally linear algebra + calculus + probability implemented as code. You need enough to read a paper's equations and know *what a model is doing*, not to prove theorems.

**Depth required for an AI Engineer (important):** You need **applied intuition + the ability to read/write the math in code**, not a math degree. Skip formal proofs, skip abstract vector spaces, skip most of theoretical statistics.

**Must Learn:**
- **Linear Algebra:** vectors, matrices, matrix multiplication, dot product, transpose, identity/inverse (concept only), norms. Eigenvalues/eigenvectors — *just enough* to understand PCA (you don't need to hand-compute them).
- **Calculus:** derivatives, the chain rule, partial derivatives, gradients — enough to understand backpropagation intuitively. You will never manually differentiate a loss function in your job (autograd does it) but you must understand *what a gradient means*.
- **Probability & Statistics:** mean/variance/std, distributions (normal, Bernoulli, categorical — since these show up as output distributions in classifiers and LLMs), conditional probability, Bayes' theorem (crucial for Naive Bayes and intuition about LLM next-token prediction), basic hypothesis testing (know what a p-value is, don't obsess over it).

**Skip initially:** Proof-based linear algebra, measure theory, most of frequentist hypothesis-testing machinery, multivariable calculus beyond partial derivatives/gradients.

**Prerequisites:** None beyond high-school math + willingness to see math as code, not symbols. Implement everything in NumPy as you learn it — that's the AI-engineer way to learn math, not doing it on paper.

**Resources:**
- [Must] **3Blue1Brown "Essence of Linear Algebra" + "Essence of Calculus" (YouTube)** — best visual intuition, ~4 hours total
- [Must] **StatQuest with Josh Starmer (YouTube)** — for probability/statistics/ML-stats intuition, extremely practical
- [Optional] *Mathematics for Machine Learning* (free book, mml-book.github.io) — only if you want a written reference

**Project:** Implement gradient descent from scratch in NumPy to fit a line to noisy data (no sklearn). This single project cements linear algebra + calculus + Python together and is a great "I understand fundamentals" talking point in interviews.

---

### Stage 3: Data Handling and Analysis — **[Partially familiar — you already know data pipelines]**

**Why it exists:** Every ML/AI system is downstream of data quality. You already built a data pipeline in Node — the concepts transfer, the tools (NumPy/Pandas) are new but fast to learn.

**Must Learn:**
- **NumPy:** arrays, vectorized ops, broadcasting, indexing/slicing
- **Pandas:** DataFrames, filtering, groupby, merge/join, handling missing values, pivoting
- **EDA:** distributions, correlations, outlier detection, visualization (Matplotlib/Seaborn)
- **SQL:** SELECT/JOIN/GROUP BY/window functions — you likely know some from MongoDB work, but SQL is a distinct and essential skill (most companies still run on relational data)
- **Feature engineering:** encoding categoricals, scaling/normalization, datetime features, basic feature creation

**Should Learn:** Basics of data pipelines/ETL concepts (you already intuitively know this from your Node pipeline project — just map it to `pandas`/Airflow-style thinking, don't learn Airflow yet)

**Skip initially:** Spark/big-data tools, advanced SQL window-function gymnastics, dbt — revisit only if a job posting demands it.

**Resources:**
- [Must] **Kaggle's free "Pandas" and "Intro to SQL" micro-courses** — short, hands-on, exactly production-relevant
- [Must] **Official Pandas docs "10 minutes to pandas"** — best quick reference
- [Should] **Mode Analytics SQL Tutorial** (free, interactive)

**Project (Intermediate):**
1. **Name:** Real-World EDA + Cleaning Pipeline
2. **Problem:** Take a genuinely messy public dataset (e.g., a Kaggle "dirty" dataset) and produce a clean, analysis-ready dataset + insights report
3. **Tech:** Pandas, NumPy, Matplotlib/Seaborn, Jupyter
4. **Concepts:** missing value strategies, outlier handling, feature engineering, visualization storytelling
5. **Features:** automated data-quality report, before/after comparison, key insight visualizations
6. **Advanced improvement:** wrap it as a reusable Python package/CLI (leverages your Stage 1 skills)
7. **What you'll learn:** the "boring 80%" of real ML work that most tutorials skip

---

### Stage 4: Machine Learning

**Why it exists:** Classical ML is still the backbone of most production systems (tabular data, recommendation, fraud, ranking) and is the foundation every deep-learning concept builds on (loss functions, overfitting, evaluation, regularization — all reused verbatim in DL).

**Must Learn (concepts, deeply):**
- Supervised vs unsupervised learning, the train/val/test split, cross-validation
- Regression (linear, logistic) — understand the math, not just the API call
- Classification metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- Bias-variance tradeoff, overfitting/underfitting, regularization (L1/L2)
- Decision Trees, Random Forest, Gradient Boosting (XGBoost/LightGBM) — these win most tabular-data competitions and are heavily used in industry
- Clustering (K-Means), dimensionality reduction (PCA) — practical familiarity is enough
- Hyperparameter tuning (GridSearch/RandomSearch/Optuna basics)
- Handling imbalanced datasets (class weights, SMOTE, resampling)

**Should Learn:** SVM, KNN, Naive Bayes — understand *when* to use each, don't over-invest in manual math for these.

**Skip initially:** Deriving every algorithm's math from scratch (know gradient boosting's *idea*, not its full derivation), obscure classical algorithms nobody uses in production anymore.

**Depth guidance:** You should be able to explain *why* you chose an algorithm and *why* a model is overfitting — this is what separates an engineer from someone who just calls `.fit()`.

**Resources:**
- [Must] **Andrew Ng's "Machine Learning Specialization" (Coursera, audit free)** — the industry-standard foundation, teaches intuition + math together
- [Must] **Scikit-learn official documentation + "User Guide"** — genuinely one of the best-written docs in ML
- [Should] **StatQuest playlists on Random Forest/XGBoost/Gradient Boosting**

**Project (Intermediate → Advanced):**
1. **Name:** End-to-End Churn/Fraud Prediction System
2. **Problem:** Predict customer churn or transaction fraud on an imbalanced, realistic dataset
3. **Tech:** Scikit-learn, XGBoost, Pandas, Optuna (for tuning)
4. **Concepts:** class imbalance handling, feature engineering, cross-validation, model comparison, interpretability (SHAP)
5. **Features:** baseline model → tuned model → SHAP explainability dashboard
6. **Advanced improvement:** wrap in a FastAPI endpoint (bridges into Stage 9)
7. **What you'll learn:** the complete classical ML workflow companies actually use, and how to defend model choices in an interview

---

### Stage 5: Deep Learning

**Why it exists:** This is the bridge from classical ML to modern GenAI. Transformers (which power every LLM) are neural networks — you cannot understand LLMs deeply without understanding backprop, attention, and how neural nets are trained.

**Must Learn:**
- Perceptron, forward pass, loss functions (MSE, cross-entropy), backpropagation (conceptually + in code)
- Activation functions (ReLU, sigmoid, softmax) and why they matter
- Optimizers (SGD, Adam) — know what "learning rate" actually controls
- CNNs — convolution, pooling, why they suit images (practical familiarity is enough unless you go into computer vision specifically)
- RNNs/LSTMs/GRUs — understand *why* they existed and *why* they were replaced by attention (this history matters for interviews)
- **Attention mechanism** and **Transformer architecture** — this is the single most important concept in this entire roadmap. Spend real time here: self-attention, multi-head attention, positional encoding, encoder/decoder structure
- Transfer learning and fine-tuning (concept: reuse pretrained weights, adapt final layers)

**Framework:** **PyTorch** as primary (dominant in research and increasingly in industry/GenAI tooling). Know Keras/TensorFlow exists and can read it, but don't split focus.

**Skip initially:** Exotic architectures (GANs, diffusion model internals) unless you specifically want a generative-vision path — for LLM/GenAI Engineering, Transformers are what matter most.

**Resources:**
- [Must] **"Neural Networks: Zero to Hero" by Andrej Karpathy (YouTube)** — builds backprop and a mini-GPT from scratch in code; this is the single best resource for genuinely understanding transformers, not just using them
- [Must] **PyTorch official 60-minute blitz + docs**
- [Should] **fast.ai "Practical Deep Learning for Coders"** — very hands-on, project-first

**Project (Advanced):**
1. **Name:** Build-Your-Own Mini-GPT (character-level)
2. **Problem:** Implement a small Transformer from scratch in PyTorch and train it on a text corpus
3. **Tech:** PyTorch, tokenization (simple char-level), attention implementation
4. **Concepts:** self-attention, positional encoding, training loop, loss curves
5. **Features:** train on a small corpus, generate text, visualize attention weights
6. **Advanced improvement:** scale up to subword tokenization (BPE) and compare
7. **What you'll learn:** this project alone will make you understand LLMs at a level most "prompt engineers" never reach — it's an exceptional resume/portfolio piece for GenAI roles

---

### Stage 6: Generative AI and Large Language Models

**Why it exists:** This is where "AI Engineer" as a 2025-2026 job title actually lives day-to-day — building with, around, and sometimes fine-tuning LLMs.

**Must Learn:**
- What Generative AI is vs discriminative models
- Tokenization (BPE), embeddings, context windows — and their cost/latency implications
- Prompt engineering: zero/few-shot, chain-of-thought, system prompts, structured output (JSON mode), function/tool calling
- LLM APIs (Anthropic, OpenAI) — request/response structure, streaming, token limits, cost management
- Open-source models via **Hugging Face** (transformers library, `pipeline()`, model hub)
- Inference concepts: temperature, top-p/top-k sampling, latency vs quality tradeoffs
- **Fine-tuning basics, LoRA/PEFT, quantization (4-bit/8-bit)** — know when fine-tuning is actually necessary (rare — most problems are solved by prompting or RAG first)
- Hallucinations, guardrails, basic LLM safety/red-teaming awareness
- **The three-way distinction (a common interview question):**
  - *Using an API* — fastest, no infra, pay-per-token, best for 90% of real products
  - *Fine-tuning* — needed when you must change model *behavior/style/domain knowledge* consistently and prompting/RAG can't achieve it cost-effectively
  - *Training from scratch* — essentially never done by an "AI Engineer"; reserved for research labs with massive compute; know this exists but you will not do it in industry

**Should Learn:** Model evaluation basics (benchmarks, LLM-as-judge concept)

**Skip initially:** Full pretraining runs, distributed training infrastructure (DeepSpeed/Megatron) — that's a different job (ML Research Engineer / Infra Engineer), not AI Engineer.

**Resources:**
- [Must] **Anthropic's own prompt engineering docs** (docs.claude.com) — directly relevant since you're likely to be interviewed by/build on Claude-family or GPT-family APIs
- [Must] **Hugging Face NLP Course (free, huggingface.co/course)** — the standard for transformers-library fluency
- [Should] **DeepLearning.AI short courses** (e.g., "Finetuning Large Language Models", "How Diffusion Models Work" only if relevant) — free, 1-2 hours each, very practical

**Project (Advanced):**
1. **Name:** Fine-Tune a Small Open-Source LLM with LoRA
2. **Problem:** Adapt a small model (e.g., a 1-3B parameter open model) to a niche task/domain (e.g., resume-bullet rewriting, given your resume-tailoring experience)
3. **Tech:** Hugging Face Transformers + PEFT, LoRA, bitsandbytes (quantization), a free-tier GPU (Colab/Kaggle)
4. **Concepts:** parameter-efficient fine-tuning, quantization tradeoffs, before/after eval
5. **Features:** dataset prep, training loop, quantitative + qualitative eval vs base model
6. **Advanced improvement:** deploy the fine-tuned model behind a small API
7. **What you'll learn:** genuine fine-tuning experience — a strong differentiator since most "AI Engineers" only ever call APIs and never fine-tune anything

---

### Stage 7: RAG and AI Applications

**Why it exists:** RAG is the most commonly *actually deployed* GenAI pattern in industry — it lets an LLM answer using your private/current data without retraining.

**Must Learn:**
- Embeddings for retrieval (distinct from the "embeddings" concept in Stage 6 — here it's about semantic similarity)
- Vector databases: **FAISS** (learn this first — free, local, teaches the mechanics) then **Chroma** or **Pinecone** (managed, production-friendly)
- Chunking strategies (fixed-size, semantic, recursive) — and why bad chunking is the #1 cause of bad RAG systems
- Semantic search vs keyword search vs **hybrid search**
- Reranking (cross-encoders) — improves retrieval precision significantly
- Retrieval evaluation (precision@k, recall@k) and RAG evaluation (faithfulness, answer relevance — tools like RAGAS)
- Context engineering: how much/what to stuff into the prompt, context window management

**Tool guidance (don't learn everything blindly):**
- **LangChain** — Should Learn: good for quick prototyping and has the widest ecosystem/integrations; learn its core abstractions (loaders, splitters, retrievers, chains) but don't memorize every integration
- **LlamaIndex** — Should Learn: purpose-built for RAG/data-indexing specifically, often cleaner for pure retrieval pipelines; learn it *after* LangChain so you can compare
- **LangGraph** — save for Stage 8 (agents), it's an orchestration layer, not core RAG
- **Recommendation:** start with **FAISS + raw Python** to understand the mechanics with no framework magic, *then* learn LangChain or LlamaIndex to move faster in real projects. This order matters — framework-first learners often can't debug RAG systems because they never saw the mechanics.

**Skip initially:** Every vector DB (you don't need to learn Pinecone, Weaviate, Qdrant, Milvus all at once) — pick FAISS (local/free) + one managed option (Pinecone or Chroma), that covers 95% of interview and job needs.

**Resources:**
- [Must] **LlamaIndex + LangChain official "Getting Started" docs** — both have excellent free docs with runnable examples
- [Must] **DeepLearning.AI "Building Applications with Vector Databases" / "LangChain for LLM Application Development"** (free short courses)
- [Should] **RAGAS documentation** for evaluation

**Project (Production-level portfolio project):**
1. **Name:** Production-Grade PDF/Document Q&A System
2. **Problem:** Upload documents (PDFs, docs) and ask natural-language questions with cited, accurate answers
3. **Tech:** FastAPI backend (leverages your backend skills!), LlamaIndex or LangChain, FAISS/Chroma, an embedding model, an LLM API
4. **Concepts:** chunking strategy comparison, hybrid search, reranking, RAGAS evaluation, citation of sources
5. **Features:** multi-document upload, source citations in answers, chat history, evaluation dashboard showing retrieval quality
6. **Advanced improvement:** add hybrid search + reranking, add streaming responses, add a simple React frontend (you already know React!)
7. **What you'll learn:** this is your single strongest portfolio project — it combines your existing backend/frontend strength with new RAG skills, is directly what most "GenAI Engineer" job descriptions ask for, and is demoable in an interview

---

### Stage 8: AI Agents

**Why it exists:** Agents are the current frontier of GenAI engineering — systems that plan, use tools, and act autonomously rather than just answering one-shot questions. This is increasingly a distinct hiring category.

**Must Learn:**
- What makes something an "agent" vs a simple LLM call (planning + tool use + memory + iteration)
- Agent architecture: the reasoning loop (e.g., ReAct pattern — reason, act, observe, repeat)
- Tool/function calling — you've already used this conceptually via API design; now the LLM decides *when* to call your tools
- Memory (short-term/conversation vs long-term/persistent)
- Multi-agent systems: when and why to split work across specialized agents vs one agent
- Human-in-the-loop patterns (approval gates before risky actions)
- Agent evaluation and reliability — agents fail in different ways than single LLM calls (infinite loops, wrong tool choice, hallucinated tool args) — must know how to test for these
- **Agent security:** prompt injection via tool outputs/retrieved content, over-permissioned tools, data exfiltration risks — genuinely important, not optional

**Frameworks:**
- **LangGraph** — Must Learn: graph-based agent orchestration, the current industry standard for controllable, production agent workflows; its explicit-state-machine approach also matches your Node/Express mental model of explicit control flow
- **MCP (Model Context Protocol)** — Should Learn: the emerging standard for connecting agents/LLMs to tools and data sources in a provider-agnostic way; given your backend experience, building an MCP server is a very natural, resume-worthy project
- Simple single-purpose "Agents SDK"-style frameworks — Optional/Learn Later: useful to know exist, but LangGraph + raw function-calling covers the core skill

**Skip initially:** Complex multi-agent frameworks beyond 2-3 agents, fully autonomous long-horizon agents — these are still research-y and rarely production-stable; focus on reliable, scoped, tool-using agents first.

**Resources:**
- [Must] **LangGraph official documentation + tutorials** (langchain-ai.github.io/langgraph)
- [Must] **Anthropic's "Building Effective Agents" engineering blog post** — one of the best practical, non-hype guides to when/how to use agents at all
- [Should] **modelcontextprotocol.io docs** for MCP

**Project (Production-level portfolio project):**
1. **Name:** AI Coding/Research Assistant Agent with MCP Tools
2. **Problem:** An agent that can search the web, read/write files, run code, and answer multi-step technical questions — with an MCP server exposing your own tools (e.g., querying your resume-project database)
3. **Tech:** LangGraph, an LLM API, MCP server (Python), FastAPI
4. **Concepts:** ReAct loop, tool calling, state management, human-in-the-loop approval for destructive actions, error handling/retries
5. **Features:** multi-step task planning, tool use with logging/tracing, a safety check-in before any "write" action
6. **Advanced improvement:** add a second specialized sub-agent (e.g., a "verifier" agent that checks the main agent's output)
7. **What you'll learn:** real agent-reliability engineering — the difference between a flashy demo and something you could defend as "production-ready" in an interview

---

### Stage 9: MLOps and AI Deployment

**Why it exists:** A model or agent that only runs in a notebook isn't a product. This stage is where your existing backend strength (Node/Express/MongoDB/JWT/REST) becomes a major asset — the concepts transfer almost 1:1, just with Python/FastAPI as the new surface.

**Must Learn (essential for a beginner AI Engineer):**
- **FastAPI** — build REST APIs around models/LLMs/agents (directly analogous to your Express experience — you will learn this fast)
- **Docker** — containerize your app; know Dockerfile basics, multi-stage builds, docker-compose
- Basic CI/CD (GitHub Actions) — lint/test/build/deploy pipeline
- Cloud fundamentals — pick **one** provider deeply rather than all three shallowly; AWS is the safest default for job-market breadth (EC2/S3/Lambda basics, IAM concepts)
- Model/LLM serving basics — synchronous vs streaming responses, batching
- **Experiment tracking with MLflow** — logging metrics/params/models, comparing runs
- Logging, monitoring, basic observability (structured logs, request tracing)
- Model/data drift — conceptual understanding: why deployed models degrade over time
- Background jobs and message queues (Celery + Redis, or a cloud queue) — for long-running AI tasks (e.g., document ingestion pipelines)
- Caching (Redis) — critical for LLM cost/latency control (cache repeated queries/embeddings)

**Should Learn:** Kubernetes overview (concepts only — pods, deployments, services; don't need to run a cluster yourself yet), serverless deployment (AWS Lambda/Cloud Run) — genuinely useful for lightweight LLM endpoints, GPU basics (what a GPU actually accelerates, VRAM constraints for local model serving)

**Learn Later:** Deep Kubernetes administration, multi-cloud strategy, advanced distributed-systems-scale serving (vLLM internals, Triton Inference Server) — revisit only if a specific job needs it.

**Resources:**
- [Must] **FastAPI official documentation** — exceptionally good docs, tutorial-first
- [Must] **Docker's official "Get Started" guide**
- [Should] **MLflow documentation quickstart**

**Project (Production-level portfolio project — this is your capstone-adjacent deployment project):**
1. **Name:** Deploy Your RAG/Agent System as a Real Product
2. **Problem:** Take your Stage 7 or Stage 8 project and make it genuinely production-grade
3. **Tech:** FastAPI, Docker, GitHub Actions CI/CD, MLflow (for any fine-tuned model tracking), Redis caching, deployed on a free/cheap cloud tier (AWS free tier / Render / Railway)
4. **Concepts:** containerization, CI/CD, monitoring/logging, caching, rate limiting, basic auth (JWT — you already know this!)
5. **Features:** Dockerized API, automated tests in CI, structured logging + a simple monitoring dashboard, rate-limited endpoints, cached LLM responses
6. **Advanced improvement:** add a staging/production environment split, add load testing (Locust)
7. **What you'll learn:** the "last mile" that separates a tutorial project from a resume-line that survives interview scrutiny — and it's the stage where your existing backend skills give you a real edge over typical ML-only candidates

---

### Stage 10: AI System Design

**Why it exists:** Senior/mid-level AI Engineer interviews increasingly include a system-design round specific to AI systems (not generic distributed-systems design). This stage teaches you to reason about tradeoffs at the architecture level.

**Must Learn — how to structure any AI system design answer:**
1. Clarify requirements (scale, latency, cost constraints, accuracy needs)
2. Data flow design (ingestion → processing → storage → retrieval/inference → response)
3. Component choices with tradeoffs (which vector DB, which model size, cache layers)
4. Scalability (horizontal scaling of stateless API layers, async processing for heavy tasks)
5. Latency optimization (streaming, caching, smaller models for simple queries + routing to larger models for hard ones)
6. Cost optimization (token usage, caching, model routing/cascading, batch vs real-time)
7. Security (auth, rate limiting, prompt-injection defenses, PII handling)
8. Observability (logging, tracing requests through the pipeline, eval pipelines in production)
9. Failure handling and fallback strategies (what happens when the LLM API is down or returns garbage — retries, fallback models, graceful degradation)

**Example problems to practice (do these on a whiteboard/paper, out loud, timed):**
- Design a ChatGPT-like application (multi-turn chat, streaming, conversation memory, rate limits)
- Design a PDF question-answering system (ingestion pipeline, chunking, retrieval, citation)
- Design an AI customer support agent (tool use, escalation to humans, guardrails)
- Design an AI coding assistant (context management across a codebase, tool use for running code, safety)
- Design a recommendation system (feature pipeline, model serving, A/B testing, cold-start problem)

**Resources:**
- [Must] **"System Design for Machine Learning" style resources** — search current 2026 write-ups from ML system design interview guides (this space evolves fast — search for the latest before your interview cycle)
- [Should] Practice writing out full design docs for your own Stage 7/8/9 projects — the best practice is designing systems you actually built

---

## 4. Technology Priority Matrix

| Category | Must Learn | Should Learn | Learn Later |
|---|---|---|---|
| Language | Python | — | Rust/Go (only for high-perf inference roles) |
| ML | Scikit-learn, XGBoost | Optuna | AutoML tools |
| DL | PyTorch | — | TensorFlow/JAX (read-only familiarity) |
| GenAI | Hugging Face Transformers, one LLM API (Anthropic/OpenAI) | PEFT/LoRA, quantization basics | Full pretraining, distributed training (DeepSpeed) |
| RAG | FAISS, one framework (LangChain **or** LlamaIndex) | Reranking, hybrid search, RAGAS | Every vector DB (Pinecone/Weaviate/Qdrant/Milvus) |
| Agents | LangGraph | MCP | Multiple competing agent SDKs |
| Deployment | FastAPI, Docker, Git/GitHub Actions | MLflow, Redis, AWS basics | Kubernetes administration, multi-cloud |
| Data | NumPy, Pandas, SQL | Basic ETL concepts | Spark, Airflow, dbt |

---

## 5. Weekly Study Structure & Learning Strategy

**Ratio:** ~30% theory / 70% hands-on practice and building — you already lean practical from your engineering background, keep that instinct.

**Weekly rhythm (assuming ~15-20 hrs/week):**
- 2 sessions: focused learning (course/docs/video) — take notes in your own words, don't just watch
- 2-3 sessions: hands-on coding along with what you learned
- 1 session: build/extend a personal project *without* a tutorial open
- 1 session: revision — re-explain last week's core concept out loud or in writing (teaching yourself is the strongest retention method)

**The cycle for every stage:** `Learn → Practice small exercises → Build a project (tutorial-assisted) → Rebuild/extend it independently → Deploy it → Document it on GitHub with a proper README`

**When to start building projects:** Immediately — from Stage 1 onward, one small project per stage minimum, as listed above. Don't wait until you "know enough."

**When to start GitHub contributions:** Start pushing every project from day one (private repos are fine initially, go public once you're comfortable). Begin contributing to open-source (small docs fixes, bug reports) once you finish Stage 6 — you'll have enough context to meaningfully engage with GenAI-related repos (LangChain, LlamaIndex, Hugging Face have great first-issue labels).

**When to prepare for interviews:** Start light interview prep (behavioral + reviewing your own projects deeply) from Stage 7 onward — that's when your portfolio starts looking job-ready. Do focused technical + system-design interview prep in the final 4-6 weeks before you start applying seriously.

**DSA:** You're already doing this in C++ — keep a light, steady cadence (3-4 problems/week) throughout the *entire* roadmap rather than a separate phase. Don't stop it for AI learning; most AI Engineer interviews at bigger companies still include a DSA round.

**Avoiding tutorial hell — the hard rule for every stage:** After finishing the stage's primary resource, close it, and rebuild that stage's project from memory + docs only. If you get stuck, look up *just* the specific thing you're stuck on — never rewatch the whole tutorial. This is the single highest-leverage habit in this entire roadmap.

---

## 6. Job-Readiness Section

### Skills checklist (by target role)

| Role | Core requirement beyond general AI Engineering |
|---|---|
| AI Engineer (general) | Strong RAG + deployment (Stages 6-9), solid ML fundamentals |
| Machine Learning Engineer | Deeper Stage 3-4 (classical ML, feature engineering, MLOps rigor) |
| Generative AI Engineer | Deep Stage 5-7 (Transformers, fine-tuning, RAG), less classical-ML emphasis |
| LLM Engineer | Deepest Stage 5-6 (tokenization, fine-tuning, quantization, inference optimization) |

### Portfolio checklist
- [ ] One classical ML project with proper evaluation + explainability (SHAP)
- [ ] One from-scratch deep learning / mini-Transformer project (shows real understanding, not just API usage)
- [ ] One production-grade RAG application (deployed, with citations + evaluation)
- [ ] One AI agent project (tool use, LangGraph, ideally with MCP)
- [ ] At least one project fully deployed (Docker + CI/CD + live URL), not just "runs on my machine"
- [ ] Every project has a clear README: problem, architecture diagram, tradeoffs, how to run it, what you'd improve

### GitHub checklist
- [ ] Consistent commit history (not one giant commit per project)
- [ ] Clean repo structure, `requirements.txt`/`pyproject.toml`, tests included
- [ ] Pinned, working demo links or GIFs/screenshots for anything not trivially runnable
- [ ] Pin your 3-4 strongest projects to your profile

### Resume project ideas (leveraging your background specifically)
- "Production PDF Q&A system" (Stage 7 project) — pairs naturally with your resume-tailoring/LaTeX work you already do
- "MCP server + LangGraph agent" — leverages your backend/API design experience directly
- "Fine-tuned LoRA model for [niche task]" — shows depth beyond "just called an API"

### Interview topics to prepare
- Classical ML: bias/variance, regularization, evaluation metric selection for imbalanced data, why XGBoost often beats deep learning on tabular data
- Deep learning: backprop intuition, why attention replaced RNNs, transformer architecture walkthrough
- GenAI: tokenization/embeddings, when to prompt vs RAG vs fine-tune, hallucination mitigation, prompt injection
- RAG: chunking tradeoffs, retrieval evaluation, why RAG systems fail in practice
- Agents: ReAct loop, tool-calling reliability, agent security
- System design: the Stage 10 framework applied live
- Behavioral: be ready to walk through your resume-tailoring/job-hunt experience and your Node.js pipeline project — genuine past engineering work is a strong signal

---

## 7. Condensed Final Roadmap (One-Line View)

```
Python (fast, revision)
 → Math (applied: LinAlg + Calc + Probability, via NumPy)
   → NumPy/Pandas/SQL/EDA
     → Classical ML (Scikit-learn, XGBoost)
       → Deep Learning (PyTorch: NN → CNN/RNN → Attention/Transformer)
         → GenAI/LLMs (APIs, prompting, Hugging Face, LoRA fine-tuning)
           → RAG (FAISS → LangChain/LlamaIndex, chunking, reranking, eval)
             → AI Agents (LangGraph, tool calling, MCP)
               → MLOps/Deployment (FastAPI, Docker, CI/CD, MLflow, monitoring)
                 → AI System Design (interview-ready)
                   → JOB-READY
```

---

## 8. Personalized 6–12 Month Plan

Given your existing backend/full-stack strength, C++ DSA momentum, and June 2026 graduation, here's a realistic sequencing (starting now, late Aug 2026):

| Months | Focus |
|---|---|
| Month 1 | Python fluency (fast) + applied Math (Stages 1-2). Keep DSA cadence going. |
| Months 2-3 | Data handling + full Classical ML stage (Stages 3-4), first two portfolio-quality projects |
| Months 3-4.5 | Deep Learning through Transformers (Stage 5) — this is the densest stage, don't rush the Karpathy series |
| Months 4.5-6 | GenAI/LLMs + RAG (Stages 6-7) — build your strongest portfolio project here (PDF Q&A system) |
| Months 6-7 | AI Agents (Stage 8) — build the MCP + LangGraph agent project |
| Months 7-8.5 | MLOps/Deployment (Stage 9) — deploy your RAG and/or agent project properly; this is also when to start applying to a few roles to calibrate |
| Months 8.5-10 | System design practice (Stage 10) + focused interview prep, while continuing to apply |
| Ongoing throughout | Light DSA practice (C++, 3-4/week), GitHub documentation, resume iteration |

This gets you to a genuinely strong, differentiated AI Engineer portfolio — backed by real backend/deployment skills most bootcamp-only ML candidates lack — in roughly 8-10 months, well-positioned relative to your June 2026 graduation timeline.
