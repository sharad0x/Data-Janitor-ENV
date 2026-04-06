---
title: Data Janitor Agent Pipeline
emoji: 🧹
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Autonomous Data Janitor 🧹

This is a complete hackathon submission for the OpenEnv Data Janitor track. It features an autonomous LLM-powered agent designed to systematically clean, impute, and transform messy real-world datasets into ML-ready pipelines.

The agent interacts with an OpenEnv physics engine to dynamically inspect dataset schema, missing values, skewness, and outliers, and takes sequential actions to maximize the performance of a downstream Random Forest classifier.

## 🌟 Key Features
* **Dynamic Environment Loading:** Automatically loads the correct Kaggle datasets based on the task (Easy: Credit Card, Medium: Stroke, Hard: Spaceship Titanic).
* **Robust Agent Architecture:** Features exponential backoff for API rate limits and a "Rate Limit Fallback" to prevent pipeline crashes during peak traffic.
* **Smart Data Heuristics:** Implements targeted outlier clipping, log1p transformations for skewed data, and automated high-cardinality string dropping for identifiers.
* **Strict Evaluation Compliance:** Emits perfectly formatted `[START]`, `[STEP]`, and `[END]` logs for automated regex graders.

## 📂 Project Structure
```text
data_janitor/
├── data/                  # Raw messy datasets (easy, medium, hard)
├── server/
│   ├── app.py             # FastAPI OpenEnv Server entry point
│   └── data_janitor_env.py# Data Physics & ML Grader logic
├── Dockerfile             # Container definitions (Moved to root for build context)
├── inference.py           # The LLM Agent Logic & Client Setup
├── models.py              # Pydantic schemas for Actions & Observations
├── openenv.yaml           # Task metadata and benchmark configs
├── pyproject.toml         # Modern dependency management via 'uv'
└── README.md              # Documentation & HF Metadata
```

## 🛠️ Local Testing & Deployment
Ensure you have uv and Docker installed.

1. Install dependencies:
```bash
uv sync
```

2. Set your environment variables to point to the Docker image:
```bash
export LOCAL_IMAGE_NAME="data-janitor-env"
export TASK_NAME="hard"
export HF_TOKEN="your_hugging_face_token"
```

3. Run the inference agent against the isolated container:
```bash
python inference.py
```

---

## 📈 Baseline Agent Scores
The included `inference.py` script uses the OpenAI client to run a **meta/llama-3.1-70b-instruct agent (via NVIDIA NIM)** against the environment. Due to the strict "Zero NaN" evaluation policy, agents must successfully impute or drop all problematic columns before the final step.

| Task Difficulty | Kaggle Dataset | Max Steps | Expected Baseline Score (F1) |
| :--- | :--- | :--- | :--- |
| **Easy** | Credit Card Fraud | 20 | ~0.94 |
| **Medium** | Stroke Risk | 30 | ~0.58 |
| **Hard** | Spaceship Titanic | 40 | ~0.74 |

*(Note: If the agent fails to clean all NaNs within the step limit, the environment strictly returns a `0.0` score to enforce data pipeline integrity).*

---

## 📊 Evaluation Logic
The environment uses a Random Forest Classifier with a macro-F1 score metric.
Note: A strict "Zero NaN" policy is enforced—any dataset submitted with remaining missing values results in a score of 0.0 to ensure total data integrity.
