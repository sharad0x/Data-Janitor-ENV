---
title: Data Janitor Agent Pipeline
emoji: 🧹
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
docker_path: server/Dockerfile
tags:
  - openenv
---

# Autonomous Data Janitor 🧹

This is a complete hackathon submission for the OpenEnv Data Janitor track. It features an autonomous LLM-powered agent designed to systematically clean, impute, and transform messy real-world datasets into ML-ready pipelines.

The agent interacts with an OpenEnv physics engine to dynamically inspect dataset schema, missing values, skewness, and outliers, and takes sequential actions to maximize the performance of a downstream Random Forest classifier.

## Features & Capabilities
* **Dynamic Environment Loading:** Automatically loads the correct Kaggle datasets (Easy: Credit Card, Medium: Stroke, Hard: Spaceship Titanic).
* **Robust Error Handling:** Features exponential backoff for API rate limits and safe fallback actions to prevent pipeline crashes.
* **Smart Heuristics:** Implements targeted outlier handling, log transformations for skewed data, and high-cardinality string dropping.
* **Strict Evaluation Compliance:** Emits perfectly formatted `[START]`, `[STEP]`, and `[END]` logs for automated regex graders.

## Project Structure
```text
data_janitor/
├── data/                           # Messy Kaggle datasets (easy, medium, hard)
├── server/
│   ├── app.py                      # FastAPI OpenEnv Server
│   ├── data_janitor_environment.py # Data Physics & ML Grader logic
│   ├── requirements.txt            # Docker container dependencies
│   └── Dockerfile                  # Container build definitions
├── inference.py                    # The LLM Agent Logic
├── models.py                       # Pydantic schemas for Actions & Observations
├── client.py                       # OpenEnv API communication client
├── openenv.yaml                    # Hackathon task configurations
├── pyproject.toml                  # Local `uv` dependencies
└── .dockerignore                   # Build exclusions
```

## How to Test Locally
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