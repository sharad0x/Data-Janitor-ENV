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

This is a complete, Phase 2 compliant hackathon submission for the **Meta PyTorch OpenEnv Data Janitor** track. It features an autonomous LLM-powered agent designed to systematically clean, impute, and transform messy real-world datasets into ML-ready pipelines.

The agent acts as an elite Data Engineer. It interacts with an OpenEnv physics engine to dynamically inspect dataset schemas, missing values, skewness, and outliers, taking sequential actions to maximize the performance of downstream machine learning models.

## 🌟 Key Architectural Features

* **"Curriculum Learning" Task Cycling:** To satisfy strict multi-task validation via a single container, the environment implements internal state cycling. Every call to `env.reset()` dynamically hot-swaps the dataset, automatically progressing the agent from **Easy** -> **Medium** -> **Hard** tasks in a single execution loop.
* **Dual-Layer Reward Shielding:** The environment calculates intermediate step rewards (`+0.05` for dropping junk, `-0.10` for errors) to provide rich feedback in the Hugging Face Web UI. However, the `inference.py` script acts as a shield, masking these as `0.00` in the standard output to perfectly comply with the Phase 2 Validator's strict `0.0 to 1.0` total score bounds.
* **Data Leakage Prevention:** All imputation, scaling, and encoding actions are strictly `fit` on the 80% training split and safely `transformed` on the 20% test split. 
* **Robust Agent Inference:** Features exponential backoff for API timeouts and a graceful fallback submission mechanism to prevent pipeline crashes during peak traffic.

---

## 🛠️ The Environment Physics

### Observation Space
The environment provides the agent with a dense, mathematically rich state representation of the **Training Data only**, including:
* `dataset_schema`: Pandas data types (numeric vs. categorical).
* `missing_values`: Exact NaN counts per column.
* `skewness`: Skewness of numerical columns (filtered for abs > 0.5).
* `outlier_counts`: IQR-based outlier detection counts.
* `categorical_cardinality`: Unique value counts for string columns to identify IDs vs. Encodable features.
* `sample_data`: A live Markdown preview of the top 3 rows.
* `feedback`: Explicit text feedback (and UI rewards) from the previous action.

---

### Action Space
The agent has access to 7 distinct tools, strictly validated via Pydantic:
1. `drop_column`: Removes high-cardinality IDs or unrecoverable features.
2. `fill_missing`: Imputes NaNs (`mean`, `median`, `mode`, `constant`).
3. `handle_outliers`: Addresses extreme values (`clip_percentile`, `drop_zscore`).
4. `transform_distribution`: Normalizes skewed data (`log1p`, `sqrt`).
5. `scale_feature`: Scales numerical data (`standard`, `robust`).
6. `encode_categorical`: Converts strings to machine-readable formats (`one_hot`, `ordinal`).
7. `submit`: Terminates the episode and triggers the ML Grader.

---

### 🎁 Intermediate Reward Heuristics (Step-by-Step Feedback)

To facilitate efficient learning and prevent redundant operations, the environment provides **dense, immediate feedback** after every action. These rewards are visible in the Hugging Face Web UI but are **shielded from the final validator** to maintain a strict `0.0–1.0` total score range.

* **Time Efficiency Penalty (`-0.01`):**  
  Every step incurs a minor penalty to encourage the agent to find the shortest path to a clean dataset.

* **Progress Bonus (`+0.05`):**  
  Awarded when an action structurally improves the data, such as:
  - Dropping high-cardinality "junk" columns  
  - Reducing missing (`NaN`) values  
  - Applying valid encodings  

* **Invalid Action Penalty (`-0.10`):**  
  Triggered when the agent attempts an invalid transformation (e.g., scaling a string column) or encounters an execution error.

* **Anti-Looping Shield (`-0.50`):**  
  A severe penalty applied if the agent repeats the same action on the same column to exploit reward gains.

---

### ✅ Success Threshold

To pass validation for any task, the combined **Dual-Model average score must be ≥ 0.85**.

---

### 🧠 The Architecture

The final reward is computed as the **average score of two distinct models** trained on the cleaned dataset:

* **Model A: Random Forest (Tree-Based)**  
  - Acts as a **robust baseline**  
  - Focuses on structural correctness (imputation, encoding)  
  - Less sensitive to feature scaling  

* **Model B: Linear / Logistic Regression (Gradient-Based)**  
  - Acts as a **strict evaluator**  
  - Highly sensitive to:
    - Unscaled features  
    - Outliers  
  - Forces the agent to apply:
    - `scale_feature`  
    - `handle_outliers`  

⚠️ If these steps are skipped, this model’s performance drops significantly, reducing the final score.

---

### 📏 Grading Logic

* **For Regression (Easy Task):**  
  - Final score = Average **R² Score**  
  - Models used:
    - RandomForestRegressor  
    - Ridge Regression  
  - Score bounded as: `max(0.0, r2)`

* **For Classification (Medium/Hard Tasks):**  
  - Final score = Average **Macro F1-Score**  
  - Models used:
    - RandomForestClassifier  
    - LogisticRegression  

---

## 🗄️ Curriculum Tasks & Datasets

The environment simulates a progressive curriculum, challenging the agent with three distinct real-world data engineering scenarios:

* **Easy: Health Insurance Charges Prediction**
  * **Dataset:** A clean mix of categorical (Sex, Smoker, Region) and numerical (Age, BMI) features used to predict continuous medical charges.  
  * **Kaggle Link:** https://www.kaggle.com/datasets/mirichoi0218/insurance  
  * **The Challenge:** A foundational regression task. The agent must accurately encode categorical text features, scale numerical distributions, and handle potential outliers in BMI to optimize the regression pipeline.

* **Medium: Stroke Risk Assessment**
  * **Dataset:** Highly imbalanced classification data with natural missing values (e.g., BMI) and mixed feature types.  
  * **Kaggle Link:** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset  
  * **The Challenge:** Accurately imputing missing health metrics without introducing bias, encoding text features into machine-readable formats, and optimizing for the minority class (stroke events).

* **Hard: Spaceship Titanic**
  * **Dataset:** A complex sci-fi dataset with high-cardinality strings (Passenger IDs, Cabin numbers, Names), multiple missing values scattered across columns, and diverse feature types.  
  * **Kaggle Link:** https://www.kaggle.com/competitions/spaceship-titanic  
  * **The Challenge:** Extensive feature pruning (dropping unrecoverable/ID columns), robust multi-column imputation, and complex categorical encoding. The agent must thoroughly clean the schema before the ML model will accept it.

---

## 📊 Final Scoring: Dual-Model Evaluator

When the agent calls `submit` (or exhausts its step limit), the environment triggers the `_evaluate_final_pipeline()` method. Unlike standard benchmarks, this system uses a **Dual-Model Evaluator** to ensure comprehensive data engineering quality.

### Baseline Agent Scores
The included `inference.py` script can be run locally using the NVIDIA NIM API (e.g., `meta/llama-3.1-70b-instruct`) or via Hugging Face Serverless.

| Task Difficulty | Dataset | Max Steps | Target Type | Expected Baseline Score |
| :--- | :--- | :--- | :--- | :--- |
| **Easy** | Health Insurance | 20 | Regression | ~0.82+ (R²) |
| **Medium** | Stroke Risk | 30 | Classification | ~0.48+ (F1) |
| **Hard** | Spaceship Titanic | 40 | Classification | Evaluated on submission |

---

## 🚀 Setup & Local Testing

You can test the agent locally while connecting to the live Hugging Face Space.

**1. Install Dependencies**
Using `uv` (recommended):
```bash
uv sync
```
Or using standard pip: pip install -r requirements.txt

**2. Configure Environment Variables**
Create a .env file in the root directory (alongside inference.py) with your credentials:
```
# Your AI Provider API key (Hugging Face, NVIDIA NIM, etc.). This example uses NVIDIA NIM
API_KEY=your_api_key_here
API_BASE_URL=https://integrate.api.nvidia.com/v1
MODEL_NAME=meta/llama-3.1-70b-instruct

# Direct URL to your deployed Hugging Face Space
SPACE_URL=https://sharad0x-data-janitor-env.hf.space
```

**3. Run the Evaluation Loop**
```
python inference.py
```
The script will sequentially connect to the space, complete the Easy, Medium, and Hard tasks, and output perfectly formatted [START], [STEP], and [END] logs.
---

Built for the Meta PyTorch OpenEnv Hackathon.
