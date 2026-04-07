import os
import uuid
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True) 
from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        DataJanitorAction, DataJanitorObservation, DataJanitorState,
        DropColumnAction, FillMissingAction, HandleOutliersAction, 
        TransformDistributionAction, EncodeCategoricalAction, 
        ScaleFeatureAction, SubmitDatasetAction
    )
except ImportError:
    from models import (
        DataJanitorAction, DataJanitorObservation, DataJanitorState,
        DropColumnAction, FillMissingAction, HandleOutliersAction, 
        TransformDistributionAction, EncodeCategoricalAction, 
        ScaleFeatureAction, SubmitDatasetAction
    )

class DataJanitorEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, target_column: str = "target", **kwargs):
        super().__init__(**kwargs) 
        self.target_column = target_column
        
        # 1. Read from environment variable and force lowercase to match mapping keys
        self.difficulty = kwargs.get("difficulty", os.environ.get("TASK_NAME", "hard")).lower()
        
        # Internal state
        self.episode_id = ""
        self.current_step = 0
        
        # FIX: Dynamically set max_steps based on the task difficulty
        if self.difficulty == "easy":
            self.max_steps = 20
        elif self.difficulty == "medium":
            self.max_steps = 30
        else:
            self.max_steps = 40
            
        self.action_history = []
        self.last_feedback = "Initialized."
        self.final_score = 0.0
        self.cleaned_columns = set()

        # 2. Map the difficulty to the correct Kaggle CSV file
        dataset_mapping = {
            "easy": "data/easy_messy.csv",
            "medium": "data/medium_messy.csv",
            "hard": "data/hard_messy.csv"
        }
        
        # Default to hard if an invalid task name is somehow passed
        file_path = dataset_mapping.get(self.difficulty, "data/hard_messy.csv")
        
        # 3. Load the data and create the train/test split
        try:
            raw_df = pd.read_csv(file_path)
            
            # Split into 80% train and 20% test
            self.train_df = raw_df.sample(frac=0.8, random_state=42)
            self.test_df = raw_df.drop(self.train_df.index)
            
            print(f"[DEBUG] Loaded '{self.difficulty}' dataset from {file_path}")
        except Exception as e:
            print(f"[ERROR] Could not load dataset {file_path}: {e}")
            self.train_df = None
            self.test_df = None

    def reset(self) -> DataJanitorObservation:
        self.episode_id = str(uuid.uuid4())
        self.current_step = 0
        self.action_history = []
        self.final_score = 0.0
        self.cleaned_columns = set()
        
        # 1. DYNAMIC TASK LOADING
        # Detect difficulty from OpenEnv metadata if available, else default to hard
        diff = getattr(self, "difficulty", "hard")
        dataset_path = f"data/{diff}_messy.csv"
        
        if not os.path.exists(dataset_path):
            # Ensure generate_data.py has been run
            dataset_path = "data/hard_messy.csv"

        full_df = pd.read_csv(dataset_path)
        from sklearn.model_selection import train_test_split
        
        # 2. STATEFUL SPLIT (Leakage Prevention)
        if self.target_column in full_df.columns and full_df[self.target_column].nunique() < 20:
            self.train_df, self.test_df = train_test_split(full_df, test_size=0.2, stratify=full_df[self.target_column], random_state=42)
        else:
            self.train_df, self.test_df = train_test_split(full_df, test_size=0.2, random_state=42)

        self.train_df = self.train_df.copy()
        self.test_df = self.test_df.copy()
        
        self.last_feedback = f"Loaded {diff.upper()} task. Train/Test split initialized."
        return self._generate_observation(reward=0.0, done=False)

    @property
    def state(self) -> DataJanitorState:
        return DataJanitorState(
            episode_id=self.episode_id,
            step_count=self.current_step,
            task_difficulty=getattr(self, "difficulty", "hard"),
            original_dataset_path=f"data/{getattr(self, 'difficulty', 'hard')}_messy.csv",
            max_steps=self.max_steps
        )

    def step(self, action: DataJanitorAction) -> DataJanitorObservation:
        self.current_step += 1
        done = False
        cmd = action.command
        
        self.action_history.append(cmd.action_type)
        self.action_history = self.action_history[-5:]
        
        # ==========================================
        # REWARD SHAPING (Requirement #4)
        # ==========================================
        step_reward = -0.01 # Constant step penalty to encourage efficiency
        is_manual_submit = False
        
        try:
            if isinstance(cmd, SubmitDatasetAction):
                is_manual_submit = True
                done = True
                step_reward = 0.0 # No penalty on final submit
                
            elif isinstance(cmd, DropColumnAction):
                if cmd.column_name in self.train_df.columns:
                    self.train_df = self.train_df.drop(columns=[cmd.column_name])
                    self.test_df = self.test_df.drop(columns=[cmd.column_name], errors='ignore')
                    self.last_feedback = f"Dropped '{cmd.column_name}'."
                    step_reward += 0.05 # Reward for feature pruning
                else:
                    self.last_feedback = "Error: Column not found."
                    step_reward -= 0.05 # Penalty for invalid actions
                    
            elif isinstance(cmd, FillMissingAction):
                col = cmd.column_name
                if col in self.train_df.columns:
                    # Partial Progress Reward: Reward if this is the first time cleaning this column
                    if self.train_df[col].isna().any():
                        step_reward += 0.10
                        
                    if cmd.strategy == "mean": fill_val = self.train_df[col].mean()
                    elif cmd.strategy == "median": fill_val = self.train_df[col].median()
                    elif cmd.strategy == "mode": fill_val = self.train_df[col].mode()[0]
                    else: fill_val = cmd.constant_value if cmd.constant_value else "Unknown"
                    
                    self.train_df[col] = self.train_df[col].fillna(fill_val)
                    self.test_df[col] = self.test_df[col].fillna(fill_val)
                    self.last_feedback = f"Imputed '{col}'."
                else:
                    self.last_feedback = "Error: Column not found."

            elif isinstance(cmd, HandleOutliersAction):
                col = cmd.column_name
                if col in self.train_df.select_dtypes(include=[np.number]).columns:
                    if cmd.strategy == "clip_percentile":
                        step_reward += 0.05
                        lower_val = self.train_df[col].quantile(cmd.lower_percentile)
                        upper_val = self.train_df[col].quantile(cmd.upper_percentile)
                        self.train_df[col] = self.train_df[col].clip(lower=lower_val, upper=upper_val)
                        self.test_df[col] = self.test_df[col].clip(lower=lower_val, upper=upper_val)
                        self.last_feedback = f"Clipped '{col}'."
                    elif cmd.strategy == "drop_zscore":
                        mean_val, std_val = self.train_df[col].mean(), self.train_df[col].std()
                        z_scores = np.abs((self.train_df[col] - mean_val) / (std_val + 1e-9))
                        self.train_df = self.train_df[z_scores < cmd.zscore_threshold]
                        self.last_feedback = f"Dropped outlier rows in '{col}'."
                else:
                    self.last_feedback = "Error: Column must be numeric."

            elif isinstance(cmd, TransformDistributionAction):
                col = cmd.column_name
                if col in self.train_df.select_dtypes(include=[np.number]).columns:
                    step_reward += 0.05
                    # Clip lower bounds at 0 to prevent math errors on negative numbers
                    if cmd.strategy == "log1p":
                        self.train_df[col] = np.log1p(self.train_df[col].clip(lower=0))
                        self.test_df[col] = np.log1p(self.test_df[col].clip(lower=0))
                    elif cmd.strategy == "sqrt":
                        self.train_df[col] = np.sqrt(self.train_df[col].clip(lower=0))
                        self.test_df[col] = np.sqrt(self.test_df[col].clip(lower=0))
                    self.last_feedback = f"Transformed '{col}' using {cmd.strategy}."
                else:
                    self.last_feedback = "Error: Column must be numeric."

            elif isinstance(cmd, ScaleFeatureAction):
                col = cmd.column_name
                if col in self.train_df.select_dtypes(include=[np.number]).columns:
                    # Prevent redundant scaling loop reward
                    if col not in self.cleaned_columns:
                        step_reward += 0.05
                        self.cleaned_columns.add(col)
                        
                    from sklearn.preprocessing import StandardScaler, RobustScaler
                    scaler = StandardScaler() if cmd.strategy == "standard" else RobustScaler()
                    self.train_df[col] = scaler.fit_transform(self.train_df[[col]])
                    self.test_df[col] = scaler.transform(self.test_df[[col]])
                    self.last_feedback = f"Scaled '{col}'."
                else:
                    self.last_feedback = "Error: Column must be numeric."

            elif isinstance(cmd, EncodeCategoricalAction):
                col = cmd.column_name
                if col in self.train_df.columns:
                    step_reward += 0.10
                    if cmd.strategy == "one_hot":
                        from sklearn.preprocessing import OneHotEncoder
                        # handle_unknown='ignore' prevents crashes on unseen test data
                        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                        
                        # Fit on Train only
                        train_encoded = enc.fit_transform(self.train_df[[col]].astype(str))
                        train_encoded_df = pd.DataFrame(train_encoded, columns=enc.get_feature_names_out([col]), index=self.train_df.index)
                        self.train_df = pd.concat([self.train_df.drop(columns=[col]), train_encoded_df], axis=1)
                        
                        # Transform Test
                        test_encoded = enc.transform(self.test_df[[col]].astype(str))
                        test_encoded_df = pd.DataFrame(test_encoded, columns=enc.get_feature_names_out([col]), index=self.test_df.index)
                        self.test_df = pd.concat([self.test_df.drop(columns=[col]), test_encoded_df], axis=1)
                    else:
                        from sklearn.preprocessing import OrdinalEncoder
                        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        self.train_df[col] = enc.fit_transform(self.train_df[[col]].astype(str))
                        self.test_df[col] = enc.transform(self.test_df[[col]].astype(str))
                    self.last_feedback = f"Encoded '{col}'."
                else:
                    self.last_feedback = "Error: Column not found."

        except Exception as e:
            self.last_feedback = f"Execution Error: {str(e)}"
            step_reward -= 0.10 # Explicit penalty for crashing the physics engine

        # ==========================================
        # STRICT HACKATHON GRADING (0.0 to 1.0)
        # ==========================================
        # The evaluator requires the total episode reward to fall strictly
        # between 0.0 and 1.0. We must discard all intermediate step_rewards.
        total_reward = 0.0 
        
        if is_manual_submit or self.current_step >= self.max_steps:
            done = True
            self.final_score = self._evaluate_final_pipeline()
            total_reward = float(self.final_score) # Only the terminal ML score matters

        return self._generate_observation(total_reward, done)

    def _evaluate_final_pipeline(self) -> float:
        """Standardized ML Grader for Easy, Medium, and Hard tasks."""
        if self.target_column not in self.train_df.columns:
            return 0.0
        
        # Check if agent fulfilled the Task objective
        # Easy: Date conversion, Medium: No NaNs, Hard: Performance
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import f1_score
            
            y_train = self.train_df[self.target_column]
            X_train = self.train_df.drop(columns=[self.target_column])
            y_test = self.test_df[self.target_column]
            X_test = self.test_df.drop(columns=[self.target_column])
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

            # Strict check for remaining NaNs (Requirement #3)
            if X_train.isna().sum().sum() > 0:
                self.last_feedback = "Evaluation Failed: Dataset contains NaNs."
                return 0.0

            # Encode any remaining strings to prevent full crash
            for col in X_train.select_dtypes(exclude=[np.number]).columns:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))

            model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            score = f1_score(y_test, preds, average='macro')
            self.last_feedback = f"Evaluation Complete. Score: {score:.3f}"
            return float(score)
        except Exception:
            return 0.0

    def _generate_observation(self, reward: float, done: bool) -> DataJanitorObservation:
        """Generates raw mathematical state using only Train data statistics."""
        num_cols = self.train_df.select_dtypes(include=[np.number]).drop(columns=[self.target_column], errors='ignore')
        cat_cols = self.train_df.select_dtypes(exclude=[np.number]).drop(columns=[self.target_column], errors='ignore')
            
        return DataJanitorObservation(
            dataset_schema={str(c): ("numeric" if pd.api.types.is_numeric_dtype(t) else "categorical") for c, t in self.train_df.dtypes.items()},
            missing_values={str(k): int(v) for k, v in self.train_df.isna().sum().items() if v > 0},
            skewness={str(k): float(v) for k, v in num_cols.skew().fillna(0).round(2).items() if abs(v) >= 0.5},
            outlier_counts={str(col): int(((num_cols[col] < (num_cols[col].quantile(0.25) - 1.5 * (num_cols[col].quantile(0.75) - num_cols[col].quantile(0.25)))) | (num_cols[col] > (num_cols[col].quantile(0.75) + 1.5 * (num_cols[col].quantile(0.75) - num_cols[col].quantile(0.25))))).sum()) for col in num_cols.columns if int(((num_cols[col] < (num_cols[col].quantile(0.25) - 1.5 * (num_cols[col].quantile(0.75) - num_cols[col].quantile(0.25)))) | (num_cols[col] > (num_cols[col].quantile(0.75) + 1.5 * (num_cols[col].quantile(0.75) - num_cols[col].quantile(0.25))))).sum()) > 0},
            zero_counts={str(col): int((num_cols[col] == 0).sum()) for col in num_cols.columns if int((num_cols[col] == 0).sum()) > 0},
            negative_counts={},
            categorical_cardinality={str(col): int(cat_cols[col].nunique()) for col in cat_cols.columns},
            action_history=self.action_history,
            total_rows=int(len(self.train_df)),
            sample_data=str(self.train_df[[self.target_column] + list(cat_cols.columns) + list(num_cols.columns)[:3]].head(3).to_markdown()),
            feedback=str(self.last_feedback),
            attempts_left=int(self.max_steps - self.current_step),
            reward=float(reward), 
            done=bool(done),
            final_score=float(self.final_score) 
        )