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
    SUPPORTED_TASKS = ["easy", "medium", "hard"]

    def __init__(self, target_column: str = "target", **kwargs):
        super().__init__(**kwargs) 
        self.target_column = target_column
        
        self.task_cycle = ["easy", "medium", "hard"]
        self.cycle_index = 0
        
        self.difficulty = "hard" 
        self.episode_id = ""
        self.current_step = 0
        self.max_steps = 40
        self.action_history = []
        self.last_feedback = "Initialized."
        # FIXED: Initialize safely above 0.0
        self.final_score = 0.001 
        self.cleaned_columns = set()
        self.applied_actions = set()
        
        self.train_df = None
        self.test_df = None

    def reset(self, **kwargs) -> DataJanitorObservation:
        self.difficulty = self.task_cycle[self.cycle_index % 3]
        self.cycle_index += 1
        
        if self.difficulty == "easy":
            self.max_steps = 20
        elif self.difficulty == "medium":
            self.max_steps = 30
        else:
            self.max_steps = 40

        from sklearn.model_selection import train_test_split

        dataset_path = f"data/{self.difficulty}_messy.csv"
        if not os.path.exists(dataset_path):
            dataset_path = "data/hard_messy.csv"

        try:
            full_df = pd.read_csv(dataset_path)
            
            if self.target_column in full_df.columns and full_df[self.target_column].nunique() < 20:
                self.train_df, self.test_df = train_test_split(
                    full_df, test_size=0.2, stratify=full_df[self.target_column], random_state=42
                )
            else:
                self.train_df, self.test_df = train_test_split(
                    full_df, test_size=0.2, random_state=42
                )

            self.train_df = self.train_df.copy()
            self.test_df = self.test_df.copy()
        except Exception:
            # FIXED: Safe dummy data to prevent fatal container crashes
            self.train_df = pd.DataFrame({self.target_column: [0, 1, 0, 1], "dummy": [1, 2, 3, 4]})
            self.test_df = pd.DataFrame({self.target_column: [0, 1], "dummy": [1, 2]})

        self.episode_id = str(uuid.uuid4())
        self.current_step = 0
        self.action_history = []
        # FIXED: Initialize safely above 0.0
        self.final_score = 0.001 
        self.cleaned_columns = set()
        self.applied_actions = set()
        self.last_feedback = f"Loaded {self.difficulty.upper()} task. Train/Test split initialized."

        # FIXED: Return safe reward
        return self._generate_observation(reward=0.001, done=False)

    @property
    def state(self) -> DataJanitorState:
        return DataJanitorState(
            episode_id=self.episode_id,
            step_count=self.current_step,
            task_difficulty=self.difficulty,
            original_dataset_path=f"data/{self.difficulty}_messy.csv",
            max_steps=self.max_steps
        )

    def step(self, action: DataJanitorAction) -> DataJanitorObservation:
        self.current_step += 1
        done = False
        cmd = action.command
        
        self.action_history.append(cmd.action_type)
        self.action_history = self.action_history[-5:]
        
        step_reward = -0.01 
        is_manual_submit = False

        col_name = getattr(cmd, 'column_name', None)
        if col_name and cmd.action_type not in ["submit", "drop_column"]:
            action_sig = f"{cmd.action_type}_{col_name}"
            if action_sig in self.applied_actions:
                self.last_feedback = f"FATAL PENALTY: You already applied {cmd.action_type} to '{col_name}'. DO NOT REPEAT ACTIONS ON THE SAME COLUMN."
                step_reward -= 0.50 
                return self._generate_observation(float(step_reward), done)
            else:
                self.applied_actions.add(action_sig)
        
        try:
            if isinstance(cmd, SubmitDatasetAction):
                is_manual_submit = True
                done = True
                step_reward = 0.001 
                
            elif isinstance(cmd, DropColumnAction):
                if cmd.column_name in self.train_df.columns:
                    self.train_df = self.train_df.drop(columns=[cmd.column_name])
                    self.test_df = self.test_df.drop(columns=[cmd.column_name], errors='ignore')
                    self.last_feedback = f"Dropped '{cmd.column_name}'."
                    step_reward += 0.05 
                else:
                    self.last_feedback = "Error: Column not found."
                    step_reward -= 0.05 
                    
            elif isinstance(cmd, FillMissingAction):
                col = cmd.column_name
                if col in self.train_df.columns:
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
                        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                        train_encoded = enc.fit_transform(self.train_df[[col]].astype(str))
                        train_encoded_df = pd.DataFrame(train_encoded, columns=enc.get_feature_names_out([col]), index=self.train_df.index)
                        self.train_df = pd.concat([self.train_df.drop(columns=[col]), train_encoded_df], axis=1)
                        
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
            step_reward -= 0.10 

        total_reward = 0.0 
        
        if is_manual_submit or self.current_step >= self.max_steps:
            done = True
            self.final_score = self._evaluate_final_pipeline()
            total_reward = float(self.final_score) 
        else:
            total_reward = float(step_reward)

        return self._generate_observation(total_reward, done)

    def _evaluate_final_pipeline(self) -> float:
        if self.target_column not in self.train_df.columns:
            return 0.001 
        
        try:
            y_train = self.train_df[self.target_column]
            X_train = self.train_df.drop(columns=[self.target_column])
            y_test = self.test_df[self.target_column]
            X_test = self.test_df.drop(columns=[self.target_column])
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

            if X_train.isna().sum().sum() > 0:
                self.last_feedback = "Evaluation Failed: Dataset contains NaNs."
                return 0.001 

            for col in X_train.select_dtypes(exclude=[np.number]).columns:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))

            is_regression = False
            if pd.api.types.is_float_dtype(y_train):
                is_regression = True
            elif pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() > 20:
                is_regression = True

            if is_regression:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.linear_model import Ridge
                from sklearn.metrics import r2_score
                
                rf = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                rf_score = max(0.0, r2_score(y_test, rf.predict(X_test)))
                
                try:
                    ridge = Ridge(random_state=42)
                    ridge.fit(X_train, y_train)
                    linear_score = max(0.0, r2_score(y_test, ridge.predict(X_test)))
                except Exception:
                    linear_score = 0.0
                
                final_score = (rf_score + linear_score) / 2.0
                self.last_feedback = f"Eval Complete (Regression). RF R²: {rf_score:.3f} | Ridge R²: {linear_score:.3f} | Final: {final_score:.3f}"
                return float(final_score)
                
            else:
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import f1_score
                from sklearn.preprocessing import LabelEncoder
                import warnings
                from sklearn.exceptions import ConvergenceWarning
                
                target_le = LabelEncoder()
                y_train_enc = target_le.fit_transform(y_train.astype(str))
                try:
                    y_test_enc = target_le.transform(y_test.astype(str))
                except ValueError:
                    most_frequent = y_train_enc.mode()[0] if hasattr(y_train_enc, 'mode') else 0
                    y_test_enc = [target_le.transform([str(x)])[0] if str(x) in target_le.classes_ else most_frequent for x in y_test]
                
                rf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train_enc)
                rf_score = f1_score(y_test_enc, rf.predict(X_test), average='macro')
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        lr = LogisticRegression(max_iter=200, random_state=42)
                        lr.fit(X_train, y_train_enc)
                        linear_score = f1_score(y_test_enc, lr.predict(X_test), average='macro')
                except Exception:
                    linear_score = 0.0
                
                final_score = (rf_score + linear_score) / 2.0
                self.last_feedback = f"Eval Complete (Class). RF F1: {rf_score:.3f} | LogReg F1: {linear_score:.3f} | Final: {final_score:.3f}"
                return float(final_score)

        except Exception:
            return 0.001 

    def _generate_observation(self, reward: float, done: bool) -> DataJanitorObservation:
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
            # THE ULTIMATE SHIELD: Force everything sent over the wire into the strictly valid bounds
            reward=float(np.clip(reward, 0.001, 0.999)), 
            done=bool(done),
            final_score=float(np.clip(self.final_score, 0.001, 0.999))
        )