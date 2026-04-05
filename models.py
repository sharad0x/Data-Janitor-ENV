from typing import Dict, Literal, Optional, Union
from pydantic import BaseModel, Field
from openenv.core import Action, Observation
from openenv.core.env_server.types import State

# ==========================================
# 1. THE ACTION SPACE
# ==========================================

class DropColumnAction(BaseModel):
    action_type: Literal["drop_column"] = "drop_column"
    column_name: str = Field(..., description="The exact name of the column to drop.")

class FillMissingAction(BaseModel):
    action_type: Literal["fill_missing"] = "fill_missing"
    column_name: str = Field(..., description="The column containing missing (NaN) values to impute.")
    strategy: Literal["mean", "median", "mode", "constant", "forward_fill", "backward_fill"] = Field(...)
    constant_value: Optional[str] = Field(default=None)

class ChangeDataTypeAction(BaseModel):
    action_type: Literal["change_dtype"] = "change_dtype"
    column_name: str = Field(..., description="The name of the column to modify.")
    new_type: Literal["int64", "float64", "datetime64[ns]", "object", "bool"] = Field(...)

class HandleOutliersAction(BaseModel):
    action_type: Literal["handle_outliers"] = "handle_outliers"
    column_name: str = Field(..., description="The numerical column to check for outliers.")
    strategy: Literal["clip_percentile", "drop_zscore", "nan_zeros"] = Field(...)
    lower_percentile: float = Field(0.01, ge=0.0, le=0.49, description="Lower clip bound (for clip_percentile)")
    upper_percentile: float = Field(0.99, ge=0.51, le=1.0, description="Upper clip bound (for clip_percentile)")
    zscore_threshold: float = Field(3.0, ge=1.0, le=5.0, description="Z-score cutoff (for drop_zscore)")

class FeatureEngineeringAction(BaseModel):
    action_type: Literal["feature_engineering"] = "feature_engineering"
    column_a: str = Field(..., description="First numerical column.")
    column_b: str = Field(..., description="Second numerical column.")
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(...)
    new_column_name: str = Field(..., description="Name for the newly created feature.")

class TransformDistributionAction(BaseModel):
    action_type: Literal["transform_distribution"] = "transform_distribution"
    column_name: str = Field(..., description="The highly skewed numerical column to normalize.")
    strategy: Literal["log1p", "sqrt"] = Field(...)

class EncodeCategoricalAction(BaseModel):
    action_type: Literal["encode_categorical"] = "encode_categorical"
    column_name: str = Field(..., description="The string/categorical column to encode into numbers.")
    strategy: Literal["one_hot", "ordinal", "target_encode"] = Field(...)

class ScaleFeatureAction(BaseModel):
    action_type: Literal["scale_feature"] = "scale_feature"
    column_name: str = Field(..., description="The numerical column to scale.")
    strategy: Literal["standard", "minmax", "robust"] = Field(...)

class ReduceDimensionsAction(BaseModel):
    action_type: Literal["reduce_dimensions"] = "reduce_dimensions"
    strategy: Literal["pca", "drop_collinear"] = Field(...)

class SubmitDatasetAction(BaseModel):
    action_type: Literal["submit"] = "submit"
    notes: str = Field(..., description="Summary of the transformations applied.")

class DataJanitorAction(Action):
    command: Union[
        DropColumnAction, 
        FillMissingAction, 
        ChangeDataTypeAction,
        HandleOutliersAction,
        TransformDistributionAction,
        EncodeCategoricalAction,
        ScaleFeatureAction,
        FeatureEngineeringAction,
        ReduceDimensionsAction,
        SubmitDatasetAction
    ] = Field(..., description="The data engineering action to perform.")

# ==========================================
# 2. THE OBSERVATION SPACE
# ==========================================
class DataJanitorObservation(Observation):
    dataset_schema: Dict[str, str] = Field(..., description="Pandas data types.")
    missing_values: Dict[str, int] = Field(..., description="NaN counts per column.")
    skewness: Dict[str, float] = Field(..., description="Skewness of numerical columns.")
    outlier_counts: Dict[str, int] = Field(..., description="Count of potential outliers (using IQR method).")
    zero_counts: Dict[str, int] = Field(..., description="Count of exact 0.0 values.")
    negative_counts: Dict[str, int] = Field(..., description="Count of negative values.")
    categorical_cardinality: Dict[str, int] = Field(..., description="Number of unique values in string/categorical columns.")
    
    total_rows: int = Field(..., description="Total rows in current training dataset.")
    sample_data: str = Field(..., description="Markdown sample of train data.")
    action_history: list[str] = Field(default_factory=list, description="Last 5 actions taken.")
    feedback: str = Field(..., description="Feedback from previous action execution.")
    attempts_left: int = Field(..., description="Actions left before termination.")
    
    reward: float = Field(0.0)
    done: bool = Field(False)
    final_score: float = Field(0.0)

# ==========================================
# 3. THE STATE
# ==========================================
class DataJanitorState(State):
    episode_id: str
    step_count: int
    task_difficulty: Literal["easy", "medium", "hard"]
    original_dataset_path: str
    max_steps: int