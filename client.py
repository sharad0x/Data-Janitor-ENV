from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
try:
    # When imported as a package
    from .models import DataJanitorAction, DataJanitorObservation, DataJanitorState
except ImportError:
    # When run directly via standalone scripts
    from models import DataJanitorAction, DataJanitorObservation, DataJanitorState
    
class DataJanitorEnv(EnvClient[DataJanitorAction, DataJanitorObservation, DataJanitorState]):
    """
    Client for the Data Janitor Environment.
    """
    def _step_payload(self, action: DataJanitorAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[DataJanitorObservation]:
        obs_data = payload.get("observation", {})
        if "reward" not in obs_data: obs_data["reward"] = payload.get("reward", 0.0)
        if "done" not in obs_data: obs_data["done"] = payload.get("done", False)

        return StepResult(
            observation=DataJanitorObservation(**obs_data),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: Dict) -> DataJanitorState:
        return DataJanitorState(**payload)