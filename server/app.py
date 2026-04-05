import os
import uvicorn
from openenv.core.env_server.http_server import create_app

try:
    from ..models import DataJanitorAction, DataJanitorObservation
    from .data_janitor_environment import DataJanitorEnvironment
except ImportError:
    from models import DataJanitorAction, DataJanitorObservation
    from server.data_janitor_environment import DataJanitorEnvironment

# Enable the built-in debugging UI
os.environ["ENABLE_WEB_INTERFACE"] = "true"

app = create_app(
    DataJanitorEnvironment,
    DataJanitorAction,
    DataJanitorObservation,
    env_name="data_janitor",
    max_concurrent_envs=1,
)

def main():
    """Required by OpenEnv for multi-mode deployment validation."""
    # Pass the 'app' object directly instead of a string
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()