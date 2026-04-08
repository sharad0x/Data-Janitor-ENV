import os
import uvicorn
from dotenv import load_dotenv
from openenv.core.env_server.http_server import create_app

load_dotenv()

try:
    from ..models import DataJanitorAction, DataJanitorObservation
    from .data_janitor_environment import DataJanitorEnvironment
except ImportError:
    from models import DataJanitorAction, DataJanitorObservation
    from server.data_janitor_environment import DataJanitorEnvironment

os.environ["ENABLE_WEB_INTERFACE"] = "true"

# HACKATHON FIX: Explicitly map the class to the YAML tasks so the validator 
# recognizes all 3 graders as valid.
DataJanitorEnvironment.SUPPORTED_TASKS = ["easy", "medium", "hard"]

app = create_app(
    DataJanitorEnvironment,
    DataJanitorAction,
    DataJanitorObservation,
    env_name="data_janitor",
    max_concurrent_envs=1,
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

def main():
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()