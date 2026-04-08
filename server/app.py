import os
import uvicorn
from dotenv import load_dotenv
from openenv.core.env_server.http_server import create_app

# Load local environment variables safely
load_dotenv()

try:
    from ..models import DataJanitorAction, DataJanitorObservation
    from .data_janitor_environment import DataJanitorEnvironment
except ImportError:
    from models import DataJanitorAction, DataJanitorObservation
    from server.data_janitor_environment import DataJanitorEnvironment

# Enable the built-in debugging UI
os.environ["ENABLE_WEB_INTERFACE"] = "true"

# HACKATHON FIX: Use standard OpenEnv registration. 
# The framework will route tasks internally via the /reset payload.
app = create_app(
    DataJanitorEnvironment,
    DataJanitorAction,
    DataJanitorObservation,
    env_name="data_janitor",
    max_concurrent_envs=1,
)

# ==========================================
# HEALTHCHECK FIX
# ==========================================
@app.get("/health")
def health_check():
    return {"status": "ok"}

def main():
    """Required by OpenEnv for multi-mode deployment validation."""
    # PORT FIX: Dynamically grab the port if the platform provides one, 
    # otherwise default strictly to 8000 for Hugging Face Spaces.
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()