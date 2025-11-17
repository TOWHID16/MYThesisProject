import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- Model Configuration ---
# You can easily switch models here
# Supported providers: "openai", "gemini", "anthropic"
DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "models/gemini-2.5-flash" 


# --- Experiment Configuration ---
# To avoid high costs, let's run on a small sample of the dataset first
# Set to -1 to run on the full dataset
NUM_SAMPLES = 20
# Path for saving results
RESULTS_PATH = "results/comparison_results.csv"

# --- API Handling Configuration ---
# Number of times to retry a failed API call
MAX_RETRIES = 5
# Initial delay in seconds before retrying a failed API call
INITIAL_RETRY_DELAY = 5 # seconds
# A small delay between each successful API call to avoid hitting rate limits
REQUEST_DELAY = 10 # seconds