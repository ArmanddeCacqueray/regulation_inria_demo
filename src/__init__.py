from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.json"

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)
