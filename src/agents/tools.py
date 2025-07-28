# src/agents/tools.py

import subprocess
import requests
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ── WFC & Unity Tools ─────────────────────────────────────────────────────────

def run_wfc(biome: str) -> str:
    """Generate a WFC layout for a given biome."""
    cfg = Path("configs/biomes") / f"{biome}.json"
    out_json = Path("data/samples") / f"wfc_{biome}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "venv\\Scripts\\python.exe",
        "src\\wfc\\base_wfc.py",
        "--config", str(cfg),
        "--export-json", str(out_json)
    ]
    subprocess.run(cmd, check=True)
    return f"Generated WFC layout → {out_json}"

def reload_unity(biome: str) -> str:
    """Notify Unity via HTTP to reload the given biome layout."""
    url = "http://127.0.0.1:5005/reload"
    resp = requests.post(url, json={"biome": biome})
    resp.raise_for_status()
    return f"Unity reloaded biome {biome}"

# ── Memory Tools (Chroma) ──────────────────────────────────────────────────────

# PersistentClient will create & manage a DuckDB+Parquet store under data/chroma
_chroma_client = chromadb.PersistentClient(
     path=str(Path("data/chroma").absolute()),
     settings=Settings(),
     tenant=DEFAULT_TENANT,
     database=DEFAULT_DATABASE,
)
_collection = _chroma_client.get_or_create_collection("world_memory")

def save_memory(key: str, document: dict) -> str:
    """Save a document under a given key in Chroma memory."""
    _collection.add(
        documents=[json.dumps(document)],
        metadatas=[{"key": key}],
        ids=[f"{key}-{_collection.count()}"]
    )
    return f"Saved memory under key={key}"

def query_memory(query: str, top_k: int = 3) -> list:
    """Retrieve up to top_k similar documents from memory."""
    results = _collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results["documents"][0]  # list of JSON strings

# ── Narrative & Character Tools ────────────────────────────────────────────────

def generate_dialog(context: str) -> str:
    """Produce narrative dialog or mission text based on context."""
    # This function will be wrapped by LangChain; here we simply return the prompt.
    return f"Dialog generated for context: {context}"

def update_character(state_updates: dict) -> str:
    """Apply state updates to the CharacterAgent’s internal state."""
    # In a real system you’d persist this; for now we echo back.
    return f"Character state updated: {json.dumps(state_updates)}"

def log_event(message: str) -> str:
    """Log a global event."""
    log_path = Path("logs/events.log")
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        f.write(message + "\n")
    return f"Logged event: {message}"

# if __name__ == "__main__":
#     # Example usage
#     print(run_wfc("forest"))
#     print(reload_unity("forest"))
#     print(save_memory("test_key", {"example": "data"}))
#     print(query_memory("example query"))
#     print(generate_dialog("This is a test context."))
#     print(update_character({"health": 100, "mana": 50}))
#     print(log_event("Test event logged."))