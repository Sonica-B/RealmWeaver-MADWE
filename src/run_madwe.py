"""
Entry point for running the MADWE multi-agent system
Day 5: Production runner
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agents.multi_agent_coordinator import run_multi_agent_system


# Configuration for the multi-agent system
MADWE_CONFIG = {
    'environment_agent': {
        'biomes': ['forest', 'desert', 'cyberpunk'],
        'chunk_size': (32, 32),
        'max_cache_size': 100,
        'generation_timeout': 5.0,
        'max_concurrent': 3
    },
    'nwsg': {
        'd_model': 512,
        'd_agent': 256,
        'cell_size': 10.0,
        'max_temporal_length': 100
    }
}


if __name__ == "__main__":
    asyncio.run(run_multi_agent_system(MADWE_CONFIG))