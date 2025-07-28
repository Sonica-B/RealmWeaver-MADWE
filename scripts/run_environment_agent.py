import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import asyncio
from agents.environment_agent import EnvironmentAgent
from unity_bridge.communication import UnityBridge
from models.nwsg.graph_network import NeuralWorldStateGraph

async def run_environment_agent():
    # Initialize Unity bridge
    bridge = UnityBridge()
    bridge.start()
    
    # Initialize NWSG
    nwsg_config = {
        'd_model': 512,
        'd_agent': 256,
        'cell_size': 10.0,
        'max_temporal_length': 100
    }
    nwsg = NeuralWorldStateGraph(nwsg_config)
    
    # Create environment agent config
    env_config = {
        'biomes': ['forest', 'desert', 'snow'],
        'chunk_size': (32, 32),
        'generation_timeout': 5.0
    }
    
    # Create environment agent - BaseAgent expects agent_id, agent_type, config
    env_agent = EnvironmentAgent(
        agent_id="env_agent_01",
        agent_type="environment",
        config=env_config
    )
    
    # Set bridge and nwsg as attributes after initialization
    env_agent.unity_bridge = bridge
    env_agent.world_state_graph = nwsg
    
    print("Starting Environment Agent...")
    await env_agent.initialize()
    await env_agent.start()
    
    print("Waiting for Unity connection...")
    while not bridge.connected:
        await asyncio.sleep(0.1)
    
    print("Unity connected! Generating world chunks...")
    
    # Generate chunks for different biomes
    biomes = ['forest', 'desert', 'snow']
    positions = [(0, 0), (32, 0), (64, 0)]
    
    for biome, pos in zip(biomes, positions):
        print(f"\nGenerating {biome} chunk at {pos}...")
        
        # Set current biome if the agent has this attribute
        if hasattr(env_agent, 'current_biome'):
            env_agent.current_biome = biome
        
        # Generate chunk - check if method exists
        if hasattr(env_agent, 'generate_chunk'):
            chunk_result = await env_agent.generate_chunk(pos)
            
            if chunk_result:
                print(f"Generated {biome} chunk in {chunk_result['generation_time']:.1f}ms")
            else:
                print(f"Failed to generate {biome} chunk")
    
    # Keep running
    print("\nEnvironment agent running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        await env_agent.shutdown()
        bridge.stop()

if __name__ == "__main__":
    asyncio.run(run_environment_agent())