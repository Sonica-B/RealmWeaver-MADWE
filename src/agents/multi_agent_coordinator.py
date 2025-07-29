"""
Multi-Agent Coordinator for MADWE
Day 5: Orchestrates all agents in the system
"""

import asyncio
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from .base_agent import BaseAgent, MessageBus, AgentState, MessageType
from .environment_agent import EnvironmentAgent
from ..unity_bridge.communication import UnityBridge
from ..models.nwsg.graph_network import NeuralWorldStateGraph


class MultiAgentCoordinator:
    """Coordinates multiple agents in the MADWE system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("MultiAgentCoordinator")
        
        # Core components
        self.message_bus = MessageBus()
        self.unity_bridge = UnityBridge()
        self.world_state_graph = None
        
        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_priorities = {
            'environment': 1,
            'asset': 2,
            'character': 3,
            'narrative': 4
        }
        
        # System state
        self.running = False
        self.start_time = None
        self.coordination_tasks = []
        
    async def initialize(self):
        """Initialize the multi-agent system"""
        self.logger.info("Initializing multi-agent coordinator")
        
        # Start message bus
        await self.message_bus.start()
        
        # Initialize Unity bridge
        self.unity_bridge.start()
        
        # Initialize world state graph
        nwsg_config = self.config.get('nwsg', {
            'd_model': 512,
            'd_agent': 256,
            'cell_size': 10.0,
            'max_temporal_length': 100
        })
        self.world_state_graph = NeuralWorldStateGraph(nwsg_config)
        
        # Create and initialize agents
        await self._create_agents()
        
        self.start_time = datetime.now()
        self.logger.info("Multi-agent coordinator initialized")
        
    async def _create_agents(self):
        """Create and initialize all agents"""
        # Environment Agent
        env_config = self.config.get('environment_agent', {
            'biomes': ['forest', 'desert', 'cyberpunk'],
            'chunk_size': (32, 32),
            'max_cache_size': 100,
            'generation_timeout': 5.0
        })
        
        env_agent = EnvironmentAgent(
            agent_id="env_agent_01",
            config=env_config,
            message_bus=self.message_bus
        )
        
        # Set external connections
        env_agent.unity_bridge = self.unity_bridge
        env_agent.world_state_graph = self.world_state_graph
        
        await self._register_agent(env_agent)
        
        # Note: Asset, Character, and Narrative agents would be created here
        # For Day 5, we focus on the environment agent
        
    async def _register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator"""
        await agent.initialize()
        self.agents[agent.agent_id] = agent
        
        # Set up agent monitoring
        self.coordination_tasks.append(
            asyncio.create_task(self._monitor_agent(agent))
        )
        
        self.logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type})")
        
    async def start(self):
        """Start the multi-agent system"""
        self.logger.info("Starting multi-agent system")
        self.running = True
        
        # Start all agents
        for agent in self.agents.values():
            await agent.start()
            
        # Start coordination tasks
        self.coordination_tasks.append(
            asyncio.create_task(self._coordinate_agents())
        )
        self.coordination_tasks.append(
            asyncio.create_task(self._monitor_system_health())
        )
        
        self.logger.info("Multi-agent system started")
        
    async def _coordinate_agents(self):
        """Main coordination loop"""
        while self.running:
            try:
                # Check for cross-agent dependencies
                await self._check_dependencies()
                
                # Balance workload
                await self._balance_workload()
                
                # Sync world state
                await self._sync_world_state()
                
                await asyncio.sleep(1)  # Coordination cycle
                
            except Exception as e:
                self.logger.error(f"Coordination error: {e}")
                
    async def _check_dependencies(self):
        """Check and resolve agent dependencies"""
        # For now, simple dependency checking
        # In full system, would handle complex agent interactions
        
        env_agent = self.agents.get("env_agent_01")
        if env_agent and env_agent.state == AgentState.READY:
            # Check if other agents need terrain data
            # Would send coordination messages here
            pass
            
    async def _balance_workload(self):
        """Balance workload across agents"""
        agent_loads = {}
        
        for agent_id, agent in self.agents.items():
            metrics = agent.get_metrics()
            # Simple load metric based on queue size and processing time
            load = metrics.get('avg_processing_time', 0) * 100
            agent_loads[agent_id] = load
            
        # If any agent is overloaded, redistribute work
        avg_load = sum(agent_loads.values()) / len(agent_loads) if agent_loads else 0
        
        for agent_id, load in agent_loads.items():
            if load > avg_load * 1.5:  # 50% above average
                self.logger.warning(f"Agent {agent_id} is overloaded: {load:.1f}")
                # Would implement load balancing here
                
    async def _sync_world_state(self):
        """Synchronize world state across agents"""
        if not self.world_state_graph:
            return
            
        # Broadcast world state updates
        state_summary = await self.world_state_graph.get_state_summary()
        
        for agent in self.agents.values():
            await agent.send_message(
                recipient=agent.agent_id,
                message_type=MessageType.SYNC,
                payload={
                    'sync_type': 'world_state',
                    'state_summary': state_summary,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
    async def _monitor_agent(self, agent: BaseAgent):
        """Monitor individual agent health"""
        while self.running:
            try:
                metrics = agent.get_metrics()
                
                # Check for errors
                if metrics['errors'] > 10:
                    self.logger.error(f"Agent {agent.agent_id} has high error rate")
                    
                # Check for stalls
                if agent.state == AgentState.BUSY:
                    # Track how long agent has been busy
                    pass
                    
                await asyncio.sleep(5)  # Monitor interval
                
            except Exception as e:
                self.logger.error(f"Error monitoring agent {agent.agent_id}: {e}")
                
    async def _monitor_system_health(self):
        """Monitor overall system health"""
        while self.running:
            try:
                # Collect system metrics
                total_messages = sum(
                    agent.get_metrics()['messages_sent'] 
                    for agent in self.agents.values()
                )
                
                active_agents = sum(
                    1 for agent in self.agents.values()
                    if agent.state == AgentState.READY
                )
                
                # Unity connection status
                unity_connected = self.unity_bridge.connected
                
                # Log system status
                self.logger.info(
                    f"System Health - Active agents: {active_agents}/{len(self.agents)}, "
                    f"Total messages: {total_messages}, Unity: {'Connected' if unity_connected else 'Disconnected'}"
                )
                
                await asyncio.sleep(30)  # Health check interval
                
            except Exception as e:
                self.logger.error(f"Error monitoring system health: {e}")
                
    async def request_chunk_generation(self, position: tuple[int, int], 
                                     biome: Optional[str] = None) -> bool:
        """Request chunk generation from environment agent"""
        env_agent = self.agents.get("env_agent_01")
        if not env_agent:
            return False
            
        result = await env_agent.generate_chunk(position, biome)
        return result is not None
        
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        agent_statuses = {}
        
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = {
                'state': agent.state.value,
                'metrics': agent.get_metrics()
            }
            
        return {
            'running': self.running,
            'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'agents': agent_statuses,
            'unity_connected': self.unity_bridge.connected,
            'message_bus_queue_size': self.message_bus.message_queue.qsize()
        }
        
    async def shutdown(self):
        """Shutdown the multi-agent system"""
        self.logger.info("Shutting down multi-agent system")
        self.running = False
        
        # Stop all agents
        shutdown_tasks = []
        for agent in self.agents.values():
            shutdown_tasks.append(agent.shutdown())
            
        await asyncio.gather(*shutdown_tasks)
        
        # Cancel coordination tasks
        for task in self.coordination_tasks:
            task.cancel()
            
        # Stop Unity bridge
        self.unity_bridge.stop()
        
        self.logger.info("Multi-agent system shutdown complete")


# Convenience function for running the system
async def run_multi_agent_system(config: Optional[Dict[str, Any]] = None):
    """Run the complete multi-agent system"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and initialize coordinator
    coordinator = MultiAgentCoordinator(config)
    await coordinator.initialize()
    await coordinator.start()
    
    # Keep running until interrupted
    try:
        while True:
            await asyncio.sleep(1)
            
            # Periodic status check
            if int(datetime.now().timestamp()) % 60 == 0:
                status = await coordinator.get_system_status()
                logging.info(f"System status: {status}")
                
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
        await coordinator.shutdown()