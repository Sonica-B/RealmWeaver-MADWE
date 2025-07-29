"""
Narrative Generation Agent for MADWE
Handles NPC dialogue, quest generation, and narrative coherence
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_agent import BaseAgent, AgentConfig, AgentState
from unity_bridge.communication import Message, MessageType, MessagePriority, MessageRouter
from .protocols import NarrativeGenerationRequest, RequestType

logger = logging.getLogger(__name__)


@dataclass
class NPCProfile:
    """Profile for an NPC character"""
    npc_id: str
    name: str
    personality_traits: List[str]
    backstory: str
    current_emotion: str = "neutral"
    relationship_scores: Dict[str, float] = None
    dialogue_history: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.relationship_scores is None:
            self.relationship_scores = {}
        if self.dialogue_history is None:
            self.dialogue_history = []


@dataclass
class Quest:
    """Quest structure"""
    quest_id: str
    title: str
    description: str
    objectives: List[str]
    rewards: Dict[str, Any]
    prerequisites: List[str] = None
    dialogue_triggers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.dialogue_triggers is None:
            self.dialogue_triggers = {}


class NarrativeAgent(BaseAgent):
    """Agent responsible for narrative generation and dialogue"""
    
    def __init__(self, config: AgentConfig, router: MessageRouter,
                 model_name: str = "microsoft/DialoGPT-medium"):
        config.unity_compatible = True
        super().__init__(config, router)
        
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # Narrative state
        self.npc_profiles: Dict[str, NPCProfile] = {}
        self.active_quests: Dict[str, Quest] = {}
        self.world_lore: Dict[str, str] = {}
        self.dialogue_context: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        
        # Performance tracking
        self.generation_times = []
        self.dialogue_count = 0
        
    def _initialize(self):
        """Initialize language models"""
        try:
            # Load dialogue model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            # Load narrative templates
            self._load_narrative_templates()
            
            # Subscribe to events
            self.subscribe_event("player_interaction", self.on_player_interaction)
            self.subscribe_event("quest_update", self.on_quest_update)
            self.subscribe_event("world_event", self.on_world_event)
            
            logger.info("Narrative agent initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize narrative agent: {e}")
            self.state = AgentState.ERROR
    
    def handle_command(self, message: Message):
        """Handle narrative generation commands"""
        command = message.payload.get('command')
        
        if command == 'generate_dialogue':
            request_data = message.payload.get('request', {})
            request = NarrativeGenerationRequest(**request_data)
            self._handle_dialogue_request(request, message)
            
        elif command == 'create_npc':
            npc_data = message.payload.get('npc_data')
            self._create_npc(npc_data)
            
        elif command == 'generate_quest':
            quest_params = message.payload.get('quest_params')
            self._generate_quest(quest_params, message)
    
    def handle_query(self, message: Message):
        """Handle narrative queries"""
        query_type = message.payload.get('query_type')
        
        if query_type == 'npc_info':
            npc_id = message.payload.get('npc_id')
            info = self._get_npc_info(npc_id)
            self.send_message(
                message.sender_id,
                MessageType.RESPONSE,
                {'npc_info': info},
                correlation_id=message.message_id
            )
            
        elif query_type == 'quest_status':
            quest_id = message.payload.get('quest_id')
            status = self._get_quest_status(quest_id)
            self.send_message(
                message.sender_id,
                MessageType.RESPONSE,
                {'quest_status': status},
                correlation_id=message.message_id
            )
    
    def handle_state_update(self, message: Message):
        """Handle narrative state updates"""
        update_type = message.payload.get('update_type')
        
        if update_type == 'emotion_change':
            npc_id = message.payload.get('npc_id')
            new_emotion = message.payload.get('emotion')
            self._update_npc_emotion(npc_id, new_emotion)
            
        elif update_type == 'quest_progress':
            quest_id = message.payload.get('quest_id')
            progress = message.payload.get('progress')
            self._update_quest_progress(quest_id, progress)
    
    def _handle_dialogue_request(self, request: NarrativeGenerationRequest,
                                original_message: Message):
        """Generate dialogue based on request"""
        start_time = time.time()
        
        try:
            if request.narrative_type == 'dialogue':
                response = self._generate_dialogue(request)
            elif request.narrative_type == 'quest':
                response = self._generate_quest_dialogue(request)
            elif request.narrative_type == 'lore':
                response = self._generate_lore(request)
            else:
                response = self._generate_description(request)
            
            generation_time = time.time() - start_time
            self.generation_times.append(generation_time)
            self.dialogue_count += 1
            
            # Send response
            self.send_message(
                original_message.sender_id,
                MessageType.RESPONSE,
                {
                    'narrative': response,
                    'generation_time': generation_time
                },
                correlation_id=original_message.message_id
            )
            
            # Update dialogue history if NPC
            if request.character_id:
                self._update_dialogue_history(request.character_id, response)
            
        except Exception as e:
            logger.error(f"Dialogue generation failed: {e}")
            self.send_message(
                original_message.sender_id,
                MessageType.RESPONSE,
                {'error': str(e)},
                correlation_id=original_message.message_id
            )
    
    def _generate_dialogue(self, request: NarrativeGenerationRequest) -> Dict[str, Any]:
        """Generate NPC dialogue"""
        npc_profile = self.npc_profiles.get(request.character_id)
        
        if not npc_profile:
            # Create temporary profile
            npc_profile = NPCProfile(
                npc_id=request.character_id or "temp_npc",
                name="Stranger",
                personality_traits=request.personality_traits or ["neutral"],
                backstory="A mysterious figure",
                current_emotion=request.emotional_state
            )
        
        # Build context
        context = self._build_dialogue_context(npc_profile, request)
        
        # Generate response
        response_text = self._generate_text(context, max_length=100)
        
        # Apply personality and emotion filters
        response_text = self._apply_personality_filter(response_text, npc_profile)
        
        return {
            'speaker': npc_profile.name,
            'text': response_text,
            'emotion': npc_profile.current_emotion,
            'actions': self._extract_actions(response_text)
        }
    
    def _generate_quest_dialogue(self, request: NarrativeGenerationRequest) -> Dict[str, Any]:
        """Generate quest-related dialogue"""
        quest_context = request.context.get('quest', {})
        quest_id = quest_context.get('quest_id')
        
        if quest_id and quest_id in self.active_quests:
            quest = self.active_quests[quest_id]
            dialogue_type = quest_context.get('dialogue_type', 'offer')
            
            if dialogue_type == 'offer':
                text = f"Greetings, traveler. I have a task that requires someone of your skills. {quest.description} Will you help?"
            elif dialogue_type == 'progress':
                completed = quest_context.get('completed_objectives', 0)
                text = f"You've made progress! {completed} of {len(quest.objectives)} objectives completed."
            elif dialogue_type == 'complete':
                text = f"Excellent work! You've completed {quest.title}. Here's your reward."
            else:
                text = "How goes your quest?"
        else:
            text = "I might have work for an adventurer like you."
        
        return {
            'speaker': request.context.get('npc_name', 'Quest Giver'),
            'text': text,
            'quest_update': quest_context
        }
    
    def _generate_lore(self, request: NarrativeGenerationRequest) -> Dict[str, Any]:
        """Generate world lore"""
        lore_topic = request.context.get('topic', 'general')
        
        # Check existing lore
        if lore_topic in self.world_lore:
            base_lore = self.world_lore[lore_topic]
        else:
            # Generate new lore
            prompt = f"Ancient lore about {lore_topic}: "
            base_lore = self._generate_text(prompt, max_length=200)
            self.world_lore[lore_topic] = base_lore
        
        return {
            'topic': lore_topic,
            'text': base_lore,
            'related_topics': self._get_related_topics(lore_topic)
        }
    
    def _generate_description(self, request: NarrativeGenerationRequest) -> Dict[str, Any]:
        """Generate environmental or item descriptions"""
        desc_type = request.context.get('description_type', 'environment')
        target = request.context.get('target', 'unknown')
        
        if desc_type == 'environment':
            biome = request.context.get('biome', 'generic')
            text = self._generate_environment_description(biome, target)
        elif desc_type == 'item':
            text = self._generate_item_description(target, request.context)
        else:
            text = f"You see {target}."
        
        return {
            'type': desc_type,
            'target': target,
            'text': text
        }
    
    def _generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using language model"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                num_return_sequences=1,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from generated text
        response = generated[len(prompt):].strip()
        
        return response
    
    def _build_dialogue_context(self, npc: NPCProfile, 
                               request: NarrativeGenerationRequest) -> str:
        """Build context for dialogue generation"""
        context_parts = []
        
        # Add personality
        traits = ", ".join(npc.personality_traits)
        context_parts.append(f"{npc.name} is {traits}.")
        
        # Add emotional state
        context_parts.append(f"Currently feeling {npc.current_emotion}.")
        
        # Add recent dialogue history
        if npc.dialogue_history:
            recent = npc.dialogue_history[-3:]
            for entry in recent:
                context_parts.append(f"{entry['speaker']}: {entry['text']}")
        
        # Add player input
        if request.dialogue_history:
            last_player = request.dialogue_history[-1]
            context_parts.append(f"Player: {last_player.get('text', '')}")
        
        # Add response prompt
        context_parts.append(f"{npc.name}:")
        
        return "\n".join(context_parts)
    
    def _apply_personality_filter(self, text: str, npc: NPCProfile) -> str:
        """Apply personality-based modifications to text"""
        # Simple personality filters
        if "grumpy" in npc.personality_traits:
            text = text.replace("!", "...")
            text = text.replace("Hello", "Hmph")
        elif "cheerful" in npc.personality_traits:
            if not text.endswith("!"):
                text += "!"
            text = text.replace(".", "!")
        elif "mysterious" in npc.personality_traits:
            text = text.replace("I ", "One ")
            text += "..."
        
        return text
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract implied actions from dialogue"""
        actions = []
        
        # Simple action extraction
        if "give" in text.lower() or "take this" in text.lower():
            actions.append("give_item")
        if "follow" in text.lower():
            actions.append("follow_player")
        if "attack" in text.lower() or "fight" in text.lower():
            actions.append("initiate_combat")
        
        return actions
    
    def _create_npc(self, npc_data: Dict[str, Any]):
        """Create new NPC profile"""
        npc = NPCProfile(
            npc_id=npc_data['npc_id'],
            name=npc_data['name'],
            personality_traits=npc_data.get('traits', ['neutral']),
            backstory=npc_data.get('backstory', 'A resident of this world'),
            current_emotion=npc_data.get('emotion', 'neutral')
        )
        
        self.npc_profiles[npc.npc_id] = npc
        
        # Publish event
        self.publish_event('npc_created', {
            'npc_id': npc.npc_id,
            'name': npc.name
        })
    
    def _generate_quest(self, params: Dict[str, Any], message: Message):
        """Generate a new quest"""
        quest_type = params.get('type', 'fetch')
        difficulty = params.get('difficulty', 'medium')
        context_biome = params.get('biome', 'generic')
        
        # Generate quest components
        if quest_type == 'fetch':
            title = f"The Lost {self._random_item()}"
            description = f"Retrieve the ancient {self._random_item()} from the {context_biome} depths."
            objectives = [
                f"Find the entrance to the {context_biome} depths",
                f"Defeat the guardian",
                f"Retrieve the {self._random_item()}"
            ]
        elif quest_type == 'eliminate':
            enemy = self._random_enemy()
            title = f"Eliminate the {enemy}"
            description = f"The {enemy} has been terrorizing the local village."
            objectives = [
                f"Track down the {enemy}",
                f"Defeat the {enemy}",
                "Return to the quest giver"
            ]
        else:
            title = "A Mysterious Task"
            description = "Complete this unusual request."
            objectives = ["Investigate the strange occurrence"]
        
        quest = Quest(
            quest_id=f"quest_{int(time.time())}",
            title=title,
            description=description,
            objectives=objectives,
            rewards={'experience': 100 * (1 if difficulty == 'easy' else 2 if difficulty == 'medium' else 3)}
        )
        
        self.active_quests[quest.quest_id] = quest
        
        # Send response
        self.send_message(
            message.sender_id,
            MessageType.RESPONSE,
            {'quest': asdict(quest)},
            correlation_id=message.message_id
        )
    
    def _update_dialogue_history(self, character_id: str, response: Dict[str, Any]):
        """Update dialogue history for character"""
        if character_id in self.npc_profiles:
            entry = {
                'speaker': response['speaker'],
                'text': response['text'],
                'timestamp': time.time()
            }
            self.npc_profiles[character_id].dialogue_history.append(entry)
            
            # Keep only recent history
            if len(self.npc_profiles[character_id].dialogue_history) > 20:
                self.npc_profiles[character_id].dialogue_history = \
                    self.npc_profiles[character_id].dialogue_history[-20:]
    
    def on_player_interaction(self, message: Message):
        """Handle player interaction events"""
        event_data = message.payload.get('event_data', {})
        npc_id = event_data.get('npc_id')
        interaction_type = event_data.get('type')
        
        if npc_id and interaction_type == 'talk':
            # Generate appropriate dialogue
            request = NarrativeGenerationRequest(
                narrative_type='dialogue',
                character_id=npc_id,
                context=event_data
            )
            self._handle_dialogue_request(request, message)
    
    def on_quest_update(self, message: Message):
        """Handle quest update events"""
        event_data = message.payload.get('event_data', {})
        quest_id = event_data.get('quest_id')
        update_type = event_data.get('update_type')
        
        if quest_id in self.active_quests:
            quest = self.active_quests[quest_id]
            
            if update_type == 'objective_complete':
                # Generate completion dialogue
                self.publish_event('quest_dialogue_needed', {
                    'quest_id': quest_id,
                    'dialogue_type': 'progress'
                })
    
    def on_world_event(self, message: Message):
        """Handle world events that affect narrative"""
        event_data = message.payload.get('event_data', {})
        event_type = event_data.get('event_type')
        
        # Update world state based on events
        if event_type == 'time_change':
            time_of_day = event_data.get('time')
            # Adjust NPC behaviors/availability
            
        elif event_type == 'weather_change':
            weather = event_data.get('weather')
            # Adjust dialogue to reference weather
    
    def _load_narrative_templates(self):
        """Load narrative generation templates"""
        # Load from files or define templates
        self.templates = {
            'greeting': [
                "Well met, traveler!",
                "Greetings, adventurer.",
                "Ah, a new face!"
            ],
            'farewell': [
                "Safe travels!",
                "May your path be clear.",
                "Until we meet again."
            ]
        }
    
    def _random_item(self) -> str:
        """Generate random item name"""
        items = ["Artifact", "Relic", "Tome", "Crystal", "Amulet"]
        return items[int(time.time()) % len(items)]
    
    def _random_enemy(self) -> str:
        """Generate random enemy name"""
        enemies = ["Bandit Leader", "Dark Sorcerer", "Giant Spider", "Corrupted Elemental"]
        return enemies[int(time.time()) % len(enemies)]
    
    def _get_related_topics(self, topic: str) -> List[str]:
        """Get related lore topics"""
        # Simple relationship mapping
        relations = {
            'history': ['ancient_war', 'founding', 'heroes'],
            'magic': ['spells', 'artifacts', 'mages'],
            'geography': ['regions', 'dungeons', 'landmarks']
        }
        
        for category, topics in relations.items():
            if topic in topics:
                return [t for t in topics if t != topic]
        
        return []
    
    def _generate_environment_description(self, biome: str, target: str) -> str:
        """Generate environment description"""
        descriptions = {
            'forest': f"The {target} is surrounded by towering trees, their leaves rustling in the gentle breeze.",
            'desert': f"The {target} stands stark against the endless dunes, shimmering in the heat.",
            'cyberpunk': f"The {target} glows with neon lights, holographic advertisements flickering nearby.",
            'dungeon': f"The {target} looms in the shadows, ancient stones worn by centuries."
        }
        
        return descriptions.get(biome, f"You see the {target}.")
    
    def _generate_item_description(self, item: str, context: Dict[str, Any]) -> str:
        """Generate item description"""
        rarity = context.get('rarity', 'common')
        item_type = context.get('item_type', 'generic')
        
        if rarity == 'legendary':
            return f"The {item} radiates with ancient power, its surface covered in mystical runes."
        elif rarity == 'rare':
            return f"A finely crafted {item}, clearly the work of a master artisan."
        else:
            return f"A standard {item}, well-maintained and functional."
    
    def _get_npc_info(self, npc_id: str) -> Dict[str, Any]:
        """Get NPC information"""
        if npc_id in self.npc_profiles:
            npc = self.npc_profiles[npc_id]
            return {
                'name': npc.name,
                'traits': npc.personality_traits,
                'emotion': npc.current_emotion,
                'dialogue_count': len(npc.dialogue_history)
            }
        return {'error': 'NPC not found'}
    
    def _get_quest_status(self, quest_id: str) -> Dict[str, Any]:
        """Get quest status"""
        if quest_id in self.active_quests:
            quest = self.active_quests[quest_id]
            return {
                'title': quest.title,
                'objectives': quest.objectives,
                'active': True
            }
        return {'error': 'Quest not found'}
    
    def _update_npc_emotion(self, npc_id: str, emotion: str):
        """Update NPC emotional state"""
        if npc_id in self.npc_profiles:
            self.npc_profiles[npc_id].current_emotion = emotion
            logger.info(f"Updated {npc_id} emotion to {emotion}")
    
    def _update_quest_progress(self, quest_id: str, progress: Dict[str, Any]):
        """Update quest progress"""
        if quest_id in self.active_quests:
            # Update quest state based on progress
            logger.info(f"Quest {quest_id} progress updated")