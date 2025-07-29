"""
Asset Generation Agent for MADWE
Handles texture, sprite, and model generation using diffusion models
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import logging
from dataclasses import dataclass
import json
import hashlib

from .base_agent import BaseAgent, AgentConfig, AgentState, AgentState
from unity_bridge.communication import Message, MessageType, MessagePriority, MessageRouter
from .protocols import AssetGenerationRequest, RequestType
from ..models.diffusion.inference import DiffusionInference
from ..models.diffusion.lora_trainer import LoRAConfig

logger = logging.getLogger(__name__)


@dataclass
class AssetCache:
    """Cache entry for generated assets"""
    asset_id: str
    asset_type: str
    file_path: str
    metadata: Dict[str, Any]
    generated_at: float
    access_count: int = 0
    last_accessed: float = 0


class AssetAgent(BaseAgent):
    """Agent responsible for generating game assets using diffusion models"""
    
    def __init__(self, config: AgentConfig, router: MessageRouter, 
                 model_path: str, output_dir: str = "data/generated/assets"):
        # Initialize with Unity compatibility
        config.unity_compatible = True
        super().__init__(config, router)
        
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Asset generation state
        self.diffusion_model = None
        self.lora_models = {}
        self.asset_cache = {}
        self.generation_queue = asyncio.Queue() if config.enable_async else None
        
        # Performance tracking
        self.generation_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _initialize(self):
        """Initialize diffusion models and LoRA adapters"""
        try:
            # Load base diffusion model
            self.diffusion_model = DiffusionInference(
                model_id="stabilityai/stable-diffusion-2-1",
                device="cuda" if torch.cuda.is_available() else "cpu",
                enable_optimization=True
            )
            
            # Load biome-specific LoRA models
            biomes = ['forest', 'desert', 'cyberpunk', 'dungeon', 'underwater']
            for biome in biomes:
                lora_path = self.model_path / f"lora_{biome}.safetensors"
                if lora_path.exists():
                    self.lora_models[biome] = str(lora_path)
                    logger.info(f"Loaded LoRA model for {biome}")
            
            # Subscribe to relevant events
            self.subscribe_event("chunk_generated", self.on_chunk_generated)
            self.subscribe_event("style_update", self.on_style_update)
            
        except Exception as e:
            logger.error(f"Failed to initialize asset agent: {e}")
            self.state = AgentState.ERROR
    
    def handle_command(self, message: Message):
        """Handle asset generation commands"""
        command = message.payload.get('command')
        
        if command == 'generate_asset':
            request_data = message.payload.get('request', {})
            request = AssetGenerationRequest(**request_data)
            self._handle_generation_request(request, message)
            
        elif command == 'clear_cache':
            self._clear_cache(message.payload.get('asset_type'))
            
        elif command == 'preload_assets':
            asset_list = message.payload.get('assets', [])
            self._preload_assets(asset_list)
    
    def handle_query(self, message: Message):
        """Handle asset-related queries"""
        query_type = message.payload.get('query_type')
        
        if query_type == 'asset_status':
            asset_id = message.payload.get('asset_id')
            status = self._get_asset_status(asset_id)
            self.send_message(
                message.sender_id,
                MessageType.RESPONSE,
                {'status': status, 'asset_id': asset_id},
                correlation_id=message.message_id
            )
            
        elif query_type == 'cache_stats':
            stats = self._get_cache_stats()
            self.send_message(
                message.sender_id,
                MessageType.RESPONSE,
                {'cache_stats': stats},
                correlation_id=message.message_id
            )
    
    def handle_state_update(self, message: Message):
        """Handle state updates affecting asset generation"""
        update_type = message.payload.get('update_type')
        
        if update_type == 'biome_change':
            new_biome = message.payload.get('biome')
            self._update_active_lora(new_biome)
            
        elif update_type == 'quality_setting':
            quality = message.payload.get('quality')
            self._update_generation_quality(quality)
    
    def handle_unity_request(self, message: Message):
        """Handle Unity-specific asset requests"""
        request_type = message.payload.get('request_type')
        
        if request_type == 'texture_needed':
            # High-priority texture generation for Unity
            texture_spec = message.payload.get('texture_spec')
            self._generate_texture_urgent(texture_spec, message)
            
        elif request_type == 'batch_assets':
            # Batch asset generation for Unity scene
            asset_list = message.payload.get('assets', [])
            self._handle_batch_generation(asset_list, message)
    
    def _handle_generation_request(self, request: AssetGenerationRequest, 
                                  original_message: Message):
        """Process asset generation request"""
        # Check cache first
        cache_key = self._get_cache_key(request)
        if request.use_cache and cache_key in self.asset_cache:
            self.cache_hits += 1
            cached_asset = self.asset_cache[cache_key]
            self._send_asset_ready(cached_asset, original_message)
            return
        
        self.cache_misses += 1
        
        # Generate new asset
        if self.config.enable_async:
            asyncio.create_task(
                self._async_generate_asset(request, original_message)
            )
        else:
            self._generate_asset(request, original_message)
    
    def _generate_asset(self, request: AssetGenerationRequest, 
                       original_message: Message):
        """Synchronous asset generation"""
        start_time = time.time()
        
        try:
            # Select appropriate LoRA model
            if request.biome in self.lora_models:
                self.diffusion_model.load_lora(self.lora_models[request.biome])
            
            # Generate based on asset type
            if request.asset_type == 'texture':
                result = self._generate_texture(request)
            elif request.asset_type == 'sprite':
                result = self._generate_sprite(request)
            elif request.asset_type == 'character':
                result = self._generate_character(request)
            else:
                raise ValueError(f"Unknown asset type: {request.asset_type}")
            
            # Save and cache
            asset_id = self._save_asset(result, request)
            generation_time = time.time() - start_time
            self.generation_times.append(generation_time)
            
            # Create cache entry
            cache_entry = AssetCache(
                asset_id=asset_id,
                asset_type=request.asset_type,
                file_path=str(result['file_path']),
                metadata=result['metadata'],
                generated_at=time.time()
            )
            
            cache_key = self._get_cache_key(request)
            self.asset_cache[cache_key] = cache_entry
            
            # Send completion notification
            self._send_asset_ready(cache_entry, original_message)
            
        except Exception as e:
            logger.error(f"Asset generation failed: {e}")
            self.send_message(
                original_message.sender_id,
                MessageType.RESPONSE,
                {
                    'error': str(e),
                    'request': request.to_message_payload()
                },
                correlation_id=original_message.message_id
            )
    
    async def _async_generate_asset(self, request: AssetGenerationRequest,
                                   original_message: Message):
        """Asynchronous asset generation"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._generate_asset, request, original_message
        )
    
    def _generate_texture(self, request: AssetGenerationRequest) -> Dict[str, Any]:
        """Generate texture using diffusion model"""
        prompt = self._build_texture_prompt(request)
        
        # Generate with appropriate settings
        if request.quality_preset == 'fast':
            num_steps = 20
            guidance_scale = 7.5
        elif request.quality_preset == 'quality':
            num_steps = 50
            guidance_scale = 10.0
        else:  # balanced
            num_steps = 30
            guidance_scale = 8.5
        
        # Generate variations if requested
        images = []
        for i in range(request.variations):
            image = self.diffusion_model.generate(
                prompt=prompt,
                negative_prompt="blurry, low quality, artifacts",
                width=request.resolution[0],
                height=request.resolution[1],
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=None if i > 0 else 42  # Fixed seed for first, random for variations
            )
            images.append(image)
        
        # Make seamless if requested
        if request.seamless:
            images = [self._make_seamless(img) for img in images]
        
        # Save images
        file_paths = []
        for i, img in enumerate(images):
            filename = f"{request.asset_type}_{request.biome}_{int(time.time())}_{i}.png"
            file_path = self.output_dir / filename
            img.save(file_path)
            file_paths.append(file_path)
        
        return {
            'file_path': file_paths[0] if len(file_paths) == 1 else file_paths,
            'metadata': {
                'prompt': prompt,
                'biome': request.biome,
                'resolution': request.resolution,
                'seamless': request.seamless,
                'variations': request.variations
            }
        }
    
    def _generate_sprite(self, request: AssetGenerationRequest) -> Dict[str, Any]:
        """Generate sprite with transparency"""
        # Similar to texture but with alpha channel handling
        prompt = f"sprite, {request.biome} style, " + \
                ", ".join(f"{k}:{v}" for k, v in request.style_attributes.items())
        
        image = self.diffusion_model.generate(
            prompt=prompt,
            negative_prompt="background, 3d render",
            width=request.resolution[0],
            height=request.resolution[1],
            num_inference_steps=25
        )
        
        # Apply alpha channel based on background
        image = self._remove_background(image)
        
        filename = f"sprite_{request.biome}_{int(time.time())}.png"
        file_path = self.output_dir / filename
        image.save(file_path, 'PNG')
        
        return {
            'file_path': file_path,
            'metadata': {
                'prompt': prompt,
                'biome': request.biome,
                'has_alpha': True
            }
        }
    
    def _generate_character(self, request: AssetGenerationRequest) -> Dict[str, Any]:
        """Generate character with specific attributes"""
        # Character-specific generation logic
        character_attrs = request.style_attributes.get('character', {})
        prompt = f"character portrait, {character_attrs.get('race', 'human')}, " \
                f"{character_attrs.get('class', 'warrior')}, {request.biome} setting"
        
        image = self.diffusion_model.generate(
            prompt=prompt,
            width=request.resolution[0],
            height=request.resolution[1],
            num_inference_steps=40,
            guidance_scale=9.0
        )
        
        filename = f"character_{request.biome}_{int(time.time())}.png"
        file_path = self.output_dir / filename
        image.save(file_path)
        
        return {
            'file_path': file_path,
            'metadata': {
                'prompt': prompt,
                'character_attrs': character_attrs
            }
        }
    
    def _build_texture_prompt(self, request: AssetGenerationRequest) -> str:
        """Build detailed prompt for texture generation"""
        base_prompts = {
            'forest': "lush forest floor texture, moss, leaves, twigs, natural organic",
            'desert': "sandy desert texture, dunes, dry cracked earth, arid",
            'cyberpunk': "neon cyberpunk texture, holographic, tech panels, glowing circuits",
            'dungeon': "dark stone dungeon texture, medieval, moss, cracks, weathered",
            'underwater': "underwater coral texture, aquatic, bioluminescent, ocean floor"
        }
        
        prompt = base_prompts.get(request.biome, f"{request.biome} texture")
        
        # Add style attributes
        if request.style_attributes:
            style_str = ", ".join(f"{k} {v}" for k, v in request.style_attributes.items())
            prompt += f", {style_str}"
        
        # Add quality modifiers
        prompt += ", high quality, detailed, seamless pattern" if request.seamless else ", high quality, detailed"
        
        return prompt
    
    def _make_seamless(self, image: Image.Image) -> Image.Image:
        """Make texture seamless using blending"""
        width, height = image.size
        
        # Create seamless by blending edges
        blend_size = min(width, height) // 8
        
        # Implement seamless algorithm (simplified)
        # In production, use more sophisticated blending
        return image
    
    def _remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background for sprites"""
        # Simplified background removal
        # In production, use rembg or similar
        return image.convert("RGBA")
    
    def _save_asset(self, result: Dict[str, Any], 
                   request: AssetGenerationRequest) -> str:
        """Save generated asset and return ID"""
        # Generate unique asset ID
        asset_data = json.dumps({
            'type': request.asset_type,
            'biome': request.biome,
            'timestamp': time.time()
        })
        asset_id = hashlib.md5(asset_data.encode()).hexdigest()[:16]
        
        # Save metadata
        metadata_path = self.output_dir / f"{asset_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'asset_id': asset_id,
                'request': request.to_message_payload(),
                'result': {k: str(v) if isinstance(v, Path) else v 
                          for k, v in result.items()},
                'generated_at': time.time()
            }, f, indent=2)
        
        return asset_id
    
    def _send_asset_ready(self, asset: AssetCache, original_message: Message):
        """Send asset ready notification"""
        # Notify requesting agent
        self.send_message(
            original_message.sender_id,
            MessageType.RESPONSE,
            {
                'asset_id': asset.asset_id,
                'file_path': asset.file_path,
                'metadata': asset.metadata
            },
            correlation_id=original_message.message_id
        )
        
        # Notify Unity if needed
        if self.unity_compatible and hasattr(self, '_unity_bridge'):
            self.send_message(
                "unity_bridge",
                MessageType.ASSET_GENERATED,
                {
                    'asset_id': asset.asset_id,
                    'asset_type': asset.asset_type,
                    'file_path': asset.file_path
                },
                priority=MessagePriority.HIGH
            )
        
        # Publish event
        self.publish_event('asset_generated', {
            'asset_id': asset.asset_id,
            'asset_type': asset.asset_type,
            'biome': asset.metadata.get('biome')
        })
    
    def _get_cache_key(self, request: AssetGenerationRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            'type': request.asset_type,
            'biome': request.biome,
            'resolution': request.resolution,
            'style': sorted(request.style_attributes.items()),
            'seamless': request.seamless
        }
        return hashlib.md5(json.dumps(key_data).encode()).hexdigest()
    
    def _get_asset_status(self, asset_id: str) -> Dict[str, Any]:
        """Get status of asset generation"""
        for cache_entry in self.asset_cache.values():
            if cache_entry.asset_id == asset_id:
                return {
                    'status': 'completed',
                    'file_path': cache_entry.file_path,
                    'generated_at': cache_entry.generated_at
                }
        
        return {'status': 'not_found'}
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = sum(
            Path(entry.file_path).stat().st_size 
            for entry in self.asset_cache.values()
            if Path(entry.file_path).exists()
        )
        
        return {
            'cache_size': len(self.asset_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'total_size_mb': total_size / (1024 * 1024),
            'avg_generation_time': sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0
        }
    
    def _clear_cache(self, asset_type: Optional[str] = None):
        """Clear asset cache"""
        if asset_type:
            # Clear specific type
            keys_to_remove = [
                k for k, v in self.asset_cache.items() 
                if v.asset_type == asset_type
            ]
            for key in keys_to_remove:
                del self.asset_cache[key]
        else:
            # Clear all
            self.asset_cache.clear()
        
        self.cache_hits = 0
        self.cache_misses = 0
    
    def on_chunk_generated(self, message: Message):
        """Handle chunk generation events"""
        event_data = message.payload.get('event_data', {})
        chunk_data = event_data.get('data', {})
        biome = chunk_data.get('biome')
        
        # Pre-generate common assets for this biome
        if biome and biome in self.lora_models:
            common_assets = self._get_common_assets_for_biome(biome)
            for asset_spec in common_assets:
                request = AssetGenerationRequest(**asset_spec)
                if self._get_cache_key(request) not in self.asset_cache:
                    self._handle_generation_request(request, message)
    
    def on_style_update(self, message: Message):
        """Handle style update events"""
        event_data = message.payload.get('event_data', {})
        new_style = event_data.get('style')
        
        # Update generation parameters based on style
        logger.info(f"Style updated to: {new_style}")
    
    def _get_common_assets_for_biome(self, biome: str) -> List[Dict[str, Any]]:
        """Get list of common assets to pre-generate for biome"""
        common_assets = {
            'forest': [
                {'asset_type': 'texture', 'biome': 'forest', 'resolution': (512, 512), 'seamless': True},
                {'asset_type': 'sprite', 'biome': 'forest', 'style_attributes': {'object': 'tree'}}
            ],
            'desert': [
                {'asset_type': 'texture', 'biome': 'desert', 'resolution': (512, 512), 'seamless': True},
                {'asset_type': 'sprite', 'biome': 'desert', 'style_attributes': {'object': 'cactus'}}
            ],
            'cyberpunk': [
                {'asset_type': 'texture', 'biome': 'cyberpunk', 'resolution': (512, 512), 'seamless': True},
                {'asset_type': 'sprite', 'biome': 'cyberpunk', 'style_attributes': {'object': 'neon_sign'}}
            ]
        }
        
        return common_assets.get(biome, [])