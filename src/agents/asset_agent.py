"""
Asset Generation Agent for MADWE
Day 5: Friday, June 7 - Multi-Agent Foundation
Implements asset generation agent using diffusion models with multi-agent coordination
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import torch
import json
from datetime import datetime
import numpy as np
from PIL import Image
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from abc import ABC, abstractmethod
import logging

# LangGraph imports for agent orchestration
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

from diffusers import StableDiffusionPipeline
from ..models.diffusion.inference import DiffusionInference, FastInference


class AssetType(Enum):
    """Types of assets that can be generated"""

    TEXTURE = "texture"
    SPRITE = "sprite"
    PROP = "prop"
    BACKGROUND = "background"
    EFFECT = "effect"


class AssetPriority(Enum):
    """Priority levels for asset generation"""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class AssetRequest:
    """Request for asset generation"""

    request_id: str
    asset_type: AssetType
    biome: str
    description: str
    style_attributes: Dict[str, Any]
    resolution: Tuple[int, int]
    priority: AssetPriority
    seamless: bool = False
    variations: int = 1
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.metadata["created_at"] = datetime.now().isoformat()


@dataclass
class AssetResponse:
    """Response from asset generation"""

    request_id: str
    asset_id: str
    asset_type: AssetType
    biome: str
    image_paths: List[Path]
    generation_time: float
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class AssetAgentState:
    """State management for asset generation agent"""

    def __init__(self):
        self.pending_requests: List[AssetRequest] = []
        self.active_generations: Dict[str, AssetRequest] = {}
        self.completed_assets: Dict[str, AssetResponse] = {}
        self.generation_history: List[Dict[str, Any]] = []
        self.biome_models: Dict[str, Any] = {}
        self.generation_stats: Dict[str, Any] = {
            "total_generated": 0,
            "total_time": 0,
            "by_type": {},
            "by_biome": {},
        }


class AssetGenerationAgent:
    """Asset generation agent using diffusion models"""

    def __init__(
        self,
        model_dir: Path,
        output_dir: Path,
        config_path: Optional[Path] = None,
        use_fast_inference: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize state
        self.state = AssetAgentState()

        # Setup diffusion models
        self.use_fast_inference = use_fast_inference
        self.inference_engine = None
        self._setup_models()

        # Setup agent graph
        self.workflow = self._setup_workflow()

        # Message queue for inter-agent communication
        self.message_queue = asyncio.Queue()

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load agent configuration"""
        default_config = {
            "max_concurrent_generations": 2,
            "generation_timeout": 30,
            "quality_threshold": 0.7,
            "cache_size": 100,
            "biome_models": {
                "forest": "lora_textures_forest",
                "desert": "lora_textures_desert",
                "snow": "lora_textures_snow",
                "volcanic": "lora_textures_volcanic",
                "underwater": "lora_textures_underwater",
                "sky": "lora_textures_sky",
            },
        }

        if config_path and config_path.exists():
            with open(config_path, "r") as f:
                custom_config = json.load(f)
                default_config.update(custom_config)

        return default_config

    def _setup_models(self):
        """Initialize diffusion models"""
        self.logger.info("Setting up diffusion models...")

        if self.use_fast_inference:
            self.inference_engine = FastInference(
                model_id="runwayml/stable-diffusion-v1-5",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            self.inference_engine = DiffusionInference(
                model_id="runwayml/stable-diffusion-v1-5"
            )

        # Load biome-specific LoRA models
        for biome, lora_name in self.config["biome_models"].items():
            lora_path = self.model_dir / lora_name / "final"
            if lora_path.exists():
                self.logger.info(f"Loading LoRA for {biome}: {lora_path}")
                # Store path for on-demand loading
                self.state.biome_models[biome] = lora_path

    def _setup_workflow(self) -> StateGraph:
        """Setup LangGraph workflow for asset generation"""
        workflow = StateGraph()

        # Define nodes
        workflow.add_node("validate_request", self._validate_request)
        workflow.add_node("prepare_generation", self._prepare_generation)
        workflow.add_node("generate_asset", self._generate_asset)
        workflow.add_node("post_process", self._post_process)
        workflow.add_node("quality_check", self._quality_check)
        workflow.add_node("save_asset", self._save_asset)

        # Define edges
        workflow.add_edge("validate_request", "prepare_generation")
        workflow.add_edge("prepare_generation", "generate_asset")
        workflow.add_edge("generate_asset", "post_process")
        workflow.add_edge("post_process", "quality_check")

        # Conditional edge based on quality check
        workflow.add_conditional_edges(
            "quality_check",
            lambda x: "save_asset" if x["quality_passed"] else "generate_asset",
            {"save_asset": "save_asset", "generate_asset": "generate_asset"},
        )

        workflow.add_edge("save_asset", END)

        # Set entry point
        workflow.set_entry_point("validate_request")

        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def _validate_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate incoming asset request"""
        request = state["request"]

        # Validation checks
        if request.resolution[0] > 2048 or request.resolution[1] > 2048:
            state["error"] = "Resolution too high (max 2048x2048)"
            state["valid"] = False
            return state

        if request.asset_type not in AssetType:
            state["error"] = f"Invalid asset type: {request.asset_type}"
            state["valid"] = False
            return state

        if request.biome not in self.state.biome_models:
            state["error"] = f"Unsupported biome: {request.biome}"
            state["valid"] = False
            return state

        state["valid"] = True
        return state

    async def _prepare_generation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare for asset generation"""
        request = state["request"]

        # Load appropriate LoRA model
        if request.biome in self.state.biome_models:
            lora_path = self.state.biome_models[request.biome]
            self.inference_engine.load_lora(lora_path, adapter_name=request.biome)
            state["lora_loaded"] = request.biome

        # Construct prompt based on request
        prompt_parts = [
            f"{request.asset_type.value} asset",
            f"{request.biome} biome",
            request.description,
        ]

        # Add style attributes to prompt
        for attr, value in request.style_attributes.items():
            prompt_parts.append(f"{attr}: {value}")

        if request.seamless:
            prompt_parts.append("seamless tiling texture")

        prompt_parts.extend(["high quality", "game asset", "professional", "detailed"])

        state["prompt"] = ", ".join(prompt_parts)
        state["negative_prompt"] = "blurry, low quality, amateur, ugly, distorted"

        return state

    async def _generate_asset(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the actual asset"""
        request = state["request"]
        prompt = state["prompt"]
        negative_prompt = state["negative_prompt"]

        try:
            # Record start time
            start_time = asyncio.get_event_loop().time()

            # Generate variations
            images = []
            for i in range(request.variations):
                # Add variation to seed
                seed = hash(request.request_id) + i

                if self.use_fast_inference:
                    # Fast 4-step generation
                    image = await asyncio.to_thread(
                        self.inference_engine.generate_fast,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=request.resolution[0],
                        height=request.resolution[1],
                        guidance_scale=7.5,
                        seed=seed,
                    )
                else:
                    # Standard generation
                    image = await asyncio.to_thread(
                        self.inference_engine.generate,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=request.resolution[0],
                        height=request.resolution[1],
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        seed=seed,
                    )

                images.append(image)

            # Record generation time
            generation_time = asyncio.get_event_loop().time() - start_time

            state["generated_images"] = images
            state["generation_time"] = generation_time
            state["generation_success"] = True

        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            state["generation_success"] = False
            state["error"] = str(e)

        return state

    async def _post_process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process generated assets"""
        if not state.get("generation_success", False):
            return state

        request = state["request"]
        images = state["generated_images"]
        processed_images = []

        for image in images:
            # Convert to PIL if needed
            if isinstance(image, torch.Tensor):
                image = self._tensor_to_pil(image)

            # Apply seamless tiling if requested
            if request.seamless and request.asset_type == AssetType.TEXTURE:
                image = self._make_seamless(image)

            # Apply additional filters based on asset type
            if request.asset_type == AssetType.SPRITE:
                # Ensure transparency for sprites
                if image.mode != "RGBA":
                    image = image.convert("RGBA")
                # Remove background (simple version)
                image = self._remove_background(image)

            processed_images.append(image)

        state["processed_images"] = processed_images
        return state

    async def _quality_check(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check quality of generated assets"""
        if not state.get("generation_success", False):
            state["quality_passed"] = False
            return state

        images = state.get("processed_images", [])
        quality_scores = []

        for image in images:
            score = self._calculate_quality_score(image)
            quality_scores.append(score)

        avg_quality = np.mean(quality_scores)
        state["quality_scores"] = quality_scores
        state["avg_quality"] = avg_quality
        state["quality_passed"] = avg_quality >= self.config["quality_threshold"]

        if not state["quality_passed"]:
            state["retry_count"] = state.get("retry_count", 0) + 1
            if state["retry_count"] >= 3:
                # Max retries reached
                state["quality_passed"] = True  # Accept anyway
                self.logger.warning(
                    f"Max retries reached for request {state['request'].request_id}"
                )

        return state

    async def _save_asset(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Save generated assets to disk"""
        request = state["request"]
        images = state.get("processed_images", [])

        # Create output directory structure
        biome_dir = self.output_dir / request.biome / request.asset_type.value
        biome_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        asset_id = str(uuid.uuid4())

        for i, image in enumerate(images):
            # Generate filename
            filename = f"{request.biome}_{request.asset_type.value}_{asset_id}_{i}.png"
            filepath = biome_dir / filename

            # Save image
            image.save(filepath, optimize=True)
            saved_paths.append(filepath)

        # Create response
        response = AssetResponse(
            request_id=request.request_id,
            asset_id=asset_id,
            asset_type=request.asset_type,
            biome=request.biome,
            image_paths=saved_paths,
            generation_time=state.get("generation_time", 0),
            metadata={
                "prompt": state.get("prompt", ""),
                "quality_scores": state.get("quality_scores", []),
                "resolution": request.resolution,
                "seamless": request.seamless,
            },
        )

        # Update state
        self.state.completed_assets[request.request_id] = response
        self._update_statistics(request, response)

        state["response"] = response
        return state

    def _make_seamless(self, image: Image.Image, blend_width: int = 64) -> Image.Image:
        """Make texture seamlessly tileable"""
        width, height = image.size

        # Create a copy to work with
        seamless = image.copy()

        # Blend horizontal edges
        for x in range(blend_width):
            alpha = x / blend_width
            for y in range(height):
                # Get pixels from both edges
                left_pixel = image.getpixel((x, y))
                right_pixel = image.getpixel((width - blend_width + x, y))

                # Blend pixels
                blended = tuple(
                    int((1 - alpha) * r + alpha * l)
                    for l, r in zip(left_pixel, right_pixel)
                )

                # Set blended pixels
                seamless.putpixel((x, y), blended)
                seamless.putpixel((width - blend_width + x, y), blended)

        # Blend vertical edges
        for y in range(blend_width):
            alpha = y / blend_width
            for x in range(width):
                # Get pixels from both edges
                top_pixel = seamless.getpixel((x, y))
                bottom_pixel = seamless.getpixel((x, height - blend_width + y))

                # Blend pixels
                blended = tuple(
                    int((1 - alpha) * b + alpha * t)
                    for t, b in zip(top_pixel, bottom_pixel)
                )

                # Set blended pixels
                seamless.putpixel((x, y), blended)
                seamless.putpixel((x, height - blend_width + y), blended)

        return seamless

    def _remove_background(self, image: Image.Image) -> Image.Image:
        """Simple background removal for sprites"""
        # Convert to RGBA if needed
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Get image data
        data = np.array(image)

        # Simple threshold-based removal (assumes white/light background)
        # In production, use a proper segmentation model
        threshold = 240
        mask = np.all(data[:, :, :3] > threshold, axis=2)

        # Set alpha channel
        data[:, :, 3] = np.where(mask, 0, 255)

        return Image.fromarray(data, "RGBA")

    def _calculate_quality_score(self, image: Image.Image) -> float:
        """Calculate quality score for generated image"""
        # Convert to numpy array
        img_array = np.array(image)

        # Basic quality metrics
        scores = []

        # 1. Sharpness (using Laplacian variance)
        gray = (
            np.mean(img_array[:, :, :3], axis=2)
            if len(img_array.shape) > 2
            else img_array
        )
        laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
        sharpness = laplacian.var()
        scores.append(min(sharpness / 1000, 1.0))  # Normalize

        # 2. Contrast
        contrast = gray.std() / 128
        scores.append(min(contrast, 1.0))

        # 3. Color distribution (avoid flat colors)
        if len(img_array.shape) > 2:
            color_variance = np.mean([img_array[:, :, i].std() for i in range(3)])
            scores.append(min(color_variance / 80, 1.0))

        # 4. Edge density (detail level)
        edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        scores.append(min(edge_density * 10, 1.0))

        return np.mean(scores)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert torch tensor to PIL Image"""
        # Assume tensor is in [-1, 1] range
        tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1)

        # Convert to numpy
        if tensor.dim() == 4:
            tensor = tensor[0]  # Remove batch dimension

        array = tensor.cpu().numpy()
        array = (array * 255).astype(np.uint8)

        # Rearrange channels if needed
        if array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))

        return Image.fromarray(array)

    def _update_statistics(self, request: AssetRequest, response: AssetResponse):
        """Update generation statistics"""
        stats = self.state.generation_stats

        # Update totals
        stats["total_generated"] += len(response.image_paths)
        stats["total_time"] += response.generation_time

        # Update by type
        asset_type = request.asset_type.value
        if asset_type not in stats["by_type"]:
            stats["by_type"][asset_type] = {"count": 0, "time": 0}
        stats["by_type"][asset_type]["count"] += len(response.image_paths)
        stats["by_type"][asset_type]["time"] += response.generation_time

        # Update by biome
        if request.biome not in stats["by_biome"]:
            stats["by_biome"][request.biome] = {"count": 0, "time": 0}
        stats["by_biome"][request.biome]["count"] += len(response.image_paths)
        stats["by_biome"][request.biome]["time"] += response.generation_time

        # Add to history
        self.state.generation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "request_id": request.request_id,
                "asset_type": asset_type,
                "biome": request.biome,
                "generation_time": response.generation_time,
                "quality_scores": response.metadata.get("quality_scores", []),
            }
        )

    async def process_request(self, request: AssetRequest) -> AssetResponse:
        """Process a single asset generation request"""
        # Add to active generations
        self.state.active_generations[request.request_id] = request

        try:
            # Run through workflow
            initial_state = {"request": request}
            final_state = await self.workflow.ainvoke(initial_state)

            if "response" in final_state:
                response = final_state["response"]
            else:
                # Create error response
                response = AssetResponse(
                    request_id=request.request_id,
                    asset_id="",
                    asset_type=request.asset_type,
                    biome=request.biome,
                    image_paths=[],
                    generation_time=0,
                    metadata={},
                    success=False,
                    error_message=final_state.get("error", "Unknown error"),
                )

            return response

        finally:
            # Remove from active generations
            if request.request_id in self.state.active_generations:
                del self.state.active_generations[request.request_id]

    async def process_batch(self, requests: List[AssetRequest]) -> List[AssetResponse]:
        """Process multiple requests with concurrency control"""
        # Sort by priority
        sorted_requests = sorted(requests, key=lambda r: r.priority.value)

        # Process with limited concurrency
        semaphore = asyncio.Semaphore(self.config["max_concurrent_generations"])

        async def process_with_limit(request):
            async with semaphore:
                return await self.process_request(request)

        # Process all requests
        tasks = [process_with_limit(req) for req in sorted_requests]
        responses = await asyncio.gather(*tasks)

        return responses

    async def handle_message(self, message: Dict[str, Any]):
        """Handle inter-agent messages"""
        msg_type = message.get("type")

        if msg_type == "asset_request":
            # Convert to AssetRequest
            request_data = message["data"]
            request = AssetRequest(**request_data)

            # Add to queue
            self.state.pending_requests.append(request)

            # Send acknowledgment
            await self.send_message(
                {
                    "type": "asset_request_ack",
                    "request_id": request.request_id,
                    "agent_id": "asset_generator",
                }
            )

        elif msg_type == "status_query":
            # Return current status
            await self.send_message(
                {
                    "type": "status_response",
                    "agent_id": "asset_generator",
                    "active_generations": len(self.state.active_generations),
                    "pending_requests": len(self.state.pending_requests),
                    "completed_assets": len(self.state.completed_assets),
                    "statistics": self.state.generation_stats,
                }
            )

    async def send_message(self, message: Dict[str, Any]):
        """Send message to other agents"""
        message["sender"] = "asset_generator"
        message["timestamp"] = datetime.now().isoformat()
        await self.message_queue.put(message)

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        stats = self.state.generation_stats.copy()

        # Calculate averages
        if stats["total_generated"] > 0:
            stats["avg_generation_time"] = (
                stats["total_time"] / stats["total_generated"]
            )
        else:
            stats["avg_generation_time"] = 0

        return stats

    async def run(self):
        """Main agent loop"""
        self.logger.info("Asset Generation Agent started")

        while True:
            # Process pending requests
            if self.state.pending_requests:
                # Get batch of requests
                batch_size = min(
                    len(self.state.pending_requests),
                    self.config["max_concurrent_generations"],
                )
                batch = self.state.pending_requests[:batch_size]
                self.state.pending_requests = self.state.pending_requests[batch_size:]

                # Process batch
                responses = await self.process_batch(batch)

                # Send completion messages
                for response in responses:
                    await self.send_message(
                        {"type": "asset_completed", "response": asdict(response)}
                    )

            # Check for messages
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
                await self.handle_message(message)
            except asyncio.TimeoutError:
                pass

            # Brief pause
            await asyncio.sleep(0.01)


class PerformanceMonitor:
    """Monitor agent performance"""

    def __init__(self):
        self.metrics = {
            "generation_times": [],
            "quality_scores": [],
            "memory_usage": [],
            "gpu_utilization": [],
        }

    def record_generation(self, time: float, quality: float):
        """Record generation metrics"""
        self.metrics["generation_times"].append(time)
        self.metrics["quality_scores"].append(quality)

        # Record GPU metrics if available
        if torch.cuda.is_available():
            self.metrics["memory_usage"].append(
                torch.cuda.memory_allocated() / 1024**3  # GB
            )
            # GPU utilization would require nvidia-ml-py

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}

        for metric, values in self.metrics.items():
            if values:
                summary[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }

        return summary


# Import OpenCV for quality metrics
try:
    import cv2
except ImportError:
    print("Warning: OpenCV not available, some quality metrics will be limited")
    cv2 = None


async def main():
    """Test the asset generation agent"""
    import argparse

    parser = argparse.ArgumentParser(description="Run Asset Generation Agent")
    parser.add_argument(
        "--model-dir", type=str, default="data/models", help="Model directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/generated_assets",
        help="Output directory",
    )
    parser.add_argument("--fast", action="store_true", help="Use fast inference")

    args = parser.parse_args()

    # Create agent
    agent = AssetGenerationAgent(
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
        use_fast_inference=args.fast,
    )

    # Create test request
    test_request = AssetRequest(
        request_id=str(uuid.uuid4()),
        asset_type=AssetType.TEXTURE,
        biome="forest",
        description="mossy rock texture with ferns",
        style_attributes={
            "detail_level": "high",
            "color_palette": "natural greens and browns",
            "lighting": "soft forest lighting",
        },
        resolution=(512, 512),
        priority=AssetPriority.HIGH,
        seamless=True,
        variations=3,
    )

    # Process request
    print("Processing test request...")
    response = await agent.process_request(test_request)

    if response.success:
        print(f"Success! Generated {len(response.image_paths)} assets")
        print(f"Generation time: {response.generation_time:.2f}s")
        print(f"Saved to: {response.image_paths[0].parent}")
    else:
        print(f"Failed: {response.error_message}")

    # Print statistics
    stats = agent.get_statistics()
    print(f"\nStatistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
