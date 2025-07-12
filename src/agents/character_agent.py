"""
Character Generation Agent for MADWE
Day 8: Wednesday, June 12 - Agent Communication
Implements character generation with pose control, equipment variation, and environment consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from PIL import Image, ImageDraw
import asyncio
import uuid
from collections import defaultdict

from ..agents.base_agent import BaseAgent, Message, MessageType, AgentState
from ..models.diffusion.inference import DiffusionInference, FastInference
from diffusers import StableDiffusionPipeline, ControlNetModel
from transformers import pipeline


class CharacterType(Enum):
    """Types of characters that can be generated"""

    HERO = "hero"
    NPC = "npc"
    ENEMY = "enemy"
    BOSS = "boss"
    MERCHANT = "merchant"
    QUEST_GIVER = "quest_giver"
    COMPANION = "companion"


class PoseType(Enum):
    """Standard pose types for characters"""

    IDLE = "idle"
    WALKING = "walking"
    RUNNING = "running"
    ATTACKING = "attacking"
    DEFENDING = "defending"
    CASTING = "casting"
    INTERACTING = "interacting"
    SITTING = "sitting"
    DEAD = "dead"


@dataclass
class CharacterAttributes:
    """Attributes defining a character"""

    character_type: CharacterType
    race: str  # human, elf, dwarf, etc.
    class_type: str  # warrior, mage, rogue, etc.
    gender: str
    age_category: str  # young, adult, elderly
    build: str  # slim, average, muscular, heavy
    skin_tone: str
    hair_style: str
    hair_color: str
    facial_features: Dict[str, str]
    personality_traits: List[str]

    def to_prompt_description(self) -> str:
        """Convert attributes to prompt description"""
        desc_parts = [
            f"{self.age_category} {self.gender} {self.race} {self.class_type}",
            f"{self.build} build",
            f"{self.skin_tone} skin",
            f"{self.hair_style} {self.hair_color} hair",
        ]

        # Add facial features
        for feature, value in self.facial_features.items():
            desc_parts.append(f"{value} {feature}")

        return ", ".join(desc_parts)


@dataclass
class Equipment:
    """Equipment/clothing for a character"""

    slot: str  # head, chest, legs, feet, hands, weapon, accessory
    name: str
    material: str
    color_scheme: List[str]
    enchantment: Optional[str] = None
    wear_level: float = 0.0  # 0 = pristine, 1 = heavily worn

    def to_prompt_description(self) -> str:
        """Convert equipment to prompt description"""
        desc = f"{self.name} made of {self.material}"
        if self.color_scheme:
            desc += f" in {' and '.join(self.color_scheme)} colors"
        if self.enchantment:
            desc += f" with {self.enchantment} enchantment"
        if self.wear_level > 0.7:
            desc += ", heavily worn"
        elif self.wear_level > 0.3:
            desc += ", slightly worn"
        return desc


@dataclass
class CharacterRequest:
    """Request for character generation"""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    character_type: CharacterType = CharacterType.NPC
    attributes: Optional[CharacterAttributes] = None
    pose: PoseType = PoseType.IDLE
    equipment: List[Equipment] = field(default_factory=list)
    environment_context: Optional[Dict[str, Any]] = None
    style_reference: Optional[str] = None
    resolution: Tuple[int, int] = (512, 768)  # Portrait orientation
    variations: int = 1
    priority: int = 5


@dataclass
class CharacterResponse:
    """Response from character generation"""

    request_id: str
    character_id: str
    character_type: CharacterType
    image_paths: List[Path]
    pose_skeleton: Optional[np.ndarray]
    equipment_masks: Optional[Dict[str, np.ndarray]]
    generation_time: float
    consistency_score: float
    metadata: Dict[str, Any]


class PoseController:
    """Handles pose control for character generation"""

    def __init__(self):
        self.joint_connections = [
            # Head and spine
            (0, 1),  # nose -> neck
            (1, 2),  # neck -> right shoulder
            (1, 5),  # neck -> left shoulder
            (1, 8),  # neck -> spine
            (8, 11),  # spine -> pelvis
            # Right arm
            (2, 3),  # right shoulder -> right elbow
            (3, 4),  # right elbow -> right wrist
            # Left arm
            (5, 6),  # left shoulder -> left elbow
            (6, 7),  # left elbow -> left wrist
            # Right leg
            (11, 12),  # pelvis -> right hip
            (12, 13),  # right hip -> right knee
            (13, 14),  # right knee -> right ankle
            # Left leg
            (11, 15),  # pelvis -> left hip
            (15, 16),  # left hip -> left knee
            (16, 17),  # left knee -> left ankle
        ]

        self.pose_templates = self._load_pose_templates()

    def _load_pose_templates(self) -> Dict[PoseType, np.ndarray]:
        """Load predefined pose templates"""
        templates = {}

        # Define keypoint positions for each pose (normalized 0-1)
        # Format: [x, y] for each of 18 keypoints

        templates[PoseType.IDLE] = np.array(
            [
                [0.5, 0.1],  # 0: nose
                [0.5, 0.2],  # 1: neck
                [0.4, 0.25],  # 2: right shoulder
                [0.4, 0.4],  # 3: right elbow
                [0.4, 0.55],  # 4: right wrist
                [0.6, 0.25],  # 5: left shoulder
                [0.6, 0.4],  # 6: left elbow
                [0.6, 0.55],  # 7: left wrist
                [0.5, 0.4],  # 8: spine
                [0.5, 0.5],  # 9: (unused)
                [0.5, 0.5],  # 10: (unused)
                [0.5, 0.6],  # 11: pelvis
                [0.45, 0.65],  # 12: right hip
                [0.45, 0.8],  # 13: right knee
                [0.45, 0.95],  # 14: right ankle
                [0.55, 0.65],  # 15: left hip
                [0.55, 0.8],  # 16: left knee
                [0.55, 0.95],  # 17: left ankle
            ]
        )

        templates[PoseType.WALKING] = np.array(
            [
                [0.5, 0.1],  # nose
                [0.5, 0.2],  # neck
                [0.4, 0.25],  # right shoulder
                [0.35, 0.4],  # right elbow (forward)
                [0.3, 0.5],  # right wrist
                [0.6, 0.25],  # left shoulder
                [0.65, 0.35],  # left elbow (back)
                [0.7, 0.45],  # left wrist
                [0.5, 0.4],  # spine
                [0.5, 0.5],  # unused
                [0.5, 0.5],  # unused
                [0.5, 0.6],  # pelvis
                [0.45, 0.65],  # right hip
                [0.4, 0.8],  # right knee (forward)
                [0.35, 0.95],  # right ankle
                [0.55, 0.65],  # left hip
                [0.6, 0.8],  # left knee (back)
                [0.65, 0.95],  # left ankle
            ]
        )

        templates[PoseType.ATTACKING] = np.array(
            [
                [0.5, 0.08],  # nose
                [0.5, 0.18],  # neck
                [0.35, 0.22],  # right shoulder (pulled back)
                [0.25, 0.25],  # right elbow (raised)
                [0.2, 0.15],  # right wrist (weapon raised)
                [0.6, 0.25],  # left shoulder
                [0.65, 0.4],  # left elbow (forward)
                [0.7, 0.35],  # left wrist
                [0.48, 0.38],  # spine (twisted)
                [0.5, 0.5],  # unused
                [0.5, 0.5],  # unused
                [0.48, 0.58],  # pelvis
                [0.4, 0.63],  # right hip
                [0.35, 0.78],  # right knee
                [0.3, 0.93],  # right ankle (back foot)
                [0.55, 0.63],  # left hip
                [0.6, 0.78],  # left knee
                [0.65, 0.93],  # left ankle (front foot)
            ]
        )

        return templates

    def get_pose_skeleton(
        self, pose_type: PoseType, resolution: Tuple[int, int]
    ) -> np.ndarray:
        """Get pose skeleton for given pose type"""
        if pose_type not in self.pose_templates:
            pose_type = PoseType.IDLE

        template = self.pose_templates[pose_type]

        # Scale to resolution
        width, height = resolution
        skeleton = template.copy()
        skeleton[:, 0] *= width
        skeleton[:, 1] *= height

        return skeleton

    def generate_pose_control_image(
        self, pose_type: PoseType, resolution: Tuple[int, int], thickness: int = 5
    ) -> Image.Image:
        """Generate control image for pose"""
        skeleton = self.get_pose_skeleton(pose_type, resolution)

        # Create blank image
        img = Image.new("RGB", resolution, color="black")
        draw = ImageDraw.Draw(img)

        # Draw skeleton
        for start_idx, end_idx in self.joint_connections:
            start = tuple(skeleton[start_idx].astype(int))
            end = tuple(skeleton[end_idx].astype(int))
            draw.line([start, end], fill="white", width=thickness)

        # Draw joints
        for point in skeleton:
            x, y = point.astype(int)
            draw.ellipse(
                [x - thickness, y - thickness, x + thickness, y + thickness], fill="red"
            )

        return img

    def interpolate_poses(
        self,
        pose1: PoseType,
        pose2: PoseType,
        alpha: float,
        resolution: Tuple[int, int],
    ) -> np.ndarray:
        """Interpolate between two poses"""
        skeleton1 = self.get_pose_skeleton(pose1, resolution)
        skeleton2 = self.get_pose_skeleton(pose2, resolution)

        # Linear interpolation
        interpolated = skeleton1 * (1 - alpha) + skeleton2 * alpha

        return interpolated


class EquipmentManager:
    """Manages equipment generation and layering"""

    def __init__(self):
        self.equipment_templates = self._load_equipment_templates()
        self.material_properties = {
            "leather": {"shine": 0.3, "texture": "rough", "flexibility": 0.8},
            "metal": {"shine": 0.9, "texture": "smooth", "flexibility": 0.1},
            "cloth": {"shine": 0.1, "texture": "soft", "flexibility": 1.0},
            "fur": {"shine": 0.2, "texture": "fuzzy", "flexibility": 0.9},
            "crystal": {"shine": 1.0, "texture": "faceted", "flexibility": 0.0},
            "wood": {"shine": 0.4, "texture": "grainy", "flexibility": 0.2},
        }

    def _load_equipment_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load equipment templates"""
        return {
            "warrior": {
                "head": ["helmet", "chainmail_coif", "none"],
                "chest": ["plate_armor", "chainmail", "leather_armor"],
                "legs": ["plate_greaves", "chainmail_leggings", "leather_pants"],
                "weapon": ["sword", "axe", "mace", "spear"],
            },
            "mage": {
                "head": ["wizard_hat", "circlet", "hood"],
                "chest": ["robe", "tunic", "cloak"],
                "legs": ["cloth_pants", "robe_bottom"],
                "weapon": ["staff", "wand", "orb", "tome"],
            },
            "rogue": {
                "head": ["hood", "bandana", "none"],
                "chest": ["leather_vest", "cloak", "tunic"],
                "legs": ["leather_pants", "cloth_pants"],
                "weapon": ["dagger", "bow", "throwing_knives"],
            },
        }

    def generate_equipment_set(
        self, class_type: str, level: int = 1, style: str = "standard"
    ) -> List[Equipment]:
        """Generate a complete equipment set"""
        if class_type not in self.equipment_templates:
            class_type = "warrior"

        templates = self.equipment_templates[class_type]
        equipment_set = []

        # Determine quality based on level
        quality_chance = min(0.1 + (level * 0.05), 0.8)

        for slot, options in templates.items():
            if np.random.random() < 0.8:  # 80% chance to have equipment in slot
                item_name = np.random.choice(options)

                # Determine material based on class and quality
                if np.random.random() < quality_chance:
                    materials = (
                        ["metal", "crystal"]
                        if class_type == "warrior"
                        else ["cloth", "crystal"]
                    )
                else:
                    materials = ["leather", "cloth"]

                material = np.random.choice(materials)

                # Generate color scheme based on style
                color_scheme = self._generate_color_scheme(class_type, style)

                # Add enchantment for higher levels
                enchantment = None
                if level > 5 and np.random.random() < 0.3:
                    enchantments = ["glowing", "frost", "fire", "shadow", "holy"]
                    enchantment = np.random.choice(enchantments)

                equipment = Equipment(
                    slot=slot,
                    name=item_name,
                    material=material,
                    color_scheme=color_scheme,
                    enchantment=enchantment,
                    wear_level=np.random.uniform(0, 0.5),
                )

                equipment_set.append(equipment)

        return equipment_set

    def _generate_color_scheme(self, class_type: str, style: str) -> List[str]:
        """Generate appropriate color scheme"""
        color_palettes = {
            "warrior": {
                "standard": ["silver", "gray", "brown"],
                "royal": ["gold", "crimson", "white"],
                "dark": ["black", "dark gray", "red"],
            },
            "mage": {
                "standard": ["blue", "purple", "gray"],
                "royal": ["gold", "white", "azure"],
                "dark": ["black", "purple", "green"],
            },
            "rogue": {
                "standard": ["brown", "dark gray", "black"],
                "royal": ["black", "gold", "crimson"],
                "dark": ["black", "dark green", "gray"],
            },
        }

        palette = color_palettes.get(class_type, color_palettes["warrior"])
        colors = palette.get(style, palette["standard"])

        # Select 2-3 colors
        num_colors = np.random.randint(2, 4)
        return list(np.random.choice(colors, num_colors, replace=False))

    def create_equipment_mask(
        self, equipment: Equipment, character_image: Image.Image
    ) -> np.ndarray:
        """Create mask for equipment piece"""
        # This would use segmentation model in practice
        # For now, create simple masks based on slot
        width, height = character_image.size
        mask = np.zeros((height, width), dtype=np.uint8)

        # Simple geometric masks based on slot
        if equipment.slot == "head":
            # Head region
            y_start = int(height * 0.05)
            y_end = int(height * 0.25)
            x_start = int(width * 0.3)
            x_end = int(width * 0.7)
            mask[y_start:y_end, x_start:x_end] = 255

        elif equipment.slot == "chest":
            # Torso region
            y_start = int(height * 0.25)
            y_end = int(height * 0.6)
            x_start = int(width * 0.2)
            x_end = int(width * 0.8)
            mask[y_start:y_end, x_start:x_end] = 255

        elif equipment.slot == "legs":
            # Legs region
            y_start = int(height * 0.6)
            y_end = int(height * 0.95)
            x_start = int(width * 0.25)
            x_end = int(width * 0.75)
            mask[y_start:y_end, x_start:x_end] = 255

        return mask


class CharacterEnvironmentConsistency:
    """Ensures character-environment consistency"""

    def __init__(self):
        self.biome_adaptations = {
            "forest": {
                "color_adjustments": {"saturation": 0.9, "brightness": 0.95},
                "suggested_materials": ["leather", "cloth", "wood"],
                "weathering_types": ["moss", "dirt"],
            },
            "desert": {
                "color_adjustments": {"saturation": 0.8, "brightness": 1.1},
                "suggested_materials": ["cloth", "leather"],
                "weathering_types": ["sand", "sun_bleaching"],
            },
            "snow": {
                "color_adjustments": {"saturation": 0.7, "brightness": 1.05},
                "suggested_materials": ["fur", "leather", "metal"],
                "weathering_types": ["frost", "ice"],
            },
            "volcanic": {
                "color_adjustments": {"saturation": 1.1, "brightness": 0.9},
                "suggested_materials": ["metal", "crystal"],
                "weathering_types": ["soot", "heat_damage"],
            },
            "underwater": {
                "color_adjustments": {"saturation": 0.85, "brightness": 0.85},
                "suggested_materials": ["crystal", "metal"],
                "weathering_types": ["coral_growth", "rust"],
            },
        }

    def adapt_character_to_environment(
        self,
        character_attrs: CharacterAttributes,
        equipment: List[Equipment],
        environment: Dict[str, Any],
    ) -> Tuple[CharacterAttributes, List[Equipment]]:
        """Adapt character appearance to environment"""
        biome = environment.get("biome", "forest")

        if biome in self.biome_adaptations:
            adaptation = self.biome_adaptations[biome]

            # Adapt equipment materials
            suggested_materials = adaptation["suggested_materials"]
            for eq in equipment:
                if eq.material not in suggested_materials and np.random.random() < 0.5:
                    eq.material = np.random.choice(suggested_materials)

            # Add environmental weathering
            weathering_types = adaptation["weathering_types"]
            for eq in equipment:
                if np.random.random() < 0.3:
                    eq.wear_level = min(eq.wear_level + 0.2, 1.0)

        return character_attrs, equipment

    def check_consistency_score(
        self, character_image: torch.Tensor, environment_features: torch.Tensor
    ) -> float:
        """Check how well character fits the environment"""
        # Extract color statistics
        char_colors = self._extract_color_stats(character_image)
        env_colors = self._extract_color_stats(environment_features)

        # Compare color distributions
        color_similarity = F.cosine_similarity(
            char_colors.unsqueeze(0), env_colors.unsqueeze(0)
        ).item()

        # Check lighting consistency
        char_brightness = torch.mean(character_image).item()
        env_brightness = torch.mean(environment_features).item()
        brightness_diff = abs(char_brightness - env_brightness)
        brightness_score = 1.0 - min(brightness_diff * 2, 1.0)

        # Combined score
        consistency_score = (color_similarity + brightness_score) / 2

        return consistency_score

    def _extract_color_stats(self, image: torch.Tensor) -> torch.Tensor:
        """Extract color statistics from image"""
        if image.dim() == 4:
            image = image[0]  # Take first in batch

        # Compute color histogram
        hist_bins = 16
        color_stats = []

        for c in range(3):  # RGB channels
            hist = torch.histc(image[c], bins=hist_bins, min=0, max=1)
            hist = hist / hist.sum()  # Normalize
            color_stats.append(hist)

        return torch.cat(color_stats)


class CharacterGenerationAgent(BaseAgent):
    """Agent for generating game characters"""

    def __init__(
        self,
        agent_id: str,
        model_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        message_bus=None,
    ):
        super().__init__(agent_id, "character_generator", config, message_bus)

        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.pose_controller = PoseController()
        self.equipment_manager = EquipmentManager()
        self.consistency_checker = CharacterEnvironmentConsistency()

        # Generation models
        self.diffusion_model = None
        self.controlnet_model = None

        # Request queue
        self.request_queue: List[CharacterRequest] = []

        # Character database
        self.character_database: Dict[str, CharacterResponse] = {}

        # Statistics
        self.generation_stats = defaultdict(int)

    async def _initialize(self):
        """Initialize character generation models"""
        # Setup diffusion model
        self.diffusion_model = DiffusionInference(
            model_id="runwayml/stable-diffusion-v1-5"
        )

        # Setup ControlNet for pose control
        try:
            self.controlnet_model = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
            )
            self.logger.info("ControlNet model loaded for pose control")
        except Exception as e:
            self.logger.warning(f"ControlNet not available: {e}")
            self.controlnet_model = None

        # Register message handlers
        self.register_handler(MessageType.REQUEST, self._handle_character_request)
        self.register_handler(MessageType.UPDATE, self._handle_environment_update)

    async def _start(self):
        """Start character generation loop"""
        generation_task = asyncio.create_task(self._generation_loop())
        self._tasks.append(generation_task)

    async def _generation_loop(self):
        """Main generation loop"""
        while self._running:
            if self.request_queue:
                # Get highest priority request
                self.request_queue.sort(key=lambda r: r.priority)
                request = self.request_queue.pop(0)

                # Generate character
                response = await self._generate_character(request)

                # Store in database
                self.character_database[response.character_id] = response

                # Send response
                await self.send_message(
                    Message(
                        type=MessageType.RESPONSE,
                        recipient=request.metadata.get("sender"),
                        correlation_id=request.request_id,
                        payload={"character_response": response.__dict__},
                    )
                )
            else:
                await asyncio.sleep(0.1)

    async def _generate_character(self, request: CharacterRequest) -> CharacterResponse:
        """Generate a character based on request"""
        start_time = asyncio.get_event_loop().time()

        # Generate or use provided attributes
        if request.attributes is None:
            request.attributes = self._generate_random_attributes(
                request.character_type
            )

        # Generate equipment if not provided
        if not request.equipment:
            level = (
                request.environment_context.get("player_level", 1)
                if request.environment_context
                else 1
            )
            request.equipment = self.equipment_manager.generate_equipment_set(
                request.attributes.class_type, level
            )

        # Adapt to environment if context provided
        if request.environment_context:
            request.attributes, request.equipment = (
                self.consistency_checker.adapt_character_to_environment(
                    request.attributes, request.equipment, request.environment_context
                )
            )

        # Generate pose control
        pose_control = self.pose_controller.generate_pose_control_image(
            request.pose, request.resolution
        )

        # Build prompt
        prompt = self._build_character_prompt(request)

        # Generate character images
        images = []
        for i in range(request.variations):
            # Generate with pose control if available
            if self.controlnet_model:
                image = await self._generate_with_pose_control(
                    prompt, pose_control, request.resolution
                )
            else:
                image = await self._generate_standard(prompt, request.resolution)

            images.append(image)

        # Save images
        character_id = str(uuid.uuid4())
        image_paths = []

        for i, image in enumerate(images):
            filename = f"{request.character_type.value}_{character_id}_{i}.png"
            filepath = self.output_dir / filename
            image.save(filepath)
            image_paths.append(filepath)

        # Generate equipment masks
        equipment_masks = {}
        if request.equipment and images:
            for eq in request.equipment:
                mask = self.equipment_manager.create_equipment_mask(eq, images[0])
                equipment_masks[eq.slot] = mask

        # Check environment consistency
        consistency_score = 1.0
        if (
            request.environment_context
            and "environment_features" in request.environment_context
        ):
            # Convert image to tensor
            img_tensor = transforms.ToTensor()(images[0])
            env_features = request.environment_context["environment_features"]
            consistency_score = self.consistency_checker.check_consistency_score(
                img_tensor, env_features
            )

        generation_time = asyncio.get_event_loop().time() - start_time

        # Update statistics
        self.generation_stats[request.character_type.value] += 1
        self.generation_stats["total"] += 1

        return CharacterResponse(
            request_id=request.request_id,
            character_id=character_id,
            character_type=request.character_type,
            image_paths=image_paths,
            pose_skeleton=self.pose_controller.get_pose_skeleton(
                request.pose, request.resolution
            ),
            equipment_masks=equipment_masks,
            generation_time=generation_time,
            consistency_score=consistency_score,
            metadata={
                "attributes": request.attributes.__dict__,
                "equipment": [eq.__dict__ for eq in request.equipment],
                "pose": request.pose.value,
                "prompt": prompt,
            },
        )

    def _generate_random_attributes(
        self, character_type: CharacterType
    ) -> CharacterAttributes:
        """Generate random character attributes"""
        races = ["human", "elf", "dwarf", "orc", "halfling"]
        classes = ["warrior", "mage", "rogue", "cleric", "ranger"]
        genders = ["male", "female"]
        ages = ["young", "adult", "elderly"]
        builds = ["slim", "average", "muscular", "heavy"]
        skin_tones = ["pale", "fair", "tan", "brown", "dark"]
        hair_styles = ["short", "long", "braided", "bald", "ponytail"]
        hair_colors = ["black", "brown", "blonde", "red", "gray", "white"]

        # Character type influences some choices
        if character_type == CharacterType.MERCHANT:
            classes = ["merchant", "trader"]
            ages = ["adult", "elderly"]  # Usually not young
        elif character_type == CharacterType.BOSS:
            builds = ["muscular", "heavy"]  # Usually imposing

        attributes = CharacterAttributes(
            character_type=character_type,
            race=np.random.choice(races),
            class_type=np.random.choice(classes),
            gender=np.random.choice(genders),
            age_category=np.random.choice(ages),
            build=np.random.choice(builds),
            skin_tone=np.random.choice(skin_tones),
            hair_style=np.random.choice(hair_styles),
            hair_color=np.random.choice(hair_colors),
            facial_features={
                "eyes": np.random.choice(["blue", "green", "brown", "gray", "amber"]),
                "nose": np.random.choice(["straight", "curved", "broad", "narrow"]),
                "scars": (
                    np.random.choice(["none", "facial scar", "battle scars"])
                    if character_type in [CharacterType.HERO, CharacterType.ENEMY]
                    else "none"
                ),
            },
            personality_traits=np.random.choice(
                ["brave", "cunning", "wise", "aggressive", "friendly", "mysterious"],
                size=2,
                replace=False,
            ).tolist(),
        )

        return attributes

    def _build_character_prompt(self, request: CharacterRequest) -> str:
        """Build detailed prompt for character generation"""
        prompt_parts = []

        # Character description
        prompt_parts.append(request.attributes.to_prompt_description())

        # Pose description
        pose_descriptions = {
            PoseType.IDLE: "standing in a relaxed pose",
            PoseType.WALKING: "walking forward",
            PoseType.RUNNING: "running dynamically",
            PoseType.ATTACKING: "in an aggressive attack pose",
            PoseType.DEFENDING: "in a defensive stance with shield raised",
            PoseType.CASTING: "casting a spell with magical energy",
            PoseType.INTERACTING: "reaching out to interact",
            PoseType.SITTING: "sitting down",
            PoseType.DEAD: "lying defeated on the ground",
        }
        prompt_parts.append(pose_descriptions.get(request.pose, "standing"))

        # Equipment description
        for eq in request.equipment:
            prompt_parts.append(eq.to_prompt_description())

        # Style and quality
        style_desc = "fantasy RPG character art"
        if request.style_reference:
            style_desc = f"{request.style_reference} style"

        prompt_parts.extend(
            [
                style_desc,
                "detailed",
                "high quality",
                "game asset",
                "full body portrait",
                "centered composition",
            ]
        )

        # Environment context
        if request.environment_context:
            biome = request.environment_context.get("biome", "")
            if biome:
                prompt_parts.append(f"in a {biome} environment")

        return ", ".join(prompt_parts)

    async def _generate_with_pose_control(
        self, prompt: str, pose_control: Image.Image, resolution: Tuple[int, int]
    ) -> Image.Image:
        """Generate character with pose control"""
        # This would use ControlNet pipeline
        # For now, fallback to standard generation
        return await self._generate_standard(prompt, resolution)

    async def _generate_standard(
        self, prompt: str, resolution: Tuple[int, int]
    ) -> Image.Image:
        """Standard character generation"""
        negative_prompt = "low quality, blurry, deformed, ugly, bad anatomy, bad proportions, extra limbs, missing limbs"

        image = await asyncio.to_thread(
            self.diffusion_model.generate,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=resolution[0],
            height=resolution[1],
            num_inference_steps=30,
            guidance_scale=7.5,
        )

        return image

    async def _handle_character_request(self, message: Message):
        """Handle character generation request"""
        payload = message.payload

        # Create request from payload
        request = CharacterRequest(
            character_type=CharacterType(payload.get("character_type", "npc")),
            pose=PoseType(payload.get("pose", "idle")),
            environment_context=payload.get("environment_context"),
            style_reference=payload.get("style"),
            resolution=tuple(payload.get("resolution", [512, 768])),
            variations=payload.get("variations", 1),
            priority=payload.get("priority", 5),
        )

        # Add metadata
        request.metadata = {"sender": message.sender}

        # Add to queue
        self.request_queue.append(request)

        # Send acknowledgment
        await self.send_message(
            Message(
                type=MessageType.RESPONSE,
                recipient=message.sender,
                correlation_id=message.id,
                payload={"status": "queued", "position": len(self.request_queue)},
            )
        )

    async def _handle_environment_update(self, message: Message):
        """Handle environment context update"""
        # Update any pending requests with new environment context
        env_context = message.payload.get("environment_context", {})

        for request in self.request_queue:
            if request.environment_context is None:
                request.environment_context = env_context

    async def _pause(self):
        """Pause character generation"""
        self.logger.info("Pausing character generation")

    async def _resume(self):
        """Resume character generation"""
        self.logger.info("Resuming character generation")

    async def _shutdown(self):
        """Shutdown character generation"""
        # Save any pending work
        if self.request_queue:
            self.logger.info(f"Saving {len(self.request_queue)} pending requests")

    async def _get_custom_status(self) -> Dict[str, Any]:
        """Get character agent status"""
        return {
            "pending_requests": len(self.request_queue),
            "characters_generated": self.generation_stats["total"],
            "generation_stats": dict(self.generation_stats),
            "database_size": len(self.character_database),
        }

    def _get_custom_state(self) -> Dict[str, Any]:
        """Get character agent state for saving"""
        return {
            "generation_stats": dict(self.generation_stats),
            "request_queue": [r.__dict__ for r in self.request_queue],
        }

    def _load_custom_state(self, state: Dict[str, Any]):
        """Load character agent state"""
        self.generation_stats = defaultdict(int, state.get("generation_stats", {}))


# Import required libraries
from torchvision import transforms


async def test_character_agent():
    """Test the character generation agent"""
    import logging

    logging.basicConfig(level=logging.INFO)

    # Create agent
    agent = CharacterGenerationAgent(
        agent_id="char_gen_001",
        model_dir=Path("data/models"),
        output_dir=Path("data/generated_characters"),
    )

    # Initialize
    await agent.initialize()
    await agent.start()

    # Create test request
    test_message = Message(
        type=MessageType.REQUEST,
        sender="test_client",
        payload={
            "character_type": "hero",
            "pose": "attacking",
            "environment_context": {"biome": "forest", "player_level": 5},
            "variations": 2,
        },
    )

    # Send request
    await agent._handle_character_request(test_message)

    # Wait for generation
    await asyncio.sleep(10)

    # Get status
    status = await agent.get_status()
    print(f"Agent status: {json.dumps(status, indent=2)}")

    # Shutdown
    await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(test_character_agent())
