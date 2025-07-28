# src/agents/character_agent_lc.py

import json
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline as hf_pipeline,
)
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)
from langchain import LLMChain
from langchain.agents import Tool, initialize_agent
from langchain.llms import HuggingFacePipeline

from src.agents.tools import save_memory, query_memory, reload_unity

# ————————————————————————————————————————————————————————
# 1) Reuse your core types & controllers from character_agent.py
# ————————————————————————————————————————————————————————

from enum import Enum
from dataclasses import dataclass, field

# Types of characters that can be generated:
class CharacterType(Enum):
    HERO       = "hero"
    NPC        = "npc"
    ENEMY      = "enemy"
    BOSS       = "boss"
    MERCHANT   = "merchant"
    QUEST_GIVER= "quest_giver"
    COMPANION  = "companion"  # :contentReference[oaicite:10]{index=10}

# Standard pose types for characters:
class PoseType(Enum):
    IDLE       = "idle"
    WALKING    = "walking"
    RUNNING    = "running"
    ATTACKING  = "attacking"
    DEFENDING  = "defending"
    CASTING    = "casting"
    INTERACTING= "interacting"
    SITTING    = "sitting"
    DEAD       = "dead"      # :contentReference[oaicite:11]{index=11}

@dataclass
class CharacterAttributes:
    character_type: CharacterType
    race: str
    class_type: str
    gender: str
    age_category: str
    build: str
    skin_tone: str
    hair_style: str
    hair_color: str
    facial_features: Dict[str,str]
    personality_traits: List[str]

    def to_prompt(self) -> str:
        desc = f"{self.age_category} {self.gender} {self.race} {self.class_type}, {self.build} build, {self.skin_tone} skin, {self.hair_style} hair in {self.hair_color}"
        for feat, val in self.facial_features.items():
            desc += f", {val} {feat}"
        return desc        # :contentReference[oaicite:12]{index=12}

@dataclass
class Equipment:
    slot: str
    name: str
    material: str
    color_scheme: List[str]
    enchantment: str = ""
    wear_level: float = 0.0

    def to_prompt(self) -> str:
        txt = f"{self.name} made of {self.material}"
        if self.color_scheme:
            txt += " in " + " and ".join(self.color_scheme)
        if self.enchantment:
            txt += f" with {self.enchantment}"
        if self.wear_level>0.7: txt += ", heavily worn"
        elif self.wear_level>0.3: txt += ", slightly worn"
        return txt        # :contentReference[oaicite:13]{index=13}

class PoseController:
    """Use ControlNet to adapt a stable-diffusion model to a given PoseType"""
    def __init__(self):
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
        ).to("cuda" if torch.cuda.is_available() else "cpu")  # :contentReference[oaicite:14]{index=14}

    def generate(self, attributes: CharacterAttributes, pose: PoseType, variations: int=1) -> List[Path]:
        prompt = attributes.to_prompt() + f", {pose.value} pose, pixel art sprite"
        outs = []
        for i in range(variations):
            img = self.pipe(prompt, num_inference_steps=20).images[0]
            out_path = Path("data/generated_characters")/f"{attributes.character_type.value}_{pose.value}_{i}.png"
            out_path.parent.mkdir(exist_ok=True, parents=True)
            img.save(out_path)
            outs.append(out_path)
        return outs

# ————————————————————————————————————————————————————————
# 2) Define LangChain Tools for each capability
# ————————————————————————————————————————————————————————

# (a) Generate random attributes via a small HF model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model     = AutoModelForCausalLM.from_pretrained("google/flan-t5-base")
txt_pipe  = hf_pipeline(
    "text2text-generation", model=model, tokenizer=tokenizer,
    max_new_tokens=100, do_sample=True, temperature=0.7
)
llm_txt   = HuggingFacePipeline(pipeline=txt_pipe)

# 2) Tool definitions
def generate_attributes(params: Dict[str,Any]) -> str:
    prompt = (
        f"Generate JSON attributes for a {params.get('character_type','npc')}."
    )
    return llm_txt(prompt)

def generate_equipment(params: Dict[str,Any]) -> str:
    prompt = (
        f"Generate JSON equipment list for a level "
        f"{params.get('level',1)} {params.get('class_type','warrior')}."
    )
    return llm_txt(prompt)

def adapt_to_environment(params: Dict[str,Any]) -> str:
    prompt = (
        f"Adapt this character JSON {json.dumps(params['attributes'])} "
        f"and equipment {json.dumps(params['equipment'])} "
        f"to environment {params.get('environment_context','')}"
    )
    return llm_txt(prompt)

def generate_pose_images(params: Dict[str,Any]) -> List[str]:
    attrs = CharacterAttributes(**params["attributes"])
    pose  = PoseType(params.get("pose","idle"))
    outs  = PoseController().generate(attrs, pose, params.get("variations",1))
    return [str(p) for p in outs]

tools = [
    Tool("generate_attributes", generate_attributes, "Generate character attributes JSON."),
    Tool("generate_equipment",   generate_equipment,   "Generate equipment JSON."),
    Tool("adapt_to_environment", adapt_to_environment, "Adapt to specific biome."),
    Tool("generate_pose_images", generate_pose_images, "Render character sprite images."),
    Tool("save_memory",          lambda a: save_memory(a["key"], a["doc"]),    "Save to memory."),
    Tool("query_memory",         lambda a: query_memory(a["query"], a.get("top_k",3)), "Recall from memory."),
    Tool("reload_unity",         lambda a: reload_unity(a["biome"]),           "Notify Unity to reload."),
]

# 3) Build the agent without callback_manager
agent = initialize_agent(
    tools=tools,
    llm=llm_txt,
    agent="zero-shot-react-description",
    verbose=False,      # set True only if you want basic console output
    max_iterations=6,
)

# 4) Standalone entry
if __name__ == "__main__":
    prompt = """
    Create a level-5 hero for a forest biome:
    1) Generate its attributes.
    2) Generate its equipment.
    3) Adapt both to environment context: {"biome":"forest","player_level":5}.
    4) Render two pixel-art idle poses.
    5) Save memory under key "hero_forest".
    6) Notify Unity to reload the forest biome.
    """
    print(agent.run(prompt))