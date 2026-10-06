"""NPC agent runtime (E7): personas, memory stream, JSON actions, grounding verifier and local LLM adapters."""

from realmweaver.npc.actions import (
    Action,
    ActionError,
    End,
    Give,
    OfferQuest,
    Refuse,
    ReportCrime,
    Say,
    action_schema,
    parse_action,
)
from realmweaver.npc.llm_local import FakeLlm, LlmCall, LocalLlm, load_local_llm
from realmweaver.npc.memory import Memory, MemoryStream
from realmweaver.npc.persona import Persona, ScheduleBlock, sample_facts, sample_personas, schedule_for
from realmweaver.npc.runtime import (
    Knowledge,
    NpcRuntime,
    TurnRecord,
    build_prompt,
    facts_from_graph,
    unknown_names,
    verify_grounding,
)

__all__ = [
    "Action",
    "ActionError",
    "End",
    "FakeLlm",
    "Give",
    "Knowledge",
    "LlmCall",
    "LocalLlm",
    "Memory",
    "MemoryStream",
    "NpcRuntime",
    "OfferQuest",
    "Persona",
    "Refuse",
    "ReportCrime",
    "Say",
    "ScheduleBlock",
    "TurnRecord",
    "action_schema",
    "build_prompt",
    "facts_from_graph",
    "load_local_llm",
    "parse_action",
    "sample_facts",
    "sample_personas",
    "schedule_for",
    "unknown_names",
    "verify_grounding",
]
