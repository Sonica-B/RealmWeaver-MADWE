"""NPC agent runtime (E7): personas, memory stream, action schema, grounding verifier and local LLM adapters.

`NpcRuntime(llm, facts)` runs one dialogue turn; `facts_for` reads a region's facts from the world state graph;
`FakeLlm` and `LocalLlm` are the two `Callable[[str], str]` adapters it accepts."""

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
from realmweaver.npc.grounding import Fact, Knowledge, facts_for, verify_grounding
from realmweaver.npc.llm_local import FakeLlm, LocalLlm, load_local_llm
from realmweaver.npc.memory import Memory, MemoryStream
from realmweaver.npc.persona import Persona, ScheduleBlock
from realmweaver.npc.runtime import NpcRuntime, TurnRecord

__all__ = [
    "Action",
    "ActionError",
    "End",
    "Fact",
    "FakeLlm",
    "Give",
    "Knowledge",
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
    "facts_for",
    "load_local_llm",
    "parse_action",
    "verify_grounding",
]
