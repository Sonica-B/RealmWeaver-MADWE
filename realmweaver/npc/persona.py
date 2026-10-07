"""Persona: an NPC's identity, goals, voice and memory seed (spec section 8), with the 24 h schedule the prompt reads
the NPC's place and activity from. Personas are data: authored records, test fixtures and the spike harness
(`tools/npc-spike/personas.py`) build them; this module holds none."""

from __future__ import annotations

from dataclasses import dataclass

HOURS = 24


@dataclass(frozen=True)
class ScheduleBlock:
    """Where the NPC is and what it does from `start` (inclusive) to `end` (exclusive), in whole hours of the day."""

    start: int
    end: int
    place: str
    activity: str


@dataclass(frozen=True)
class Persona:
    """`schedule` must cover every hour of the day (`schedule_gaps` says where it does not). `voice_id` names the
    TTS voice for the milestone that adds voice and stays empty until then; `fallback_line` is the authored reply
    delivered when the model's proposals are all rejected."""

    name: str
    role: str
    faction: str
    home: str
    work: str
    schedule: tuple[ScheduleBlock, ...]
    goals: tuple[str, ...]
    memory_seed: tuple[str, ...]
    voice_id: str = ""
    fallback_line: str = "I have nothing to say about that, traveller."

    def block_at(self, hour: float) -> ScheduleBlock:
        """The schedule block covering `hour` (wraps past 24); raises ValueError when the schedule has a gap."""
        h = int(hour) % HOURS
        for block in self.schedule:
            if block.start <= h < block.end:
                return block
        raise ValueError(f"{self.name}: no schedule block covers hour {h}")

    def schedule_gaps(self) -> list[int]:
        """Hours of the day no block covers; an authored schedule returns []."""
        covered = {h for block in self.schedule for h in range(block.start, block.end)}
        return [h for h in range(HOURS) if h not in covered]
