"""Memory stream (Generative Agents pattern): timestamped memories scored by importance, retrieved by
recency x importance x relevance, and folded into reflections every N observations.

Time is game hours. Each component is min-max normalised over the stream as in Park et al. 2023; relevance is
weighted `RELEVANCE_WEIGHT` because the word-overlap stand-in for embeddings is coarse.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger(__name__)

MemoryKind = Literal["observation", "reflection"]
MemorySource = Literal["world", "self", "player"]

RECENCY_DECAY = 0.9  # per game hour
RELEVANCE_WEIGHT = 2.0  # word overlap is coarse, so an on-topic memory must beat a merely recent one
DEFAULT_REFLECT_EVERY = 10

_WORD = re.compile(r"[a-z0-9']+")
_STOP_WORDS = (
    "a an the and or but if of to in on at by for with from about as is are was were be been am i you he she it we "
    "they me him her us them my your his its our their this that these those what which who whom where when why how "
    "do does did have has had not no yes so than then there here very just can could would should will shall may "
    "might must said says say asked tell told ask like want need go went come came get got"
)
_STOP = frozenset(_STOP_WORDS.split())

# ponytail: keyword importance; the upgrade path is an LLM rating (Generative Agents' 1-10 "poignancy" prompt).
_IMPORTANCE = {
    "killed": 1.0, "murder": 1.0, "dead": 0.9, "stole": 0.9, "theft": 0.9, "thief": 0.9, "attack": 0.9,
    "threat": 0.8, "fire": 0.8, "bounty": 0.8, "crime": 0.8, "wolves": 0.7, "quest": 0.7, "promise": 0.7,
    "secret": 0.7, "reward": 0.6, "gave": 0.6, "lost": 0.6, "paid": 0.5, "guard": 0.5, "reeve": 0.5,
    "fever": 0.6, "sick": 0.5, "rats": 0.5,
}  # fmt: skip
_BASE_IMPORTANCE = 0.3


def heuristic_importance(text: str) -> float:
    """0..1 from the strongest keyword in the text; `_BASE_IMPORTANCE` when none matches."""
    words = set(_WORD.findall(text.lower()))
    return max((w for k, w in _IMPORTANCE.items() if k in words), default=_BASE_IMPORTANCE)


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def extractive_summary(texts: list[str]) -> str:
    """Reflection stub: the memories joined into one sentence; an LLM summariser is the upgrade path."""
    return "Looking back: " + "; ".join(t.rstrip(".") for t in texts) + "."


@dataclass
class Memory:
    text: str
    t: float  # game hours
    importance: float  # 0..1
    kind: MemoryKind = "observation"
    source: MemorySource = "world"


class MemoryStream:
    """An NPC's memories in arrival order. `retrieve` ranks them for a query; `add` reflects every N observations."""

    def __init__(
        self,
        seed: Iterable[str] = (),
        *,
        now: float = 0.0,
        reflect_every: int = DEFAULT_REFLECT_EVERY,
        importance: Callable[[str], float] = heuristic_importance,
        summarise: Callable[[list[str]], str] = extractive_summary,
    ) -> None:
        self.memories: list[Memory] = []
        self.reflect_every, self._importance, self._summarise = reflect_every, importance, summarise
        self._since_reflection = 0
        for text in seed:
            self.add(text, t=now, source="world")

    def __len__(self) -> int:
        return len(self.memories)

    def add(
        self,
        text: str,
        *,
        t: float,
        importance: float | None = None,
        source: MemorySource = "world",
    ) -> Memory:
        memory = Memory(text, t, self._importance(text) if importance is None else importance, source=source)
        self.memories.append(memory)
        self._since_reflection += 1
        if self.reflect_every > 0 and self._since_reflection >= self.reflect_every:
            self.reflect(t)
        return memory

    def reflect(self, t: float) -> Memory:
        """Fold the NPC's own observations since the last reflection (what it saw and said, never what the player
        said: a reflection is a trusted `self` memory, so player speech must not launder into one) into one
        high-importance reflection memory."""
        since = self.memories[len(self.memories) - self._since_reflection :]
        recent = [m for m in since if m.kind == "observation" and m.source != "player"]
        texts = [m.text for m in sorted(recent, key=lambda m: m.importance, reverse=True)[:5]]
        importance = max((m.importance for m in recent), default=_BASE_IMPORTANCE)
        reflection = Memory(
            self._summarise(texts), t, min(1.0, importance + 0.1), kind="reflection", source="self"
        )
        self.memories.append(reflection)
        self._since_reflection = 0
        log.debug("reflection over %d memories: %s", len(recent), reflection.text)
        return reflection

    def retrieve(self, query: str, *, now: float, k: int = 5) -> list[Memory]:
        """Top-k memories by normalised recency + importance + relevance (query content-word overlap)."""
        if not self.memories:
            return []
        q = content_words(query)
        recency = [RECENCY_DECAY ** max(0.0, now - m.t) for m in self.memories]
        importance = [m.importance for m in self.memories]
        # ponytail: relevance is the share of query content words the memory contains; the upgrade path is an
        # injected embedder (the same seam the world graph uses for style vectors).
        relevance = [len(q & content_words(m.text)) / len(q) if q else 0.0 for m in self.memories]
        score = [
            r + i + RELEVANCE_WEIGHT * v
            for r, i, v in zip(_minmax(recency), _minmax(importance), _minmax(relevance), strict=True)
        ]
        order = sorted(range(len(self.memories)), key=lambda i: (-score[i], -self.memories[i].t))
        return [self.memories[i] for i in order[:k]]


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    return [0.0 for _ in values] if hi - lo < 1e-12 else [(v - lo) / (hi - lo) for v in values]
