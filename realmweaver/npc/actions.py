"""Action schema: the JSON tool calls an NPC may emit (spec section 8), validated with pydantic.

The LLM proposes one action per turn; the runtime's grounding verifier and, later, the game rules decide whether it
takes effect. `parse_action` is lenient about the text around the JSON (code fences, a stray sentence, a leftover
`<think>` block) and strict about the JSON itself: unknown actions and extra fields are errors.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

TEXT_MAX = 400
ACTION_NAMES: tuple[str, ...] = ("say", "give", "offer_quest", "refuse", "report_crime", "end")

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class ActionError(ValueError):
    """The LLM output holds no valid action; the message says why (fed back to the model on retry)."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Say(_Base):
    act: Literal["say"]
    text: str = Field(min_length=1, max_length=TEXT_MAX)


class Give(_Base):
    act: Literal["give"]
    item: str = Field(min_length=1, max_length=80)
    quantity: int = Field(default=1, ge=1, le=99)
    text: str = Field(default="", max_length=TEXT_MAX)


class OfferQuest(_Base):
    act: Literal["offer_quest"]
    quest: str = Field(min_length=1, max_length=80)
    text: str = Field(default="", max_length=TEXT_MAX)


class Refuse(_Base):
    act: Literal["refuse"]
    text: str = Field(min_length=1, max_length=TEXT_MAX)
    reason: str = Field(default="", max_length=120)


class ReportCrime(_Base):
    act: Literal["report_crime"]
    crime: str = Field(min_length=1, max_length=120)
    suspect: str = Field(default="the traveller", max_length=80)
    text: str = Field(default="", max_length=TEXT_MAX)


class End(_Base):
    act: Literal["end"]
    text: str = Field(default="", max_length=TEXT_MAX)


Action = Annotated[Say | Give | OfferQuest | Refuse | ReportCrime | End, Field(discriminator="act")]
_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def action_schema() -> dict:
    """JSON schema of the action union: the action schema the engine side validates against."""
    return _ADAPTER.json_schema()


def _first_json_object(raw: str) -> dict:
    text = _THINK.sub("", raw)
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    raise ActionError("no JSON object found in the reply")


def parse_action(raw: str) -> Action:
    """The first JSON object in `raw` as a validated action; ActionError names what is wrong."""
    obj = _first_json_object(raw)
    try:
        return _ADAPTER.validate_python(obj)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'action'}: {err['msg']}" for err in e.errors()
        )
        raise ActionError(f"invalid action ({problems})") from None
