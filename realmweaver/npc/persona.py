"""Persona: an NPC's identity, goals, voice and memory seed (spec section 8), the per-role schedule templates the
runtime authors 24 h schedules from (E7, M1), and the five Emberfall personas plus the authored facts the Phase 0
spike measures against. The facts stand in for the NPC / Quest / Item graph nodes that arrive with the M0 data model.
"""

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
    name: str
    role: str
    faction: str
    home: str
    work: str
    schedule: tuple[ScheduleBlock, ...]
    goals: tuple[str, ...]
    voice_id: str
    memory_seed: tuple[str, ...]
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


# (start, end, place, activity) per role; "{home}" and "{work}" are filled per persona.
SCHEDULE_TEMPLATES: dict[str, tuple[tuple[int, int, str, str], ...]] = {
    "innkeeper": (
        (0, 6, "{home}", "sleeping"),
        (6, 10, "{work}", "serving breakfast and airing the rooms"),
        (10, 14, "the market", "buying provisions"),
        (14, 23, "{work}", "serving ale and renting rooms"),
        (23, 24, "{home}", "counting the day's takings"),
    ),
    "guard": (
        (0, 5, "{home}", "sleeping"),
        (5, 6, "{home}", "arming up"),
        (6, 14, "{work}", "standing watch"),
        (14, 16, "the inn", "eating"),
        (16, 22, "{work}", "patrolling"),
        (22, 24, "{home}", "sleeping"),
    ),
    "merchant": (
        (0, 7, "{home}", "sleeping"),
        (7, 8, "{home}", "preparing stock"),
        (8, 18, "{work}", "trading"),
        (18, 20, "the inn", "eating supper"),
        (20, 24, "{home}", "resting"),
    ),
    "farmer": (
        (0, 5, "{home}", "sleeping"),
        (5, 12, "{work}", "working the fields"),
        (12, 13, "{home}", "eating the midday meal"),
        (13, 19, "{work}", "working the fields"),
        (19, 24, "{home}", "resting"),
    ),
    "priest": (
        (0, 5, "{home}", "sleeping"),
        (5, 7, "{work}", "holding the dawn rite"),
        (7, 12, "{work}", "hearing petitions"),
        (12, 14, "the market", "tending the sick"),
        (14, 20, "{work}", "preparing the dusk rite"),
        (20, 24, "{home}", "reading and sleeping"),
    ),
}


def schedule_for(role: str, home: str, work: str) -> tuple[ScheduleBlock, ...]:
    """The role's template with home and work filled in; KeyError for a role without a template."""
    rows = SCHEDULE_TEMPLATES[role]
    return tuple(ScheduleBlock(s, e, place.format(home=home, work=work), act) for s, e, place, act in rows)


# -- Emberfall spike fixtures -------------------------------------------------------------------------------

# Facts are "<kind>: <Name> - <detail>"; the verifier reads the kind and the name before " - ".
SHARED_FACTS: tuple[str, ...] = (
    "place: Emberfall - a village on the Ashfield road at the edge of the Greywood",
    "place: Market Square - the centre of Emberfall; the Gilded Flagon stands on its north side",
    "place: Greywood - the forest east of the village; wolves have been heard there this autumn",
    "person: Reeve Harlan Dusk - the village reeve, who collects the road toll",
    "faction: Ember Watch - Emberfall's guards, captained by Tomas Rook",
    "faction: Hearthguild - the guild of Emberfall's innkeepers and merchants",
    "item: copper coin - Emberfall's everyday currency; silver is rare here",
)

_PERSONA_FACTS: dict[str, tuple[str, ...]] = {
    "Mara Vell": (
        "place: Gilded Flagon - Mara Vell's inn on Market Square; a room costs 5 copper a night",
        "item: Barley Ale - 2 copper a mug at the Gilded Flagon",
        "item: Rusty Lantern - lost property kept behind the Flagon's bar; free to anyone going down to the cellar",
        "quest: Rats in the Cellar - Mara pays 10 copper to whoever clears the rats out of the Flagon's cellar",
        "person: Pell - the stable boy at the Gilded Flagon",
        "person: Tomas Rook - captain of the Ember Watch; drinks at the Flagon most evenings",
    ),
    "Tomas Rook": (
        "place: North Gate - the toll gate on the Ashfield road, manned by the Ember Watch",
        "place: Watch Barracks - the Ember Watch's quarters behind the North Gate",
        "item: Watch Token - the brass badge every Ember Watch guard carries; Tomas lost his last night",
        "quest: Missing Watch Token - Tomas pays 15 copper for the return of his Watch Token",
        "person: Mara Vell - innkeeper of the Gilded Flagon",
        "lore: curfew - the North Gate closes at dusk; nobody passes after that without the reeve's seal",
    ),
    "Edda Thorn": (
        "place: Mill Cottage - Edda Thorn's cottage by the old mill, west of Market Square",
        "place: Herb Stall - Edda's stall on Market Square, open from the eighth hour to the eighteenth",
        "item: Greywood Honey - 4 copper a jar; good for coughs",
        "item: Fever Moss - a Greywood moss that breaks fevers; Edda has none left in stock",
        "quest: Fever Moss for the Chapel - Edda pays 8 copper a bundle for Fever Moss gathered in the Greywood",
        "person: Sister Ilse - priest of the Cinder Chapel; buys Edda's herbs for the sick",
    ),
    "Bram Ashfield": (
        "place: Ashfield Farm - Bram's farmstead south of Emberfall: barley and two cows",
        "place: Barley Fields - Bram's fields along the Ashfield road",
        "item: Iron Ploughshare - Bram's ploughshare, cracked and waiting for a smith",
        "item: barley sack - 3 copper a sack from Bram",
        "quest: Wolves at the Barley - Bram pays 20 copper to whoever drives the wolves off the Barley Fields",
        "person: Pell - the stable boy at the Gilded Flagon, Bram's nephew",
    ),
    "Sister Ilse": (
        "place: Cinder Chapel - the chapel of the Circle of the Cinder on the hill above Market Square",
        "faction: Circle of the Cinder - the order that keeps the Cinder Chapel's flame",
        "item: Cinder Candle - a chapel candle; every sick villager is given one to keep by the bed",
        "quest: Light the Cinder Candles - Ilse asks for help lighting the twelve candles before the dusk rite",
        "person: Edda Thorn - the herbalist who supplies the chapel's remedies",
        "lore: the Flame - the Circle teaches that the chapel's flame has burned since Emberfall was founded",
    ),
}


def sample_personas() -> list[Persona]:
    """The five Emberfall personas of the Phase 0 spike, one per schedule template."""
    return [
        Persona(
            name="Mara Vell",
            role="innkeeper",
            faction="Hearthguild",
            home="the loft above the Gilded Flagon",
            work="the Gilded Flagon",
            schedule=schedule_for("innkeeper", "the loft above the Gilded Flagon", "the Gilded Flagon"),
            goals=("keep every room let through the harvest fair", "get the rats out of the cellar"),
            voice_id="kokoro:af_heart",
            memory_seed=(
                "Rats gnawed through three grain sacks in the cellar this week",
                "Tomas Rook paid his tab last night and left early",
                "Pell has not mucked out the stable in two days",
                "The harvest fair is in nine days and every room is booked",
            ),
            fallback_line="Can't help you with that, traveller. Ale's two copper if you're staying.",
        ),
        Persona(
            name="Tomas Rook",
            role="guard",
            faction="Ember Watch",
            home="the Watch Barracks",
            work="the North Gate",
            schedule=schedule_for("guard", "the Watch Barracks", "the North Gate"),
            goals=("find the Watch Token before the reeve notices", "keep the gate shut after dusk"),
            voice_id="kokoro:am_michael",
            memory_seed=(
                "I lost my Watch Token somewhere between the Flagon and the barracks last night",
                "Reeve Harlan Dusk ordered the gate closed at dusk all this week",
                "A stranger in a grey hood asked about the toll road yesterday",
            ),
            fallback_line="That's Watch business, traveller. Move along.",
        ),
        Persona(
            name="Edda Thorn",
            role="merchant",
            faction="Hearthguild",
            home="Mill Cottage",
            work="the Herb Stall on Market Square",
            schedule=schedule_for("merchant", "Mill Cottage", "the Herb Stall on Market Square"),
            goals=("restock Fever Moss before the fever spreads", "sell the honey before it crystallises"),
            voice_id="kokoro:bf_emma",
            memory_seed=(
                "The last bundle of Fever Moss went to Sister Ilse on Tuesday",
                "Greywood Honey sold well at last year's fair",
                "The wolves kept me out of the Greywood all week",
            ),
            fallback_line="I only know herbs and honey, traveller. Ask me about those.",
        ),
        Persona(
            name="Bram Ashfield",
            role="farmer",
            faction="Ashfield Commons",
            home="Ashfield Farm",
            work="the Barley Fields",
            schedule=schedule_for("farmer", "Ashfield Farm", "the Barley Fields"),
            goals=("drive the wolves off the Barley Fields", "get the ploughshare mended before sowing"),
            voice_id="kokoro:am_adam",
            memory_seed=(
                "Wolves took a ewe from the Barley Fields two nights ago",
                "The ploughshare cracked on a stone in the south field",
                "Reeve Harlan Dusk raised the road toll again; market day is barely worth the trip",
            ),
            fallback_line="Don't know about that. Barley's three copper a sack if you want some.",
        ),
        Persona(
            name="Sister Ilse",
            role="priest",
            faction="Circle of the Cinder",
            home="the cell behind the Cinder Chapel",
            work="the Cinder Chapel",
            schedule=schedule_for("priest", "the cell behind the Cinder Chapel", "the Cinder Chapel"),
            goals=("keep the chapel flame lit", "break the fever among the sick"),
            voice_id="kokoro:bf_isabella",
            memory_seed=(
                "Three of the sick in the chapel have the fever and Edda Thorn is out of Fever Moss",
                "The dusk rite needs twelve candles lit and I have no acolyte",
                "Mara Vell sends bread for the sick every morning",
            ),
            fallback_line="The Flame keeps its own counsel on that, traveller.",
        ),
    ]


def sample_facts(persona: Persona) -> list[str]:
    """The authored facts an Emberfall persona knows: the shared village facts plus its own."""
    return [*SHARED_FACTS, *_PERSONA_FACTS.get(persona.name, ())]
