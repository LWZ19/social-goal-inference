"""Dressing a layout: one layout of the placeholder generator made into a scene description for a renderer
and the ground truth that goes with it.

The scene stays the twin of the placeholder scene: the same people in the same places, the same robot and
camera, the same targets, bounds and episodes. The layout is not changed, and what is not said here is read
from it as it is. Dressing adds or changes four things, and the ground truth describes each as it is drawn:

- staff: one at each staff spot of a construction, a person with the role "staff" and no group, after the
  layout's own people, as s0, s1, ...: first whoever serves at a target, then the vendor of the second stall
  (the furnishing of the kind "shop", which no request names), so that the place a request asks for is not
  the only staffed place of its scene. The second stall stands where the twin has it, which is where the
  observation shows all of it or nothing of it: in the scenes of seed 0 it is never in the observation, and
  the vendor is seen only round the tour, in a frame or two of eight in two scenes of three. The vendor is
  cast from avatars tagged for a second stall, in everyday dress;
- the obstacles under a staffed target place and under a second stall: what the constructions of its items
  list, in place of the one rectangle its twin has there and at that rectangle's place in the list, in every
  variant, with staff or without. Every other obstacle is the twin's, polygon for polygon;
- one text of an episode: where a member of staff stands at the target of an episode with nobody at it, the
  rationale no longer says of the people that nobody stands close to the target, but that nobody does on the
  visitors' side and that a member of staff stands behind it (Dressed.episodes, _episodes). This is the one
  field in which the episodes of a scene with staff differ from its twin's: ids, requests, contexts, request
  types, plans and the rest of the rationale are the twin's, and so is every episode of a variant without
  staff;
- how things look: a construction for each item, an avatar in a pose for each person, small things on free
  tops, a floor and a view through the windows.

A Variant is a preset of three switches: the signage, whether there are staff, and whether there are small
things. All variants of a scene are dressed from the same draws. The random numbers come from five generators,
each seeded with `seed` and a number of its own, in the order of _STREAMS: one for the constructions, one for
the cast and their poses, one for the staff, one for the small things and one for the room's look. So a
variant that leaves the staff or the small things out shows the same constructions and the same people.

Each target has an object group of its own, 1, 2, ... in the order of the targets, which is carried by what
draws it: the constructions of a place's items, with the small things on them; the person; the construction
of the item that holds an object. Everything else is in group 0.

Casting: one avatar for each person and none twice in a scene. A visitor wears casual or business
dress. Only a person whom a target's label asks for in red wears red, and nobody wears pink, in any scene:
the scenes and their variants stay comparable. A person is as tall as the placeholder's figure of them, a
little broader or slighter than their avatar, and posed by what they are doing (_SUITS): the members of a
conversation about a person who is asked for in poses for talking, none of them with a phone and not all of
them only listening (_listens). No pose reaches further than 0.45 m from the body's axis; people next to each
other do not share a pose, nor two takes of one pose, while others are free, and neither do any two people of
a scene, which keeps the poses of a group apart however far its members stand from one another; and where
the manifest measured the poses, nobody's pose reaches into the person next to them: the people of a line
stand 0.6 m apart, and a hand held out would lie in the back of whoever stands before.

Besides what a Library holds, two things are read from the manifest it was loaded from: an avatar's
`max_width_scale` and a pose's `measured` front, back and side.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from sgi.geometry import wrap_angle
from sgi.polygons import Polygon, distance_to_polygon
from sgi.scene import EpisodeModel, Person, PolygonModel, SceneGT, Target
from sgi.synth.realistic.constructions import STAFFED_KINDS, Construction, Lettered, Surface, construct
from sgi.synth.realistic.library import AvatarAsset, Library, ModelAsset, PoseAsset, manifest_entries
from sgi.synth.realistic.spec import ModelRef, Opening, PersonRef, RoomSpec, SceneSpec
from sgi.synth.recipes import Item, Layout
from sgi.synth.render import Box, Color, Solid, Sphere

Block = tuple[float, float, float, float, float, float]    # x, y, yaw, half_x, half_y, top: see Dressed.target_blocks

# What each generator of a dressing is for: generator k is seeded with [*seed, k].
_STREAMS = ("constructions", "cast", "staff", "small things", "room")

# --- The cast. ---
_VISITOR_TAGS = frozenset({"casual", "business"})  # how a visitor is dressed; a uniform is staff dress
_RED = "red"                                       # the colour a target's label may ask for, and the tag that wears it
_KEPT_TAGS = frozenset({_RED, "pinkish"})          # what nobody wears but the person asked for in red
_BUILD = (0.92, 1.15)                              # how much wider than the avatar a visitor is made, drawn between
_WIDTH_SCALE = 1.07                                # the most an avatar is widened in all, height over natural height
#                                                    times build, where its manifest entry gives no max_width_scale
_REACH_M = 0.45                                    # no pose gets further than this from the body's axis, as placed
_NEXT_TO_M = 1.5                                   # two people this near stand next to each other
_STAFF_ROLE = "staff"
_STAFF_POSE = "listen_upright"                     # at the three staffed kinds it touches nothing; the other serving
#                                                    poses put toes or hands a few centimetres into the furniture
_STAFF_SHORTER_M = (0.02, 0.05)                    # how far under the avatar's natural height a staff member stands
_LISTENS = "listen_"                               # how the library names a pose in which someone only listens: arms
#                                                    down, no gesture. A conversation is not made of those alone
# What a person is doing, and the tags of the poses that suit it. In a line: waiting, perhaps with a phone. In
# the circle about a person who is asked for: talking, perhaps with a cup, and never with a phone, whatever
# else the pose is tagged for: whoever holds a phone to an ear or looks at one is not in the conversation, and
# the group reads as the others with a bystander. In another group: any of those. Alone: no pose that is for
# talking only, which is a gesture made to nobody. Asked for and alone: waiting.
_LINE, _CIRCLE, _PAIR, _ALONE, _ASKED = "line", "circle", "pair", "alone", "asked"
_PHONE, _CUP = "phone", "cup"                      # the two things a pose may hold, each the tag of its poses
_SUITS = {_LINE: ("waiting", _PHONE), _CIRCLE: ("talking", _CUP), _PAIR: ("waiting", "talking", _PHONE, _CUP),
          _ALONE: ("waiting", _PHONE, _CUP), _ASKED: ("waiting",)}
_NEVER = {_CIRCLE: (_PHONE,)}                      # the tags of the poses that do not suit it, whatever else they have

# --- The room. ---
_FLOORS = ("floor_tile", "floor_terrazzo", "floor_laminate")
_CEILING = "ceiling_tile"
_LIGHTING = "ceiling"

# --- Small things. ---
_SMALL_KINDS = ("cup", "book", "box")              # what reads as what it is at the size a scene shows it. Not a
#                                                    bottle: on a desk or a counter the library's reads as a flask of
#                                                    oil or of something with a hazard label
_SMALL_MOST = 2                                    # on one free top
_SMALL_OF_A_KIND = 2                               # on all the tops of one construction: a desk with four cups on it
#                                                    and nothing else is a shelf of cups
_SMALL_SCALE = (0.85, 1.15)                        # of its life size, as a construction uses a model
_SMALL_TURN_RAD = 0.4                              # how far one is turned from square to its top, either way
_SMALL_GAP_M = 0.02                                # between two on one top
_SMALL_CLEAR_M = 0.3                               # how high one may be: a Surface has nothing within this over it
_SMALL_TRIES = 8                                   # at turning one until it is on its top, and at finding it a place

_ON_M = 1e-6                                       # a point is on a polygon when it is no further than this outside

# --- The episodes. ---
_NOBODY = "no_one"                                 # the situation of an episode with nobody at its target (recipes)
# What its rationale says of the people where a member of staff stands at the target: nobody but them.
_NOBODY_BUT_STAFF = ("Nobody is waiting at the target or standing close to it on the visitors' side; a member of "
                     "staff stands behind it.")


@dataclass(frozen=True)
class Variant:
    """A preset of what a scene is dressed with."""
    name: str
    signs: str                                     # one of constructions.SIGN_MODES
    staff: bool
    small_things: bool


VARIANTS = {"real": Variant("real", "natural", True, True), "plain": Variant("plain", "labels", False, False),
            "staffed": Variant("staffed", "labels", True, False)}


@dataclass(frozen=True)
class Dressed:
    spec: SceneSpec
    scene_gt: SceneGT                              # the layout's, with staff added and the obstacles as drawn
    target_groups: Mapping[str, tuple[int, ...]]   # target id -> object groups that draw it
    target_blocks: Mapping[str, tuple[Block, ...]]
    #                                                target id -> upright blocks holding it: x, y, yaw, half_x, half_y,
    #                                                top. A place: one for each of its items, on the item's own ground
    #                                                rectangle (a plate stands 3 mm out of it), up to the top of what
    #                                                stands there; what a sign over it hangs from the ceiling by is
    #                                                no part of it. A person: round their reach. A target without a
    #                                                footprint: round its match radius, as high as the thing itself
    lettering: Mapping[str, tuple[Lettered, ...]]  # target id -> lettered rectangles on it
    episodes: tuple[EpisodeModel, ...]             # the layout's, each as it is but for what staff make untrue of
    #                                                the people in its rationale: see _episodes


def _middle(polygon: Polygon) -> tuple[float, float]:
    xs, ys = zip(*polygon)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _top(solid: Solid) -> float:
    if isinstance(solid, Sphere):
        return solid.z + solid.radius
    return solid.z0 + (solid.sz if isinstance(solid, Box) else solid.height)


def _furthest(solid: Solid, x: float, y: float) -> float:
    """How far from the ground point (x, y) the ground under a solid gets: the furthest corner of a box's
    rectangle, or the far side of a round solid's disc."""
    if isinstance(solid, Box):
        c, s = math.cos(solid.yaw), math.sin(solid.yaw)
        return max(math.hypot(solid.cx + c * i * solid.sx / 2 - s * j * solid.sy / 2 - x,
                              solid.cy + s * i * solid.sx / 2 + c * j * solid.sy / 2 - y)
                   for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1)))
    reach = solid.radius if isinstance(solid, Sphere) else max(solid.radius, solid.top_radius or 0.0)
    return math.hypot(solid.x - x, solid.y - y) + reach


def _solids(one: Construction, more: Sequence[ModelRef] = ()) -> list[Solid]:
    """All that a construction stands in the room, and what stands on it: its parts, and the box each model fills."""
    return [*(part.solid for part in one.parts),
            *(Box(model.x, model.y, model.yaw, *model.size, z0=model.z) for model in (*one.models, *more))]


def _owners(layout: Layout) -> list[str | None]:
    """For each item of the layout, the id of the target it draws, or None for one that draws none.

    Only what the scene is about, an item of `boxes`, draws a target: a place whose footprint the item's ground
    lies on, or else a thing whose anchor is over the item's ground. A person is drawn by no item."""
    owners: list[str | None] = []
    for item in layout.items:
        owner = None
        if item.source == "boxes":
            for target in layout.scene_gt.targets:
                if target.kind == "person":
                    continue
                if target.footprint is not None:
                    on_it = distance_to_polygon(_middle(item.footprint), target.footprint) <= _ON_M
                else:
                    on_it = distance_to_polygon(target.anchor[:2], item.footprint) <= _ON_M
                if on_it:
                    owner = target.id
                    break
        owners.append(owner)
    return owners


# --- The people. ---

def _doing(layout: Layout) -> dict[str, str]:
    """What each person of the layout is doing, by id, as a key of _SUITS.

    The group at a target, as an episode names it, stands in a line when the target is a place or a thing, and
    in a circle when the target is a person. Another group is a pair, and whoever is in no group is alone, or
    asked for and alone."""
    truth = layout.scene_gt
    kinds = {target.id: target.kind for target in truth.targets}
    at: dict[str, str] = {}                        # a group at a target: the kind of that target
    for episode in layout.episodes:
        named = episode.ground_truth.group_at_target
        for target_id in episode.ground_truth.target_ids:
            group = named.get(target_id) if isinstance(named, dict) else named
            if group is not None and target_id in kinds:
                at.setdefault(group, kinds[target_id])
    asked_for = {target.person_id for target in truth.targets if target.kind == "person"}
    doing = {}
    for person in truth.people:
        if person.group in at:
            doing[person.id] = _CIRCLE if at[person.group] == "person" else _LINE
        elif person.group is not None:
            doing[person.id] = _PAIR
        else:
            doing[person.id] = _ASKED if person.id in asked_for else _ALONE
    return doing


def _widest(entries: Sequence[dict]) -> dict[str, float]:
    """For each avatar whose manifest entry says it, the most it may be widened in all (`max_width_scale`):
    beyond that its body no longer shows on the axis that the scene check samples."""
    return {entry["id"]: float(entry["max_width_scale"]) for entry in entries
            if "id" in entry and "max_width_scale" in entry}


def _measured(entries: Sequence[dict]) -> dict[str, tuple[float, float, float]]:
    """For each pose whose manifest entry measured it, how far the body gets from its axis ahead, behind and to
    the side, at the avatar's own size (`measured`: front, back, side)."""
    found = {}
    for entry in entries:
        measures = entry.get("measured")
        sides = [measures.get(key) for key in ("front", "back", "side")] if isinstance(measures, dict) else [None]
        if "id" in entry and all(isinstance(side, (int, float)) and side > 0 for side in sides):
            found[entry["id"]] = (float(sides[0]), float(sides[1]), float(sides[2]))
    return found


def _towards(measures: tuple[float, float, float], off: float) -> float:
    """How far a body of these measures (ahead, behind, to the side) gets from its axis the way that lies
    `off` radians from its facing: as an ellipse through the measures does."""
    along = measures[0] if math.cos(off) >= 0 else measures[1]
    return 1.0 / math.hypot(math.cos(off) / along, math.sin(off) / measures[2])


def _held(pose: PoseAsset) -> str | None:
    """What a pose is made to hold: the renderer refuses any other pairing."""
    return _PHONE if _PHONE in pose.tags else _CUP if _CUP in pose.tags else None


def _take(pose_id: str) -> str:
    """The pose that a pose is a take of. The library names the takes of one pose alike but for a last letter,
    relaxed_a and relaxed_b; a pose that is not named so is a pose of its own."""
    return pose_id[:-2] if re.fullmatch(r".+_[a-z]", pose_id) else pose_id


def _listens(pose_id: str) -> bool:
    """Whether a pose is one in which someone only listens, by the name the library gives such a pose."""
    return pose_id.startswith(_LISTENS)


def _pose(library: Library, tags: Sequence[str], without: Sequence[str], avatar: AvatarAsset, scale: float, who: str,
          clear: Callable[[PoseAsset], bool], alike: Callable[[PoseAsset], tuple],
          draws: np.random.Generator, speaking: bool = False) -> PoseAsset:
    """A pose for one person: one with one of `tags` and none of `without` that suits the avatar and, on a
    body `scale` times as wide as the avatar's, keeps within reach of its axis.

    Of those, with `speaking`, one in which they do not only listen, where there is one. Of those, one that is
    `clear` of the people next to them, where there is one: the layout fixes how near people stand, so where
    every pose touches someone, one that does is taken. And of those, one that is least `alike` what the
    others have: of the poses for which `alike` gives the least, one is drawn."""
    suiting = [pose for pose in library.poses.values()
               if pose.tags & set(tags) and not pose.tags & set(without) and pose.sex in (None, avatar.sex)
               and pose.reach * scale <= _REACH_M]
    if not suiting:
        raise ValueError(f"no pose tagged {' or '.join(tags)} keeps within {_REACH_M} m of the body's axis on "
                         f"{who}, who is cast as {avatar.id} at {scale:.2f} of its width")
    if speaking:
        suiting = [pose for pose in suiting if not _listens(pose.id)] or suiting
    suiting = [pose for pose in suiting if clear(pose)] or suiting
    least = min(alike(pose) for pose in suiting)
    free = [pose for pose in suiting if alike(pose) == least]
    return free[int(draws.integers(len(free)))]


def _cast(layout: Layout, library: Library, groups: Mapping[str, int], draws: np.random.Generator) -> list[PersonRef]:
    """The layout's people, each as an avatar in a pose, in the layout's order."""
    people, targets = layout.scene_gt.people, layout.scene_gt.targets
    if len(layout.figures) != len(people):
        raise ValueError(f"the layout {layout.name} has {len(people)} people and {len(layout.figures)} figures: "
                         "a figure gives each person their height")
    asked_for = {target.person_id: target for target in targets if target.kind == "person"}
    in_red = {name for name, target in asked_for.items() if re.search(rf"\b{_RED}\b", target.label.lower())}
    visitors = [avatar for avatar in library.avatars.values() if avatar.tags & _VISITOR_TAGS]
    plain = [avatar for avatar in visitors if not avatar.tags & _KEPT_TAGS]
    red = [avatar for avatar in visitors if _RED in avatar.tags]
    others = sum(person.id not in in_red for person in people)
    if others > len(plain):
        raise ValueError(f"the scene has {others} visitors and the library has {len(plain)} avatars for them")
    # What the manifest says beyond what a Library holds. It is the one the library was loaded from, which lies
    # beside its prepared assets; a library made by hand has none, and then nothing is listed.
    entries = manifest_entries(library.prepared.parent.parent)
    widest, measured, doing = _widest(entries["avatars"]), _measured(entries["poses"]), _doing(layout)

    def width(one: PersonRef) -> float:            # of a person as placed, in widths of their avatar
        return one.height / library.avatars[one.avatar].natural_height * one.build

    cast: list[PersonRef] = []
    for person, figure in zip(people, layout.figures):
        worn = {one.avatar for one in cast}
        free = [avatar for avatar in (red if person.id in in_red else plain) if avatar.id not in worn]
        if not free:                               # the others were counted: only the red ones can run out
            raise ValueError(f"no avatar tagged {_RED} is left for the target {asked_for[person.id].id}, "
                             f"{asked_for[person.id].label!r}")
        avatar = free[int(draws.integers(len(free)))]
        height = float(figure.height)
        most = widest.get(avatar.id, _WIDTH_SCALE) * avatar.natural_height / height
        build = min(float(draws.uniform(*_BUILD)), _BUILD[1], most)
        scale = height / avatar.natural_height * build
        # Those who are cast already, nearest first, each with how far off they stand; and of those next to this
        # person whose pose is measured, the way to them off this person's facing and the room they leave that way.
        near = sorted(((math.hypot(one.x - person.x, one.y - person.y), one) for one in cast), key=lambda each: each[0])
        next_to = [(gap, one) for gap, one in near if gap <= _NEXT_TO_M]
        beside = []
        for gap, one in next_to:
            if one.pose in measured:
                way = math.atan2(one.y - person.y, one.x - person.x)
                theirs = _towards(measured[one.pose], way + math.pi - one.yaw) * width(one)
                beside.append((way - person.yaw, gap - theirs))

        def clear(pose: PoseAsset) -> bool:
            """Whether the pose keeps out of everyone next to this person: a hand held out is not in the back
            of whoever stands before them in a line. A pose that the manifest did not measure is let be."""
            return pose.id not in measured or all(_towards(measured[pose.id], off) * scale <= room
                                                  for off, room in beside)

        # What the others have, from what matters most: the pose of the nearest person and of those next to this
        # one, then a take of the pose of those, then the pose, or a take of the pose, of anyone in the scene. The
        # last two keep the poses of a group apart as well, whose members may stand further from one another than
        # people next to each other do.
        nearest, around = [one.pose for _, one in near[:1]], [one.pose for _, one in next_to]
        had = [(of, {of(name) for name in names}) for of, names in (
            (str, nearest), (str, around), (_take, nearest), (_take, around), (str, [one.pose for one in cast]),
            (_take, [one.pose for one in cast]))]

        def alike(pose: PoseAsset) -> tuple[bool, ...]:
            """How a pose is like what the others have: where two poses differ first, the one with False is
            less like it. So people next to each other do not share a pose while another is free."""
            return tuple(of(pose.id) in names for of, names in had)

        # A conversation is not made of listeners alone: where everyone else of a circle only listens, the last of
        # them to be posed does not.
        suits, never = _SUITS[doing[person.id]], _NEVER.get(doing[person.id], ())
        circle = [each.id for each in people if doing[person.id] == _CIRCLE and each.group == person.group]
        others = [one.pose for one in cast if one.person_id in circle]
        speaking = len(circle) > 1 and len(others) == len(circle) - 1 and all(_listens(name) for name in others)
        pose = _pose(library, suits, never, avatar, scale, person.id, clear, alike, draws, speaking)
        group = groups[asked_for[person.id].id] if person.id in asked_for else 0
        cast.append(PersonRef(person.id, avatar.id, pose.id, float(person.x), float(person.y), float(person.yaw),
                              height, build, _held(pose), group))
    return cast


def _staff(spots: Sequence[tuple[str, tuple[float, float, float]]], library: Library, worn: Sequence[str],
           draws: np.random.Generator) -> list[PersonRef]:
    """Someone on each staff spot, each given as the kind of the place and (x, y, yaw): an avatar tagged for
    that kind which nobody wears yet, upright, a little under its natural height and no wider than it is."""
    if not spots:
        return []
    pose = library.poses.get(_STAFF_POSE)
    if pose is None:
        raise ValueError(f"the library has no pose {_STAFF_POSE}, which the staff stand in")
    worn, staff = set(worn), []
    for number, (kind, (x, y, yaw)) in enumerate(spots):
        tag, place = f"staff_{kind}", kind.replace("_", " ")
        tagged = [avatar for avatar in library.avatars.values() if tag in avatar.tags and not avatar.tags & _KEPT_TAGS
                  and pose.sex in (None, avatar.sex)]
        free = [avatar for avatar in tagged if avatar.id not in worn]
        if not free:
            raise ValueError(f"no avatar tagged {tag} is left for the staff of the {place}: the visitors wear all "
                             f"{len(tagged)}" if tagged else
                             f"no avatar tagged {tag}: the library has nobody to stand behind the {place}")
        avatar = free[int(draws.integers(len(free)))]
        worn.add(avatar.id)
        height = avatar.natural_height - float(draws.uniform(*_STAFF_SHORTER_M))
        if pose.reach * height / avatar.natural_height > _REACH_M:
            raise ValueError(f"the pose {_STAFF_POSE} gets {pose.reach} m from the body's axis, further than the "
                             f"{_REACH_M} m a free top keeps from a staff spot")
        staff.append(PersonRef(f"s{number}", avatar.id, pose.id, float(x), float(y), float(yaw), height, 1.0, None, 0))
    return staff


# --- Small things. ---

def _stood(model: ModelAsset, surface: Surface, there: list[tuple[float, float, float, float]], group: int,
           draws: np.random.Generator) -> ModelRef | None:
    """The model stood on the free top, turned a little and wholly on it, clear of what is `there` already; or
    None when no place is found for it. `there` holds each thing's middle and half sides on the top, along the
    top's depth and across, and gains this one's."""
    most = min(_SMALL_SCALE[1], surface.depth / model.size[0], surface.length / model.size[1],
               _SMALL_CLEAR_M / model.size[2])
    scale = float(draws.uniform(_SMALL_SCALE[0], most))
    along, across = scale * model.size[0], scale * model.size[1]
    turn = float(draws.uniform(-_SMALL_TURN_RAD, _SMALL_TURN_RAD))
    for _ in range(_SMALL_TRIES):                  # turned, it takes more of the top each way: turned back by halves
        half_u = (along * abs(math.cos(turn)) + across * abs(math.sin(turn))) / 2
        half_v = (along * abs(math.sin(turn)) + across * abs(math.cos(turn))) / 2
        if half_u <= surface.depth / 2 and half_v <= surface.length / 2:
            break
        turn /= 2
    else:
        turn, half_u, half_v = 0.0, along / 2, across / 2
    for _ in range(_SMALL_TRIES):
        u = float(draws.uniform(half_u - surface.depth / 2, surface.depth / 2 - half_u))
        v = float(draws.uniform(half_v - surface.length / 2, surface.length / 2 - half_v))
        if all(abs(u - other_u) >= half_u + other_half_u + _SMALL_GAP_M
               or abs(v - other_v) >= half_v + other_half_v + _SMALL_GAP_M
               for other_u, other_v, other_half_u, other_half_v in there):
            there.append((u, v, half_u, half_v))
            c, s = math.cos(surface.yaw), math.sin(surface.yaw)
            return ModelRef(model.id, float(surface.x + c * u - s * v), float(surface.y + s * u + c * v),
                            float(surface.z), wrap_angle(surface.yaw + turn), (scale, scale, scale),
                            tuple(scale * side for side in model.size), group)
    return None


def _small_things(made: Sequence[Construction], groups: Sequence[int], library: Library,
                  draws: np.random.Generator) -> list[list[ModelRef]]:
    """For each construction, what is stood on its free tops: on each top up to _SMALL_MOST of the library's
    small things that go on it, and on all its tops together no more than _SMALL_OF_A_KIND of one kind, in the
    construction's object group. They stand on what is blocked already."""
    models = [model for kind in _SMALL_KINDS for model in library.models_of(kind)]
    things: list[list[ModelRef]] = []
    for one, group in zip(made, groups):
        stood: list[ModelRef] = []
        kinds: dict[str, int] = {}                 # how many of each kind stand on this construction so far
        for surface in one.surfaces:
            fitting = [model for model in models
                       if _SMALL_SCALE[0] * model.size[0] <= surface.depth
                       and _SMALL_SCALE[0] * model.size[1] <= surface.length
                       and _SMALL_SCALE[0] * model.size[2] <= _SMALL_CLEAR_M]
            if not fitting:
                continue
            there: list[tuple[float, float, float, float]] = []
            for _ in range(int(draws.integers(_SMALL_MOST + 1))):
                model = fitting[int(draws.integers(len(fitting)))]
                if kinds.get(model.kind, 0) >= _SMALL_OF_A_KIND:
                    continue                       # the place is left bare: nothing else is drawn for it
                thing = _stood(model, surface, there, group, draws)
                if thing is not None:
                    stood.append(thing)
                    kinds[model.kind] = kinds.get(model.kind, 0) + 1
        things.append(stood)
    return things


# --- The room and the ground truth. ---

def _colour(colour: Color) -> Color:
    return tuple(int(channel) for channel in colour)


def _room(layout: Layout, openings: Sequence[Opening], draws: np.random.Generator) -> RoomSpec:
    """The layout's room, with what the constructions cut into its walls and a floor of the look's choosing."""
    room = layout.room
    return RoomSpec(tuple((float(x), float(y)) for x, y in room.polygon), float(room.height),
                    _FLOORS[int(draws.integers(len(_FLOORS)))], _CEILING, _colour(room.wall),
                    None if room.dado is None else _colour(room.dado), float(room.dado_height), float(room.lights),
                    tuple(openings))


def _obstacles(layout: Layout, made: Sequence[Construction], owners: Sequence[str | None]) -> list[PolygonModel]:
    """The layout's obstacles, with the one rectangle under each staffed place replaced, where it stands in the
    list, by what the constructions of that place's items list: the floor behind its counter is free, and its
    staff stand there. A target place is staffed when its front, the first of its items, is of a staffed kind;
    its rectangle is its footprint. A furnishing of a staffed kind, the second stall, is one item, and its
    rectangle is that item's."""
    twin = layout.scene_gt.obstacles
    drawn: dict[int, list[Polygon]] = {}           # the place in the twin's list: what stands there instead
    staffed: list[tuple[str, str, Polygon, list[int]]] = []     # what it is called, its kind, its ground, its items
    for target in layout.scene_gt.targets:
        own = [index for index, owner in enumerate(owners) if owner == target.id]
        if target.footprint is not None and own and layout.items[own[0]].kind in STAFFED_KINDS:
            staffed.append((f"footprint of the target {target.id}", layout.items[own[0]].kind, target.footprint, own))
    for index, item in enumerate(layout.items):
        if item.source != "boxes" and item.kind in STAFFED_KINDS:
            staffed.append((f"rectangle of the {item.kind}", item.kind, item.footprint, [index]))
    for name, kind, rectangle, own in staffed:
        ground = np.asarray(rectangle, dtype=np.float64)
        under = [index for index, obstacle in enumerate(twin) if index not in drawn
                 and np.shape(obstacle.polygon) == ground.shape and np.allclose(obstacle.polygon, ground, atol=1e-9)]
        if not under:
            raise ValueError(f"the {name} is not among the obstacles of the layout "
                             f"{layout.name}: nothing says what its {kind} blocks there")
        drawn[under[0]] = [polygon for index in own for polygon in made[index].obstacles]
    obstacles: list[PolygonModel] = []
    for index, obstacle in enumerate(twin):
        if index in drawn:
            obstacles += [PolygonModel(polygon=[(float(x), float(y)) for x, y in polygon]) for polygon in drawn[index]]
        else:
            obstacles.append(obstacle)
    return obstacles


def _episodes(layout: Layout, staffed: Sequence[str]) -> tuple[EpisodeModel, ...]:
    """The layout's episodes, as they are but for one text. `staffed` are the ids of the targets that someone
    on duty stands at in this dressing.

    The placeholder says of a target with nobody at it that nobody is waiting there or standing close to it.
    With a member of staff behind the target that is no longer so, and an episode of that situation at such a
    target says instead what the picture shows (_NOBODY_BUT_STAFF). Its id, its request, its context, its
    request type, its plan and the rest of its rationale are the twin's."""
    told = []
    for episode in layout.episodes:
        truth = episode.ground_truth
        if truth.situation == _NOBODY and truth.rationale is not None and set(truth.target_ids) & set(staffed):
            rationale = truth.rationale.model_copy(update={"people": _NOBODY_BUT_STAFF})
            episode = episode.model_copy(update={"ground_truth": truth.model_copy(update={"rationale": rationale})})
        told.append(episode)
    return tuple(told)


def _item_block(item: Item, solids: Sequence[Solid], ceiling: float) -> Block:
    """The upright block on an item's own ground rectangle, up to the top of the highest of the solids that end
    under the room's ceiling. One that reaches the ceiling is what a sign over the place hangs by: a hanger, and
    not the place, which the placeholder builds no higher than a robot's view of it holds."""
    solids = [solid for solid in solids if _top(solid) < ceiling - _ON_M]
    c, s = math.cos(item.yaw), math.sin(item.yaw)
    own = [(c * (x - item.x) + s * (y - item.y), c * (y - item.y) - s * (x - item.x)) for x, y in item.footprint]
    xs, ys = zip(*own)
    x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return (float(item.x + c * x - s * y), float(item.y + s * x + c * y), float(item.yaw), float(max(xs) - min(xs)) / 2,
            float(max(ys) - min(ys)) / 2, float(max(map(_top, solids), default=0.0)))


def _round_block(target: Target, solids: Sequence[Solid]) -> Block:
    """The upright block round a target's match radius, up to the top of the highest of the solids that stand
    wholly within that radius of its anchor: the thing itself, and none of what stands beside it."""
    x, y, radius = float(target.anchor[0]), float(target.anchor[1]), float(target.match_radius)
    on_it = [solid for solid in solids if _furthest(solid, x, y) <= radius + _ON_M]
    return (x, y, 0.0, radius, radius, float(max(map(_top, on_it or solids), default=target.anchor[2])))


def _person_block(one: PersonRef, library: Library) -> Block:
    """The upright block round a person: as far from their axis as their pose reaches at the size they stand at."""
    reach = library.poses[one.pose].reach * one.height / library.avatars[one.avatar].natural_height * one.build
    return (one.x, one.y, 0.0, reach, reach, one.height)


def dress(layout: Layout, variant: Variant, library: Library, seed: Sequence[int]) -> Dressed:
    """The layout as a scene description and its ground truth, dressed as the variant says.

    `seed` is a sequence of whole numbers, none below 0; the same arguments give the same result. ValueError
    for a signage the constructions do not know, for a layout without a room, and for a library that cannot
    serve the scene: too few avatars for its visitors, none in red for a person asked for in red, none tagged
    for a kind of place that is staffed, or no pose for what someone is doing."""
    if layout.room is None:
        raise ValueError(f"the layout {layout.name} has no room: a scene is dressed in the room of its layout")
    numbers = [int(number) for number in seed]
    building, casting, staffing, placing, look = (np.random.default_rng([*numbers, k]) for k in range(len(_STREAMS)))
    truth = layout.scene_gt
    groups = {target.id: number for number, target in enumerate(truth.targets, start=1)}
    owners = _owners(layout)
    of_item = [groups[owner] if owner is not None else 0 for owner in owners]
    # One number of the generator for each item, whatever the signage: every variant has the same constructions.
    made = [construct(item, getattr(layout, item.source)[item.start:item.stop], variant.signs, library, building, group)
            for item, group in zip(layout.items, of_item)]

    people = _cast(layout, library, groups, casting)
    spots = [(item.kind, spot) for item, one in zip(layout.items, made) for spot in one.staff_spots]
    staff = _staff(spots, library, [one.avatar for one in people], staffing) if variant.staff else []
    if variant.small_things:
        things = _small_things(made, of_item, library, placing)
    else:
        things = [[] for _ in made]

    # The small things come after the constructions' own models, and the staff after the layout's people.
    models = (*(model for one in made for model in one.models), *(thing for stood in things for thing in stood))
    spec = SceneSpec(room=_room(layout, [opening for one in made for opening in one.openings], look),
                     parts=tuple(part for one in made for part in one.parts), models=models,
                     people=(*people, *staff), lighting=_LIGHTING, hdri=library.hdris[0] if library.hdris else None,
                     seed=int(look.integers(2 ** 31)))
    on_duty = [Person(id=one.person_id, x=one.x, y=one.y, yaw=one.yaw, role=_STAFF_ROLE) for one in staff]
    scene_gt = SceneGT(people=[*truth.people, *on_duty], targets=list(truth.targets), bounds=truth.bounds,
                       obstacles=_obstacles(layout, made, owners), occupancy=truth.occupancy)

    blocks: dict[str, tuple[Block, ...]] = {}
    lettering: dict[str, tuple[Lettered, ...]] = {}
    for target in truth.targets:
        own = [index for index, owner in enumerate(owners) if owner == target.id]
        lettering[target.id] = tuple(lettered for index in own for lettered in made[index].lettering)
        if target.kind == "person":
            blocks[target.id] = tuple(_person_block(one, library) for one in people
                                      if one.person_id == target.person_id)
        elif target.footprint is None and target.match_radius is not None:
            solids = [solid for index in own for solid in _solids(made[index], things[index])]
            blocks[target.id] = (_round_block(target, solids),) if own else ()
        else:
            blocks[target.id] = tuple(_item_block(layout.items[index], _solids(made[index], things[index]),
                                                  float(layout.room.height)) for index in own)
    # The targets that someone on duty stands at: those an item with a staff spot draws, where there are staff.
    at = [owner for owner, one in zip(owners, made) if owner is not None and one.staff_spots] if staff else []
    return Dressed(spec, scene_gt, {target.id: (groups[target.id],) for target in truth.targets}, blocks, lettering,
                   _episodes(layout, at))
