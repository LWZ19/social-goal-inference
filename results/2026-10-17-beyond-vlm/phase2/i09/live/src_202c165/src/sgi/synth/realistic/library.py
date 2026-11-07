"""The asset library of the realistic set, as its manifest describes it: which models, avatars, poses,
materials, HDRIs and fonts there are, with their measurements and tags.

Only the manifest is read, so a library loads on a machine that has downloaded nothing. The manifest is the
folder `manifest/` under the assets folder, with one JSON file per asset class, named after it (ASSET_CLASSES):
`models.json`, `materials.json`, `hdris.json`, `fonts.json`, `avatars.json`, `poses.json`. A file is a list
of entries, each an object with an `id` that no other entry of its class has. A class without a file has no
entries, and a file of another name is left alone. What an entry must hold beside its `id`:

    models     kind, size ([along its front, across, up] in metres, at life size once prepared), tags, brand
    avatars    sex, natural_height (metres), tags
    poses      reach (metres from the body's upright axis), tags; sex if it suits one sex only
    materials  kind, tile_m (metres that one repeat of its texture covers)
    hdris      nothing more
    fonts      nothing more; file, its path under the prepared folder, if that is not fonts/<id>.ttf

Each side of a size, a natural_height and a tile_m are above 0, and a reach is 0 or more; none of them is NaN
or infinite. A font's file is written from the prepared folder and stays under it, whether the entry gives it
or its id does: it is not absolute, and no `..` of it leads out. A value of the wrong kind or out of its
range is a ValueError that names the file, the entry and the key.

An entry of any class also carries what the fetching and the attribution need (`sources`, `licence`,
`author`, `redistributable`, `brand`, `prepare`). But for a model's `brand`, those and any other key are not
read here; manifest_entries gives the entries whole.

Prepared assets lie under `_cache/prepared/`, which is not committed. The preparation ends by writing
PREPARED_STAMP there: a JSON object that lists, for each class, the ids it prepared, as in
{"models": ["plant_a"], "avatars": [...], ...}. is_prepared compares the library with that list alone and
opens no asset.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, TypeVar

from sgi.config import REPO_ROOT

ASSETS_DIR = REPO_ROOT / "assets" / "realistic"   # manifest/, ATTRIBUTION.md, _cache/{downloads,prepared}
ASSET_CLASSES = ("models", "materials", "hdris", "fonts", "avatars", "poses")     # each a file of the manifest
PREPARED_STAMP = "prepared.json"                   # in the prepared folder: what has been prepared, by class

# The names the tasks share. A manifest may hold others; these are the ones the scenes ask for.
MODEL_KINDS = ("plant", "bench", "cooler", "lockers", "office_chair", "monitor", "keyboard", "espresso_machine", "cup",
               "bottle", "book", "box", "clock", "bag")
AVATAR_TAGS = ("female", "male", "red", "pinkish", "casual", "business", "staff_stall", "staff_desk",
               "staff_ticket_counter", "staff_shop")
POSE_TAGS = ("waiting", "talking", "serving", "phone", "cup")

_T = TypeVar("_T")


@dataclass(frozen=True)
class ModelAsset:
    id: str
    kind: str
    size: tuple[float, float, float]               # at life size, once prepared: along its front, across, up
    tags: frozenset[str]
    brand: bool                                    # it shows a real brand: listed, and never offered by models_of


@dataclass(frozen=True)
class AvatarAsset:
    id: str
    sex: str                                       # "female" | "male"
    natural_height: float                          # as modelled, standing upright
    tags: frozenset[str]


@dataclass(frozen=True)
class PoseAsset:
    id: str
    reach: float                                   # the furthest the body gets from its upright axis
    tags: frozenset[str]
    sex: str | None = None                         # the one sex it suits; None for either


@dataclass(frozen=True)
class MaterialAsset:
    id: str
    kind: str
    tile_m: float                                  # what one repeat of its texture covers


@dataclass(frozen=True)
class Library:
    models: Mapping[str, ModelAsset]               # each by id, in the order of the manifest
    avatars: Mapping[str, AvatarAsset]
    poses: Mapping[str, PoseAsset]
    materials: Mapping[str, MaterialAsset]
    hdris: tuple[str, ...]                         # ids
    fonts: Mapping[str, Path]                      # id to the font file, once prepared
    prepared: Path                                 # the folder of the prepared assets, there or not

    def models_of(self, kind: str) -> list[ModelAsset]:
        """The models of a kind that may be shown, in the order of the manifest; none for a kind without one.

        A model that shows a real brand is left out: no picture may show one."""
        return [model for model in self.models.values() if model.kind == kind and not model.brand]


def manifest_entries(root: Path = ASSETS_DIR) -> dict[str, list[dict]]:
    """The entries of each class as the manifest under `root` writes them, every key kept.

    A class without a file, or a root without a manifest, has none."""
    found: dict[str, list[dict]] = {}
    for name in ASSET_CLASSES:
        path = root / "manifest" / f"{name}.json"
        if not path.is_file():
            found[name] = []
            continue
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: not JSON: {error}") from error
        if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
            raise ValueError(f"{path}: expected a list of entries, each an object")
        found[name] = entries
    return found


def _text(entry: dict, key: str) -> str:
    if not isinstance(entry[key], str) or not entry[key]:
        raise ValueError(f"{key} is {entry[key]!r}: expected a name")
    return entry[key]


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(entry: dict, key: str) -> float:
    if not _is_number(entry[key]):
        raise ValueError(f"{key} is {entry[key]!r}: expected a number")
    return float(entry[key])


def _above_zero(entry: dict, key: str) -> float:
    """A measure that things are scaled by or divided by: at zero or less it describes nothing. Nor does NaN or
    infinity, which Python reads from JSON as numbers, and which no comparison with 0 turns away by itself."""
    value = _number(entry, key)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{key} is {entry[key]!r}: expected a finite number above 0")
    return value


def _zero_or_more(entry: dict, key: str) -> float:
    value = _number(entry, key)
    if not (math.isfinite(value) and value >= 0):
        raise ValueError(f"{key} is {entry[key]!r}: expected a finite number of 0 or more")
    return value


def _size(entry: dict, key: str) -> tuple[float, float, float]:
    sides = entry[key]
    if not isinstance(sides, list) or len(sides) != 3 or not all(_is_number(side) for side in sides):
        raise ValueError(f"{key} is {sides!r}: expected three numbers")
    if not all(math.isfinite(side) and side > 0 for side in sides):       # each side: min() keeps a leading NaN
        raise ValueError(f"{key} is {sides!r}: expected three finite numbers above 0")
    return float(sides[0]), float(sides[1]), float(sides[2])


def _under_prepared(name: str, said: str) -> str:
    """A path as the manifest gives it for a prepared file: one that stays under the prepared folder once joined
    to it, so not absolute and with no `..` that leads out. Judged by its spelling alone; no file is looked at.
    `said` is what the error calls it."""
    spelled = os.path.normpath(name)
    if os.path.isabs(name) or spelled == os.pardir or spelled.startswith(os.pardir + os.sep):
        raise ValueError(f"{said} is {name!r}: expected a path under the prepared folder")
    return name


def _prepared_path(entry: dict, key: str) -> str:
    return _under_prepared(_text(entry, key), key)


def _tags(entry: dict, key: str) -> frozenset[str]:
    if not isinstance(entry[key], list) or not all(isinstance(tag, str) for tag in entry[key]):
        raise ValueError(f"{key} is {entry[key]!r}: expected a list of names")
    return frozenset(entry[key])


def _flag(entry: dict, key: str) -> bool:
    if not isinstance(entry[key], bool):
        raise ValueError(f"{key} is {entry[key]!r}: expected true or false")
    return entry[key]


def _by_id(path: Path, entries: list[dict], read: Callable[[dict], _T]) -> dict[str, _T]:
    """What `read` makes of each entry, by the entry's id. An entry that lacks a key `read` asks for, holds a value
    of the wrong kind or out of its range, or has the id of an earlier one is an error that names the file and
    the entry."""
    found: dict[str, _T] = {}
    for entry in entries:
        name = entry.get("id", "(without an id)")
        try:
            key, value = _text(entry, "id"), read(entry)
        except KeyError as error:
            raise ValueError(f"{path}: the entry {name!r} has no {error.args[0]!r}") from None
        except ValueError as error:
            raise ValueError(f"{path}: in the entry {name!r}, {error}") from None
        if key in found:
            raise ValueError(f"{path}: two entries have the id {key!r}")
        found[key] = value
    return found


def _model(entry: dict) -> ModelAsset:
    return ModelAsset(entry["id"], _text(entry, "kind"), _size(entry, "size"), _tags(entry, "tags"),
                      _flag(entry, "brand"))


def _avatar(entry: dict) -> AvatarAsset:
    return AvatarAsset(entry["id"], _text(entry, "sex"), _above_zero(entry, "natural_height"), _tags(entry, "tags"))


def _pose(entry: dict) -> PoseAsset:
    sex = _text(entry, "sex") if entry.get("sex") is not None else None
    return PoseAsset(entry["id"], _zero_or_more(entry, "reach"), _tags(entry, "tags"), sex)


def _material(entry: dict) -> MaterialAsset:
    return MaterialAsset(entry["id"], _text(entry, "kind"), _above_zero(entry, "tile_m"))


def load_library(root: Path = ASSETS_DIR) -> Library:
    """The library that the manifest under `root` describes; an empty one where there is no manifest."""
    entries, prepared = manifest_entries(root), root / "_cache" / "prepared"

    def read(name: str, make: Callable[[dict], _T]) -> dict[str, _T]:
        return _by_id(root / "manifest" / f"{name}.json", entries[name], make)

    def font_file(entry: dict) -> Path:
        if "file" in entry:
            return prepared / _prepared_path(entry, "file")
        # The id is part of the path then, and is held to what a file is held to.
        return prepared / _under_prepared(f"fonts/{entry['id']}.ttf", "the file its id gives")

    return Library(models=read("models", _model), avatars=read("avatars", _avatar), poses=read("poses", _pose),
                   materials=read("materials", _material), hdris=tuple(read("hdris", lambda entry: None)),
                   fonts=read("fonts", font_file), prepared=prepared)


def is_prepared(library: Library) -> bool:
    """Whether the preparation has been run for everything the library names. Never raises.

    True when the stamp in the prepared folder lists every id of every class of the library. False without
    the stamp, with one that cannot be read, and after the manifest has gained an asset that the last
    preparation did not know. It opens no asset: the `verify` action is what checks the files."""
    named = {"models": library.models, "materials": library.materials, "hdris": library.hdris, "fonts": library.fonts,
             "avatars": library.avatars, "poses": library.poses}
    try:
        stamp = json.loads((library.prepared / PREPARED_STAMP).read_text(encoding="utf-8"))
        return all(set(named[name]) <= set(stamp.get(name, ())) for name in ASSET_CLASSES)
    except Exception:                              # no stamp, not JSON, not an object of lists: not prepared
        return False
