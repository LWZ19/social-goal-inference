"""The asset library as its manifest describes it. No file of any asset is needed, and none is read."""
import inspect
import json
from pathlib import Path

import pytest

from sgi.config import REPO_ROOT
from sgi.synth.realistic.library import (ASSET_CLASSES, ASSETS_DIR, AVATAR_TAGS, MODEL_KINDS, POSE_TAGS, PREPARED_STAMP,
                                         AvatarAsset, Library, MaterialAsset, ModelAsset, PoseAsset, is_prepared,
                                         load_library, manifest_entries)

NAN, INF = float("nan"), float("inf")              # json writes them as NaN and Infinity, and reads those back

# What every entry also carries, and the library does not read.
RIGHTS = {"sources": [{"url": "https://example.org/a.zip", "sha256": "0" * 64, "bytes": 10}], "licence": "CC0",
          "author": "A. Maker", "redistributable": True, "brand": False, "prepare": {"scale": 0.01, "front": "-Y"}}


def model(name: str, kind: str, **changes) -> dict:
    return {"id": name, "kind": kind, "size": [0.5, 0.4, 1.2], "tags": ["leafy"], **RIGHTS, **changes}


def without(entry: dict, key: str) -> dict:
    return {name: value for name, value in entry.items() if name != key}


def write_manifest(root: Path, **files) -> Path:
    """A manifest folder under `root` with one file per name given, holding its entries."""
    (root / "manifest").mkdir(parents=True, exist_ok=True)
    for name, entries in files.items():
        (root / "manifest" / f"{name}.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return root


def whole_manifest(root: Path) -> Path:
    return write_manifest(
        root,
        models=[model("plant_a", "plant", future_key=[1, 2]),
                model("bench_a", "bench", size=[1.6, 0.5, 0.45], tags=[])],
        materials=[{"id": "wood", "kind": "pbr", "tile_m": 1.2, **RIGHTS},
                   {"id": "glass", "kind": "glass", "tile_m": 1, **RIGHTS}],
        hdris=[{"id": "street_a", **RIGHTS}, {"id": "park_b", **RIGHTS}],
        fonts=[{"id": "sans", "file": "fonts/Sans-Regular.ttf", **RIGHTS}, {"id": "mono", **RIGHTS}],
        avatars=[{"id": "f_casual_01", "sex": "female", "natural_height": 1.68, "tags": ["female", "casual"],
                  **RIGHTS}],
        poses=[{"id": "wait_01", "reach": 0.31, "tags": ["waiting"], **RIGHTS},
               {"id": "serve_f", "reach": 0.4, "tags": ["serving"], "sex": "female", **RIGHTS}])


def test_the_names_other_tasks_use():
    assert ASSETS_DIR == REPO_ROOT / "assets" / "realistic"
    assert inspect.signature(load_library).parameters["root"].default == ASSETS_DIR
    assert ASSET_CLASSES == ("models", "materials", "hdris", "fonts", "avatars", "poses")
    assert PREPARED_STAMP == "prepared.json"
    assert MODEL_KINDS == ("plant", "bench", "cooler", "lockers", "office_chair", "monitor", "keyboard",
                           "espresso_machine", "cup", "bottle", "book", "box", "clock", "bag")
    assert AVATAR_TAGS == ("female", "male", "red", "pinkish", "casual", "business", "staff_stall", "staff_desk",
                           "staff_ticket_counter", "staff_shop")
    assert POSE_TAGS == ("waiting", "talking", "serving", "phone", "cup")
    assert PoseAsset("p", 0.3, frozenset()).sex is None


def test_an_absent_manifest_is_an_empty_library(tmp_path):
    root = tmp_path / "nowhere"
    library = load_library(root)
    assert library == Library(models={}, avatars={}, poses={}, materials={}, hdris=(), fonts={},
                              prepared=root / "_cache" / "prepared")
    assert library.models_of("plant") == [] and is_prepared(library) is False
    assert manifest_entries(root) == {name: [] for name in ASSET_CLASSES}
    assert not root.exists()                                               # reading a library makes nothing
    some = load_library(write_manifest(tmp_path / "some", hdris=[{"id": "street_a"}]))      # a file of one class alone
    assert some.hdris == ("street_a",) and some.models == {} and some.fonts == {}


def test_the_manifest_files_are_merged_and_unknown_keys_ignored(tmp_path):
    root = whole_manifest(tmp_path)
    (root / "manifest" / "notes.json").write_text('{"about": "not an asset class"}')      # of no class: left alone
    library = load_library(root)
    fonts = root / "_cache" / "prepared" / "fonts"
    assert library.models == {"plant_a": ModelAsset("plant_a", "plant", (0.5, 0.4, 1.2), frozenset({"leafy"}), False),
                              "bench_a": ModelAsset("bench_a", "bench", (1.6, 0.5, 0.45), frozenset(), False)}
    assert library.avatars == {
        "f_casual_01": AvatarAsset("f_casual_01", "female", 1.68, frozenset({"female", "casual"}))}
    assert library.poses == {"wait_01": PoseAsset("wait_01", 0.31, frozenset({"waiting"})),
                             "serve_f": PoseAsset("serve_f", 0.4, frozenset({"serving"}), sex="female")}
    assert library.materials == {"wood": MaterialAsset("wood", "pbr", 1.2),
                                 "glass": MaterialAsset("glass", "glass", 1.0)}
    assert library.hdris == ("street_a", "park_b")
    assert library.prepared == root / "_cache" / "prepared"
    assert library.fonts == {"sans": fonts / "Sans-Regular.ttf", "mono": fonts / "mono.ttf"}
    assert list(library.models) == ["plant_a", "bench_a"]                  # in the order of the file
    plant, glass = library.models["plant_a"], library.materials["glass"]
    assert isinstance(plant.size, tuple) and isinstance(plant.tags, frozenset) and isinstance(glass.tile_m, float)


def test_the_entries_are_also_given_as_they_are_written(tmp_path):
    entries = manifest_entries(whole_manifest(tmp_path))
    assert list(entries) == list(ASSET_CLASSES)
    assert [len(entries[name]) for name in ASSET_CLASSES] == [2, 2, 2, 2, 1, 2]
    plant = entries["models"][0]
    assert plant["sources"] == RIGHTS["sources"] and plant["prepare"] == {"scale": 0.01, "front": "-Y"}
    assert plant["future_key"] == [1, 2] and plant["licence"] == "CC0"


def test_models_of_a_kind(tmp_path):
    library = load_library(write_manifest(tmp_path, models=[
        model("plant_a", "plant"), model("bench_a", "bench"), model("plant_b", "plant", tags=["tall"])]))
    assert [found.id for found in library.models_of("plant")] == ["plant_a", "plant_b"]
    assert library.models_of("bench") == [library.models["bench_a"]]
    assert library.models_of("cooler") == []                               # a kind without a model: the caller restyles


def test_a_model_that_shows_a_brand_is_listed_and_never_offered(tmp_path):
    library = load_library(write_manifest(tmp_path, models=[
        model("cooler_logo", "cooler", brand=True), model("cooler_plain", "cooler")]))
    assert library.models["cooler_logo"].brand is True and library.models["cooler_plain"].brand is False
    assert [found.id for found in library.models_of("cooler")] == ["cooler_plain"]


@pytest.mark.parametrize("name, entries, said", [
    ("models", [without(model("plant_a", "plant"), "size")], ("plant_a", "'size'")),
    ("models", [without(model("plant_a", "plant"), "brand")], ("plant_a", "'brand'")),
    ("models", [model("plant_a", "plant", size=[0.5, 0.4])], ("plant_a", "size")),
    ("models", [model("plant_a", "plant", size=[0.5, 0.4, "tall"])], ("plant_a", "size")),
    ("models", [model("plant_a", "plant", tags="leafy")], ("plant_a", "tags")),
    ("models", [model("plant_a", "plant", brand="no")], ("plant_a", "brand")),
    ("models", [model("plant_a", "plant"), model("plant_a", "plant")], ("plant_a", "two")),
    ("avatars", [{"id": "f_01", "sex": "female", "tags": []}], ("f_01", "'natural_height'")),
    ("poses", [{"reach": 0.3, "tags": []}], ("'id'",)),
    ("poses", [{"id": "wait_01", "reach": 0.3, "tags": [], "sex": 1}], ("wait_01", "sex")),
    ("materials", [{"id": "wood", "kind": "pbr", "tile_m": "wide"}], ("wood", "tile_m")),
    ("materials", [{"id": "wood", "kind": "pbr", "tile_m": True}], ("wood", "tile_m")),
    ("fonts", [{"id": "sans", "file": 3}], ("sans", "file")),
    ("hdris", {"id": "street_a"}, ("list",)),
    ("fonts", ["sans"], ("list",)),
    # Out of range: a length that something is divided by, a reach, a path that leaves the prepared folder.
    ("models", [model("plant_a", "plant", size=[0.5, 0, 1.2])], ("plant_a", "size", "above 0")),
    ("models", [model("plant_a", "plant", size=[0.5, 0.4, -1.2])], ("plant_a", "size", "above 0")),
    ("materials", [{"id": "wood", "kind": "pbr", "tile_m": 0}], ("wood", "tile_m", "above 0")),
    ("materials", [{"id": "wood", "kind": "pbr", "tile_m": -1.2}], ("wood", "tile_m", "above 0")),
    ("avatars", [{"id": "f_01", "sex": "female", "natural_height": 0.0, "tags": []}],
     ("f_01", "natural_height", "above 0")),
    ("avatars", [{"id": "f_01", "sex": "female", "natural_height": -1.68, "tags": []}],
     ("f_01", "natural_height", "above 0")),
    ("poses", [{"id": "wait_01", "reach": -0.01, "tags": []}], ("wait_01", "reach", "0 or more")),
    ("fonts", [{"id": "sans", "file": "/usr/share/fonts/Sans-Regular.ttf"}], ("sans", "file", "under the prepared")),
    ("fonts", [{"id": "sans", "file": "fonts/../../Sans-Regular.ttf"}], ("sans", "file", "under the prepared")),
    # No measure at all: NaN and infinity, which JSON as Python reads it lets through, wherever they stand.
    ("models", [model("plant_a", "plant", size=[NAN, 0.4, 1.2])], ("plant_a", "size", "finite")),
    ("models", [model("plant_a", "plant", size=[0.5, NAN, 1.2])], ("plant_a", "size", "finite")),
    ("models", [model("plant_a", "plant", size=[0.5, 0.4, NAN])], ("plant_a", "size", "finite")),
    ("models", [model("plant_a", "plant", size=[INF, 0.4, 1.2])], ("plant_a", "size", "finite")),
    ("models", [model("plant_a", "plant", size=[0.5, 0.4, INF])], ("plant_a", "size", "finite")),
    ("materials", [{"id": "wood", "kind": "pbr", "tile_m": NAN}], ("wood", "tile_m", "finite")),
    ("materials", [{"id": "wood", "kind": "pbr", "tile_m": INF}], ("wood", "tile_m", "finite")),
    ("avatars", [{"id": "f_01", "sex": "female", "natural_height": NAN, "tags": []}],
     ("f_01", "natural_height", "finite")),
    ("avatars", [{"id": "f_01", "sex": "female", "natural_height": INF, "tags": []}],
     ("f_01", "natural_height", "finite")),
    ("poses", [{"id": "wait_01", "reach": NAN, "tags": []}], ("wait_01", "reach", "finite")),
    ("poses", [{"id": "wait_01", "reach": INF, "tags": []}], ("wait_01", "reach", "finite")),
    # A font without a file takes fonts/<id>.ttf, and an id can lead that out of the prepared folder as a file can.
    ("fonts", [{"id": "../../sans"}], ("'../../sans'", "fonts/../../sans.ttf", "under the prepared")),
    ("fonts", [{"id": "sans"}, {"id": "../../../etc/sans"}], ("'../../../etc/sans'", "under the prepared")),
])
def test_a_malformed_entry_is_reported_with_its_file_and_what_is_wrong(tmp_path, name, entries, said):
    root = write_manifest(tmp_path, **{name: entries})
    with pytest.raises(ValueError) as raised:
        load_library(root)
    assert f"{name}.json" in str(raised.value) and [word for word in said if word not in str(raised.value)] == []


def test_what_is_just_inside_each_range_is_read(tmp_path):
    library = load_library(write_manifest(
        tmp_path,
        models=[model("card", "book", size=[0.001, 0.2, 0.3])],
        materials=[{"id": "paper", "kind": "pbr", "tile_m": 0.01}],
        avatars=[{"id": "child", "sex": "female", "natural_height": 0.9, "tags": []}],
        poses=[{"id": "upright", "reach": 0, "tags": []}],
        fonts=[{"id": "sans", "file": "fonts/../type/Sans-Regular.ttf"}, {"id": "../type/sans.bold"}]))
    assert library.models["card"].size == (0.001, 0.2, 0.3) and library.materials["paper"].tile_m == 0.01
    assert library.avatars["child"].natural_height == 0.9
    assert library.poses["upright"].reach == 0.0                           # a body may keep to its upright axis
    assert library.fonts["sans"] == library.prepared / "fonts" / ".." / "type" / "Sans-Regular.ttf"   # still inside
    assert library.fonts["../type/sans.bold"] == library.prepared / "fonts" / ".." / "type" / "sans.bold.ttf"   # too


def test_a_manifest_file_that_is_not_json_is_reported_with_its_name(tmp_path):
    root = write_manifest(tmp_path)
    (root / "manifest" / "poses.json").write_text("[{'id': ", encoding="utf-8")
    with pytest.raises(ValueError, match="poses.json"):
        load_library(root)


def test_the_library_is_prepared_once_the_stamp_names_all_of_it(tmp_path):
    root = whole_manifest(tmp_path)
    library = load_library(root)
    stamp = library.prepared / PREPARED_STAMP
    assert is_prepared(library) is False                                   # nothing fetched, nothing prepared
    library.prepared.mkdir(parents=True)
    assert is_prepared(library) is False                                   # a folder without the stamp
    done = {"models": ["plant_a", "bench_a"], "materials": ["wood", "glass"], "hdris": ["street_a", "park_b"],
            "fonts": ["sans", "mono"], "avatars": ["f_casual_01"], "poses": ["wait_01", "serve_f", "one_dropped_since"]}
    stamp.write_text(json.dumps(done))
    assert is_prepared(library) is True
    for name in ASSET_CLASSES:                                             # each class counts
        stamp.write_text(json.dumps({**done, name: done[name][1:]}))
        assert is_prepared(library) is False, name
    stamp.write_text(json.dumps(done))
    grown = load_library(write_manifest(root, hdris=[{"id": "street_a"}, {"id": "park_b"}, {"id": "added_since"}]))
    assert is_prepared(grown) is False                                     # the manifest has grown since
    for unreadable in ("", "[1, 2]", '{"models": 5}', "{not json"):
        stamp.write_text(unreadable)
        assert is_prepared(library) is False                               # and it never raises
