"""The assets of the realistic set: the manifest as it is committed, and the command that fetches, verifies and
prepares what it names and writes the attribution file.

Nothing here uses the network: every download is a fake that the test serves from memory, and the real one is
made unusable. The tests of the Blender side build a library of their own from a few files written here, and
are skipped where no Blender of the series answers."""
import ast
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sgi.config import REPO_ROOT
from sgi.synth.realistic import assets, blender
from sgi.synth.realistic.library import (ASSET_CLASSES, ASSETS_DIR, MODEL_KINDS, PREPARED_STAMP, is_prepared,
                                         load_library,
                                         manifest_entries)
from sgi.synth.realistic.spec import MATERIALS

THINGS = ("models", "materials", "hdris", "fonts")             # the classes this part of the library holds
SRC = REPO_ROOT / "src" / "sgi"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------------- a fake web
GLTF, DIFF = "https://models.example.org/plant/plant_1k.gltf", "https://models.example.org/plant/jpg/plant_diff_1k.jpg"
BENCH = "https://models.example.org/bench/bench.glb"
WOOD = "https://textures.example.org/get?file=Wood_1K-JPG.zip"              # its name is in the query, not the path
SKY, FONT, FBX = "https://skies.example.org/street_2k.hdr", "https://fonts.example.org/sans/Sans-Regular.ttf", \
    "https://people.example.org/avatars/f_01.fbx"
MOVE = "https://people.example.org/moves/idle%20wait.fbx"                    # a name the address writes with an escape
SERVED = {GLTF: b'{"asset": {"version": "2.0"}}', DIFF: b"\xff\xd8 a picture of leaves", BENCH: b"glTF a bench",
          WOOD: b"PK a zip of maps", SKY: b"#?RADIANCE a street", FONT: b"\x00\x01\x00\x00 a font",
          FBX: b"Kaydara FBX Binary  a body", MOVE: b"Kaydara FBX Binary  a move"}


class Web:
    """Stands in for assets.download: writes what it serves for an address, and keeps what it was asked for."""

    def __init__(self):
        self.serve, self.calls = dict(SERVED), []

    def __call__(self, url: str, path: Path) -> None:
        self.calls.append(url)
        Path(path).write_bytes(self.serve[url])


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **options):
        raise AssertionError(f"a test reached for the network: {args}")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


@pytest.fixture
def web(monkeypatch) -> Web:
    fake = Web()
    monkeypatch.setattr(assets, "download", fake)
    return fake


def source(url: str, **more) -> dict:
    return {"url": url, "sha256": sha(SERVED[url]), "bytes": len(SERVED[url]), **more}


def entry(name: str, *sources: dict, **more) -> dict:
    return {"id": name, "title": name.replace("_", " ").title(), "sources": list(sources),
            "licence": 'CC0 1.0 ("all our assets are CC0", https://models.example.org/licence)', "author": "A. Maker",
            "page": f"https://models.example.org/a/{name}", "redistributable": True, "brand": False, "prepare": {},
            **more}


def write_manifest(root: Path, **files) -> Path:
    (root / "manifest").mkdir(parents=True, exist_ok=True)
    for name, entries in files.items():
        (root / "manifest" / f"{name}.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return root


def a_manifest(root: Path, **changed) -> Path:
    """A manifest of all six classes: seven entries with eight files to fetch, and one thing made in code."""
    files = dict(
        models=[entry("plant_a", source(GLTF), source(DIFF, file="textures/plant_diff_1k.jpg"), kind="plant",
                      size=[0.5, 0.4, 1.2], tags=[]),
                entry("bench_a", source(BENCH), kind="bench", size=[1.5, 0.4, 0.8], tags=[])],
        materials=[entry("wood", source(WOOD, file="Wood_1K-JPG.zip"), kind="pbr", tile_m=1.5),
                   entry("glass", kind="glass", tile_m=1.0, licence="none: made in code", author="this project",
                         page=None)],
        hdris=[entry("street", source(SKY))],
        fonts=[entry("sans", source(FONT),
                     licence="OFL 1.1 (SIL Open Font License, https://fonts.example.org/OFL.txt)")],
        avatars=[entry("f_01", source(FBX), sex="female", natural_height=1.7, tags=["female"],
                       licence="MIT (LICENSE.md, Copyright (c) 2020 A. Maker)")],
        poses=[entry("wait_01", source(MOVE), reach=0.3, tags=["waiting"], licence="MIT (LICENSE.md)")])
    return write_manifest(root, **{**files, **changed})


def downloads(root: Path) -> Path:
    return root / "_cache" / "downloads"


def fetched(root: Path) -> dict[str, bytes]:
    """Every file under the downloads folder, by its path from there."""
    return {str(path.relative_to(downloads(root))): path.read_bytes()
            for path in sorted(downloads(root).rglob("*")) if path.is_file()}


EVERYTHING = {"plant_a/plant_1k.gltf": SERVED[GLTF], "plant_a/textures/plant_diff_1k.jpg": SERVED[DIFF],
              "bench_a/bench.glb": SERVED[BENCH], "wood/Wood_1K-JPG.zip": SERVED[WOOD],
              "street/street_2k.hdr": SERVED[SKY],
              "sans/Sans-Regular.ttf": SERVED[FONT], "f_01/f_01.fbx": SERVED[FBX],
              "wait_01/idle wait.fbx": SERVED[MOVE]}


# ------------------------------------------------------------------------------- the manifest as committed
def test_every_manifest_entry_names_its_sources_licence_and_checksums():
    entries = manifest_entries()
    assert [name for name in THINGS if not entries[name]] == []            # the four classes of this part are there
    assert assets.manifest_problems() == []                                # the rules the command itself holds to
    seen: dict[Path, str] = {}
    for name in ASSET_CLASSES:                                             # and, read here once more, the bare facts
        for item in entries[name]:
            said = f"{name} {item.get('id')!r}"
            assert isinstance(item["licence"], str) and len(item["licence"]) >= 3, said
            assert isinstance(item["author"], str) and item["author"].strip(), said
            assert type(item["redistributable"]) is bool and type(item["brand"]) is bool, said
            assert isinstance(item["prepare"], dict), said
            assert item["sources"] or item["licence"].startswith("none"), said       # nothing unsourced but our own
            for one in item["sources"]:
                assert re.fullmatch(r"https://[^\s/]+/\S+", one["url"]), said
                assert re.fullmatch(r"[0-9a-f]{64}", one["sha256"]), said
                assert type(one["bytes"]) is int and one["bytes"] > 0, said
                path = assets.source_path(ASSETS_DIR, item, one)
                assert downloads(ASSETS_DIR) in path.parents and ".." not in path.parts, said
                assert seen.setdefault(path, one["sha256"]) == one["sha256"], f"{said}: {path} is claimed twice"
    assert len(seen) >= 60                                                 # a model alone is several files


def rows_by_class(text: str) -> dict[str, list[str]]:
    """The table rows of an attribution text, under the class whose heading they stand under."""
    parts = {part.split("\n", 1)[0].strip().lower(): part for part in text.split("\n## ")[1:]}
    return {name: [line for line in parts.get(name, "").splitlines() if line.startswith("| `")]
            for name in ASSET_CLASSES}


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def test_the_attribution_file_lists_every_asset_and_is_up_to_date():
    written = (ASSETS_DIR / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert written == assets.attribution()                                 # regenerating it changes nothing
    rows = rows_by_class(written)
    for name, entries in manifest_entries().items():
        assert len(rows[name]) == len(entries), name                       # a row for each asset, and no other
        for item in entries:
            (row,) = [line for line in rows[name] if line.startswith(f"| `{item['id']}` |")]
            assert _cell(item["author"]) in row and _cell(item["licence"]) in row, item["id"]
            assert not item["sources"] or item.get("page", item["sources"][0]["url"]) in row, item["id"]


def test_brand_assets_are_marked_and_none_is_redistributable_by_mistake():
    entries, library = manifest_entries(), load_library()
    for name in ASSET_CLASSES:
        for item in entries[name]:
            said = f"{name} {item['id']}"
            assert type(item["brand"]) is bool, said                       # each says whether it shows one
            assert not (item["brand"] and item["redistributable"]), said   # a licence does not cover another's mark
            if item["redistributable"]:                                    # only under a licence known to allow it
                assert assets.licence_family(item["licence"]) in assets.REDISTRIBUTABLE, said
    # what a model shows was looked at, and is said
    for item in entries["models"]:
        assert isinstance(item.get("lettering"), str) and item["lettering"].strip(), item["id"]
    shown = {model.id for kind in {model.kind for model in library.models.values()}
             for model in library.models_of(kind)}
    assert shown == {item["id"] for item in entries["models"] if not item["brand"]}       # a brand is never offered
    # The rule can be broken: a model that shows a maker's name and is passed on all the same.
    marked = entry("monitor_b", source(BENCH), kind="monitor", size=[0.2, 0.5, 0.4], tags=[], brand=True)
    assert [said for said in assets.entry_problems(marked) if "brand" in said and "redistributable" in said]
    assert assets.entry_problems({**marked, "redistributable": False}) == []
    unclear = entry("plant_b", source(GLTF), licence="free for personal use (the uploader's page)")
    assert [said for said in assets.entry_problems(unclear) if "redistributable" in said and "licence" in said]
    assert assets.entry_problems({**unclear, "redistributable": False}) == []


# The library is to hold a model of each kind "where an acceptable one exists". For the machine of a coffee stall
# none does: a stall's counter takes a machine about 0.3 m deep, and the one clean model of a coffee machine found
# is a commercial brewer twice that. The kind may therefore be empty, and what it holds all the same says in its
# `note` why no stall shows it.
KINDS_WITHOUT_AN_ACCEPTABLE_MODEL = ("espresso_machine",)


def test_the_library_has_the_materials_and_models_the_scenes_need():
    entries, library = manifest_entries(), load_library()
    assert [name for name in MATERIALS if name not in library.materials] == []            # every shared material name
    # and nothing a scene cannot ask for
    assert list(library.materials) == list(MATERIALS)
    by_id = {item["id"]: item for item in entries["materials"]}
    assert {library.materials[name].kind for name in MATERIALS} <= {"pbr", "plain", "glass", "screen"}
    for name in MATERIALS:
        made_in_code = library.materials[name].kind != "pbr"
        assert (by_id[name]["sources"] == []) == made_in_code, name
        assert by_id[name]["licence"].startswith("none: made in code") == made_in_code, name
    assert library.materials["glass"].kind == "glass" and library.materials["screen"].kind == "screen"
    for name in ("floor_tile", "floor_terrazzo", "floor_laminate", "wall_paint", "ceiling_tile"):
        # no small repeat
        assert library.materials[name].kind == "pbr" and library.materials[name].tile_m >= 1.5, name
    outdoor = [item["id"] for item in entries["hdris"] if "outdoor" in item.get("tags", [])]
    assert len(outdoor) >= 2 and set(outdoor) <= set(library.hdris)                       # what a window shows
    open_fonts = [item for item in entries["fonts"] if assets.licence_family(item["licence"]) == "OFL 1.1"]
    assert len(open_fonts) >= 2 and len({item["title"] for item in open_fonts}) >= 2      # one sans, two weights
    assert "sans" in library.fonts and "sans_bold" in library.fonts
    for item in open_fonts:
        assert library.fonts[item["id"]].suffix == ".ttf"
        assert [one for one in item["sources"] if one["url"].endswith("OFL.txt")], item["id"]     # its licence text
    kinds = {model.kind for model in library.models.values()}
    # no kind a scene never asks for
    assert kinds <= set(MODEL_KINDS)
    wanting = KINDS_WITHOUT_AN_ACCEPTABLE_MODEL
    assert set(wanting) < set(MODEL_KINDS)
    assert [kind for kind in MODEL_KINDS if kind not in wanting and not library.models_of(kind)] == []
    for item in entries["models"]:                                         # kept for a place that could take it,
        if item["kind"] in wanting:                                        # and not passed off as one that fits
            assert isinstance(item.get("note"), str) and item["note"].strip(), item["id"]
    assert [kind for kind in ("plant", "bench", "office_chair") if len(library.models_of(kind)) < 2] == []
    for model in library.models.values():                                  # life size: nothing a room could not hold
        assert max(model.size) <= 2.5 and min(model.size) >= 0.01, model.id


# ------------------------------------------------------------------------- what an entry has to carry
def test_what_an_entry_lacks_is_said_key_by_key():
    whole = entry("plant_a", source(GLTF), source(DIFF, file="textures/plant_diff_1k.jpg"))
    assert assets.entry_problems(whole) == []
    for key in ("sources", "licence", "author", "redistributable", "brand", "prepare"):
        (said,) = assets.entry_problems({name: value for name, value in whole.items() if name != key})
        assert key in said
    for change, word in (({"licence": ""}, "licence"), ({"author": 3}, "author"), ({"redistributable": "yes"},
                                                                                   "redist"),
                         ({"brand": 0}, "brand"), ({"prepare": []}, "prepare"), ({"sources": {}}, "sources"),
                         ({"sources": []}, "sources"), ({"id": "../plant"}, "id"), ({"id": ""}, "id"),
                         ({"id": ".hidden"}, "id"), ({"id": "a/b"}, "id"), ({"id": "--force"}, "id"),
                         ({"id": 7}, "id")):
        problems = assets.entry_problems({**whole, **change})
        assert problems and word in problems[0], change
    for change, word in (({"sha256": "0" * 63}, "sha256"), ({"sha256": sha(b"x").upper()}, "sha256"),
                         ({"bytes": 0}, "bytes"),
                         ({"bytes": "12"}, "bytes"), ({"bytes": True}, "bytes"),
                         ({"url": "http://models.example.org/a.glb"}, "https"),
                         ({"url": "file:///etc/passwd"}, "https"), ({"url": "https://models.example.org/"}, "file"),
                         ({"file": "../other/plant.gltf"}, "file"), ({"file": "/tmp/plant.gltf"}, "file"),
                         ({"file": "textures/../../plant.gltf"}, "file"), ({"file": ""}, "file")):
        problems = assets.entry_problems({**whole, "sources": [{**source(GLTF), **change}]})
        assert problems and word in problems[0], change
    # one name, twice
    assert assets.entry_problems({**whole, "sources": [source(GLTF), source(DIFF, file="plant_1k.gltf")]})
    made = entry("glass", licence="none: made in code", author="this project")
    assert assets.entry_problems(made) == []                               # nothing to fetch for what is made in code
    assert assets.entry_problems({**made, "sources": [source(GLTF)]})      # and then nothing is fetched for it


def test_where_a_source_lies_once_fetched(tmp_path):
    item = entry("plant_a", source(GLTF), source(DIFF, file="textures/plant_diff_1k.jpg"))
    folder = tmp_path / "_cache" / "downloads" / "plant_a"
    # the address's last part
    assert assets.source_path(tmp_path, item, item["sources"][0]) == folder / "plant_1k.gltf"
    assert assets.source_path(tmp_path, item, item["sources"][1]) == folder / "textures" / "plant_diff_1k.jpg"
    assert assets.source_path(tmp_path, entry("wait_01"), source(MOVE)).name == "idle wait.fbx"


def test_a_manifest_whose_entries_claim_one_file_differently_is_refused(tmp_path, web):
    """Ids are unique within a class only, and the downloads folder is by id: an avatar and a pose of one id
    share a folder."""
    other = {**source(FBX), "sha256": sha(b"another body")}
    root = a_manifest(tmp_path, poses=[entry("f_01", other, reach=0.3, tags=[])])
    (said,) = assets.manifest_problems(root)
    assert said.startswith("poses 'f_01'") and "f_01/f_01.fbx" in said and "avatars 'f_01'" in said
    with pytest.raises(assets.AssetError, match="f_01.fbx"):
        assets.fetch(root)
    assert web.calls == []                                                 # nothing is fetched by a manifest like that
    agreeing = a_manifest(tmp_path, poses=[entry("f_01", source(FBX), reach=0.3, tags=[])])
    assert assets.manifest_problems(agreeing) == []
    assets.fetch(agreeing)
    assert web.calls.count(FBX) == 1                                       # and a file they share is fetched once


# ------------------------------------------------------------------------------------------------ fetch
def test_fetch_verifies_checksums_and_refuses_a_wrong_file(tmp_path, web):
    root = a_manifest(tmp_path)
    got = assets.fetch(root)
    # each once, each in its folder
    assert fetched(root) == EVERYTHING and sorted(web.calls) == sorted(SERVED)
    assert sorted(got) == sorted(downloads(root) / name for name in EVERYTHING)
    # all there and intact: nothing asked
    assert assets.fetch(root) == [] and len(web.calls) == len(SERVED)
    # A file that is there by other means is taken when it is the file the manifest names, and replaced when not.
    (downloads(root) / "street" / "street_2k.hdr").write_bytes(b"#?RADIANCE cut sho")
    (downloads(root) / "bench_a" / "bench.glb").unlink()
    assert assets.fetch(root) == [downloads(root) / "bench_a" / "bench.glb",
                                  downloads(root) / "street" / "street_2k.hdr"]
    assert fetched(root) == EVERYTHING and web.calls[len(SERVED):] == [BENCH, SKY]
    # A download that is not the file: refused, nothing of it kept, and what was fetched before is left alone.
    asked = len(web.calls)
    (downloads(root) / "bench_a" / "bench.glb").unlink()
    web.serve[BENCH] = b"glTF a bench, tampered with"
    with pytest.raises(assets.AssetError) as raised:
        assets.fetch(root)
    said = str(raised.value)
    assert "bench_a" in said and BENCH in said and sha(SERVED[BENCH]) in said and sha(web.serve[BENCH]) in said
    assert fetched(root) == {name: data for name, data in EVERYTHING.items() if name != "bench_a/bench.glb"}
    assert web.calls[asked:] == [BENCH]                                    # it stopped there
    web.serve[BENCH] = SERVED[BENCH][:-1] + b"X"                           # the right length and other bytes
    with pytest.raises(assets.AssetError, match="sha256"):
        assets.fetch(root)
    assert "bench_a/bench.glb" not in fetched(root)
    web.serve[BENCH] = SERVED[BENCH]
    assert assets.fetch(root) == [downloads(root) / "bench_a" / "bench.glb"] and fetched(root) == EVERYTHING


def test_fetch_takes_the_named_assets_alone_and_everything_again_when_forced(tmp_path, web):
    root = a_manifest(tmp_path)
    assets.fetch(root, only=["plant_a", "sans"])
    assert sorted(fetched(root)) == ["plant_a/plant_1k.gltf", "plant_a/textures/plant_diff_1k.jpg",
                                     "sans/Sans-Regular.ttf"]
    with pytest.raises(assets.AssetError, match="no_such_plant"):
        assets.fetch(root, only=["bench_a", "no_such_plant"])
    assert len(web.calls) == 3                                             # an unknown name: nothing is fetched
    assets.fetch(root, only=["plant_a"], force=True)
    assert web.calls[3:] == [GLTF, DIFF] and len(fetched(root)) == 3
    # Forced, a download that is wrong still leaves the good file where it was.
    web.serve[GLTF] = b"{}"
    with pytest.raises(assets.AssetError):
        assets.fetch(root, only=["plant_a"], force=True)
    assert fetched(root)["plant_a/plant_1k.gltf"] == SERVED[GLTF]
    assert [path for path in downloads(root).rglob("*") if path.name.endswith(".part")] == []


def test_fetch_refuses_a_manifest_that_does_not_say_what_to_expect(tmp_path, web):
    for change in ({"sha256": "unknown"}, {"file": "../../outside.gltf"},
                   {"url": "http://models.example.org/plant.gltf"}):
        root = a_manifest(tmp_path, models=[entry("plant_a", {**source(GLTF), **change}, kind="plant", size=[1, 1, 1],
                                                  tags=[])])
        with pytest.raises(assets.AssetError, match="plant_a"):
            assets.fetch(root)
    assert web.calls == [] and not (tmp_path / "_cache").exists() and not (tmp_path.parent / "outside.gltf").exists()


def test_the_downloader_asks_over_https_and_writes_what_it_is_given(tmp_path, monkeypatch):
    asked = []

    class Answer(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout=None):
        asked.append((request.full_url, dict(request.header_items()), timeout))
        return Answer(b"x" * 3_000_000)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assets.download("https://models.example.org/big.glb", tmp_path / "big.glb.part")
    assert (tmp_path / "big.glb.part").read_bytes() == b"x" * 3_000_000
    ((url, headers, timeout),) = asked
    assert url == "https://models.example.org/big.glb" and headers.get("User-agent") and timeout
    for url in ("http://models.example.org/big.glb", "file:///etc/hostname", "ftp://models.example.org/big.glb"):
        with pytest.raises(assets.AssetError, match="https"):
            assets.download(url, tmp_path / "other.part")
    assert len(asked) == 1 and not (tmp_path / "other.part").exists()


# ----------------------------------------------------------------------------------------------- verify
def made_from(root: Path, name: str, asset: str) -> dict:
    """What a receipt keeps of the entry its asset was prepared from, as prepare_models.py writes it: the values
    the manifest under `root` gives now for everything the prepared form depends on."""
    (item,) = [one for one in manifest_entries(root)[name] if one["id"] == asset]
    return {key: item.get(key) for key in assets.PREPARED_FROM}


def a_receipt(root: Path, name: str, asset: str, files: dict[str, bytes], **changed) -> None:
    """What the preparation leaves for one asset: its files under the prepared folder and the receipt that
    lists them and says what entry of the manifest they were made from. `changed` is what the receipt is to
    say otherwise."""
    prepared = root / "_cache" / "prepared"
    for rel, data in files.items():
        (prepared / rel).parent.mkdir(parents=True, exist_ok=True)
        (prepared / rel).write_bytes(data)
    (prepared / name).mkdir(parents=True, exist_ok=True)
    receipt = {"id": asset, "made_from": made_from(root, name, asset),
               "files": {rel: {"sha256": sha(data), "bytes": len(data)} for rel, data in files.items()}, **changed}
    (prepared / name / f"{asset}.json").write_text(json.dumps(receipt))


def prepare_by_hand(root: Path) -> Path:
    a_receipt(root, "models", "plant_a", {"models/plant_a.blend": b"BLENDER plant"})
    a_receipt(root, "models", "bench_a", {"models/bench_a.blend": b"BLENDER bench"})
    a_receipt(root, "materials", "wood", {"materials/wood/color.jpg": b"jpg", "materials/wood/roughness.jpg": b"jpg2"})
    a_receipt(root, "materials", "glass", {})
    a_receipt(root, "hdris", "street", {"hdris/street.hdr": b"#?RADIANCE"})
    a_receipt(root, "fonts", "sans", {"fonts/sans.ttf": b"font"})
    (root / "_cache" / "prepared" / "avatars").mkdir()
    (root / "_cache" / "prepared" / "avatars" / "f_01.blend").write_bytes(b"BLENDER body")
    stamp = root / "_cache" / "prepared" / PREPARED_STAMP
    stamp.write_text(json.dumps({"models": ["plant_a", "bench_a"], "materials": ["wood", "glass"], "hdris": ["street"],
                                 "fonts": ["sans"], "avatars": ["f_01"], "poses": ["wait_01"]}))
    return stamp


def test_verify_reports_what_is_missing(tmp_path, web):
    root = a_manifest(tmp_path)
    nothing = assets.verify(root)
    assert len([line for line in nothing if "not fetched" in line]) == len(EVERYTHING)    # a line for each file
    for name in EVERYTHING:
        asset, file = name.split("/", 1)
        assert [line for line in nothing if asset in line and file in line and "not fetched" in line], name
    assert len([line for line in nothing if "not prepared" in line]) == 8                 # and one for each asset
    assert not (root / "_cache").exists()                                  # looking makes nothing
    assets.fetch(root)
    unprepared = assets.verify(root)
    assert len(unprepared) == 8 and all("not prepared" in line for line in unprepared)
    assert [line for line in unprepared if line.startswith("materials glass")]            # made in code, and still made
    stamp = prepare_by_hand(root)
    assert assets.verify(root) == [] and is_prepared(load_library(root))
    assert len(web.calls) == len(SERVED)                                   # verify itself fetches nothing

    # Each thing that can go missing or wrong is said, with the asset it belongs to.
    (downloads(root) / "wood" / "Wood_1K-JPG.zip").write_bytes(b"PK another zip")
    (said,) = assets.verify(root)
    assert said.startswith("materials wood") and "Wood_1K-JPG.zip" in said and "sha256" in said
    assert assets.verify(root, only=["plant_a", "glass"]) == []            # the named ones alone
    with pytest.raises(assets.AssetError, match="no_such"):
        assets.verify(root, only=["no_such"])
    assets.fetch(root)
    (root / "_cache" / "prepared" / "models" / "bench_a.blend").unlink()
    (root / "_cache" / "prepared" / "materials" / "wood" / "roughness.jpg").write_bytes(b"jpg3")
    (root / "_cache" / "prepared" / "hdris" / "street.json").unlink()
    (root / "_cache" / "prepared" / "avatars" / "f_01.blend").unlink()
    broken = assets.verify(root)
    assert len(broken) == 4
    assert [line for line in broken if line.startswith("models bench_a") and "bench_a.blend" in line
            and "missing" in line]
    assert [line for line in broken if line.startswith("materials wood") and "roughness.jpg" in line
            and "sha256" in line]
    assert [line for line in broken if line.startswith("hdris street") and "receipt" in line]
    assert [line for line in broken if line.startswith("avatars f_01") and "f_01.blend" in line and "missing" in line]
    stamp.write_text(json.dumps({"models": ["plant_a"]}))                  # a stamp that no longer covers the manifest
    assert len([line for line in assets.verify(root) if "not prepared" in line]) == 7
    stamp.write_text("{not json")
    assert len([line for line in assets.verify(root) if "not prepared" in line]) == 8
    a_receipt(root, "fonts", "sans", {"../../outside.ttf": b"font"})       # a receipt is read as data, not trusted
    stamp.write_text(json.dumps({"fonts": ["sans"]}))
    assert [line for line in assets.verify(root, only=["sans"]) if "outside the prepared folder" in line]


def test_verify_says_when_an_entry_has_changed_since_its_asset_was_prepared(tmp_path, monkeypatch, capsys, web):
    """The stamp lists ids and a receipt's files agree with the receipt whatever the manifest says now, so an
    entry edited after the preparing is found only by what the receipt says it was made from."""
    root = a_manifest(tmp_path)
    assets.fetch(root)
    prepare_by_hand(root)
    assert assets.verify(root) == []
    entries = manifest_entries(root)

    def edited(name: str, asset: str, **changed) -> list[dict]:
        return [{**item, **changed} if item["id"] == asset else item for item in entries[name]]

    # Each thing the prepared form of an asset depends on, changed alone and nothing prepared again.
    for name, asset, changed, word in (("models", "plant_a", {"size": [0.6, 0.4, 1.2]}, "size"),
                                       ("models", "bench_a", {"prepare": {"scale": 0.5}}, "prepare"),
                                       ("materials", "wood", {"tile_m": 3.0}, "tile_m"),
                                       ("materials", "glass", {"kind": "screen"}, "kind"),
                                       ("fonts", "sans", {"file": "fonts/sans_regular.ttf"}, "file"),
                                       ("hdris", "street", {"sources": [source(SKY, file="street.hdr")]}, "sources")):
        write_manifest(root, **{name: edited(name, asset, **changed)})
        said = assets.verify(root)
        (stale,) = [line for line in said if "older entry" in line]
        assert stale.startswith(f"{name} {asset}: ") and stale.endswith(f"its {word}"), stale
        # nothing else, but that a new source is not fetched yet
        assert [line for line in said if line != stale] == (["hdris street: street/street.hdr is not fetched"]
                                                            if word == "sources" else [])
        write_manifest(root, **{name: entries[name]})
        assert assets.verify(root) == []                                   # and as it was, it is intact again
    both = edited("models", "plant_a", size=[0.6, 0.4, 1.2], prepare={"scale": 2.0}, kind="bench")
    write_manifest(root, models=both)
    assert assets.verify(root) == ["models plant_a: prepared from an older entry of the manifest, which differed in "
                                   "its prepare, kind and size"]
    # What a scene reads of an entry and the preparation does not (its tags, a note, whom to credit) leaves it be.
    write_manifest(root, models=edited("models", "plant_a", tags=["leafy"], note="for a corner", author="Another"))
    assert assets.verify(root) == [] and is_prepared(load_library(root))

    # The command says it, and says what to do about it.
    write_manifest(root, materials=edited("materials", "wood", tile_m=3.0))
    monkeypatch.setattr(assets, "ASSETS_DIR", root)
    assert assets.main(["verify"]) == 1
    out = capsys.readouterr().out
    assert "materials wood: prepared from an older entry" in out and "assets prepare" in out
    assert "assets fetch" not in out
    assert assets.main(["verify", "--only", "glass", "plant_a"]) == 0      # the others are as they were prepared
    write_manifest(root, materials=entries["materials"])

    # A receipt that does not say what it was made from, as a preparation before this rule left it: not trusted.
    a_receipt(root, "hdris", "street", {"hdris/street.hdr": b"#?RADIANCE"}, made_from="0" * 64)
    (said,) = assets.verify(root)
    assert said.startswith("hdris street: ") and "does not say" in said and "prepared from" in said
    a_receipt(root, "hdris", "street", {"hdris/street.hdr": b"#?RADIANCE"}, made_from={"by": 1})
    (said,) = assets.verify(root)                                          # nor one that says it of nothing
    assert said.startswith("hdris street: ") and "older entry" in said and "sources" in said


# ---------------------------------------------------------------------------------------------- prepare
MODELS_REPORT = {"models": ["plant_a", "bench_a"], "materials": ["wood", "glass"], "hdris": ["street"],
                 "fonts": ["sans"]}
PEOPLE_REPORT = {"avatars": ["f_01"], "poses": ["wait_01"]}


class Scripts:
    """Stands in for blender.run_script: each script writes the stamp it was told to, as the real ones do."""

    def __init__(self, stamp: Path):
        self.stamp, self.runs, self.failing = stamp, [], set()
        self.reports = {"prepare_models.py": MODELS_REPORT, "prepare_avatars.py": PEOPLE_REPORT}

    def __call__(self, script, args, timeout_s=1800.0):
        name, args = Path(script).name, [str(arg) for arg in args]
        own = Path(args[args.index("--stamp") + 1])
        self.runs.append({"script": name, "args": args, "own": own, "timeout_s": timeout_s,
                          "stamp_then": self.stamp.read_text() if self.stamp.exists() else None})
        assert Path(script).is_file() and own != self.stamp and not own.exists()
        if name in self.failing:
            raise RuntimeError(f"{name} failed in Blender with exit status 1; its log ends:\nValueError: no such bone")
        own.write_text(json.dumps(self.reports[name]))
        return subprocess.CompletedProcess([str(script)], 0, stdout="")

    def named(self) -> list[str]:
        return [run["script"] for run in self.runs]


def test_prepare_writes_the_stamp_last_and_merges_it_under_only(tmp_path, monkeypatch, web):
    root, side = a_manifest(tmp_path / "assets"), tmp_path / "blender_side"
    assets.fetch(root)
    stamp = root / "_cache" / "prepared" / PREPARED_STAMP
    side.mkdir()
    (side / "prepare_models.py").write_text("")
    scripts = Scripts(stamp)
    monkeypatch.setattr(blender, "BLENDER_SIDE", side)
    monkeypatch.setattr(blender, "run_script", scripts)

    # The people's script is not written yet: the things alone, and a stamp without the people's classes.
    assert assets.prepare(root) == MODELS_REPORT == json.loads(stamp.read_text())
    (run,) = scripts.runs
    assert run["script"] == "prepare_models.py" and run["stamp_then"] is None             # no stamp while it ran
    assert run["args"] == ["--manifest", str(root / "manifest"), "--downloads", str(downloads(root)),
                           "--prepared", str(root / "_cache" / "prepared"), "--stamp", str(run["own"])]
    assert run["timeout_s"] >= 3600 and not run["own"].exists()            # a long job; its own stamp is cleared away
    assert is_prepared(load_library(root)) is False

    # Both scripts: the things first, then the people, each with a stamp file of its own; the stamp is their union.
    (side / "prepare_avatars.py").write_text("")
    scripts.runs.clear()
    before = stamp.read_text()
    assert assets.prepare(root, force=True) == {**MODELS_REPORT, **PEOPLE_REPORT} == json.loads(stamp.read_text())
    assert scripts.named() == ["prepare_models.py", "prepare_avatars.py"]
    # written after both, not between
    assert [run["stamp_then"] for run in scripts.runs] == [before, before]
    assert scripts.runs[0]["own"] != scripts.runs[1]["own"] and all(run["args"][-1] == "--force"
                                                                    for run in scripts.runs)
    assert is_prepared(load_library(root)) is True
    assert [path.name for path in stamp.parent.iterdir()] == [PREPARED_STAMP]             # nothing left beside it

    # A script that fails: the error is passed on, and the stamp says what it said before.
    scripts.failing.add("prepare_avatars.py")
    before = stamp.read_text()
    with pytest.raises(RuntimeError, match="no such bone"):
        assets.prepare(root)
    assert stamp.read_text() == before
    scripts.failing.clear()

    # --only: an asset goes to the script of its class, which is told that name alone. What the script then lists,
    # every asset of its classes that is intact, is merged into the stamp class by class: it stands for the classes
    # it lists, and the classes of a script that did not run stay as they were.
    grown = json.loads((root / "manifest" / "models.json").read_text()) + [
        entry("bench_b", source(BENCH), kind="bench", size=[1.2, 0.4, 0.4], tags=[])]
    write_manifest(root, models=grown)
    assets.fetch(root)
    assert is_prepared(load_library(root)) is False                        # the manifest has grown since
    scripts.runs.clear()
    things = {**MODELS_REPORT, "models": ["plant_a", "bench_a", "bench_b"]}
    scripts.reports["prepare_models.py"] = things
    merged = assets.prepare(root, only=["bench_b"])
    assert scripts.named() == ["prepare_models.py"] and scripts.runs[0]["args"][-2:] == ["--only", "bench_b"]
    assert merged == json.loads(stamp.read_text()) == {**things, **PEOPLE_REPORT}         # the people are still there
    assert is_prepared(load_library(root)) is True
    # An asset the script lists no longer, its files gone or its entry changed since, is stamped no longer.
    scripts.reports["prepare_models.py"] = {**things, "models": ["plant_a", "bench_b"]}
    assert assets.prepare(root, only=["plant_a"]) == {**things, "models": ["plant_a", "bench_b"], **PEOPLE_REPORT}
    assert is_prepared(load_library(root)) is False
    scripts.reports["prepare_models.py"] = things
    assert assets.prepare(root, only=["bench_a"]) == merged and is_prepared(load_library(root)) is True
    scripts.runs.clear()
    scripts.reports.update({"prepare_models.py": {"fonts": ["sans"]}, "prepare_avatars.py": {"poses": ["wait_01"]}})
    assets.prepare(root, only=["wait_01", "sans", "f_01"], force=True)
    assert [(run["script"], run["args"][8:]) for run in scripts.runs] == [
        ("prepare_models.py", ["--only", "sans", "--force"]), ("prepare_avatars.py", ["--only", "f_01", "wait_01",
                                                                                      "--force"])]
    assert json.loads(stamp.read_text()) == merged                         # a class a script does not list stays too
    scripts.runs.clear()
    with pytest.raises(assets.AssetError, match="no_such"):
        assets.prepare(root, only=["no_such"])
    assert scripts.runs == []


def test_prepare_starts_nothing_before_its_downloads_are_the_files_the_manifest_names(tmp_path, monkeypatch, web):
    root, side = a_manifest(tmp_path / "assets"), tmp_path / "blender_side"
    side.mkdir()
    (side / "prepare_models.py").write_text("")
    scripts = Scripts(root / "_cache" / "prepared" / PREPARED_STAMP)
    monkeypatch.setattr(blender, "BLENDER_SIDE", side)
    monkeypatch.setattr(blender, "run_script", scripts)
    with pytest.raises(assets.AssetError) as raised:
        assets.prepare(root)                                               # nothing fetched yet
    assert "fetch" in str(raised.value) and "plant_1k.gltf" in str(raised.value) and scripts.runs == []
    assets.fetch(root)
    (downloads(root) / "bench_a" / "bench.glb").write_bytes(b"glTF a bench, tampered with")
    with pytest.raises(assets.AssetError, match="bench.glb"):
        # Blender opens no file that is not the named one
        assets.prepare(root)
    assert scripts.runs == [] and web.calls.count(BENCH) == 1              # and prepare itself fetches nothing
    assets.prepare(root, only=["plant_a"])                                 # what is not asked for is not looked at
    assert scripts.named() == ["prepare_models.py"]
    (side / "prepare_models.py").unlink()
    with pytest.raises(assets.AssetError, match="prepare_models.py"):
        assets.prepare(root, only=["plant_a"])


# ------------------------------------------------------------------------------------------ attribution
def test_the_attribution_names_each_asset_with_its_author_licence_and_source(tmp_path):
    root = a_manifest(tmp_path, models=[
        entry("plant_a", source(GLTF), kind="plant", size=[0.5, 0.4, 1.2], tags=[], author="Rico | Greg",
              changes="its pot recoloured"),
        entry("monitor_b", source(BENCH), kind="monitor", size=[0.2, 0.5, 0.4], tags=[], brand=True,
              redistributable=False,
              licence="CC BY 4.0 (the uploader's page)")])
    text = assets.attribution(root)
    assert text == assets.attribution(root) and text.endswith("\n") and "\n\n\n" not in text
    assert "python -m sgi.synth.realistic.assets attribution" in text      # how it is made, so nobody edits it by hand
    by_class = rows_by_class(text)
    assert {name: [line.split("|")[1].strip() for line in lines] for name, lines in by_class.items()} == {
        "models": ["`plant_a`", "`monitor_b`"], "materials": ["`wood`", "`glass`"], "hdris": ["`street`"],
        # each under its class, in order
        "fonts": ["`sans`"], "avatars": ["`f_01`"], "poses": ["`wait_01`"]}
    (plant, monitor), (wood, glass) = by_class["models"], by_class["materials"]
    assert "Rico \\| Greg" in plant and "CC0 1.0" in plant and "https://models.example.org/a/plant_a" in plant
    assert "Plant A" in plant and "its pot recoloured" in plant
    assert "shows a brand" in monitor and "never drawn" in monitor and "CC BY 4.0" in monitor
    assert "none: made in code" in glass and "https://" not in glass and "https://" in wood
    rows = [line for lines in by_class.values() for line in lines]
    assert all(line.count("|") - line.count("\\|") == rows[0].count("|") - rows[0].count("\\|") for line in rows)
    bare = entry("bench_a", source(BENCH), kind="bench", size=[1, 1, 1], tags=[])
    del bare["page"], bare["title"]                                        # without them: its address, and its id
    assert BENCH in assets.attribution(a_manifest(tmp_path, models=[bare]))
    path = assets.write_attribution(root)
    assert path == root / "ATTRIBUTION.md" and path.read_text(encoding="utf-8") == assets.attribution(root)


# ------------------------------------------------------------------------------------- Blender itself
def a_blender_archive(folder: str = "blender-4.5.14-linux-x64", stray: str | None = None) -> bytes:
    """A tar.xz laid out as Blender's: one folder, with a binary in it that answers --version as Blender does."""
    binary = f"#!{sys.executable}\nprint('Blender 4.5.14 LTS (hash 000000000000 built never)')\n".encode()
    members = {f"{folder}/blender": (binary, 0o755), f"{folder}/4.5/scripts/readme.txt": (b"scripts", 0o644)}
    if stray:
        members[stray] = (b"not Blender's", 0o644)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as tar:
        for name, (data, mode) in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture
def checkout(tmp_path, monkeypatch) -> Path:
    """A checkout without Blender, on the one kind of machine the published archive is for."""
    monkeypatch.setattr(blender, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("SGI_BLENDER", raising=False)
    monkeypatch.setattr(assets.sys, "platform", "linux")
    monkeypatch.setattr(assets.platform, "machine", lambda: "x86_64")
    return tmp_path


def in_tools(checkout: Path) -> list[str]:
    """What lies in .tools/ of a checkout, but for the user folder that every start of a Blender makes there."""
    return sorted(path.name for path in (checkout / ".tools").iterdir() if path.name != "blender-user")


def test_the_blender_that_is_fetched_is_the_published_one():
    assert assets.BLENDER_URL == "https://download.blender.org/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz"
    assert assets.BLENDER_SHA256 == "9ba871ff2ecd36526b77432745980b7e6664ecd0c7ca11c48849073dcfe06da3"
    assert blender.FETCH_COMMAND.endswith("-m sgi.synth.realistic.assets blender")
    published = REPO_ROOT / ".tools" / "blender-4.5.14.sha256"             # Blender's own list, where it was kept
    if published.is_file():
        assert f"{assets.BLENDER_SHA256}  blender-4.5.14-linux-x64.tar.xz" in published.read_text().splitlines()


def test_the_blender_action_fetches_checks_and_unpacks_the_binary(checkout, monkeypatch, web):
    archive = a_blender_archive()
    web.serve[assets.BLENDER_URL] = archive
    monkeypatch.setattr(assets, "BLENDER_SHA256", sha(archive))
    assert not (checkout / ".tools").exists()
    binary = assets.fetch_blender()
    assert binary == checkout / ".tools" / "blender-4.5.14-linux-x64" / "blender" and os.access(binary, os.X_OK)
    assert blender.find_blender() == binary and blender.blender_version(binary) == "4.5.14"       # found, and it runs
    assert (binary.parent / "4.5" / "scripts" / "readme.txt").read_bytes() == b"scripts"
    assert (checkout / ".tools" / ".gitignore").read_text().split() == ["*"]              # Blender is never committed
    assert in_tools(checkout) == [".gitignore", "blender-4.5.14-linux-x64"]               # no archive left behind
    assert web.calls == [assets.BLENDER_URL]
    assert assets.fetch_blender() == binary and len(web.calls) == 1        # it is there: nothing is fetched again
    (binary.parent / "left_by_the_old_one.txt").write_text("old")
    assert assets.fetch_blender(force=True) == binary and len(web.calls) == 2
    assert not (binary.parent / "left_by_the_old_one.txt").exists()        # forced: the folder is the archive's again


def test_a_blender_archive_that_is_not_the_published_one_is_refused_and_nothing_is_unpacked(checkout, monkeypatch, web):
    web.serve[assets.BLENDER_URL] = a_blender_archive()                    # not what the real checksum names
    with pytest.raises(assets.AssetError) as raised:
        assets.fetch_blender()
    assert "sha256" in str(raised.value) and assets.BLENDER_SHA256 in str(raised.value)
    assert in_tools(checkout) == [".gitignore"]                            # no archive, no folder
    # The right checksum, and members that would land beside Blender's folder or outside .tools.
    for stray in ("other/thing.txt", "../outside.txt", "/tmp/sgi_outside.txt",
                  "blender-4.5.14-linux-x64/../../outside.txt"):
        archive = a_blender_archive(stray=stray)
        web.serve[assets.BLENDER_URL] = archive
        monkeypatch.setattr(assets, "BLENDER_SHA256", sha(archive))
        with pytest.raises(assets.AssetError, match="blender-4.5.14-linux-x64"):
            assets.fetch_blender()
        assert in_tools(checkout) == [".gitignore"], stray
    assert not (checkout / "outside.txt").exists() and not (checkout.parent / "outside.txt").exists()
    archive = a_blender_archive(folder="blender-9.9.9-linux-x64")          # the checksum's, and not this Blender
    web.serve[assets.BLENDER_URL] = archive
    monkeypatch.setattr(assets, "BLENDER_SHA256", sha(archive))
    with pytest.raises(assets.AssetError):
        assets.fetch_blender()
    assert in_tools(checkout) == [".gitignore"]


def test_the_blender_action_says_so_on_a_machine_the_archive_is_not_for(checkout, monkeypatch, web):
    monkeypatch.setattr(assets.platform, "machine", lambda: "arm64")
    with pytest.raises(assets.AssetError, match="SGI_BLENDER"):
        assets.fetch_blender()
    assert web.calls == [] and not (checkout / ".tools").exists()


# -------------------------------------------------------------------------------------------- the command
def test_the_command_runs_each_action_and_says_what_is_wrong(tmp_path, monkeypatch, capsys, web):
    root = a_manifest(tmp_path)
    monkeypatch.setattr(assets, "ASSETS_DIR", root)
    assert assets.main(["verify"]) == 1                                    # nothing fetched
    out = capsys.readouterr().out
    assert "plant_a" in out and "not fetched" in out and "assets fetch" in out
    assert assets.main(["fetch", "--only", "plant_a", "sans"]) == 0 and len(web.calls) == 3
    assert assets.main(["fetch"]) == 0 and fetched(root) == EVERYTHING
    assert "5 files" in capsys.readouterr().out
    assert assets.main(["verify"]) == 1 and "assets prepare" in capsys.readouterr().out
    prepare_by_hand(root)
    assert assets.main(["verify"]) == 0 and "all" in capsys.readouterr().out
    assert assets.main(["attribution"]) == 0 and (root / "ATTRIBUTION.md").read_text() == assets.attribution(root)
    assert assets.main(["fetch", "--only", "no_such"]) == 2                # an error is said, not thrown at the reader
    assert capsys.readouterr().err.startswith("error: ") and len(web.calls) == len(SERVED)
    for wrong in (["attribution", "--only", "plant_a"], ["verify", "--force"], ["blender", "--only", "x"], ["tidy"],
                  []):
        with pytest.raises(SystemExit) as raised:
            assets.main(wrong)
        assert raised.value.code == 2
    source_text = Path(assets.__file__).read_text(encoding="utf-8")
    for word in ("blender", "fetch", "prepare", "verify", "attribution", "--only", "--force"):
        assert f'"{word}"' in source_text                                  # as the documentation test looks for them


def test_the_command_prepares_through_blender_and_reports_a_missing_one(tmp_path, monkeypatch, capsys, web):
    root, side = a_manifest(tmp_path / "assets"), tmp_path / "blender_side"
    side.mkdir()
    (side / "prepare_models.py").write_text("")
    scripts = Scripts(root / "_cache" / "prepared" / PREPARED_STAMP)
    monkeypatch.setattr(assets, "ASSETS_DIR", root)
    monkeypatch.setattr(blender, "BLENDER_SIDE", side)
    monkeypatch.setattr(blender, "run_script", scripts)
    assets.fetch(root)
    assert assets.main(["prepare", "--only", "plant_a", "--force"]) == 0
    assert scripts.runs[0]["args"][-3:] == ["--only", "plant_a", "--force"]

    def missing(script, args, timeout_s=0.0):
        raise blender.BlenderMissing("no Blender found. Fetch Blender 4.5 into the checkout with: ...")

    monkeypatch.setattr(blender, "run_script", missing)
    assert assets.main(["prepare"]) == 2 and "no Blender found" in capsys.readouterr().err


# ------------------------------------------------------------------------------------- the Blender side
# A library of the tests' own, prepared by the real script in a real Blender and then loaded through the real
# loaders. A model is given so that each thing the preparation does can be seen in what comes out of it.
needs_blender = pytest.mark.skipif(not blender.binary_available(), reason="no Blender of the series answers here")
BOX_FACES = ((0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (3, 2, 6), (3, 6, 7), (0, 3, 7),
             (0, 7, 4),
             (1, 5, 6), (1, 6, 2))


def a_box(low, high) -> tuple[np.ndarray, np.ndarray]:
    (x0, y0, z0), (x1, y1, z1) = low, high
    corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1),
               (x0, y1, z1)]
    return np.array(corners, dtype=np.float32), np.array(BOX_FACES, dtype=np.uint16)


def a_fine_box(low, high, cuts: int) -> tuple[np.ndarray, np.ndarray]:
    """A box whose every face is a grid of cuts x cuts squares: far more triangles than its shape needs."""
    points, faces, steps = [], [], np.linspace(0.0, 1.0, cuts + 1)
    low, high = np.array(low, dtype=float), np.array(high, dtype=float)
    for axis in range(3):
        for end in (0.0, 1.0):
            start = len(points)
            for u in steps:
                for v in steps:
                    share = np.insert(np.array([u, v]), axis, end)
                    points.append(low + share * (high - low))
            for i in range(cuts):
                for j in range(cuts):
                    a = start + i * (cuts + 1) + j
                    faces += [(a, a + 1, a + cuts + 2), (a, a + cuts + 2, a + cuts + 1)]
    return np.array(points, dtype=np.float32), np.array(faces, dtype=np.uint16)


def a_gltf(meshes: list[dict], picture: bytes | None = None) -> bytes:
    """A glTF file in one piece. Each mesh: its name, its points and faces, where its node stands, and its material
    ("picture" for the embedded picture, "pane" for one that transmits)."""
    blob, views, accessors, out_meshes, nodes = b"", [], [], [], []

    def accessor(data: np.ndarray, kind: str, component: int) -> int:
        nonlocal blob
        blob += b"\0" * (-len(blob) % 4)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": data.nbytes})
        blob += data.tobytes()
        made = {"bufferView": len(views) - 1, "componentType": component, "count": len(data), "type": kind}
        if kind == "VEC3":
            made.update(min=data.min(axis=0).tolist(), max=data.max(axis=0).tolist())
        accessors.append(made)
        return len(accessors) - 1

    for mesh in meshes:
        points, faces = mesh["shape"]
        attributes = {"POSITION": accessor(points, "VEC3", 5126),
                      "TEXCOORD_0": accessor(np.ascontiguousarray(points[:, :2] % 1.0), "VEC2", 5126)}
        out_meshes.append({"name": mesh["name"], "primitives": [{
            "attributes": attributes, "indices": accessor(faces.reshape(-1), "SCALAR", 5123),
            "material": ("picture", "pane").index(mesh["material"])}]})
        nodes.append({"name": mesh["name"], "mesh": len(out_meshes) - 1,
                      "translation": list(mesh.get("at", (0, 0, 0)))})
    coded = "data:image/png;base64," + base64.b64encode(picture).decode() if picture else None
    document = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
        "meshes": out_meshes, "accessors": accessors, "bufferViews": views,
        "buffers": [{"byteLength": len(blob),
                     "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}],
        "materials": [{"name": "picture", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}} if coded else {}},
                      {"name": "pane", "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0}}}],
        "extensionsUsed": ["KHR_materials_transmission"],
        **({"textures": [{"source": 0}], "images": [{"uri": coded, "mimeType": "image/png"}]} if coded else {})}
    return json.dumps(document).encode()


def a_picture(size: tuple[int, int], colour: tuple[int, int, int], kind: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, kind)
    return buffer.getvalue()


def a_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


PROBE = '''
import json, sys
from pathlib import Path
import bpy
import numpy as np
side, prepared, out = sys.argv[sys.argv.index("--") + 1:]
sys.path.insert(0, side)
import library

def said(call):
    try:
        call()
        return None
    except Exception as error:
        return f"{type(error).__name__}: {error}"

def corners(ob):
    points = np.array([ob.matrix_world @ vertex.co for vertex in ob.data.vertices])
    return [points.min(axis=0).round(4).tolist(), points.max(axis=0).round(4).tolist()]

def nodes(tree):
    return sorted(node.bl_idname for node in tree.nodes)

facts = {"before": said(lambda: library.add_model("stand"))}
bpy.ops.wm.read_factory_settings(use_empty=True)
library.open_library(prepared)
stand, again, dense = library.add_model("stand"), library.add_model("stand"), library.add_model("dense")
mesh = stand.data
mesh.calc_loop_triangles()
pane = [slot for slot, material in enumerate(mesh.materials) if "ShaderNodeBsdfGlass" in nodes(material.node_tree)]
front = np.array([mesh.vertices[v].co[:] for face in mesh.polygons if face.material_index in pane
                  for v in face.vertices])
dense.data.calc_loop_triangles()
facts["stand"] = {
    "type": stand.type, "name": stand.name, "parent": stand.parent is not None, "children": len(stand.children),
    "location": list(stand.location), "scale": list(stand.scale), "corners": corners(stand),
    "in_scene": stand.name in bpy.context.scene.objects and again.name in bpy.context.scene.objects,
    "shared": again.data == stand.data and again != stand, "triangles": len(mesh.loop_triangles),
    "materials": {material.name: nodes(material.node_tree) for material in mesh.materials},
    "pane_corners": [front.min(axis=0).round(4).tolist(), front.max(axis=0).round(4).tolist()],
    "images": {image.name: [list(image.size), image.packed_file is not None] for image in bpy.data.images},
    "meshes": len([m for m in bpy.data.meshes if m.users])}
facts["dense"] = {"corners": corners(dense), "triangles": len(dense.data.loop_triangles)}
facts["unknown_model"] = said(lambda: library.add_model("no_such_model"))

def coat(material):
    """Of a model's material: its node kinds, what the base colour is read from, and the numbers of its nodes."""
    tree = material.node_tree
    shader = next(node for node in tree.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled")
    base = shader.inputs["Base Color"]
    return {"nodes": nodes(tree), "from": base.links[0].from_node.bl_idname if base.is_linked else None,
            "colours": [[round(v, 4) for v in node.inputs[socket].default_value[:3]] for node in tree.nodes
                        if node.bl_idname == "ShaderNodeMix" for socket in (6, 7)],
            "heights": sorted([node.operation, round(node.inputs[1].default_value, 4)] for node in tree.nodes
                              if node.bl_idname == "ShaderNodeMath" and node.operation != "MULTIPLY"),
            "coordinates": [link.from_socket.name for node in tree.nodes if node.bl_idname == "ShaderNodeSeparateXYZ"
                            for link in node.inputs[0].links]}

cup = library.add_model("cup")
others = facts["stand"]["images"]                      # the cup brings no picture of its own
facts["cup"] = {"corners": corners(cup),
                "materials": {material.name: coat(material) for material in cup.data.materials},
                "images": len([image for image in bpy.data.images if image.users and image.name not in others])}

def of(material):
    tree, found = material.node_tree, {"name": material.name, "nodes": nodes(material.node_tree)}
    shader = next((node for node in tree.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if shader is not None:
        found["base"] = [round(v, 4) for v in shader.inputs["Base Color"].default_value[:3]]
        found["linked"] = sorted(name for name in ("Base Color", "Roughness", "Metallic", "Normal")
                                 if shader.inputs[name].is_linked)
        found["emission"] = round(shader.inputs["Emission Strength"].default_value, 3)
    found["images"] = sorted((node.image.name, node.projection, node.image.colorspace_settings.name)
                             for node in tree.nodes if node.bl_idname == "ShaderNodeTexImage")
    found["scale"] = [round(v, 4) for node in tree.nodes if node.bl_idname == "ShaderNodeMapping"
                      for v in node.inputs["Scale"].default_value]
    found["factor"] = [round(v, 4) for node in tree.nodes if node.bl_idname == "ShaderNodeMix"
                       for v in node.inputs[7].default_value[:3]]
    found["glass"] = [round(v, 4) for node in tree.nodes if node.bl_idname == "ShaderNodeBsdfGlass"
                      for v in node.inputs["Color"].default_value[:3]]
    return found

facts["materials"] = {name + ("" if tint is None else "+tint"): of(library.material(name, tint))
                      for name in ("wood", "wall_paint", "steel", "plastic_dark", "glass", "screen")
                      for tint in (None, (128, 64, 255))}
facts["same"] = (library.material("wood") == library.material("wood")
                 and library.material("wood", (128, 64, 255)) == library.material("wood", (128, 64, 255))
                 and library.material("wood") != library.material("wood", (128, 64, 255)))
facts["unknown_material"] = said(lambda: library.material("no_such_material"))

def world():
    tree = bpy.context.scene.world.node_tree
    seen = bpy.context.scene.world.cycles_visibility
    return {"strength": round(tree.nodes["Background"].inputs["Strength"].default_value, 3),
            "picture": [node.image.name for node in tree.nodes if node.bl_idname == "ShaderNodeTexEnvironment"],
            "lights": seen.diffuse, "shows": seen.camera and seen.glossy and seen.transmission}

library.set_world("street", 1.5)
facts["world"] = world()
library.set_world(None, 0.5)
facts["sky"] = world()
facts["worlds"] = len(bpy.data.worlds)
facts["unknown_hdri"] = said(lambda: library.set_world("no_such_sky", 1.0))
facts["person"] = said(lambda: library.add_person("no_such_avatar", "no_such_pose"))
bpy.ops.wm.read_factory_settings(use_empty=True)       # a scene thrown away: the next call brings the model again
facts["after_reset"] = corners(library.add_model("stand"))
Path(out).write_text(json.dumps(facts))
'''


@pytest.fixture(scope="module")
def tiny_library(tmp_path_factory) -> dict:
    """A library of two models, six materials, an HDRI and a font, fetched from memory, prepared by the real
    script in Blender and verified; what a probe that runs the loaders in Blender then found."""
    root, picture = tmp_path_factory.mktemp("tiny_library"), a_picture((64, 32), (200, 40, 40))
    files = {
        "stand.gltf": a_gltf([{"name": "body", "shape": a_box((-1, 0, -0.5), (1, 3, 0.5)), "material": "picture"},
                              {"name": "nub", "shape": a_box((-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)), "material": "pane",
                               "at": (0, 1.2, 0.7)},                       # on the front, which glTF has along +Z
                              {"name": "stray", "shape": a_box((-1, -1, -1), (1, 1, 1)), "material": "pane",
                               "at": (10, 0, 0)}], picture),
        "dense.gltf": a_gltf([{"name": "block", "shape": a_fine_box((0, 0, 0), (40, 100, 60), 24),
                               "material": "pane"}]),
        "cup.gltf": a_gltf([{"name": "wall", "shape": a_box((-1, 0, -1), (1, 2, 1)), "material": "picture"}], picture),
        "wood_color.jpg": a_picture((16, 16), (160, 120, 80), "JPEG"),
        "wood_rough.png": a_picture((16, 16), (128,) * 3),
        "wood_height.png": a_picture((16, 16), (64,) * 3),
        "Paint_1K.zip": a_zip({"Paint_Color.png": a_picture((8, 8), (200, 180, 160)), "../Paint_Evil.png": b"no",
                               "Paint_Roughness.png": a_picture((8, 8), (90,) * 3)}),
        "steel.zip": a_zip({"maps/Steel_Color.png": a_picture((8, 8), (150,) * 3), "maps/Steel_Metalness.png":
                            a_picture((8, 8), (255,) * 3)}),
        "street_1k.hdr": b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n" + bytes([128, 128, 128, 129]) * 8,
        "Sans-Regular.ttf": b"\x00\x01\x00\x00 not a font, and never opened as one",
             "OFL.txt": b"SIL OPEN FONT LICENSE"}

    def fetch(name: str, **more) -> dict:
        return {"url": f"https://tiny.example.org/{name}", "sha256": sha(files[name]), "bytes": len(files[name]),
                **more}

    def code(name: str, kind: str, **prepare) -> dict:
        return entry(name, kind=kind, tile_m=1.0, licence="none: made in code", author="this project", prepare=prepare)

    write_manifest(
        root,
        models=[entry("stand", fetch("stand.gltf"), kind="clock", size=[0.7, 1.0, 1.5], tags=[], lettering="none",
                      prepare={"scale": 0.5, "front": "-Y", "drop": ["stray"], "texture_px": 16}),
                entry("dense", fetch("dense.gltf"), kind="box", size=[0.4, 0.6, 1.0], tags=[], lettering="none",
                      prepare={"scale": 0.01, "front": "+X", "max_tris": 1500}),
                entry("cup", fetch("cup.gltf"), kind="cup", size=[0.1, 0.1, 0.1], tags=[], lettering="none",
                      prepare={"scale": 0.05, "front": "-Y", "plain": {"picture": [240, 236, 226]},
                               "band": {"picture": {"colour": [176, 138, 96], "from": 0.03, "to": 0.07}}})],
        materials=[entry("wood", fetch("wood_color.jpg"), fetch("wood_rough.png"), fetch("wood_height.png"), kind="pbr",
                         tile_m=2.0, prepare={"maps": {"color": "wood_color.jpg", "roughness": "wood_rough.png",
                                                       "height": "wood_height.png"}, "bump_m": 0.002}),
                   entry("wall_paint", fetch("Paint_1K.zip"), kind="pbr", tile_m=4.0,
                         prepare={"archive": "Paint_1K.zip", "neutral": True, "colour": [240, 240, 240],
                                  "maps": {"color": "Paint_Color.png", "roughness": "Paint_Roughness.png"}}),
                   entry("steel", fetch("steel.zip"), kind="pbr", tile_m=0.5,
                         prepare={"archive": "steel.zip", "maps": {"color": "maps/Steel_Color.png",
                                                                   "metallic": "maps/Steel_Metalness.png"}}),
                   code("plastic_dark", "plain", colour=[48, 50, 54], roughness=0.4), code("glass", "glass"),
                   code("screen", "screen")],
        hdris=[entry("street", fetch("street_1k.hdr"), prepare={"strength": 0.4})],
        fonts=[entry("sans", fetch("Sans-Regular.ttf"), fetch("OFL.txt"), licence="OFL 1.1 (OFL.txt)")])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assets, "download", lambda url, path: Path(path).write_bytes(files[url.rsplit("/", 1)[1]]))
        patch.setattr(assets, "PREPARE_SCRIPTS", assets.PREPARE_SCRIPTS[:1])      # the things' script alone
        assets.fetch(root)
        def made_at() -> dict[str, int]:
            return {str(path): path.stat().st_mtime_ns for path in (root / "_cache" / "prepared").rglob("*")
                    if path.is_file() and path.name != PREPARED_STAMP}

        first, before = assets.prepare(root), made_at()
        second = assets.prepare(root)                                      # all intact: nothing is made again
        untouched = len(before) == 24 and made_at() == before              # thirteen files and eleven receipts
        problems = assets.verify(root)
    probe, out = root / "probe.py", root / "probe.json"
    probe.write_text(PROBE)
    blender.run_script(probe, [blender.BLENDER_SIDE, root / "_cache" / "prepared", out])
    return {"root": root, "first": first, "second": second, "untouched": untouched, "problems": problems,
            **json.loads(out.read_text())}


@needs_blender
def test_the_things_are_prepared_stamped_and_verified(tiny_library):
    root, prepared = tiny_library["root"], tiny_library["root"] / "_cache" / "prepared"
    listed = {"models": ["stand", "dense", "cup"], "materials": ["wood", "wall_paint", "steel", "plastic_dark", "glass",
                                                                 "screen"],
              "hdris": ["street"], "fonts": ["sans"]}
    assert tiny_library["first"] == tiny_library["second"] == listed
    assert json.loads((prepared / PREPARED_STAMP).read_text()) == listed
    assert tiny_library["untouched"] and tiny_library["problems"] == []    # what is intact is left alone, and verifies
    assert is_prepared(load_library(root))
    made = sorted(str(path.relative_to(prepared)) for path in prepared.rglob("*") if path.is_file()
                  and path.suffix != ".json")
    assert made == ["fonts/sans.OFL.txt", "fonts/sans.ttf", "hdris/street.hdr", "materials/steel/color.png",
                    "materials/steel/metallic.png", "materials/wall_paint/color.png",
                    "materials/wall_paint/roughness.png",
                    "materials/wood/color.jpg", "materials/wood/height.png", "materials/wood/roughness.png",
                    "models/cup.blend", "models/dense.blend", "models/stand.blend"]    # and nothing a zip's own paths
    #                                                                                    could add
    assert load_library(root).fonts["sans"].read_bytes().startswith(b"\x00\x01\x00\x00")
    receipt = json.loads((prepared / "models" / "stand.json").read_text())
    assert receipt["size"] == [0.7, 1.0, 1.5] and receipt["swapped"]["glass"] == ["pane"] and receipt["shrunk"]
    origin = receipt["made_from"]                                          # the entry's own values, for verify to read
    assert {key: origin[key] for key in assets.PREPARED_FROM} == made_from(root, "models", "stand")
    assert origin["prepare"] == {"scale": 0.5, "front": "-Y", "drop": ["stray"], "texture_px": 16}
    assert type(origin["by"]) is int and set(origin) == {"by", *assets.PREPARED_FROM}


@needs_blender
def test_an_asset_whose_entry_has_changed_is_stale_until_it_is_prepared_again(tiny_library, tmp_path):
    root = tmp_path / "assets"
    prepared = root / "_cache" / "prepared"
    shutil.copytree(tiny_library["root"] / "manifest", root / "manifest")
    shutil.copytree(tiny_library["root"] / "_cache", root / "_cache")
    assert assets.verify(root) == []
    materials = json.loads((root / "manifest" / "materials.json").read_text())
    assert materials[0]["id"] == "wood" and materials[0]["tile_m"] == 2.0
    materials[0]["tile_m"] = 3.0                                           # edited, and not prepared again
    write_manifest(root, materials=materials)
    (said,) = assets.verify(root)
    assert said.startswith("materials wood: ") and "older entry" in said and said.endswith("its tile_m")

    def made_at() -> dict[str, int]:
        return {str(path.relative_to(prepared)): path.stat().st_mtime_ns for path in prepared.rglob("*")
                if path.is_file() and path.name != PREPARED_STAMP}

    # Preparing another asset: the real script lists the stale one no longer, and the stamp follows it.
    before = made_at()
    assert "wood" not in assets.prepare(root, only=["sans"])["materials"]
    assert made_at() == before and is_prepared(load_library(root)) is False
    assert assets.verify(root) == ["materials wood: not prepared"]
    # Preparing it: made again from the entry as it is now, and nothing else with it.
    assert "wood" in assets.prepare(root, only=["wood"])["materials"]
    remade = sorted(path for path, at in made_at().items() if before[path] != at)
    assert remade == ["materials/wood.json", "materials/wood/color.jpg", "materials/wood/height.png",
                      "materials/wood/roughness.png"]
    receipt = json.loads((prepared / "materials" / "wood.json").read_text())
    assert receipt["tile_m"] == 3.0 and receipt["made_from"]["tile_m"] == 3.0      # what the Blender side lays it by
    assert assets.verify(root) == [] and is_prepared(load_library(root))


@needs_blender
def test_a_prepared_model_stands_on_the_origin_at_life_size_facing_x(tiny_library):
    stand, dense = tiny_library["stand"], tiny_library["dense"]
    assert tiny_library["before"].startswith("RuntimeError") and "open_library" in tiny_library["before"]
    assert (stand["type"], stand["name"], stand["parent"], stand["children"]) == ("MESH", "stand", False, 0)
    assert stand["location"] == [0, 0, 0] and stand["scale"] == [1, 1, 1] and stand["in_scene"]
    # metres, the middle of its base at the origin
    assert stand["corners"] == [[-0.35, -0.5, 0.0], [0.35, 0.5, 1.5]]
    assert stand["pane_corners"] == [[0.15, -0.1, 0.5], [0.35, 0.1, 0.7]]  # its front along +X; the stray piece is gone
    assert stand["shared"] and stand["meshes"] == 2                        # brought twice, one mesh
    assert stand["triangles"] == 24 and len(stand["materials"]) == 2
    glass = [found for found in stand["materials"].values() if "ShaderNodeBsdfGlass" in found]
    assert len(glass) == 1 and "ShaderNodeBsdfTransparent" in glass[0] and "ShaderNodeLightPath" in glass[0]
    assert list(stand["images"].values()) == [[[16, 8], True]]             # shrunk to texture_px, and in the file
    assert dense["corners"] == [[-0.2, -0.3, 0.0], [0.2, 0.3, 1.0]]        # thinned, and still its size
    assert 12 <= dense["triangles"] <= 1600
    assert tiny_library["after_reset"] == stand["corners"]
    assert "no_such_model" in tiny_library["unknown_model"] and "dense, stand" in tiny_library["unknown_model"]


@needs_blender
def test_a_model_is_painted_over_plainly_with_a_band_of_another_colour_between_two_heights(tiny_library):
    """`plain` paints a material of a model over in one colour, and `band` puts a second colour round it between
    two heights over the model's base, in metres at life size: a white paper cup with a brown sleeve."""
    cup = tiny_library["cup"]
    assert cup["corners"] == [[-0.05, -0.05, 0.0], [0.05, 0.05, 0.1]]
    receipt = json.loads((tiny_library["root"] / "_cache" / "prepared" / "models" / "cup.json").read_text())
    assert receipt["swapped"] == {"glass": [], "plain": ["picture"], "band": ["picture"]}
    assert receipt["materials"] == ["picture_plain"] and cup["images"] == 0    # its picture is gone with what showed it
    (coat,) = cup["materials"].values()
    assert coat["from"] == "ShaderNodeMix" and coat["coordinates"] == ["Object"]    # by the height in the mesh itself,
    #                                                                                 which a placed model's scale keeps
    white, brown = [0.8714, 0.8388, 0.7605], [0.4342, 0.2542, 0.1170]           # the two colours, scene-linear
    assert coat["colours"] == [pytest.approx(white, abs=2e-4), pytest.approx(brown, abs=2e-4)]
    assert coat["heights"] == [["GREATER_THAN", 0.03], ["LESS_THAN", 0.07]]
    # A model without a band is painted as before: the stand's receipt names none.
    stand = json.loads((tiny_library["root"] / "_cache" / "prepared" / "models" / "stand.json").read_text())
    assert stand["swapped"] == {"glass": ["pane"], "plain": [], "band": []}


@needs_blender
def test_a_material_is_projected_at_its_real_size_and_takes_a_tint(tiny_library):
    found = tiny_library["materials"]
    wood, tinted, paint, painted, steel = (found[name] for name in ("wood", "wood+tint", "wall_paint",
                                                                    "wall_paint+tint", "steel"))
    assert wood["scale"] == [0.5, 0.5, 0.5] and paint["scale"] == [0.25, 0.25, 0.25]      # one tile to tile_m metres
    assert [image[1:] for image in wood["images"]] == [["BOX", "sRGB"], ["BOX", "Non-Color"], ["BOX", "Non-Color"]]
    assert wood["linked"] == ["Base Color", "Normal", "Roughness"] and "ShaderNodeBump" in wood["nodes"]
    assert steel["linked"] == ["Base Color", "Metallic"]
    assert wood["factor"] == [] and tinted["factor"] == [0.2159, 0.0513, 1.0]             # its own colour, then shifted
    mean = [0.5776, 0.4564, 0.3515]                                        # of (200, 180, 160), scene-linear
    assert paint["factor"] == pytest.approx([0.8714 / m for m in mean], abs=2e-3)         # neutral: its own colour, 240
    assert painted["factor"] == pytest.approx([0.2159 / mean[0], 0.0513 / mean[1], 1.0 / mean[2]], abs=2e-3)
    plastic, coloured = found["plastic_dark"], found["plastic_dark+tint"]
    assert plastic["base"] == pytest.approx([0.0296, 0.0319, 0.0369], abs=2e-4) and plastic["images"] == []
    assert coloured["base"] == pytest.approx([0.2159, 0.0513, 1.0], abs=2e-4)
    assert found["glass"]["glass"] and found["glass+tint"]["glass"][0] < found["glass"]["glass"][0]
    assert found["screen"]["emission"] > 0 and found["screen"]["base"][0] < 0.05          # dark, and dimly lit
    assert tiny_library["same"] is True
    assert "no_such_material" in tiny_library["unknown_material"] and "wall_paint" in tiny_library["unknown_material"]


@needs_blender
def test_the_world_shows_an_hdri_or_a_plain_sky_and_lights_nothing(tiny_library):
    # An HDRI is shown at the strength asked for times its own `strength`, here 0.4: one sky is brighter than another.
    assert tiny_library["world"] == {"strength": 0.6, "picture": ["street.hdr"], "lights": False, "shows": True}
    assert tiny_library["sky"] == {"strength": 0.5, "picture": [], "lights": False, "shows": True}
    assert tiny_library["worlds"] == 1                                     # the one before is not left behind
    assert "no_such_sky" in tiny_library["unknown_hdri"]
    assert tiny_library["person"] is not None                              # no such person: said, whoever serves people


@needs_blender
def test_what_cannot_be_prepared_is_said_and_stamps_nothing(tiny_library, tmp_path):
    root = tmp_path / "assets"
    shutil.copytree(tiny_library["root"] / "manifest", root / "manifest")
    shutil.copytree(tiny_library["root"] / "_cache" / "downloads", root / "_cache" / "downloads")
    models = json.loads((root / "manifest" / "models.json").read_text())
    models[0]["size"] = [0.7, 1.0, 1.6]                                    # not what it measures
    models[1]["prepare"]["front"] = "up"
    models[2]["prepare"]["band"]["picture"]["to"] = 0.02                   # a band that ends below where it begins
    write_manifest(root, models=models)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assets, "PREPARE_SCRIPTS", assets.PREPARE_SCRIPTS[:1])
        with pytest.raises(RuntimeError) as raised:
            assets.prepare(root, only=["stand", "dense", "cup", "street"])
    said = str(raised.value)
    assert "models stand" in said and "[0.7, 1.0, 1.5]" in said and "models dense" in said and "front" in said
    assert "models cup" in said and "prepare.band runs from 0.03 to 0.02" in said
    assert not (root / "_cache" / "prepared" / PREPARED_STAMP).exists()    # the command's stamp is not written
    assert not list((root / "_cache" / "prepared").rglob("*.blend"))       # and nothing half-made is left as made
    assert (root / "_cache" / "prepared" / "hdris" / "street.hdr").is_file()              # what could be made was


# ------------------------------------------------------------------------------------- the two Pythons
def imported(path: Path) -> set[str]:
    """The top-level names of every module a file imports, wherever in it."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_both_sides_name_the_same_things_a_prepared_asset_depends_on():
    """prepare_models.py writes them into each receipt and assets.py compares them with the manifest, and neither
    can import the other: a name that only one of them had would make every asset stale, or none."""
    path = SRC / "synth" / "realistic" / "blender_side" / "prepare_models.py"
    (named,) = [ast.literal_eval(node.value) for node in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(node, ast.Assign) and [getattr(target, "id", None) for target in node.targets]
                == ["READ_OF_AN_ENTRY"]]
    assert named == assets.PREPARED_FROM and len(set(named)) == len(named) >= 6


def test_sgi_never_imports_bpy_and_the_blender_side_never_imports_sgi():
    side = sorted((SRC / "synth" / "realistic" / "blender_side").glob("*.py"))
    assert {path.name for path in side} >= {"__init__.py", "library.py", "prepare_models.py"}
    for path in side:                                                      # Blender's Python has none of these
        assert imported(path) & {"sgi", "pydantic", "PIL"} == set(), path.name
    for path in sorted(SRC.rglob("*.py")):
        if path not in side:                                               # and the project's Python has no Blender
            assert imported(path) & {"bpy", "bmesh", "mathutils"} == set(), str(path.relative_to(SRC))
