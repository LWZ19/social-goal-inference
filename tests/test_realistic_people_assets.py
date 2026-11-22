"""The people of the realistic set: the avatars and poses the manifest lists, and the scripts that prepare and
place them.

The manifest is checked as it is committed, with no file of any asset; where the people are prepared, it is
also compared with what the preparation measured. The parts of the preparing script that need no Blender (its
arguments, what a run has to make, the recolouring of a garment, how far an avatar may be widened) are loaded
from its file and called here. What Blender makes is checked in Blender: the tests that start it are skipped
without the binary (the counting of axis samples is one: it is done on bodies of boxes), and those that place
a prepared person are skipped until the assets are prepared.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from sgi.config import Thresholds
from sgi.gt import derive
from sgi.synth.realistic import blender
from sgi.synth.realistic.library import ASSETS_DIR, AVATAR_TAGS, POSE_TAGS, load_library, manifest_entries
from sgi.synth.render import HEIGHTS_M

REACH_LIMIT_M = 0.45                               # a pose that reaches further is not used
STAFFED_KINDS = ("stall", "desk", "ticket_counter", "shop")    # the kinds staff are tagged for: staff_<kind>
SECOND_STALL = "staff_shop"                        # the vendor's tag: the second stall's own, shared with no place
OUT_OF_PLACE = ("Pilot_Female_02", "Security_Male_01")     # uniforms of trades no place of a scene has: cast nowhere
SET_ASIDE = ("talk_thumb_back", "thoughtful_b")    # poses with no tag, which nobody is cast in: one reads as a fist,
#                                                    the other is the second take of a pose that came up too often
ACTIVITIES = ("waiting", "talking", "serving")     # what a person is doing; "phone" and "cup" say what they hold
DRESS = ("casual", "business", "uniform")          # one each; "uniform" is this library's own: staff clothes only
BUILDS = (0.92, 1.15)                              # what a person's build is scaled between
NEEDED_SAMPLES = 3                                 # of the nine on a person's axis, for the scene check to see them
BODY_BEHIND_AXIS_M = 0.26                          # heels, seat and hair of a person standing at ease, at the most
HANDS_BEFORE_WAIST_M = 0.31                        # hands folded before the waist; held out, they are further ahead
FEET_TOP_M = 0.30                                  # at an avatar's natural size: lower than this are feet and ankles
SOLE_M, ANKLE_M = 0.06, (0.10, 0.20)               # up to where a sole is, and from where to where the ankle over it
TOE_BEFORE_ANKLE_M = 0.05                          # a foot reaches further than this ahead of its ankle
THING_BEFORE_AXIS_M = 0.15                         # a phone that is looked at, a cup that is held: this far ahead
SOURCE = "https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/"
PREPARE = blender.BLENDER_SIDE / "prepare_avatars.py"
PEOPLE = blender.BLENDER_SIDE / "people_library.py"

needs_blender = pytest.mark.skipif(not blender.binary_available(), reason="no Blender of the series here")
needs_assets = pytest.mark.skipif(not blender.available(), reason="the assets of the realistic set are not prepared")


@pytest.fixture(scope="module")
def library():
    return load_library()


@pytest.fixture(scope="module")
def entries():
    return manifest_entries()


@pytest.fixture(scope="module")
def prepare():
    """The preparing script as a module. Outside Blender only its parts that need no bpy can be called."""
    spec = importlib.util.spec_from_file_location("prepare_avatars_under_test", PREPARE)
    module = importlib.util.module_from_spec(spec)
    before = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = before                       # the script puts its folder first, for Blender; not for the tests
    return module


def usable(library, tag: str) -> list:
    return [pose for pose in library.poses.values() if tag in pose.tags and pose.reach <= REACH_LIMIT_M]


# ------------------------------------------------------------------------------------------------ the manifest
def test_the_library_has_the_people_the_scenes_need(library):
    avatars = list(library.avatars.values())
    assert len(avatars) >= 28
    women, men = ([a for a in avatars if a.sex == sex] for sex in ("female", "male"))
    assert len(women) + len(men) == len(avatars) and abs(len(women) - len(men)) <= 4      # both sexes about evenly
    for sex in (women, men):
        assert len([a for a in sex if "red" in a.tags]) >= 2
    for kind in STAFFED_KINDS:
        staff = [a for a in avatars if f"staff_{kind}" in a.tags]
        assert len(staff) >= 3 and {a.sex for a in staff} == {"female", "male"}, kind
        # A visitor may wear any of them that is not in uniform, and the staff are cast after the visitors. So the
        # visitors of a crowded scene can wear all of a kind's, and the dressing then stops with an error that
        # says so. With five or more to a kind that takes a rare draw (about two dressings in 100,000); only a
        # stall has someone in a uniform, whom no visitor wears, and so always has its staff.
        assert len(staff) >= 5 or any("uniform" in a.tags for a in staff), kind
    assert len(library.poses) >= 20
    # within the reach the scenes allow
    assert len(usable(library, "waiting")) >= 5 and len(usable(library, "talking")) >= 4
    assert len(usable(library, "serving")) >= 3 and len(usable(library, "phone")) >= 2
    assert len(usable(library, "cup")) >= 1
    # a scene whose target is the person in red casts everyone else from the people in neither red nor pink
    plain = [a for a in avatars if not a.tags & {"red", "pinkish", "uniform"}]
    assert len(plain) >= 16 and {a.sex for a in plain} == {"female", "male"}


def test_whoever_stands_behind_a_desk_or_in_a_booth_wears_ordinary_office_or_shop_dress(library):
    """Not the uniform of another trade: a pilot behind a help desk reads as a ship's officer, a guard with a duty
    belt as an officer with handcuffs. Those two are tagged for no place, and so are cast nowhere; the chef stays
    at the coffee stall. The vendor of a second stall, which sells fruit, flowers or sweets, has a tag of their
    own and wears everyday dress: not kitchen whites, and not the face that stands behind every other counter."""
    avatars = library.avatars
    for kind in ("desk", "ticket_counter"):
        staff = [a for a in avatars.values() if f"staff_{kind}" in a.tags]
        assert staff and all("uniform" not in a.tags and a.tags & {"business", "casual"} for a in staff), kind
        assert not [a.id for a in staff if a.id.startswith(("Pilot", "Security", "Chef"))], kind
    for name in OUT_OF_PLACE:
        assert "uniform" in avatars[name].tags and not [t for t in avatars[name].tags if t.startswith("staff_")]
    assert {t for t in avatars["Chef_Female_01"].tags if t.startswith("staff_")} == {"staff_stall"}
    vendors = [a for a in avatars.values() if SECOND_STALL in a.tags]
    assert 8 <= len(vendors) <= 9 and {a.sex for a in vendors} == {"female", "male"}
    assert all(a.tags & {"casual", "business"} and "uniform" not in a.tags for a in vendors)
    # Nobody is tagged for a second stall and for another place, so that the vendor of a scene is never the one
    # face that serves in most of the others; and the man in the white shirt has one place, not three.
    assert all(len([t for t in a.tags if t.startswith("staff_")]) == 1 for a in vendors)
    assert [t for t in avatars["Business_Male_06"].tags if t.startswith("staff_")] == ["staff_ticket_counter"]
    # No avatar has more than two staff tags: the staff of 21 scenes are not three or four people.
    assert max(len([t for t in a.tags if t.startswith("staff_")]) for a in avatars.values()) <= 2


def test_a_pose_that_reads_as_a_fist_is_not_for_talking(library):
    """Seen at the distance of a scene: a hand at the chin with the other arm across the body, and a closed hand
    raised to the shoulder. Two people in a conversation who face each other so look like two squaring up. The
    first is still a way to wait, in one of its two takes: with both, a hand at the chin turned up in every
    other waiting line. The second is tagged for nothing. Nor does anything for talking hold a phone: whoever
    does is not in the conversation."""
    poses = library.poses
    assert poses["thoughtful_a"].tags == frozenset({"waiting"})
    assert all(poses[name].tags == frozenset() for name in SET_ASIDE)
    assert not [pose.id for pose in poses.values() if {"talking", "phone"} <= pose.tags]
    talking = usable(library, "talking")
    assert len(talking) >= 10 and len([pose for pose in talking if not pose.tags & {"cup", "serving"}]) >= 4


def test_every_pose_has_a_measured_reach_and_at_least_one_activity_tag(library, entries):
    assert library.poses
    for pose in library.poses.values():
        assert 0.2 < pose.reach < 0.8, pose.id                              # a body is 0.2 m at the least
        assert pose.reach != round(pose.reach, 1), f"{pose.id}: {pose.reach} looks typed, not measured"
        assert bool(pose.tags & set(ACTIVITIES)) == (pose.id not in SET_ASIDE), pose.id
        assert pose.tags <= set(POSE_TAGS), pose.id
        assert pose.sex in (None, "female", "male"), pose.id
    for entry in entries["poses"]:                                          # what it holds is what its tag says
        made_for = entry["prepare"]["prop"]
        assert made_for in (None, "phone", "cup"), entry["id"]
        assert {tag for tag in ("phone", "cup") if tag in entry["tags"]} == ({made_for} - {None}), entry["id"]


def test_no_avatar_tagged_red_is_also_tagged_for_staff(library):
    red = [avatar for avatar in library.avatars.values() if "red" in avatar.tags]
    assert red
    for avatar in red:
        assert not [tag for tag in avatar.tags if tag.startswith("staff_")], avatar.id
        assert "pinkish" not in avatar.tags and "uniform" not in avatar.tags, avatar.id


def test_every_avatar_has_one_sex_one_dress_and_a_measured_height(library):
    assert library.avatars
    for avatar in library.avatars.values():
        assert avatar.sex in ("female", "male"), avatar.id
        assert avatar.tags & {"female", "male"} == {avatar.sex}, avatar.id
        assert len(avatar.tags & set(DRESS)) == 1, avatar.id
        assert avatar.tags <= set(AVATAR_TAGS) | {"uniform"}, avatar.id
        assert 1.5 < avatar.natural_height < 2.0, avatar.id
        assert avatar.natural_height != round(avatar.natural_height, 2), f"{avatar.id}: a typed height"
        if "uniform" in avatar.tags:                                        # the clothes of a trade: at work at a
            staffs = [tag for tag in avatar.tags if tag.startswith("staff_")]   # place of that trade, or cast nowhere
            assert bool(staffs) == (avatar.id not in OUT_OF_PLACE), avatar.id


def test_every_person_and_pose_names_its_sources_licence_and_checksums(entries):
    assert entries["avatars"] and entries["poses"]
    for entry in entries["avatars"] + entries["poses"]:
        name = entry["id"]
        assert re.fullmatch(r"[A-Za-z0-9_]+", name), name                   # it is a folder and a file name
        assert entry["sources"], name
        for source in entry["sources"]:
            assert source["url"].startswith(SOURCE), name
            assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"]), name
            assert isinstance(source["bytes"], int) and source["bytes"] > 0, name
            assert source["file"] == source["url"].rsplit("/", 1)[1], name   # the default name, written out
        files = [source["file"] for source in entry["sources"]]
        assert len(set(files)) == len(files), name
        assert entry["licence"].startswith("MIT") and "Microsoft" in entry["licence"], name
        assert entry["author"] and entry["redistributable"] is True and entry["brand"] is False, name
        assert isinstance(entry["prepare"], dict), name
    for entry in entries["avatars"]:
        files = [source["file"] for source in entry["sources"]]
        assert entry["prepare"]["fbx"] in files and entry["prepare"]["texture_px"] == 1024, entry["id"]
        assert entry["looks"], entry["id"]                                  # what was seen, in words
    for entry in entries["poses"]:
        files = [source["file"] for source in entry["sources"]]
        assert files == [entry["prepare"]["animation"]], entry["id"]
        assert isinstance(entry["prepare"]["frame"], int) and entry["prepare"]["frame"] >= 1, entry["id"]


def test_a_recoloured_avatar_is_an_id_of_its_own_and_records_the_recolouring(entries):
    recoloured = [entry for entry in entries["avatars"] if entry["prepare"].get("recolour")]
    assert len(recoloured) >= 2
    for entry in recoloured:
        change = entry["prepare"]["recolour"]
        model = Path(entry["prepare"]["fbx"]).stem
        assert entry["id"] == f"{model}_red" and "red" in entry["tags"], entry["id"]
        assert change["texture"] in [source["file"] for source in entry["sources"]], entry["id"]
        low, high = change["hue_from_deg"]
        assert 0 <= low < high <= 360 and not low <= 20 <= high, entry["id"]   # the band is not the hue of skin
    # the original is not also in the library: the two would be one face in two tops, and casting goes by id
    models = [Path(entry["prepare"]["fbx"]).stem for entry in entries["avatars"]]
    assert len(set(models)) == len(models)
    for entry in entries["avatars"]:
        if not entry["prepare"].get("recolour"):
            assert entry["id"] == Path(entry["prepare"]["fbx"]).stem, entry["id"]


def measurements() -> dict[str, dict]:
    """What the preparation measured, by avatar: the file it writes beside each prepared avatar."""
    folder = ASSETS_DIR / "_cache" / "prepared" / "avatars"
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))}


def test_the_preparing_script_measures_by_the_scene_check_at_the_sizes_the_scenes_use(prepare):
    """The script cannot import sgi, so it states the check's rule and the figures' sizes again."""
    assert prepare.AXIS_HEIGHTS_M == derive._VISIBILITY_HEIGHTS_M
    assert prepare.AXIS_TOLERANCE_M == Thresholds().visibility_depth_tolerance_m
    assert prepare.AXIS_NEEDED == Thresholds().visibility_min_samples == NEEDED_SAMPLES
    assert prepare.AXIS_KEPT == NEEDED_SAMPLES + 1                          # one to spare
    heights = {height for height, _build in prepare.SIZES}
    assert heights == {min(HEIGHTS_M), max(HEIGHTS_M)}                      # the shortest and the tallest figure
    for height in heights:                                                  # a build of 0.92 to 1.15
        builds = [build for tall, build in prepare.SIZES if tall == height]
        assert {BUILDS[0], 1.0, BUILDS[1]} <= set(builds) and builds == sorted(set(builds)), height
        assert BUILDS[0] <= builds[0] and builds[-1] <= BUILDS[1], height
    assert len({prepare.size_name(*size) for size in prepare.SIZES}) == len(prepare.SIZES)
    assert prepare.size_name(1.58, 1.0) == "1.58x1.00"
    assert prepare.FEET_TOP_M == FEET_TOP_M


def test_every_pose_keeps_the_axis_the_scene_check_samples_inside_the_body(entries, prepare):
    """By what the manifest records of the measurement: of the check's nine samples on a person's axis, the fewest
    that any avatar in the pose shows from any side, at each size the scenes give a person ("<height>x<build>")."""
    assert entries["poses"]
    sizes = [prepare.size_name(*size) for size in prepare.SIZES]
    unwidened = [prepare.size_name(height, build) for height, build in prepare.SIZES if build <= 1.0]
    for entry in entries["poses"]:
        told, seen = entry["measured"], entry["measured"]["axis_seen_at"]
        assert told["over_avatars"] >= 28, entry["id"]                      # measured on the whole cast
        assert list(seen) == sizes, entry["id"]
        # never lost, however wide; and at a build of 1.0 or less with two samples to spare
        assert told["axis_seen"] == min(seen.values()) >= NEEDED_SAMPLES, entry["id"]
        assert min(seen[size] for size in unwidened) >= NEEDED_SAMPLES + 2, entry["id"]
        if "serving" in entry["tags"]:
            # behind a counter with a wall close behind: nothing reaches further back than a body itself is, and
            # nothing is held out before it; hands folded before the waist are the furthest ahead
            assert told["back"] <= BODY_BEHIND_AXIS_M and told["front_body"] <= HANDS_BEFORE_WAIST_M, entry["id"]
        for key in ("back", "front", "front_feet", "front_body", "side"):   # each is one way of reaching
            assert 0 < told[key] <= entry["reach"], (entry["id"], key)
        assert told["front"] == max(told["front_feet"], told["front_body"]), entry["id"]


def test_every_avatar_says_how_far_it_may_be_widened(entries, prepare):
    """`max_width_scale` bounds height / natural_height x build: within it an avatar shows one axis sample more
    than the scene check needs from every side in every pose. It is one of the measured sizes of the tallest
    figure; every avatar keeps it at its own proportions, and the poses' records agree with it."""
    assert entries["avatars"] and entries["poses"]
    tallest = max(height for height, _build in prepare.SIZES)
    builds = sorted(build for height, build in prepare.SIZES if height == tallest)
    within = {build: True for build in builds}                              # every avatar within its bound at it
    for entry in entries["avatars"]:
        widest, scale = entry["max_width_scale"], tallest / entry["natural_height"]
        at = [build for build in builds if widest == pytest.approx(scale * build, abs=5e-4)]
        assert len(at) == 1 and at[0] >= 1.0, entry["id"]
        for build in builds:
            within[build] = within[build] and build <= at[0]
    for entry in entries["poses"]:
        for build in builds:
            if within[build]:
                assert entry["measured"]["axis_seen_at"][prepare.size_name(tallest, build)] >= prepare.AXIS_KEPT, (
                    entry["id"], build)


def test_the_manifest_states_what_was_measured_on_the_prepared_people(library, entries, prepare):
    """Where the people are prepared from this manifest; elsewhere there is nothing to compare with."""
    measured = measurements()
    marks = {entry["id"]: prepare.pose_made_from(entry) for entry in entries["poses"]}
    other = [entry["id"] for entry in entries["avatars"]
             if measured.get(entry["id"], {}).get("made_from") != prepare.made_from(entry, marks)]
    if other:
        pytest.skip(f"{len(other)} of {len(entries['avatars'])} avatars are not prepared here from this manifest")
    for avatar in library.avatars.values():
        assert avatar.natural_height == pytest.approx(measured[avatar.id]["natural_height"], abs=5e-4), avatar.id
        assert set(library.poses) <= set(measured[avatar.id]["poses"]), avatar.id
        for pose, row in measured[avatar.id]["poses"].items():             # each pose was turned to face ahead: the
            turn = row["turn_deg"]                                          # mean heading of pelvis and chest is 0
            assert abs(turn["pelvis"] + turn["chest"]) <= 0.05 and abs(turn["chest"] - turn["pelvis"]) < 60.0, (
                avatar.id, pose)
    for entry in entries["avatars"]:                                        # how far the hanging arms are from the axis
        note = measured[entry["id"]]
        sides = [row["side"] for row in note["poses"].values()]
        assert min(sides) - 1e-3 <= entry["measured"]["half_width"] <= max(sides) + 1e-3, entry["id"]
        # how far it may be widened: as noted, and the note as the counts of its poses give it
        assert entry["max_width_scale"] == pytest.approx(note["max_width_scale"], abs=5e-4), entry["id"]
        assert note["max_width_scale"] == prepare.max_width_scale(note["poses"], note["natural_height"]), entry["id"]
    for entry in entries["poses"]:
        rows = [measured[avatar.id]["poses"][entry["id"]] for avatar in library.avatars.values()
                if entry.get("sex") in (None, avatar.sex)]                  # over everyone the pose is for
        told = entry["measured"]
        assert told["over_avatars"] == len(rows), entry["id"]
        assert entry["reach"] == pytest.approx(max(row["reach"] for row in rows), abs=5e-4), entry["id"]
        for key in ("back", "front", "front_feet", "front_body", "side"):
            assert told[key] == pytest.approx(max(row[key] for row in rows), abs=5e-4), (entry["id"], key)
        assert told["axis_seen_at"] == {size: min(row["axis_seen_at"][size] for row in rows)
                                        for size in rows[0]["axis_seen_at"]}, entry["id"]


# ------------------------------------------------------------------- the preparing script, where it needs no Blender
def test_the_blender_side_people_scripts_import_nothing_of_the_project():
    for script in (PREPARE, PEOPLE):
        text = script.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(?:import|from)\s+(?:sgi|pydantic|PIL)\b", text, re.MULTILINE), script.name


def test_the_arguments_are_read_from_behind_the_dashes(prepare):
    given = prepare.arguments(["blender", "-b", "-P", "prepare_avatars.py", "--", "--manifest", "m", "--downloads",
                               "d", "--prepared", "p", "--stamp", "s.json", "--only", "a", "b", "--force"])
    assert (given.manifest, given.downloads, given.prepared, given.stamp) == (Path("m"), Path("d"), Path("p"),
                                                                              Path("s.json"))
    assert given.only == ["a", "b"] and given.force is True
    plain = prepare.arguments(["blender", "--", "--manifest", "m", "--downloads", "d", "--prepared", "p", "--stamp",
                               "s"])
    assert plain.only is None and plain.force is False
    with pytest.raises(SystemExit):
        prepare.arguments(["blender", "--", "--manifest", "m"])            # the folders are not optional


def test_a_run_makes_what_is_missing_and_what_is_named_with_force(prepare):
    avatars, poses = ["ann", "bob"], ["wait", "talk"]
    todo = prepare.to_make
    # nothing named: whatever is not intact; with force, everything
    assert todo(avatars, poses, intact={"ann", "wait"}, only=None, force=False) == (["bob"], ["talk"])
    assert todo(avatars, poses, intact={"ann", "bob", "wait", "talk"}, only=None, force=False) == ([], [])
    assert todo(avatars, poses, intact={"ann", "bob", "wait", "talk"}, only=None, force=True) == (avatars, poses)
    # named: only those are forced, and an id of another class (a model's, say) is passed over
    assert todo(avatars, poses, intact={"ann", "bob", "wait", "talk"}, only=["bob", "plant_a"], force=True) == (
        ["bob"], [])
    assert todo(avatars, poses, intact={"wait", "talk"}, only=["bob"], force=False) == (["bob"], [])
    assert todo(avatars, poses, intact={"ann", "bob", "wait", "talk"}, only=["bob"], force=False) == ([], [])
    # a pose is baked into every avatar's file: a named avatar brings the poses it lacks, a forced pose every avatar
    assert todo(avatars, poses, intact={"ann", "wait"}, only=["bob"], force=False) == (["bob"], ["talk"])
    assert todo(avatars, poses, intact={"ann", "bob", "wait", "talk"}, only=["talk"], force=True) == (avatars,
                                                                                                    ["talk"])
    assert todo(avatars, poses, intact={"ann", "bob", "wait", "talk"}, only=["plant_a"], force=True) == ([], [])


def test_what_an_avatar_is_made_from_changes_with_its_files_its_parameters_and_its_poses(prepare):
    avatar = {"id": "ann", "sex": "female", "natural_height": 1.7, "tags": ["female"], "licence": "MIT",
              "sources": [{"url": "u", "sha256": "a" * 64, "bytes": 3, "file": "ann.fbx"}],
              "prepare": {"fbx": "ann.fbx", "texture_px": 1024}}
    poses = {"wait": "1" * 64, "talk": "2" * 64}
    same = prepare.made_from(avatar, poses)
    assert re.fullmatch(r"[0-9a-f]{64}", same)
    # what is measured or judged afterwards does not ask for a new preparation
    judged = {**avatar, "natural_height": 1.8, "tags": ["female", "red"], "licence": "x"}
    assert prepare.made_from(judged, poses) == same
    changed = [{**avatar, "sources": [{**avatar["sources"][0], "sha256": "b" * 64}]},
               {**avatar, "prepare": {**avatar["prepare"], "texture_px": 2048}},
               {**avatar, "prepare": {**avatar["prepare"], "recolour": {"texture": "t.tga"}}}]
    for other in changed:
        assert prepare.made_from(other, poses) != same
    assert prepare.made_from(avatar, {**poses, "talk": "3" * 64}) != same
    assert prepare.made_from(avatar, {"wait": "1" * 64}) != same


def test_a_pose_is_noted_by_the_larger_of_itself_with_and_without_its_thing(prepare):
    counts = {prepare.size_name(*size): 9 for size in prepare.SIZES}
    one = prepare.size_name(*prepare.SIZES[0])
    bare = {"height": 1.75, "reach": 0.30, "back": 0.20, "front": 0.25, "front_feet": 0.25, "front_body": 0.18,
            "side": 0.30, "axis_seen_at": dict(counts)}
    held = {**bare, "height": 1.76, "reach": 0.36, "back": 0.19, "front": 0.36, "front_body": 0.36,
            "axis_seen_at": {**counts, one: 5}}                             # a phone held out, which hides a sample
    told = prepare.noted({"": bare, "phone": held})
    assert told["height"] == 1.75                                           # the body's own
    assert told["reach"] == 0.36 and told["reach_with"] == {"": 0.30, "phone": 0.36}
    assert [told[key] for key in ("back", "front", "front_feet", "front_body", "side")] == [0.20, 0.36, 0.25, 0.36,
                                                                                           0.30]
    assert told["axis_seen_at"] == {**counts, one: 5} and told["axis_seen"] == 5
    alone = prepare.noted({"": bare})                                       # a pose that holds nothing
    assert alone["reach_with"] == {"": 0.30} and alone["front_body"] == 0.18 and alone["axis_seen"] == 9
    assert list(told) == ["height", "reach", "reach_with", "back", "front", "front_feet", "front_body", "side",
                          "axis_seen", "axis_seen_at"]


def test_an_avatar_may_be_widened_as_far_as_every_pose_keeps_a_sample_to_spare(prepare):
    tallest = max(height for height, _build in prepare.SIZES)
    builds = sorted(build for height, build in prepare.SIZES if height == tallest)
    assert len(builds) >= 4
    natural, kept = 1.8, prepare.AXIS_KEPT

    def rows(shown: dict[float, int], small: int = 9) -> dict[str, dict]:
        """Two poses: one that shows all nine samples at every size, and one that shows `shown[build]` of them
        at the tallest figure's builds (nine where none is given) and `small` at the shortest figure's."""
        counts = {prepare.size_name(*size): 9 for size in prepare.SIZES}
        other = {prepare.size_name(height, build): shown.get(build, 9) if height == tallest else small
                 for height, build in prepare.SIZES}
        return {"wait": {"axis_seen_at": counts}, "talk": {"axis_seen_at": other}}

    widest = prepare.max_width_scale
    assert widest(rows({}), natural) == pytest.approx(tallest / natural * builds[-1])        # all the way
    assert widest(rows({builds[-1]: kept}), natural) == pytest.approx(tallest / natural * builds[-1])
    # one pose with nothing to spare at a build: the build before it is the last
    assert widest(rows({builds[-1]: kept - 1}), natural) == pytest.approx(tallest / natural * builds[-2])
    assert widest(rows({builds[2]: kept - 1, builds[3]: 0}), natural) == pytest.approx(tallest / natural * builds[1])
    # ... and a build beyond it at which the pose shows more again is beyond it all the same
    assert widest(rows({builds[1]: kept - 1}), natural) == pytest.approx(tallest / natural * builds[0])
    assert widest(rows({builds[0]: kept - 1}), natural) == 0.0                               # not even the narrowest
    # it is read at the tallest figure; a taller avatar is scaled less to that height
    assert widest(rows({}, small=0), natural) == pytest.approx(tallest / natural * builds[-1])
    assert widest(rows({}), 2 * natural) == pytest.approx(tallest / natural * builds[-1] / 2)


GREEN_TO_RED = {"hue_from_deg": [90.0, 170.0], "hue_to_deg": 357.0, "min_saturation": 0.2, "min_value": 0.1,
                "saturation_gain": 1.2, "value_gain": 0.9}


def hue_deg(rgb) -> float:
    r, g, b = (float(v) for v in rgb[:3])
    high, low = max(r, g, b), min(r, g, b)
    if high == low:
        return 0.0
    if high == r:
        return (60.0 * ((g - b) / (high - low))) % 360.0
    return 60.0 * ((b - r) / (high - low)) + 120.0 if high == g else 60.0 * ((r - g) / (high - low)) + 240.0


def test_recolouring_turns_the_garment_red_and_leaves_skin_hair_and_other_cloth_alone(prepare):
    shirt, shade, skin, jeans, hair, white = ([0.20, 0.62, 0.36, 1.0], [0.08, 0.30, 0.15, 0.5], [0.80, 0.60, 0.48, 1.0],
                                             [0.25, 0.38, 0.62, 1.0], [0.09, 0.08, 0.07, 1.0], [0.9, 0.9, 0.9, 1.0])
    greyish_green = [0.40, 0.44, 0.41, 1.0]        # too little colour to be the garment
    picture = np.array([[shirt, shade, skin, jeans], [hair, white, greyish_green, shirt]], dtype=np.float32)
    made = prepare.recolour(picture, **GREEN_TO_RED)
    assert made.shape == picture.shape and made.dtype == np.float32
    assert np.array_equal(made[0, 2:], picture[0, 2:]) and np.array_equal(made[1, :3], picture[1, :3])
    assert np.array_equal(made[..., 3], picture[..., 3])                    # opacity is no colour
    for where in ((0, 0), (0, 1), (1, 3)):
        hue = hue_deg(made[where])
        assert hue >= 345.0 or hue <= 10.0, (where, hue)
        assert made[where][0] > 2.0 * max(made[where][1], made[where][2])   # red to the eye, not brown or pink
    assert made[0, 0][0] > made[0, 1][0] > 0.15                             # the fold in shadow stays the darker
    assert float(made.min()) >= 0.0 and float(made.max()) <= 1.0
    assert np.array_equal(prepare.recolour(picture.reshape(-1, 4), **GREEN_TO_RED), made.reshape(-1, 4))


def test_recolouring_to_another_hue_and_the_soft_rim_of_the_band(prepare):
    orange, blue = [0.9, 0.45, 0.1, 1.0], [0.1, 0.2, 0.8, 1.0]
    made = prepare.recolour(np.array([orange, blue], dtype=np.float32), hue_from_deg=[200.0, 260.0], hue_to_deg=120.0,
                            min_saturation=0.2, min_value=0.1)
    assert np.array_equal(made[0], np.array(orange, dtype=np.float32))
    assert 100.0 <= hue_deg(made[1]) <= 140.0
    # a colour at the rim of the band is moved part of the way, so no hard line is drawn through a blend
    rim = prepare.recolour(np.array([[0.35, 0.60, 0.20, 1.0]], dtype=np.float32), **GREEN_TO_RED)   # hue 98
    core = prepare.recolour(np.array([[0.20, 0.60, 0.30, 1.0]], dtype=np.float32), **GREEN_TO_RED)  # hue 135
    assert rim[0][1] > core[0][1] and rim[0][0] > 0.35


# ------------------------------------------------------------------------------------------------- in Blender
def run_prepare(root: Path, *more: str):
    args = ["--manifest", root / "manifest", "--downloads", root / "downloads", "--prepared", root / "prepared",
            "--stamp", root / "stamp.json", *more]
    return blender.run_script(PREPARE, args, timeout_s=300.0)


@needs_blender
def test_preparing_an_empty_manifest_writes_an_empty_stamp(tmp_path):
    (tmp_path / "manifest").mkdir()
    done = run_prepare(tmp_path, "--only", "plant_a", "--force")           # an id of another class is not an error
    assert json.loads((tmp_path / "stamp.json").read_text()) == {"avatars": [], "poses": []}, done.stdout[-2000:]


@needs_blender
def test_preparing_without_the_download_fails_and_names_the_file(tmp_path):
    (tmp_path / "manifest").mkdir()
    pose = {"id": "wait", "reach": 0.3, "tags": ["waiting"], "licence": "MIT", "author": "x", "redistributable": True,
            "brand": False, "sources": [{"url": SOURCE + "Animations/a/idle.max.fbx", "sha256": "0" * 64, "bytes": 5}],
            "prepare": {"animation": "idle.max.fbx", "frame": 3, "prop": None}}
    (tmp_path / "manifest" / "poses.json").write_text(json.dumps([pose]))
    with pytest.raises(RuntimeError, match=r"wait[/\\]idle\.max\.fbx") as failure:
        run_prepare(tmp_path)
    assert "fetch" in str(failure.value)                                    # it says how to get the file
    assert not (tmp_path / "stamp.json").exists()                           # a failed run stamps nothing
    # a file that is there with other bytes than the manifest states is refused as well
    (tmp_path / "downloads" / "wait").mkdir(parents=True)
    (tmp_path / "downloads" / "wait" / "idle.max.fbx").write_bytes(b"12345")
    with pytest.raises(RuntimeError, match="sha256|SHA-256"):
        run_prepare(tmp_path)


COUNTING = '''
import json, math, sys
sys.path.insert(0, {side!r})
import bpy
import prepare_avatars

def body(name, parts):
    # A mesh of upright prisms standing on the ground: a box is (x, y, half its depth, half its width, its top),
    # a round one on the axis (its radius, the number of its sides, its top).
    corners, faces = [], []
    for part in parts:
        if len(part) == 3:
            radius, sides, top = part
            ring = [(radius * math.cos(2 * math.pi * k / sides), radius * math.sin(2 * math.pi * k / sides))
                    for k in range(sides)]
        else:
            x, y, half_x, half_y, top = part
            ring = [(x + i * half_x, y + j * half_y) for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        n, m = len(corners), len(ring)
        corners += [(x, y, z) for z in (0.0, top) for x, y in ring]
        faces += [tuple(range(n + m - 1, n - 1, -1)), tuple(range(n + m, n + 2 * m))]
        faces += [(n + k, n + (k + 1) % m, n + m + (k + 1) % m, n + m + k) for k in range(m)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(corners, [], faces)
    return mesh

views = prepare_avatars.axis_views({natural!r})
json.dump({{name: prepare_avatars.axis_seen(body(name, parts), views) for name, parts in {bodies!r}.items()}},
          open({out!r}, "w"))
'''


@needs_blender
def test_the_axis_samples_are_counted_as_the_scene_check_sees_them(tmp_path, prepare):
    """On bodies of upright prisms, whose answer is plain. `natural` is the tallest figure's height, so that at
    that height a body is as tall as it is drawn and as many times as broad as the build says."""
    tallest, shortest = (pick(height for height, _build in prepare.SIZES) for pick in (max, min))
    at = {height: [prepare.size_name(height, build) for tall, build in prepare.SIZES if tall == height]
          for height in (tallest, shortest)}
    post = (0.0, 0.0, 0.1, 0.1, 2.2)               # a trunk no size leaves short, within the tolerance all round
    bodies = {
        "post": [post],
        # a head at its own height: scaled to the shortest figure, the samples at 1.6 and 1.7 m are over it
        "short": [(0.0, 0.0, 0.1, 0.1, tallest)],
        "drum": [(0.0, 0.0, 0.45, 0.45, 2.2)],     # everywhere further from its axis than the tolerance
        # an arm beside the trunk, to under 1.5 m: its outer side is 0.29 m from the axis, within the tolerance
        # until it is widened; then it hides the six samples beside it from its side
        "arm": [post, (0.0, 0.25, 0.04, 0.04, 1.45)],
        # a thin post half a metre out, 37 degrees round: it hides the whole axis from seven degrees of the round
        "wrist": [post, (0.3993, 0.3009, 0.03, 0.03, 2.2)],
        # a low wall 0.6 m out: looked over from near, and from far it stands before the lowest sample
        "sill": [post, (0.0, 0.6, 0.3, 0.01, 1.0)],
        # ... and one 2.09 m out, which a build of 0.92 brings to 1.92 m: from 10 m it stands before the two lowest
        # samples; from 9.2 m, where the camera would be if it were moved in with the body, before one
        "far sill": [post, (0.0, 2.087, 0.3, 0.01, 1.1)],
        # 72 sides, each 0.2945 m from the axis. The depth of a sample is taken along the camera's axis, which
        # looks 10 degrees down: a ray that goes down more steeply makes more of the same level distance, and from
        # 1.8 m the three lowest samples are over the tolerance
        "round": [(0.2945, 72, 2.2)],
    }
    script, out = tmp_path / "count.py", tmp_path / "counted.json"
    script.write_text(COUNTING.format(side=str(blender.BLENDER_SIDE), natural=tallest, bodies=bodies, out=str(out)))
    blender.run_script(script, [], timeout_s=300.0)
    seen = json.loads(out.read_text())
    every = at[tallest] + at[shortest]
    assert all(list(counts) == [prepare.size_name(*size) for size in prepare.SIZES] for counts in seen.values())
    assert [seen["post"][size] for size in every] == [9] * len(every)
    assert [seen["short"][size] for size in at[tallest]] == [9] * len(at[tallest])
    assert [seen["short"][size] for size in at[shortest]] == [7] * len(at[shortest])
    assert [seen["drum"][size] for size in every] == [0] * len(every)
    along = [seen["arm"][size] for size in at[tallest]]                    # from the narrowest build to the widest
    assert along[0] == 9 and along[-1] == 3 and along == sorted(along, reverse=True), along
    assert [seen["wrist"][size] for size in every] == [0] * len(every)      # the round is fine enough to find it
    assert [seen["sill"][size] for size in at[tallest]] == [8] * len(at[tallest])
    assert seen["far sill"][prepare.size_name(tallest, BUILDS[0])] == 7
    assert seen["round"][prepare.size_name(tallest, BUILDS[0])] == 9
    assert seen["round"][prepare.size_name(tallest, 1.0)] == 6


PLACING = '''
import json, sys
sys.path.insert(0, {side!r})
import bpy
import people_library

def feet(xs):
    # How far the soles reach beyond the ankles over them, ahead and behind. A foot is long before its ankle and
    # short behind it, wherever the pelvis, which is the origin, is over the feet.
    soles, ankles = [v.x for v in xs if v.z < {sole!r}], [v.x for v in xs if {ankle!r}[0] < v.z < {ankle!r}[1]]
    return {{"toe_out": max(soles) - max(ankles), "heel_out": min(ankles) - min(soles)}}

def facts(ob):
    xs = [v.co for v in ob.data.vertices]
    return {{"type": ob.type, "parent": ob.parent is not None, "location": list(ob.location), "scale": list(ob.scale),
            "rotation": list(ob.rotation_euler), "low": [min(v[i] for v in xs) for i in range(3)],
            "high": [max(v[i] for v in xs) for i in range(3)], "verts": len(xs),
            "reach": max((v.x ** 2 + v.y ** 2) ** 0.5 for v in xs), **feet(xs),
            "front_feet": max(v.x for v in xs if v.z <= {feet_top!r}),
            "front_body": max(v.x for v in xs if v.z > {feet_top!r}),
            "materials": [m.name for m in ob.data.materials], "linked": ob.name in bpy.context.scene.objects,
            "natural_height": ob["natural_height"], "height": ob["height"], "stated_reach": ob["reach"],
            "images": sorted(n.image.name for m in ob.data.materials for n in m.node_tree.nodes
                             if n.type == "TEX_IMAGE" and n.image), "mesh": ob.data.name, "name": ob.name}}

asked = {asked!r}
out = {{"refused": {{}}}}
try:
    people_library.add_person({first!r}, {plain!r})
except RuntimeError as error:
    out["refused"]["unopened"] = str(error)
people_library.open_people({prepared!r})
placed = [people_library.add_person(avatar, pose, prop) for avatar, pose, prop in asked]
out["people"] = [facts(ob) for ob in placed]
for (avatar, pose, prop), ob, told in zip(asked, placed, out["people"]):
    if prop:                                       # the thing: what the mesh has more than the pose with empty hands
        thing = [v.co for v in ob.data.vertices][len(placed[asked.index((avatar, pose, None))].data.vertices):]
        told["thing"] = [sum(v[i] for v in thing) / max(1, len(thing)) for i in range(3)]
before = (len(bpy.data.materials), len(bpy.data.images))
again = people_library.add_person(*asked[0])
out["again"] = facts(again)
out["no_new_data"] = (len(bpy.data.materials), len(bpy.data.images)) == before
for label, call in {refusals!r}.items():
    try:
        people_library.add_person(*call)
    except ValueError as error:
        out["refused"][label] = str(error)
# which way each of the two stands in every pose of the library
out["feet"] = {{avatar: {{pose: feet([v.co for v in people_library.add_person(avatar, pose).data.vertices])
                         for pose in {poses!r}}} for avatar in {both!r}}}
json.dump(out, open({out!r}, "w"))
'''


@needs_assets
def test_a_placed_person_stands_on_the_ground_point_facing_x_at_the_natural_height(tmp_path, library, entries):
    measured = measurements()
    plain = next(e["id"] for e in entries["poses"] if e["prepare"]["prop"] is None)
    phone = next(e["id"] for e in entries["poses"]          # one that is looked at: at the ear it is beside the head
                 if e["prepare"]["prop"] == "phone" and e["prepare"].get("hold") != "ear")
    cup = next(e["id"] for e in entries["poses"] if e["prepare"]["prop"] == "cup")
    hand = {e["id"]: e["prepare"].get("hand") for e in entries["poses"]}
    made_for = {e["id"]: e["prepare"]["prop"] for e in entries["poses"]}
    woman, man = (next(a.id for a in library.avatars.values() if a.sex == sex) for sex in ("female", "male"))
    asked = [(woman, plain, None), (man, phone, "phone"), (man, phone, None), (woman, cup, "cup"), (man, plain, None),
             (woman, cup, None)]
    refusals = {"avatar": ("nobody", plain, None), "pose": (woman, "no_such_pose", None), "prop": (woman, plain, "cup"),
                "thing": (woman, phone, "umbrella")}
    script, out = tmp_path / "place.py", tmp_path / "facts.json"
    script.write_text(PLACING.format(side=str(blender.BLENDER_SIDE), prepared=str(library.prepared), asked=asked,
                                     refusals=refusals, out=str(out), first=woman, plain=plain, sole=SOLE_M,
                                     ankle=ANKLE_M, feet_top=FEET_TOP_M, both=[woman, man],
                                     poses=list(library.poses)))
    blender.run_script(script, [], timeout_s=300.0)
    told = json.loads(out.read_text())
    assert "open_people" in told["refused"]["unopened"]
    assert set(told["refused"]) == {"unopened", *refusals}                  # each wrong call is a ValueError
    assert "nobody" in told["refused"]["avatar"] and "no_such_pose" in told["refused"]["pose"]
    assert "cup" in told["refused"]["prop"] and "umbrella" in told["refused"]["thing"]
    for (avatar, pose, prop), seen in zip(asked, told["people"]):
        row, label = measured[avatar]["poses"][pose], f"{avatar} {pose} {prop}"
        assert seen["type"] == "MESH" and not seen["parent"] and seen["linked"], label     # one static mesh
        assert seen["location"] == [0, 0, 0] and seen["scale"] == [1, 1, 1] and seen["rotation"] == [0, 0, 0], label
        assert seen["low"][2] == pytest.approx(0.0, abs=2e-3), label                       # the feet on the ground
        natural = library.avatars[avatar].natural_height
        assert seen["natural_height"] == pytest.approx(natural, abs=5e-4), label
        assert seen["high"][2] == pytest.approx(seen["height"], abs=2e-3), label
        assert abs(seen["high"][2] - natural) < 0.06, label                 # a pose bows a head and changes no size
        assert seen["reach"] <= library.poses[pose].reach + 1e-3, label     # the manifest states the furthest of all
        assert seen["reach"] == pytest.approx(seen["stated_reach"], abs=1e-3), label
        assert seen["reach"] == pytest.approx(row["reach_with"][prop or ""], abs=1e-3), label
        assert seen["low"][0] < 0 < seen["high"][0] and seen["low"][1] < 0 < seen["high"][1], label   # round the axis
        for key in ("front_feet", "front_body"):    # ahead: the feet, and the rest of the body with what it holds.
            if prop is not None or made_for[pose] is None:      # Noted is the pose with its thing, where it has one
                assert seen[key] == pytest.approx(row[key], abs=1e-3), (label, key)
            assert seen[key] <= row[key] + 1e-3 and seen[key] <= seen["high"][0], (label, key)
        assert seen["high"][0] == pytest.approx(max(seen["front_feet"], seen["front_body"])), label
        assert seen["images"] and all(m for m in seen["materials"]), label
        # Facing +X, by what tells a body's front from its back. A foot is long before its ankle and short behind
        # it; a person turned round has it the other way.
        assert seen["toe_out"] > TOE_BEFORE_ANKLE_M and seen["toe_out"] > seen["heel_out"], label
        if prop:                                    # and what a hand holds up to look at or to drink from is
            x, y, z = seen["thing"]                 # before the chest: not behind the back, not out at one side
            assert x > THING_BEFORE_AXIS_M and x > abs(y) and 0.5 * natural < z < natural, label
            if hand[pose] == "R":                   # the right of a person who faces +X is -Y; a thing that both
                assert y < 0.03, label              # hands hold is in the middle
    # ... in every pose of the library, on a woman and on a man: all women share one skeleton and all men another,
    # and a pose is turned to face ahead on each skeleton by itself
    for avatar in (woman, man):
        assert list(told["feet"][avatar]) == list(library.poses)
        for pose, feet in told["feet"][avatar].items():
            assert feet["toe_out"] > TOE_BEFORE_ANKLE_M and feet["toe_out"] > feet["heel_out"], f"{avatar} {pose}"
    with_phone, without = told["people"][1], told["people"][2]
    assert with_phone["verts"] > without["verts"]                           # the thing is part of the one mesh
    assert with_phone["images"] == without["images"]                        # one avatar in two poses, one set of images
    # the same person once more is an object and a mesh of its own, which the caller may change, in the same skin
    again, first = told["again"], told["people"][0]
    assert again["name"] != first["name"] and again["mesh"] != first["mesh"] and again["verts"] == first["verts"]
    assert again["materials"] == first["materials"] and told["no_new_data"]
    assert set(told["people"][4]["images"]).isdisjoint(first["images"])     # each avatar its own skin


@needs_assets
def test_a_run_that_makes_nothing_stamps_every_prepared_person_and_pose(tmp_path, library):
    """The stamp lists what is prepared and intact, earlier runs' work included. The run is given a name that
    no avatar and no pose has, so that it makes nothing whatever the state of the prepared folder is: a test
    does not prepare the real assets."""
    stamp = tmp_path / "stamp.json"
    files = {path: path.stat().st_mtime_ns for path in (library.prepared / "avatars").glob("*.blend")}
    blender.run_script(PREPARE, ["--manifest", ASSETS_DIR / "manifest", "--downloads", tmp_path / "no_downloads",
                                 "--prepared", library.prepared, "--stamp", stamp, "--only", "nobody_and_no_pose"],
                       timeout_s=900.0)
    told = json.loads(stamp.read_text())
    assert list(told) == ["avatars", "poses"]
    assert told["avatars"] == list(library.avatars) and told["poses"] == list(library.poses)   # in the manifest's order
    assert files == {path: path.stat().st_mtime_ns for path in (library.prepared / "avatars").glob("*.blend")}
