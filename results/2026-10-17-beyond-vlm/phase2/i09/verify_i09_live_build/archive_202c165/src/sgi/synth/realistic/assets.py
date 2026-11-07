"""The assets command of the realistic set: fetch what the manifest names, check it, have Blender prepare it,
and write the attribution file.

    python -m sgi.synth.realistic.assets blender        Blender itself, into .tools/
    python -m sgi.synth.realistic.assets fetch          every source of every entry, into the downloads folder
    python -m sgi.synth.realistic.assets prepare        what the scenes load, made of the downloads by Blender
    python -m sgi.synth.realistic.assets verify         what is not fetched, not prepared, or not the file it should be
    python -m sgi.synth.realistic.assets attribution    assets/realistic/ATTRIBUTION.md, from the manifest

`--only ID ...` names the assets to fetch, prepare or verify; `--force` fetches or prepares again what is
there already. Nothing but `fetch` and `blender` uses the network, and they ask for https addresses only.

The manifest (library.py says where it is and what a scene reads of it) is the one list of everything
downloaded. Beside what the library reads, every entry of every class carries:

    sources          a list of {url, sha256, bytes}, one for each file to fetch; `file` beside them when the file
                     is not to be called by the last part of its address, as a path under the entry's folder
    licence          as the source states it, beginning with its name: "CC0 1.0", "CC BY 4.0", "MIT", "OFL 1.1",
                     or "none: made in code" for what has no source (its sources are then an empty list)
    author           whom to credit
    redistributable  whether the files may be passed on; true only under one of the licences above, and never
                     for an entry that shows a brand
    brand            whether it shows a real brand; such a model is listed and never drawn
    prepare          what the preparation is to do with it: an object of parameters, read by the Blender side

and may carry `title` and `page` (its name and its page at the source, for the attribution), `changes`
(what was done to this asset beyond what is done to all of its class) and `note` (what whoever places it has
to know of it). A model also says under `lettering` what writing and marks were seen on it.

Files lie under `_cache/` beside the manifest, which is not committed: a source at `downloads/<id>/<file>`,
and what Blender made of the downloads under `prepared/`. For each model, material, HDRI and font,
prepare_models.py leaves a receipt `prepared/<class>/<id>.json` whose `files` lists what it wrote, as
{path under the prepared folder: {sha256, bytes}}, and whose `made_from` keeps what the entry said when it
was made (PREPARED_FROM). `verify` reads the receipts: it says which files are gone or changed, and which
asset was prepared from an entry that the manifest has changed since, which neither the stamp nor the files
can show. For an avatar it looks for `prepared/avatars/<id>.blend`. Each preparing script also writes, to the
file it is given as `--stamp`, the ids of its classes that are now prepared and intact; `prepare` writes
their union to `prepared/prepared.json` when every script has ended well, and not before.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

from sgi.synth.realistic import blender
from sgi.synth.realistic.library import ASSET_CLASSES, ASSETS_DIR, PREPARED_STAMP, load_library, manifest_entries

# The archive that the `blender` action fetches, and its checksum as Blender publishes it in
# https://download.blender.org/release/Blender4.5/blender-4.5.14.sha256 .
BLENDER_URL = "https://download.blender.org/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz"
BLENDER_SHA256 = "9ba871ff2ecd36526b77432745980b7e6664ecd0c7ca11c48849073dcfe06da3"

# The scripts under blender_side/ that prepare, in the order they run, each with the classes it prepares.
PREPARE_SCRIPTS = (("prepare_models.py", ("models", "materials", "hdris", "fonts")),
                   ("prepare_avatars.py", ("avatars", "poses")))
PREPARE_TIMEOUT_S = 4 * 3600.0                     # a whole library is decimated and baked in one run of each
# What of an entry the prepared form of a thing depends on. prepare_models.py, which this side cannot import,
# keeps these in a receipt's `made_from` as they were when the asset was made, under the same names
# (READ_OF_AN_ENTRY there); a key an entry has not is kept as null.
PREPARED_FROM = ("sources", "prepare", "kind", "tile_m", "size", "file")

# How a licence begins, in lower case, and the name it is known by here. The first that fits is taken.
_LICENCES = (("cc0", "CC0 1.0"), ("cc by 4.0", "CC BY 4.0"), ("cc-by-4.0", "CC BY 4.0"), ("mit", "MIT"),
             ("ofl", "OFL 1.1"), ("sil open font", "OFL 1.1"), ("none", "none"))
REDISTRIBUTABLE = ("CC0 1.0", "CC BY 4.0", "MIT", "OFL 1.1", "none")       # the licences that let a file be passed on
_LICENCE_TEXTS = {"CC0 1.0": "https://creativecommons.org/publicdomain/zero/1.0/legalcode",
                  "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/legalcode",
                  "MIT": "the text is in the source's own repository, named with each asset",
                  "OFL 1.1": "https://openfontlicense.org/open-font-license-official-text/"}

_DOWNLOADS = Path("_cache") / "downloads"          # under the assets folder
_PREPARED = Path("_cache") / "prepared"
_WITH_RECEIPT = PREPARE_SCRIPTS[0][1]              # the classes whose prepared files a receipt lists
_USER_AGENT = "sgi-assets/0.1 (course project; fetches what its manifest names, once)"
_TIMEOUT_S = 120.0                                 # of one silence of a server, not of a whole download
_HEADINGS = {"models": "Models", "materials": "Materials", "hdris": "HDRIs", "fonts": "Fonts", "avatars": "Avatars",
             "poses": "Poses"}


class AssetError(RuntimeError):
    """The manifest, a download or a prepared file is not what it has to be. The message says which and why."""


# ------------------------------------------------------------------------------------------ the manifest
def licence_family(licence: str) -> str | None:
    """The name a licence is known by here, from how its text begins; None for one that is not known."""
    text = licence.strip().lower()
    for start, family in _LICENCES:
        if text.startswith(start) and not text[len(start):len(start) + 1].isalnum():
            return family
    return None


def _plain_name(name) -> bool:
    """Whether a value can name a folder and stand on a command line: no separator, no space, and no dot or
    dash at its start, which would make it a hidden file, a way up, or an option."""
    return isinstance(name, str) and re.fullmatch(r"[^\s/\\.\-][^\s/\\]*", name) is not None


def _source_file(source: dict) -> str | None:
    """The path of a source under its entry's folder: its `file`, else the last part of its address. None when
    that is no path that stays in the folder."""
    if "file" in source:
        name = source["file"]
    else:
        name = urllib.parse.unquote(PurePosixPath(urllib.parse.urlsplit(source["url"]).path).name)
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name or name.endswith("/"):
        return None
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return None
    return str(path)


def _source_problems(source) -> list[str]:
    if not isinstance(source, dict):
        return [f"it is {source!r}: expected an object with url, sha256 and bytes"]
    missing = [key for key in ("url", "sha256", "bytes") if key not in source]
    if missing:
        return [f"it has no {key!r}" for key in missing]
    problems, url = [], source["url"]
    address = urllib.parse.urlsplit(url) if isinstance(url, str) else None
    if address is None or address.scheme != "https" or not address.netloc or re.search(r"\s", url):
        return [f"url is {url!r}: expected an https address"]
    if not isinstance(source["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
        problems.append(f"sha256 is {source['sha256']!r}: expected 64 hexadecimal digits in lower case")
    if type(source["bytes"]) is not int or source["bytes"] <= 0:
        problems.append(f"bytes is {source['bytes']!r}: expected the size of the file, a whole number above 0")
    if _source_file(source) is None:
        problems.append(f"file is {source['file']!r}: expected a path that stays in the entry's folder"
                        if "file" in source else "its address ends in no name, so `file` has to give one")
    return problems


def entry_problems(entry: dict) -> list[str]:
    """What an entry lacks of what the fetching and the attribution need, one line for each thing; none when it
    is whole. What a scene reads of an entry (its kind, its size, ...) is library.py's to check."""
    problems = []
    if not _plain_name(entry.get("id")):
        problems.append(f"id is {entry.get('id')!r}: expected a plain name, without a separator or a space, and "
                        "with no dot or dash at its start")
    for key, kind, expected in (("sources", list, "a list"), ("licence", str, "its licence, as its source states it"),
                                ("author", str, "whom to credit"), ("redistributable", bool, "true or false"),
                                ("brand", bool, "true or false"), ("prepare", dict, "an object of parameters")):
        if key not in entry:
            problems.append(f"it has no {key!r}")
        elif type(entry[key]) is not kind or (kind is str and not entry[key].strip()):
            problems.append(f"{key} is {entry[key]!r}: expected {expected}")
    sources = entry.get("sources") if isinstance(entry.get("sources"), list) else None
    licence = entry.get("licence") if isinstance(entry.get("licence"), str) and entry["licence"].strip() else None
    files: dict[str, int] = {}
    for number, source in enumerate(sources or (), start=1):
        wrong = _source_problems(source)
        problems += [f"source {number}: {said}" for said in wrong]
        if not wrong:
            first = files.setdefault(_source_file(source), number)
            if first != number:
                problems.append(f"sources {first} and {number} are both the file {_source_file(source)}")
    if sources is not None and licence is not None:
        made_here = licence_family(licence) == "none"
        if made_here and sources:
            problems.append("sources are named for what its licence says has none")
        if not made_here and not sources:
            problems.append('sources is empty: expected each file to fetch, or the licence "none: made in code"')
    if entry.get("brand") is True and entry.get("redistributable") is True:
        problems.append("brand and redistributable are both true: no licence of a model covers another's mark on it")
    if entry.get("redistributable") is True and licence is not None and licence_family(licence) not in REDISTRIBUTABLE:
        problems.append(f"redistributable is true under the licence {licence!r}, which does not begin with the name of "
                        f"one known to allow it ({', '.join(REDISTRIBUTABLE)})")
    return problems


def source_path(root: Path, entry: dict, source: dict) -> Path:
    """Where a source of an entry lies once fetched: in the entry's own folder under the downloads folder."""
    return root / _DOWNLOADS / entry["id"] / _source_file(source)


def manifest_problems(root: Path = ASSETS_DIR) -> list[str]:
    """What is wrong with the manifest under `root` for fetching and attributing, one line for each thing with
    the class and the id it is about; none for a manifest the command will work with."""
    entries = manifest_entries(root)
    problems = [f"{name} {entry.get('id')!r}: {said}" for name in ASSET_CLASSES for entry in entries[name]
                for said in entry_problems(entry)]
    if problems:
        return problems
    claimed: dict[Path, tuple[str, dict]] = {}     # entries of two classes with one id share a folder: a file
    for name in ASSET_CLASSES:                     # of that name in both has to be one file to both
        for entry in entries[name]:
            for source in entry["sources"]:
                path = source_path(root, entry, source)
                owner, first = claimed.setdefault(path, (f"{name} {entry['id']!r}", source))
                if (first["sha256"], first["bytes"]) != (source["sha256"], source["bytes"]):
                    problems.append(f"{name} {entry['id']!r}: its file {path.relative_to(root / _DOWNLOADS)} is a file "
                                    f"of {owner} too, which names another sha256 or size for it")
    return problems


def _selected(root: Path, only: Sequence[str] | None) -> list[tuple[str, dict]]:
    """The entries an action is for, each with its class, in the manifest's order: all of them, or those that
    `only` names. A manifest that does not read or has problems, and a name that no entry has, are refused
    before anything is done."""
    load_library(root)                             # ValueError for what a scene could not read of it
    problems = manifest_problems(root)
    if problems:
        more = f"\n  and {len(problems) - 12} more" if len(problems) > 12 else ""
        raise AssetError("the manifest cannot be worked with:\n  " + "\n  ".join(problems[:12]) + more)
    entries = manifest_entries(root)
    every = [(name, entry) for name in ASSET_CLASSES for entry in entries[name]]
    if only is None:
        return every
    unknown = sorted(set(only) - {entry["id"] for _, entry in every})
    if unknown:
        raise AssetError(f"no asset of the manifest is named {', '.join(unknown)}")
    return [(name, entry) for name, entry in every if entry["id"] in set(only)]


# ------------------------------------------------------------------------------------------------ files
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while block := file.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _differs(path: Path, expected: dict, named_by: str = "the manifest") -> str | None:
    """How an existing file differs from the one that {sha256, bytes} describe, as the end of a sentence about
    it; None when it is that file. A description without bytes is held to the checksum alone."""
    size, wanted = path.stat().st_size, expected.get("bytes")
    found = _sha256(path)
    if found == expected.get("sha256") and wanted in (None, size):
        return None
    return (f"is {size} bytes with the sha256 {found}, and {named_by} names "
            f"{'' if wanted is None else f'{wanted} bytes with '}the sha256 {expected.get('sha256')}")


def _write_json(path: Path, data) -> None:
    """Whole or not at all: a reader never finds half a file under the name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    os.replace(part, path)


# ------------------------------------------------------------------------------------------------ fetch
def download(url: str, path: Path) -> None:
    """Write what an https address gives to `path`. The one place that uses the network; nothing is checked
    here, and whoever calls it checks what was written."""
    if urllib.parse.urlsplit(url).scheme != "https":
        raise AssetError(f"{url}: only https addresses are fetched")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as answer, open(path, "wb") as out:
            shutil.copyfileobj(answer, out, 1 << 20)
    except OSError as error:                       # no route, no such page, a server that stops answering
        raise AssetError(f"{url}: could not be fetched ({error})") from error


def _fetch_one(url: str, path: Path, expected: dict, said: str, named_by: str = "the manifest") -> None:
    """Fetch one file to `path`, which is written only when what came is the file expected: whatever lay there
    before stays until then."""
    part = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        download(url, part)
        wrong = _differs(part, expected, named_by)
        if wrong:
            raise AssetError(f"{said}: what {url} gave {wrong}. Nothing of it was kept.")
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def fetch(root: Path = ASSETS_DIR, only: Sequence[str] | None = None, force: bool = False,
          report: Callable[[str], None] | None = None) -> list[Path]:
    """Download what is not there yet of the sources of every entry, or of the named ones, each checked against
    its size and SHA-256; the files that were fetched, in the manifest's order.

    A file that is there and is the file the manifest names is left alone, however it came there, unless
    `force` asks for everything again. The first download that is not the file it should be ends the run
    with an AssetError, and nothing of it is kept. `report` is told each file as it arrives."""
    got, looked_at = [], set()
    for name, entry in _selected(root, only):
        for source in entry["sources"]:
            path = source_path(root, entry, source)
            if path in looked_at:                  # a file that entries share
                continue
            looked_at.add(path)
            if not force and path.is_file() and _differs(path, source) is None:
                continue
            _fetch_one(source["url"], path, source, f"{name} {entry['id']}")
            if report is not None:
                report(f"fetched {path.relative_to(root / _DOWNLOADS)} ({source['bytes']} bytes)")
            got.append(path)
    return got


# ----------------------------------------------------------------------------------------------- verify
def _download_problems(root: Path, selected: list[tuple[str, dict]]) -> list[str]:
    """A line for each source of the entries that is not fetched or is not the file the manifest names."""
    problems, differs = [], {}
    for name, entry in selected:
        for source in entry["sources"]:
            path = source_path(root, entry, source)
            where = path.relative_to(root / _DOWNLOADS)
            if not path.is_file():
                problems.append(f"{name} {entry['id']}: {where} is not fetched")
                continue
            if path not in differs:                # a shared file is read once
                differs[path] = _differs(path, source)
            if differs[path]:
                problems.append(f"{name} {entry['id']}: {where} {differs[path]}")
    return problems


def _read_stamp(root: Path) -> dict[str, list[str]]:
    """The ids the stamp lists by class; nothing for a stamp that is absent or cannot be read."""
    try:
        stamp = json.loads((root / _PREPARED / PREPARED_STAMP).read_text(encoding="utf-8"))
        return {name: [item for item in listed if isinstance(item, str)]
                for name, listed in stamp.items() if isinstance(listed, list)}
    except (OSError, ValueError, AttributeError):
        return {}


def _prepared_problems(prepared: Path, name: str, entry: dict) -> list[str]:
    """What is wrong with the prepared form of one asset that the stamp lists: a receipt made from another
    entry than the manifest has now, and files that are gone or changed. A receipt is read as data: a path in
    it that leads out of the prepared folder is said, and not followed.

    The files of a receipt agree with the receipt however the manifest has changed since, and the stamp lists
    ids alone, so the first of these is the one thing that tells an asset prepared from an older entry: the
    Blender side would load it by its receipt (its size, its tile) while a scene is laid out by the manifest."""
    asset = entry["id"]
    receipt = prepared / name / f"{asset}.json"
    if name not in _WITH_RECEIPT:                  # the people: an avatar is one file, a pose has none of its own
        made = prepared / "avatars" / f"{asset}.blend"
        return [f"avatars/{asset}.blend is missing"] if name == "avatars" and not made.is_file() else []
    try:
        said = json.loads(receipt.read_text(encoding="utf-8"))
        listed = [(str(rel), dict(facts)) for rel, facts in said["files"].items()]
        origin = said.get("made_from")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return [f"its receipt {name}/{asset}.json is not there, or cannot be read"]
    problems = []
    if not isinstance(origin, dict):
        problems.append(f"its receipt {name}/{asset}.json does not say what entry of the manifest it was prepared from")
    else:
        changed = [key for key in PREPARED_FROM if origin.get(key) != entry.get(key)]
        if changed:
            problems.append("prepared from an older entry of the manifest, which differed in its "
                            + " and ".join(filter(None, (", ".join(changed[:-1]), changed[-1]))))
    for rel, facts in listed:
        if os.path.isabs(rel) or os.path.normpath(rel).split(os.sep)[0] == os.pardir:
            problems.append(f"its receipt names {rel}, outside the prepared folder")
        elif not (prepared / rel).is_file():
            problems.append(f"{rel} is missing")
        elif wrong := _differs(prepared / rel, facts, "its receipt"):
            problems.append(f"{rel} {wrong}")
    return problems


def verify(root: Path = ASSETS_DIR, only: Sequence[str] | None = None) -> list[str]:
    """What is missing or wrong of everything the manifest names, or of the named assets: one line for each
    source that is not fetched or is another file, each asset the stamp does not list, each thing that was
    prepared from an entry the manifest has changed since, and each prepared file that is gone or changed. An
    empty list when all is present and intact. Nothing is fetched or written."""
    selected = _selected(root, only)
    problems = _download_problems(root, selected)
    stamp = _read_stamp(root)
    for name, entry in selected:
        if entry["id"] not in stamp.get(name, ()):
            problems.append(f"{name} {entry['id']}: not prepared")
        else:
            wrong = _prepared_problems(root / _PREPARED, name, entry)
            problems += [f"{name} {entry['id']}: {said}" for said in wrong]
    return problems


# ---------------------------------------------------------------------------------------------- prepare
def _own_stamp(path: Path, script: str, classes: Sequence[str]) -> dict[str, list[str]]:
    """What a preparing script wrote as its stamp, of its own classes."""
    try:
        stamp = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(stamp, dict) or not all(isinstance(stamp.get(name, []), list) for name in classes):
            raise ValueError("expected an object that lists ids by class")
        return {name: [str(item) for item in stamp[name]] for name in classes if name in stamp}
    except (OSError, ValueError) as error:
        raise AssetError(f"{script} ended without a stamp that can be read ({error})") from error


def prepare(root: Path = ASSETS_DIR, only: Sequence[str] | None = None, force: bool = False) -> dict[str, list[str]]:
    """Have Blender prepare what is not prepared yet (everything, or the named assets, again with `force`), and
    then write the stamp; the stamp that was written.

    Each preparing script runs once, the things' before the people's, with a stamp file of its own; a script
    that is not written yet is passed over, and its classes are then not in the stamp. With `only`, an asset
    goes to the script of its class, and what the scripts report is merged into the stamp that is there class
    by class: a script lists every asset of its classes that is intact, those of an earlier run among them,
    so its list stands for each class it reports, and the classes of a script that did not run stay as they
    were. An asset that is stamped and no longer intact, its files gone or its entry changed since, thus
    leaves the stamp at the next run of its script, whatever that run was for. Without `only` the stamp is
    what the scripts report and nothing else. The downloads of what is to be prepared are checked first, so
    Blender opens no file that is not the one the manifest names. If a script fails, its error is passed on
    and the stamp stays as it was."""
    selected = _selected(root, only)
    runs = []
    for script, classes in PREPARE_SCRIPTS:
        named = [entry["id"] for name, entry in selected if name in classes]
        if only is not None and not named:
            continue
        if not (blender.BLENDER_SIDE / script).is_file():
            if only is None and script != PREPARE_SCRIPTS[0][0]:
                continue                           # the people's script may not be written yet
            raise AssetError(f"there is no {blender.BLENDER_SIDE / script} to prepare the {' and '.join(classes)}")
        runs.append((script, classes, named))
    wanted = {name for _, classes, _ in runs for name in classes}
    unfetched = _download_problems(root, [(name, entry) for name, entry in selected if name in wanted])
    if unfetched:
        more = f"\n  and {len(unfetched) - 12} more" if len(unfetched) > 12 else ""
        raise AssetError("the downloads are not all the files the manifest names; run the fetch action first "
                         "(python -m sgi.synth.realistic.assets fetch):\n  " + "\n  ".join(unfetched[:12]) + more)
    reports = []
    with tempfile.TemporaryDirectory(prefix="sgi-assets-") as scratch:
        for script, classes, named in runs:
            own = Path(scratch) / f"{Path(script).stem}.json"
            args = ["--manifest", root / "manifest", "--downloads", root / _DOWNLOADS, "--prepared", root / _PREPARED,
                    "--stamp", own]
            args += ["--only", *named] if only is not None else []
            args += ["--force"] if force else []
            blender.run_script(blender.BLENDER_SIDE / script, args, timeout_s=PREPARE_TIMEOUT_S)
            reports.append(_own_stamp(own, script, classes))
    stamp = _read_stamp(root) if only is not None else {}
    for report in reports:
        stamp.update(report)                       # for each class it lists, the whole truth as of now
    _write_json(root / _PREPARED / PREPARED_STAMP, stamp)
    return stamp


# ------------------------------------------------------------------------------------------ attribution
def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def attribution(root: Path = ASSETS_DIR) -> str:
    """The text of the attribution file: every asset of the manifest with its title, author, licence and
    source, under its class and in the manifest's order. The same manifest gives the same text."""
    by_class: dict[str, list[dict]] = {}
    for name, entry in _selected(root, None):
        by_class.setdefault(name, []).append(entry)
    lines = [
        "# Attribution", "",
        "The realistic scene set is drawn with the assets listed here. None of them is part of this repository:",
        "`python -m sgi.synth.realistic.assets fetch` downloads each from its source and checks it against the size",
        "and the SHA-256 that `assets/realistic/manifest/` records. This file is written from that manifest by",
        "`python -m sgi.synth.realistic.assets attribution`; change the manifest, not this file.", "",
        "Every asset is used in a changed form. A model is scaled to metres, turned to face one way and set on the",
        "ground by the middle of its base; its textures are shrunk, a dense mesh is thinned, and its glass is given",
        "another material. A texture set is projected onto surfaces at its real size, and some are tinted. An avatar",
        "is posed and baked to a still mesh with smaller textures. What was done to one asset alone is noted with it."]
    for name in ASSET_CLASSES:
        if not by_class.get(name):
            continue
        lines += ["", f"## {_HEADINGS[name]}", "", "| Asset | Title | Author | Licence | Source | Notes |",
                  "|---|---|---|---|---|---|"]
        for entry in by_class[name]:
            where = entry.get("page") or (entry["sources"][0]["url"] if entry["sources"] else "made in code")
            notes = [entry["changes"]] if entry.get("changes") else []
            notes += ["shows a brand: listed, never drawn"] if entry["brand"] else []
            notes += ["not to be passed on"] if not entry["redistributable"] else []
            cells = (f"`{entry['id']}`", entry.get("title", entry["id"]), entry["author"], entry["licence"], where,
                     "; ".join(notes))
            lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    used = [family for family in _LICENCE_TEXTS
            if any(licence_family(entry["licence"]) == family for entries in by_class.values() for entry in entries)]
    if used:
        lines += ["", "## Licences", ""] + [f"- {family}: {_LICENCE_TEXTS[family]}" for family in used]
    return "\n".join(lines) + "\n"


def write_attribution(root: Path = ASSETS_DIR) -> Path:
    path = root / "ATTRIBUTION.md"
    path.write_text(attribution(root), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------- Blender itself
def _unpack(archive: Path, into: Path, folder: str) -> None:
    """Unpack an archive whose every member lies under its one folder; anything else in it, and nothing is
    unpacked."""
    try:
        with tarfile.open(archive, "r:xz") as tar:
            for member in tar.getmembers():
                parts = PurePosixPath(member.name).parts
                if parts[:1] != (folder,) or ".." in parts:
                    raise AssetError(f"{archive.name} holds {member.name!r}, which is not under {folder}/: nothing "
                                     "of it was unpacked")
            tar.extractall(into, filter="data")    # and no link or device that leads out of it
    except (tarfile.TarError, OSError) as error:
        raise AssetError(f"{archive.name} could not be unpacked as {folder} ({error})") from error
    if not (into / folder / "blender").is_file():
        raise AssetError(f"{archive.name} holds no {folder}/blender")


def fetch_blender(force: bool = False) -> Path:
    """Blender of the series the scenes are built with, in `.tools/` of the checkout, where find_blender looks:
    downloaded, checked against the published SHA-256, unpacked, and asked for its version. The binary.

    One that is there already is left alone unless `force` asks for it again. The published archive is for
    Linux on x86-64; elsewhere the error says to install Blender and name it in SGI_BLENDER."""
    tools = blender.REPO_ROOT / ".tools"
    archive_name = PurePosixPath(urllib.parse.urlsplit(BLENDER_URL).path).name
    folder = archive_name.removesuffix(".tar.xz")
    binary = tools / folder / "blender"
    if binary.is_file() and not force:
        return binary
    if not sys.platform.startswith("linux") or platform.machine() not in ("x86_64", "AMD64"):
        raise AssetError(f"{archive_name} is Blender for Linux on x86-64, and this machine is {sys.platform} on "
                         f"{platform.machine()}: install Blender {blender.BLENDER_SERIES} yourself and name its binary "
                         "in SGI_BLENDER")
    tools.mkdir(parents=True, exist_ok=True)
    if not (tools / ".gitignore").exists():
        (tools / ".gitignore").write_text("*\n", encoding="utf-8")       # nothing under .tools/ is ever committed
    with tempfile.TemporaryDirectory(dir=tools, prefix=".fetch-") as scratch:
        archive = Path(scratch) / archive_name
        _fetch_one(BLENDER_URL, archive, {"sha256": BLENDER_SHA256}, "Blender", "Blender's published list")
        _unpack(archive, Path(scratch) / "unpacked", folder)
        if binary.parent.exists():
            shutil.rmtree(binary.parent)
        os.replace(Path(scratch) / "unpacked" / folder, binary.parent)
    blender.blender_version(binary)
    return binary


# -------------------------------------------------------------------------------------------- the command
def _files(count: int) -> str:
    return f"{count} file" + ("" if count == 1 else "s")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one action; 0 when it did what was asked, 1 when `verify` found something, 2 with an error."""
    parser = argparse.ArgumentParser(prog="python -m sgi.synth.realistic.assets",
                                     description="Fetch, verify and prepare the assets of the realistic scene set.")
    parser.add_argument("action", choices=("blender", "fetch", "prepare", "verify", "attribution"),
                        help="blender: fetch Blender into .tools/; fetch: download the assets; prepare: have Blender "
                             "make of them what the scenes load; verify: say what is missing or wrong; attribution: "
                             "write ATTRIBUTION.md")
    parser.add_argument("--only", nargs="+", metavar="ID", help="fetch, prepare or verify these assets alone")
    parser.add_argument("--force", action="store_true", help="fetch or prepare again what is there already")
    args = parser.parse_args(argv)
    if args.only is not None and args.action not in ("fetch", "prepare", "verify"):
        parser.error(f"--only is not for the action {args.action}")
    if args.force and args.action not in ("blender", "fetch", "prepare"):
        parser.error(f"--force is not for the action {args.action}")
    root = ASSETS_DIR
    try:
        if args.action == "blender":
            binary = fetch_blender(args.force)
            print(f"Blender {blender.blender_version(binary)} is at {binary}")
        elif args.action == "fetch":
            got = fetch(root, args.only, args.force, report=print)
            print(f"fetched {_files(len(got))}, {sum(path.stat().st_size for path in got) / 1e6:.1f} MB, into "
                  f"{root / _DOWNLOADS}; everything else named was there already")
        elif args.action == "prepare":
            print("preparing in Blender, which says nothing until it has ended; a whole library takes a while",
                  flush=True)
            stamp = prepare(root, args.only, args.force)
            entries = manifest_entries(root)
            for name in ASSET_CLASSES:
                done = len(set(stamp.get(name, ())) & {entry["id"] for entry in entries[name]})
                print(f"{name}: {done} of {len(entries[name])} prepared")
        elif args.action == "verify":
            problems = verify(root, args.only)
            for line in problems:
                print(line)
            if not problems:
                print(f"all {len(_selected(root, args.only))} assets are fetched and prepared")
                return 0
            if any(" is not fetched" in line or "the manifest names" in line for line in problems):
                print("to fetch what is missing: python -m sgi.synth.realistic.assets fetch")
            if any("the manifest names" not in line and " is not fetched" not in line for line in problems):
                print("to prepare what is not: python -m sgi.synth.realistic.assets prepare")
            return 1
        else:
            print(f"wrote {write_attribution(root)}")
    except (RuntimeError, ValueError) as error:    # an AssetError, no Blender, a script that failed, a manifest
        print(f"error: {error}", file=sys.stderr)  # that does not read
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
