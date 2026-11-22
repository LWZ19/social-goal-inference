"""Packing a rendered set of the realistic scenes, and fetching and verifying it again.

Rendered sets are never committed: a variant travels as one archive that is attached by hand
to a release of the private repository, and the repository holds its manifest and this command.

    python -m sgi.synth.realistic.package pack   --set FOLDER --out FILE.tar.gz [--manifest FILE] [--variant NAME]
    python -m sgi.synth.realistic.package verify --set FOLDER [--manifest FILE | --variant NAME]
    python -m sgi.synth.realistic.package fetch  (--archive FILE | --url HTTPS) --into FOLDER
                                                 (--manifest FILE | --variant NAME) [--force]

`pack` writes the archive and then the manifest, which names the variant, the archive (file name, size, SHA-256)
and every file of the set (path under the set folder, size, SHA-256) sorted by path:

    {"variant": "...", "archive": {"name", "bytes", "sha256"}, "files": [{"path", "bytes", "sha256"}, ...]}

By default the manifest is assets/realistic/sets/<variant>.json. The variant is the part after the slash of
source.simulator ("sgi-synth-<renderer>/<variant>") that every scene.json of the set carries; `--variant`
overrides it. The archive is a gzip tar by default, written to the same bytes for the same files (sorted names,
mtime, owner and mode fixed, no name or time in the gzip header). `--zstd` writes a .tar.zst when the python
module zstandard happens to be importable; it is not a dependency.

`fetch` checks the archive against the manifest, unpacks it into a temporary folder beside `--into`, checks every
file against the manifest, and only then moves the folder into place; on any failure nothing is left behind.
A member that is absolute, has a ".." or other odd component, is a link or a device, is not in the manifest, is
listed twice or is not the size the manifest says is refused before it is written. `--url` (https only) first
downloads to a temporary file, and then goes on as `--archive`. Nothing is uploaded by this code.

Exit status: 0 when it did what was asked, 1 when `verify` found something, 2 with an error.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zlib
from pathlib import Path, PurePosixPath
from typing import Iterator, Sequence

from sgi.config import REPO_ROOT

SETS_DIR = REPO_ROOT / "assets" / "realistic" / "sets"
_GZIP_SUFFIXES = (".tar.gz", ".tgz")
_ZSTD_SUFFIX = ".tar.zst"
_VARIANT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SIMULATOR = re.compile(r"sgi-synth-[^/]+/(?P<variant>.+)")
_TIMEOUT_S = 120
_USER_AGENT = "sgi-realistic-package"


class PackageError(RuntimeError):
    """Something that stops a pack, a verify or a fetch; the message says what and where."""


# ------------------------------------------------------------------------------------------------ files
def default_manifest(variant: str) -> Path:
    return SETS_DIR / f"{variant}.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while block := file.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _scan(root: Path) -> tuple[list[str], list[str]]:
    """The files under `root` as paths relative to it, sorted, and the links among what is there."""
    files, links = [], []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            links.append(relative)
        elif path.is_file():
            files.append(relative)
    return sorted(files), links


def _bad_name(name: str) -> str | None:
    """Why a path in an archive or a manifest cannot be a file inside the target folder; None when it can."""
    if not name:
        return "has no name"
    if name.startswith("/") or "\\" in name or "\0" in name or re.match(r"[A-Za-z]:", name):
        return "is absolute or not a plain relative path"
    if any(part in ("", ".", "..") for part in PurePosixPath(name).parts) or name.endswith("/") or "//" in name:
        return "has a '..', '.' or empty component"
    return None


def variant_of(folder: Path) -> str:
    """The variant the scenes under `folder` say they are, from source.simulator."""
    scenes = sorted(folder.rglob("scene.json"))
    if not scenes:
        raise PackageError(f"{folder} holds no scene.json: is it a set of scenes?")
    found = {}
    for scene in scenes:
        try:
            simulator = json.loads(scene.read_text(encoding="utf-8"))["source"]["simulator"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise PackageError(f"{scene}: source.simulator cannot be read ({error})") from error
        match = _SIMULATOR.fullmatch(str(simulator))
        if not match:
            raise PackageError(f"{scene}: source.simulator is {simulator!r}, which names no variant "
                               f"('sgi-synth-<renderer>/<variant>'); name one with --variant")
        found.setdefault(match["variant"], scene)
    if len(found) > 1:
        raise PackageError(f"the scenes of {folder} name more than one variant ({', '.join(sorted(found))}); "
                           f"pack one variant at a time, or name it with --variant")
    return next(iter(found))


def _check_variant(variant: str) -> str:
    if not _VARIANT.fullmatch(variant):
        raise PackageError(f"{variant!r} is not usable as a variant name (letters, digits, '.', '_' and '-')")
    return variant


# -------------------------------------------------------------------------------------------------- pack
def _zstandard():
    try:
        import zstandard
    except ImportError:
        return None
    return zstandard


def _write_archive(folder: Path, names: list[str], path: Path, zstd: bool) -> None:
    """Write a tar of `names` to `path`, whole or not at all. Same files, same bytes: sorted names, and the
    time, owner and mode of every member fixed."""
    part = path.with_name(path.name + ".part")
    try:
        with open(part, "wb") as raw:
            if zstd:
                compressor = _zstandard().ZstdCompressor(level=19).stream_writer(raw, closefd=False)
            else:
                compressor = gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0)
            with compressor, tarfile.open(fileobj=compressor, mode="w", format=tarfile.GNU_FORMAT) as tar:
                for name in names:
                    info = tarfile.TarInfo(name)
                    info.size, info.mtime, info.mode, info.uid, info.gid, info.uname, info.gname = (
                        (folder / name).stat().st_size, 0, 0o644, 0, 0, "", "")
                    with open(folder / name, "rb") as file:
                        tar.addfile(info, file)
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        part.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def pack(folder: Path, out: Path, manifest_path: Path | None = None, variant: str | None = None,
         zstd: bool = False) -> dict:
    """Pack the set in `folder` into the archive `out` and write its manifest; return the manifest."""
    folder, out = Path(folder), Path(out)
    if not folder.is_dir():
        raise PackageError(f"{folder} is not a folder")
    if zstd and _zstandard() is None:
        raise PackageError("--zstd needs the python module zstandard, which is not installed; use the default gzip")
    if out.name.endswith(_ZSTD_SUFFIX) != zstd or not (zstd or out.name.endswith(_GZIP_SUFFIXES)):
        raise PackageError(f"{out.name} must end in {_ZSTD_SUFFIX if zstd else ' or '.join(_GZIP_SUFFIXES)}")
    variant = _check_variant(variant if variant is not None else variant_of(folder))
    manifest_path = Path(manifest_path) if manifest_path else default_manifest(variant)
    names, links = _scan(folder)
    if links:
        raise PackageError(f"{folder} holds links ({', '.join(links[:5])}); a set is plain files")
    if not names:
        raise PackageError(f"{folder} holds no files")
    if out.resolve().is_relative_to(folder.resolve()):
        raise PackageError(f"the archive {out} would lie inside the set it packs")
    files = [{"path": name, "bytes": (folder / name).stat().st_size, "sha256": _sha256(folder / name)}
             for name in names]
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_archive(folder, names, out, zstd)
    manifest = {"variant": variant,
                "archive": {"name": out.name, "bytes": out.stat().st_size, "sha256": _sha256(out)},
                "files": files}
    _write_json(manifest_path, manifest)
    return manifest


# ------------------------------------------------------------------------------------------------ verify
def load_manifest(path: Path) -> dict:
    """The manifest in `path`, checked to have the layout this module writes."""
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise PackageError(f"the manifest {path} cannot be read ({error})") from error
    except ValueError as error:
        raise PackageError(f"the manifest {path} is not JSON ({error})") from error
    try:
        archive, files = manifest["archive"], manifest["files"]
        ok = (isinstance(manifest["variant"], str) and isinstance(archive["name"], str)
              and isinstance(archive["bytes"], int) and isinstance(archive["sha256"], str)
              and isinstance(files, list)
              and all(isinstance(f["path"], str) and isinstance(f["bytes"], int) and isinstance(f["sha256"], str)
                      for f in files))
    except (KeyError, TypeError):
        ok = False
    if not ok:
        raise PackageError(f"the manifest {path} lacks variant, archive {{name, bytes, sha256}} or files "
                           f"[{{path, bytes, sha256}}]")
    paths = [f["path"] for f in files]
    for name in paths:
        if _bad_name(name):
            raise PackageError(f"the manifest {path} names {name!r}, which {_bad_name(name)}")
    if len(set(paths)) != len(paths):
        raise PackageError(f"the manifest {path} names a file twice")
    return manifest


def verify(folder: Path, manifest: dict) -> list[str]:
    """A line for each file of the set that is missing, changed, or not in the manifest; empty when it is
    the set the manifest describes."""
    folder = Path(folder)
    if not folder.is_dir():
        return [f"{folder} is not a folder"]
    names, links = _scan(folder)
    wanted = {f["path"]: f for f in manifest["files"]}
    problems = []
    for name, entry in sorted(wanted.items()):
        path = folder / name
        if not path.is_file() or path.is_symlink():
            problems.append(f"{name}: missing")
            continue
        size, found = path.stat().st_size, _sha256(path)
        if size != entry["bytes"] or found != entry["sha256"]:
            problems.append(f"{name}: changed (is {size} bytes with the sha256 {found}, the manifest names "
                            f"{entry['bytes']} bytes with the sha256 {entry['sha256']})")
    problems += [f"{name}: not in the manifest" for name in sorted(set(names) - set(wanted))]
    problems += [f"{name}: is a link" for name in links]
    return problems


# ------------------------------------------------------------------------------------------------- fetch
def download(url: str, path: Path) -> None:
    """Write what an https address gives to `path`. The one place that uses the network; the caller checks
    what was written."""
    if urllib.parse.urlsplit(url).scheme != "https":
        raise PackageError(f"{url}: only https addresses are fetched")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as answer, open(path, "wb") as out:
            shutil.copyfileobj(answer, out, 1 << 20)
    except (OSError, http.client.HTTPException) as error:   # no route, no such page, a cut-off answer
        raise PackageError(f"{url}: could not be fetched ({error})") from error


@contextlib.contextmanager
def _work_beside(target: Path) -> Iterator[Path]:
    """A new temporary folder in the parent of `target` (on the same file system, so that a move is a rename).
    It is removed afterwards; on failure so are the parents made for it, so that nothing is left behind."""
    missing, parent = [], target.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}-"))
    try:
        yield work
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        for made in missing:                       # innermost first
            with contextlib.suppress(OSError):
                made.rmdir()
        raise
    shutil.rmtree(work, ignore_errors=True)


def _check_target(target: Path, force: bool) -> None:
    if target.name in ("", ".", ".."):
        raise PackageError(f"--into {target} has no name of its own; name the folder to create, e.g. data/photo-a")
    if (target.exists() or target.is_symlink()) and not force:
        raise PackageError(f"{target} exists; nothing is overwritten (--force replaces it once the new set "
                           f"has been checked)")


def _check_archive(manifest: dict, archive: Path) -> None:
    wanted = manifest["archive"]
    try:
        size = archive.stat().st_size
    except OSError as error:
        raise PackageError(f"the archive {archive} cannot be read ({error})") from error
    found = _sha256(archive)
    if size != wanted["bytes"] or found != wanted["sha256"]:
        raise PackageError(f"the archive {archive} is {size} bytes with the sha256 {found}, and the manifest "
                           f"names {wanted['bytes']} bytes with the sha256 {wanted['sha256']} for "
                           f"{wanted['name']}")


def _unpack(archive: Path, manifest: dict, into: Path) -> None:
    """Write the archive's files under `into`, checking each member first. Nothing is extracted by tarfile
    itself: a member is a name, a type and a size to check, and bytes to copy to a path made here."""
    wanted = {f["path"]: f for f in manifest["files"]}
    seen: set[str] = set()
    root = into.resolve()
    zstd = archive.name.endswith(_ZSTD_SUFFIX)
    if zstd and _zstandard() is None:
        raise PackageError(f"{archive.name} is zstd, which needs the python module zstandard; it is not installed")
    try:
        with open(archive, "rb") as raw:
            stream = _zstandard().ZstdDecompressor().stream_reader(raw) if zstd else raw
            with tarfile.open(fileobj=stream, mode="r|" if zstd else "r|gz") as tar:
                for member in tar:
                    name = member.name
                    if why := _bad_name(name):
                        raise PackageError(f"member {name!r} of the archive {why}")
                    if member.isdir():
                        continue
                    if not member.isreg():
                        raise PackageError(f"member {name!r} of the archive is a link, a device or another "
                                           f"kind of entry than a plain file")
                    if name not in wanted:
                        raise PackageError(f"member {name!r} of the archive is not in the manifest")
                    if name in seen:
                        raise PackageError(f"member {name!r} is listed twice in the archive")
                    if member.size != wanted[name]["bytes"]:
                        raise PackageError(f"member {name!r} of the archive is {member.size} bytes, and the "
                                           f"manifest names {wanted[name]['bytes']}")
                    path = (root / name).resolve()
                    if not path.is_relative_to(root) or path == root:
                        raise PackageError(f"member {name!r} of the archive would land outside {into}")
                    seen.add(name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "wb") as out:
                        shutil.copyfileobj(tar.extractfile(member), out, 1 << 20)
    except (tarfile.TarError, EOFError, OSError, zlib.error) as error:
        raise PackageError(f"the archive {archive.name} cannot be unpacked ({error})") from error
    except Exception as error:                     # the zstandard module's own error classes
        if isinstance(error, PackageError) or "zstandard" not in type(error).__module__:
            raise
        raise PackageError(f"the archive {archive.name} cannot be unpacked ({error})") from error


def fetch(manifest: dict, archive: Path, target: Path, force: bool = False, work: Path | None = None) -> None:
    """Unpack `archive` as the set `target`, once it and every file in it are the ones `manifest` names."""
    archive, target = Path(archive), Path(target)
    _check_target(target, force)
    _check_archive(manifest, archive)
    with _work_beside(target) if work is None else contextlib.nullcontext(work) as scratch:
        staged = scratch / "set"
        staged.mkdir()
        _unpack(archive, manifest, staged)
        problems = verify(staged, manifest)
        if problems:
            raise PackageError("the archive does not hold the set the manifest names:\n  " + "\n  ".join(problems[:12])
                               + (f"\n  ... and {len(problems) - 12} more" if len(problems) > 12 else ""))
        old = scratch / "old"
        if target.exists() or target.is_symlink():
            os.replace(target, old)
        try:
            os.replace(staged, target)
        except OSError:
            if old.exists():
                os.replace(old, target)
            raise


def fetch_url(manifest: dict, url: str, target: Path, force: bool = False) -> None:
    """Download `url` into a temporary file beside `target`, and go on as `fetch` does."""
    target = Path(target)
    _check_target(target, force)
    with _work_beside(target) as scratch:
        archive = scratch / ("download" + (_ZSTD_SUFFIX if urllib.parse.urlsplit(url).path.endswith(_ZSTD_SUFFIX)
                                           else ".tar.gz"))
        download(url, archive)
        fetch(manifest, archive, target, force, work=scratch)


# --------------------------------------------------------------------------------------------------- cli
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sgi.synth.realistic.package",
                                     description="Pack, verify and fetch a rendered set of the realistic scenes.")
    parser.add_argument("action", choices=("pack", "verify", "fetch"),
                        help="pack: archive a set and write its manifest; verify: check a set against its "
                             "manifest; fetch: unpack an archive into a folder and check it")
    parser.add_argument("--set", type=Path, metavar="FOLDER", help="the set folder (pack, verify)")
    parser.add_argument("--out", type=Path, metavar="FILE", help="the archive to write, .tar.gz (pack)")
    parser.add_argument("--manifest", type=Path, metavar="FILE",
                        help="the manifest; default assets/realistic/sets/<variant>.json")
    parser.add_argument("--variant", metavar="NAME", help="the variant; default: read from the scenes (pack, "
                                                          "verify), required without --manifest (fetch)")
    parser.add_argument("--zstd", action="store_true", help="write .tar.zst, if the module zstandard is installed")
    parser.add_argument("--archive", type=Path, metavar="FILE", help="a local archive to unpack (fetch)")
    parser.add_argument("--url", metavar="HTTPS", help="an address to download the archive from (fetch)")
    parser.add_argument("--into", type=Path, metavar="FOLDER", help="the folder to unpack the set into (fetch)")
    parser.add_argument("--force", action="store_true", help="replace the folder if it exists (fetch)")
    args = parser.parse_args(argv)
    action = args.action
    if action in ("pack", "verify") and args.set is None:
        parser.error(f"{action} needs --set")
    if action == "pack" and args.out is None:
        parser.error("pack needs --out")
    if action == "pack" and args.zstd and _zstandard() is None:
        parser.error("--zstd needs the python module zstandard, which is not installed; the default is gzip")
    if action == "pack" and args.out.name.endswith(_ZSTD_SUFFIX) != args.zstd:
        parser.error(f"the archive's name must end in {_ZSTD_SUFFIX} with --zstd and in .tar.gz without it")
    if action == "pack" and not args.zstd and not args.out.name.endswith(_GZIP_SUFFIXES):
        parser.error("the archive's name must end in .tar.gz")
    if action == "fetch":
        if (args.archive is None) == (args.url is None):
            parser.error("fetch needs exactly one of --archive and --url")
        if args.into is None:
            parser.error("fetch needs --into")
        if args.manifest is None and args.variant is None:
            parser.error("fetch needs --manifest, or --variant to find assets/realistic/sets/<variant>.json")
    if action != "pack" and (args.out or args.zstd):
        parser.error("--out and --zstd are for pack")
    if action != "fetch" and (args.archive or args.url or args.into or args.force):
        parser.error("--archive, --url, --into and --force are for fetch")
    try:
        if action == "pack":
            manifest = pack(args.set, args.out, args.manifest, args.variant, args.zstd)
            where = args.manifest or default_manifest(manifest["variant"])
            print(f"packed {len(manifest['files'])} files of {manifest['variant']} into {args.out} "
                  f"({manifest['archive']['bytes']} bytes, sha256 {manifest['archive']['sha256']}); "
                  f"manifest {where}")
            return 0
        if action == "verify":
            variant = args.variant if args.variant else None
            if args.manifest is None:
                variant = _check_variant(variant if variant is not None else variant_of(args.set))
            manifest = load_manifest(args.manifest or default_manifest(variant))
            problems = verify(args.set, manifest)
            if problems:
                print(f"{args.set} is not the set {manifest['variant']} of the manifest:", file=sys.stdout)
                for line in problems:
                    print(f"  {line}")
                return 1
            print(f"{args.set} is the set {manifest['variant']}: {len(manifest['files'])} files, all as the "
                  f"manifest names them")
            return 0
        manifest = load_manifest(args.manifest or default_manifest(_check_variant(args.variant)))
        if args.url:
            fetch_url(manifest, args.url, args.into, args.force)
        else:
            fetch(manifest, args.archive, args.into, args.force)
        print(f"unpacked {manifest['variant']} into {args.into}: {len(manifest['files'])} files, all as the "
              f"manifest names them")
        return 0
    except (PackageError, OSError, http.client.HTTPException) as error:    # an unreadable or unwritable path too
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
