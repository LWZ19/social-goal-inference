"""Packing a rendered set of the realistic scenes into one archive, and fetching and verifying it again.

Nothing here uses the network: the set is a few small files made in the test, `fetch` is run from a local
archive, and the one place that would download is made unusable."""
import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import pytest

from sgi.synth.realistic import package


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test of the package used the network")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def make_set(root: Path, simulator: str = "sgi-synth-blender/photo-a", scenes: int = 2) -> Path:
    """A set of `scenes` scene folders, with arbitrary bytes of the right kind in each file."""
    for n in range(scenes):
        folder = root / f"scene_{n:02d}"
        (folder / "tour").mkdir(parents=True)
        (folder / "scene.json").write_text(json.dumps({"scene_id": f"scene_{n:02d}", "source": {"simulator": simulator}}))
        (folder / "rgb.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([n]) * 50)
        (folder / "depth.npy").write_bytes(b"\x93NUMPY" + bytes([n + 7]) * 80)
        (folder / "tour" / "000.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"a" * (10 + n))
        (folder / "tour" / "001.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"b" * (10 + n))
    return root


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pack(tmp_path, name="set-a.tar.gz", **extra):
    folder = make_set(tmp_path / "set")
    archive, manifest = tmp_path / "out" / name, tmp_path / "manifests" / "photo-a.json"
    argv = ["pack", "--set", str(folder), "--out", str(archive), "--manifest", str(manifest)]
    for key, value in extra.items():
        argv += [f"--{key}", str(value)]
    assert package.main(argv) == 0
    return folder, archive, manifest


def write_archive(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)


def file_info(name: str, data: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info


def test_pack_writes_an_archive_and_a_manifest_listing_every_file(tmp_path):
    folder, archive, manifest_path = pack(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["variant"] == "photo-a"
    assert manifest["archive"] == {"name": "set-a.tar.gz", "bytes": archive.stat().st_size,
                                   "sha256": sha(archive.read_bytes())}
    paths = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    assert [f["path"] for f in manifest["files"]] == paths and len(paths) == 10
    for entry in manifest["files"]:
        data = (folder / entry["path"]).read_bytes()
        assert entry == {"path": entry["path"], "bytes": len(data), "sha256": sha(data)}
    with tarfile.open(archive) as tar:
        assert tar.getnames() == paths


def test_the_manifest_goes_to_the_sets_folder_of_the_repository_unless_told_otherwise(tmp_path, monkeypatch, capsys):
    assert package.default_manifest("photo-a") == package.SETS_DIR / "photo-a.json"
    assert package.SETS_DIR.parts[-3:] == ("assets", "realistic", "sets")
    sets = tmp_path / "repo" / "sets"                  # stands for the folder that does not exist yet
    monkeypatch.setattr(package, "SETS_DIR", sets)
    folder = make_set(tmp_path / "set")
    assert package.main(["pack", "--set", str(folder), "--out", str(tmp_path / "o.tar.gz")]) == 0
    assert [p.name for p in sets.iterdir()] == ["photo-a.json"]
    assert json.loads((sets / "photo-a.json").read_text())["variant"] == "photo-a"
    capsys.readouterr()
    assert package.main(["verify", "--set", str(folder)]) == 0
    (folder / "scene_00" / "rgb.png").write_bytes(b"changed")
    assert package.main(["verify", "--set", str(folder)]) == 1
    assert package.main(["fetch", "--archive", str(tmp_path / "o.tar.gz"), "--variant", "photo-a",
                         "--into", str(tmp_path / "t")]) == 0


def test_the_variant_is_read_from_the_scenes_and_can_be_overridden(tmp_path):
    folder = make_set(tmp_path / "set")
    assert package.variant_of(folder) == "photo-a"
    (folder / "scene_01" / "scene.json").write_text(
        json.dumps({"source": {"simulator": "sgi-synth-blender/photo-b"}}))
    with pytest.raises(package.PackageError, match="more than one variant"):
        package.variant_of(folder)
    out = tmp_path / "x.tar.gz"
    assert package.main(["pack", "--set", str(folder), "--out", str(out), "--variant", "photo-c",
                         "--manifest", str(tmp_path / "m.json")]) == 0
    assert json.loads((tmp_path / "m.json").read_text())["variant"] == "photo-c"


def test_a_set_with_no_variant_in_its_scenes_needs_one_to_be_named(tmp_path):
    folder = make_set(tmp_path / "set", simulator="sgi-synth")
    with pytest.raises(package.PackageError, match="--variant"):
        package.variant_of(folder)


def test_packing_the_same_set_twice_gives_the_same_archive_bytes(tmp_path, monkeypatch):
    folder = make_set(tmp_path / "set")
    copy = tmp_path / "copy"                           # same bytes, other modes and times, packed later
    shutil.copytree(folder, copy)
    for path in copy.rglob("*"):
        os.utime(path, (1_000_000_000, 1_000_000_000))
        if path.is_file():
            path.chmod(0o600)
    first, second = tmp_path / "a" / "one.tar.gz", tmp_path / "b" / "two.tar.gz"
    assert package.main(["pack", "--set", str(folder), "--out", str(first), "--manifest", str(tmp_path / "m.json")]) == 0
    later = time.time() + 5000
    monkeypatch.setattr(time, "time", lambda: later)
    assert package.main(["pack", "--set", str(copy), "--out", str(second), "--manifest", str(tmp_path / "m2.json")]) == 0
    assert first.read_bytes() == second.read_bytes()
    data = first.read_bytes()
    assert data[:2] == b"\x1f\x8b" and data[3] == 0 and data[4:8] == bytes(4)    # no name, no time in the header
    with tarfile.open(first) as tar:
        for member in tar:
            assert (member.mtime, member.uid, member.gid, member.uname, member.gname, member.mode) == (0, 0, 0, "", "", 0o644)


def test_the_default_archive_is_gzip_and_zstd_is_only_offered_when_the_module_is_there(tmp_path, monkeypatch):
    folder = make_set(tmp_path / "set")
    monkeypatch.setitem(__import__("sys").modules, "zstandard", None)      # as if it were not installed
    with pytest.raises(SystemExit):
        package.main(["pack", "--set", str(folder), "--out", str(tmp_path / "s.tar.zst"), "--zstd",
                      "--manifest", str(tmp_path / "m.json")])
    assert not (tmp_path / "s.tar.zst").exists() and not (tmp_path / "m.json").exists()
    with pytest.raises(SystemExit):                    # the name must say what the bytes are
        package.main(["pack", "--set", str(folder), "--out", str(tmp_path / "s.tar"), "--manifest", str(tmp_path / "m.json")])


def test_pack_refuses_a_link_in_the_set_and_an_empty_set(tmp_path):
    folder = make_set(tmp_path / "set")
    (folder / "link.png").symlink_to(folder / "scene_00" / "rgb.png")
    with pytest.raises(package.PackageError, match="link"):
        package.pack(folder, tmp_path / "o.tar.gz", tmp_path / "m.json")
    (tmp_path / "empty").mkdir()
    with pytest.raises(package.PackageError, match="no scene.json"):
        package.pack(tmp_path / "empty", tmp_path / "o2.tar.gz", tmp_path / "m2.json")
    assert not (tmp_path / "o.tar.gz").exists()


def test_verify_passes_on_the_packed_set_and_names_a_changed_missing_or_extra_file(tmp_path, capsys):
    folder, _, manifest = pack(tmp_path)
    assert package.main(["verify", "--set", str(folder), "--manifest", str(manifest)]) == 0
    assert "10 files" in capsys.readouterr().out
    (folder / "scene_00" / "rgb.png").write_bytes(b"changed")
    (folder / "scene_01" / "depth.npy").unlink()
    (folder / "scene_01" / "extra.txt").write_text("x")
    problems = package.verify(folder, json.loads(manifest.read_text()))
    assert len(problems) == 3
    assert any("scene_00/rgb.png" in p and "changed" in p for p in problems)
    assert any("scene_01/depth.npy" in p and "missing" in p for p in problems)
    assert any("scene_01/extra.txt" in p and "not in the manifest" in p for p in problems)
    capsys.readouterr()
    assert package.main(["verify", "--set", str(folder), "--manifest", str(manifest)]) == 1
    out = capsys.readouterr().out
    assert "scene_00/rgb.png" in out and "scene_01/depth.npy" in out and "scene_01/extra.txt" in out


def test_verify_without_a_manifest_file_says_so(tmp_path, capsys):
    folder = make_set(tmp_path / "set")
    assert package.main(["verify", "--set", str(folder), "--manifest", str(tmp_path / "none.json")]) == 2
    assert "none.json" in capsys.readouterr().err


def test_fetch_from_a_local_archive_restores_the_set_and_verifies_it(tmp_path, capsys):
    folder, archive, manifest = pack(tmp_path)
    target = tmp_path / "data" / "photo-a"
    assert package.main(["fetch", "--archive", str(archive), "--manifest", str(manifest), "--into", str(target)]) == 0
    assert package.verify(target, json.loads(manifest.read_text())) == []
    assert (target / "scene_01" / "tour" / "001.png").read_bytes() == (folder / "scene_01" / "tour" / "001.png").read_bytes()
    assert [p.name for p in target.parent.iterdir()] == ["photo-a"]            # no temporary folder is left
    capsys.readouterr()
    assert package.main(["fetch", "--archive", str(archive), "--manifest", str(manifest), "--into", str(target)]) == 2
    assert "exists" in capsys.readouterr().err                                 # nothing is overwritten unasked
    (target / "scene_00" / "rgb.png").write_bytes(b"spoilt")
    assert package.main(["fetch", "--archive", str(archive), "--manifest", str(manifest), "--into", str(target),
                         "--force"]) == 0
    assert package.verify(target, json.loads(manifest.read_text())) == []
    assert [p.name for p in target.parent.iterdir()] == ["photo-a"]


def test_fetch_finds_the_manifest_by_variant_in_the_sets_folder(tmp_path, monkeypatch):
    _, archive, manifest = pack(tmp_path)
    monkeypatch.setattr(package, "SETS_DIR", manifest.parent)
    target = tmp_path / "t"
    assert package.main(["fetch", "--archive", str(archive), "--variant", "photo-a", "--into", str(target)]) == 0
    with pytest.raises(SystemExit):                                            # neither a manifest nor a variant
        package.main(["fetch", "--archive", str(archive), "--into", str(tmp_path / "t2")])


def test_fetch_refuses_an_archive_whose_checksum_differs_and_leaves_nothing_behind(tmp_path, capsys):
    folder, archive, manifest = pack(tmp_path)
    other = tmp_path / "other.tar.gz"
    make_set(tmp_path / "set2", scenes=3)
    assert package.main(["pack", "--set", str(tmp_path / "set2"), "--out", str(other),
                         "--manifest", str(tmp_path / "m2.json")]) == 0
    target = tmp_path / "data" / "photo-a"
    target.parent.mkdir()
    capsys.readouterr()
    assert package.main(["fetch", "--archive", str(other), "--manifest", str(manifest), "--into", str(target)]) == 2
    assert "sha256" in capsys.readouterr().err
    assert list(target.parent.iterdir()) == []


def test_fetch_refuses_an_archive_that_holds_a_file_other_than_the_manifest_names(tmp_path):
    """Right checksum for its own bytes is not enough: the manifest, edited to fit, still names the files."""
    _, archive, manifest = pack(tmp_path)
    data = json.loads(manifest.read_text())
    victim = data["files"][0]
    tampered = tmp_path / "tampered.tar.gz"
    members = []
    with tarfile.open(archive) as tar:
        for member in tar:
            body = tar.extractfile(member).read()
            members.append((file_info(member.name, b"evil" if member.name == victim["path"] else body),
                            b"evil" if member.name == victim["path"] else body))
    write_archive(tampered, members)
    data["archive"] = {"name": tampered.name, "bytes": tampered.stat().st_size, "sha256": sha(tampered.read_bytes())}
    manifest.write_text(json.dumps(data))
    target = tmp_path / "data" / "photo-a"
    target.parent.mkdir()
    with pytest.raises(package.PackageError, match=victim["path"]):
        package.fetch(data, tampered, target)
    assert list(target.parent.iterdir()) == []


def test_fetch_refuses_an_archive_with_a_missing_or_an_extra_file(tmp_path):
    _, archive, manifest = pack(tmp_path)
    data = json.loads(manifest.read_text())
    with tarfile.open(archive) as tar:
        bodies = {m.name: tar.extractfile(m).read() for m in tar}
    for label, kept in (("missing", {k: v for k, v in bodies.items() if k != "scene_00/rgb.png"}),
                        ("extra", {**bodies, "scene_00/more.txt": b"x"})):
        broken = tmp_path / f"{label}.tar.gz"
        write_archive(broken, [(file_info(k, v), v) for k, v in sorted(kept.items())])
        data["archive"] = {"name": broken.name, "bytes": broken.stat().st_size, "sha256": sha(broken.read_bytes())}
        target = tmp_path / "data" / label
        target.parent.mkdir(exist_ok=True)
        with pytest.raises(package.PackageError, match="rgb.png|more.txt"):
            package.fetch(data, broken, target)
        assert list(target.parent.iterdir()) == []


@pytest.mark.parametrize("name", ["../x", "scene_00/../../x", "/etc/x", "/tmp/abs.png", "a/./../../x", "\\evil", "C:/x", ""])
def test_an_archive_member_cannot_escape_the_target_folder(tmp_path, name):
    folder, archive, manifest = pack(tmp_path)
    data = json.loads(manifest.read_text())
    with tarfile.open(archive) as tar:
        members = [(file_info(m.name, b := tar.extractfile(m).read()), b) for m in tar]
    members.append((file_info(name, b"escaped"), b"escaped"))
    bad = tmp_path / "bad.tar.gz"
    write_archive(bad, members)
    data["archive"] = {"name": "bad.tar.gz", "bytes": bad.stat().st_size, "sha256": sha(bad.read_bytes())}
    target = tmp_path / "deep" / "data" / "photo-a"
    target.parent.mkdir(parents=True)
    with pytest.raises(package.PackageError, match="member"):
        package.fetch(data, bad, target)
    assert list(target.parent.iterdir()) == []
    assert not (tmp_path / "x").exists() and not (tmp_path / "deep" / "x").exists()


@pytest.mark.parametrize("name", ["../x", "scene_00/../../x", "/tmp/abs-from-the-package-test.png"])
def test_a_manifest_naming_the_escaping_member_does_not_make_it_safe(tmp_path, name):
    """The name checks stand on their own, not on the manifest's list happening not to hold the name."""
    _, archive, manifest = pack(tmp_path)
    data = json.loads(manifest.read_text())
    with tarfile.open(archive) as tar:
        members = [(file_info(m.name, b := tar.extractfile(m).read()), b) for m in tar]
    members.append((file_info(name, b"escaped"), b"escaped"))
    bad = tmp_path / "bad.tar.gz"
    write_archive(bad, members)
    data["files"].append({"path": name, "bytes": 7, "sha256": sha(b"escaped")})
    data["archive"] = {"name": "bad.tar.gz", "bytes": bad.stat().st_size, "sha256": sha(bad.read_bytes())}
    target = tmp_path / "deep" / "data" / "photo-a"
    target.parent.mkdir(parents=True)
    with pytest.raises(package.PackageError, match="member"):
        package.fetch(data, bad, target)
    assert list(target.parent.iterdir()) == []
    assert not (tmp_path / "x").exists() and not (tmp_path / "deep" / "x").exists()
    assert not Path("/tmp/abs-from-the-package-test.png").exists()
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text(json.dumps(data))
    with pytest.raises(package.PackageError, match="manifest"):                # and the manifest itself is checked
        package.load_manifest(bad_manifest)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE])
def test_an_archive_member_that_is_a_link_or_a_device_is_refused(tmp_path, kind):
    """The special member bears a name the manifest lists, so only the check of its type can refuse it."""
    _, archive, manifest = pack(tmp_path)
    data = json.loads(manifest.read_text())
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "rgb.png").write_bytes(b"outside")
    with tarfile.open(archive) as tar:
        members = [(file_info(m.name, b := tar.extractfile(m).read()), b) for m in tar]

    def special(name, link):
        info = tarfile.TarInfo(name)
        info.type, info.linkname = kind, link
        return info, None
    cases = {"file": [m for m in members if m[0].name != "scene_00/rgb.png"] + [special("scene_00/rgb.png", str(outside / "rgb.png"))],
             "folder first": [special("scene_00", str(outside))] + members}
    for label, changed in cases.items():
        bad = tmp_path / f"{label.replace(' ', '-')}.tar.gz"
        write_archive(bad, changed)
        data["archive"] = {"name": bad.name, "bytes": bad.stat().st_size, "sha256": sha(bad.read_bytes())}
        target = tmp_path / "data" / "photo-a"
        target.parent.mkdir(exist_ok=True)
        with pytest.raises(package.PackageError, match="link, a device"):
            package.fetch(data, bad, target)
        assert list(target.parent.iterdir()) == []
        assert [p.name for p in outside.iterdir()] == ["rgb.png"] and (outside / "rgb.png").read_bytes() == b"outside"


def test_a_member_listed_twice_or_larger_than_the_manifest_says_is_refused(tmp_path):
    _, archive, manifest = pack(tmp_path)
    data = json.loads(manifest.read_text())
    with tarfile.open(archive) as tar:
        members = [(file_info(m.name, b := tar.extractfile(m).read()), b) for m in tar]
    for label, changed in (("twice", members + [members[0]]),
                           ("big", [(file_info(members[0][0].name, b"z" * 5000), b"z" * 5000)] + members[1:])):
        bad = tmp_path / f"{label}.tar.gz"
        write_archive(bad, changed)
        data["archive"] = {"name": bad.name, "bytes": bad.stat().st_size, "sha256": sha(bad.read_bytes())}
        target = tmp_path / "data" / label
        target.parent.mkdir(exist_ok=True)
        with pytest.raises(package.PackageError):
            package.fetch(data, bad, target)
        assert list(target.parent.iterdir()) == []


def test_an_archive_that_is_not_an_archive_is_refused_and_leaves_nothing(tmp_path):
    _, archive, manifest = pack(tmp_path)
    data = json.loads(manifest.read_text())
    junk = tmp_path / "junk.tar.gz"
    junk.write_bytes(b"not a tar at all" * 20)
    data["archive"] = {"name": "junk.tar.gz", "bytes": junk.stat().st_size, "sha256": sha(junk.read_bytes())}
    target = tmp_path / "data" / "photo-a"
    target.parent.mkdir()
    with pytest.raises(package.PackageError):
        package.fetch(data, junk, target)
    assert list(target.parent.iterdir()) == []


def test_fetch_with_a_url_downloads_to_a_temporary_file_and_then_unpacks_it(tmp_path, monkeypatch):
    _, archive, manifest = pack(tmp_path)
    asked = []

    def fake_urlopen(request, timeout=None):
        asked.append(request.full_url)
        return io.BytesIO(archive.read_bytes())
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    target = tmp_path / "data" / "photo-a"
    assert package.main(["fetch", "--url", "https://example.invalid/set-a.tar.gz", "--manifest", str(manifest),
                         "--into", str(target)]) == 0
    assert asked == ["https://example.invalid/set-a.tar.gz"]
    assert package.verify(target, json.loads(manifest.read_text())) == []
    assert [p.name for p in target.parent.iterdir()] == ["photo-a"]
    with pytest.raises(SystemExit):
        package.main(["fetch", "--url", "https://x/y", "--archive", str(archive), "--manifest", str(manifest),
                      "--into", str(tmp_path / "z")])
    with pytest.raises(package.PackageError, match="https"):
        package.download("http://example.invalid/a", tmp_path / "a")


needs_permissions = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")


def run_cli(*argv):
    import subprocess
    import sys
    return subprocess.run([sys.executable, "-m", "sgi.synth.realistic.package", *argv], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": ""})


def test_an_operating_system_error_is_an_error_line_and_status_2_never_a_traceback(tmp_path, capsys):
    _, archive, manifest = pack(tmp_path)
    assert package.main(["fetch", "--archive", str(tmp_path), "--manifest", str(manifest),
                         "--into", str(tmp_path / "t")]) == 2                    # a folder is not an archive
    assert "error:" in capsys.readouterr().err and not (tmp_path / "t").exists()
    for into in (".", "/", "..", str(tmp_path / "x" / "..")):                    # no name of its own
        assert package.main(["fetch", "--archive", str(archive), "--manifest", str(manifest), "--into", into,
                             "--force"]) == 2
        assert "no name of its own" in capsys.readouterr().err


@needs_permissions
def test_unwritable_and_unreadable_paths_give_status_2_in_every_action_without_a_traceback(tmp_path):
    folder, archive, manifest = pack(tmp_path)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    unreadable = folder / "scene_00" / "rgb.png"
    try:
        done = run_cli("pack", "--set", str(folder), "--out", str(locked / "x.tar.gz"), "--manifest", str(tmp_path / "m9.json"))
        assert (done.returncode, "error:" in done.stderr, "Traceback" in done.stderr) == (2, True, False), done.stderr
        done = run_cli("fetch", "--archive", str(archive), "--manifest", str(manifest), "--into", str(locked / "t"))
        assert (done.returncode, "error:" in done.stderr, "Traceback" in done.stderr) == (2, True, False), done.stderr
        unreadable.chmod(0)
        done = run_cli("verify", "--set", str(folder), "--manifest", str(manifest))
        assert (done.returncode, "error:" in done.stderr, "Traceback" in done.stderr) == (2, True, False), done.stderr
    finally:
        unreadable.chmod(0o644)
        locked.chmod(0o700)
    assert list(locked.iterdir()) == [] and not (tmp_path / "m9.json").exists()
    assert run_cli("verify", "--set", str(folder), "--manifest", str(manifest)).returncode == 0


def test_a_cut_off_download_is_an_error_line_and_leaves_nothing(tmp_path, monkeypatch, capsys):
    import http.client
    _, archive, manifest = pack(tmp_path)

    class Cut(io.BytesIO):
        def read(self, *args):
            raise http.client.IncompleteRead(b"ab")
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: Cut())
    target = tmp_path / "data" / "photo-a"
    assert package.main(["fetch", "--url", "https://example.invalid/a.tar.gz", "--manifest", str(manifest),
                         "--into", str(target)]) == 2
    assert "error:" in capsys.readouterr().err and not target.parent.exists()
