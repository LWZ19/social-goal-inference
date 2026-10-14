"""Finding Blender, checking its version and running a script in it.

Nothing here starts Blender: every binary is a fake, written for the test."""
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sgi.synth.realistic import blender
from sgi.synth.realistic.blender import (BLENDER_SERIES, BLENDER_SIDE, FETCH_COMMAND, BlenderMissing, available,
                                         binary_available, blender_version, find_blender, run_script)

# Answers --version as Blender does. Started any other way it records its arguments and what it was given of the
# environment beside itself, prints a log of 200 lines, waits, and ends with the exit status asked for.
_FAKE = """#!{python}
import json, os, sys, time
if "--version" in sys.argv:
    print({first_line!r})
    print("\\tbuild date: never")
    sys.exit(0)
with open(__file__ + ".json", "w") as out:
    json.dump({{"argv": sys.argv[1:], "pythonpath": os.environ.get("PYTHONPATH"),
               "user": os.environ.get("BLENDER_USER_RESOURCES")}}, out)
for number in range(200):
    print("log line", number)
sys.stdout.flush()
time.sleep({sleep_s})
sys.exit({exit_status})
"""


def fake_blender(path: Path, first_line: str = "Blender 4.5.14 LTS (hash 000000000000 built never)",
                 exit_status: int = 0, sleep_s: float = 0.0) -> Path:
    fake = _FAKE.format(python=sys.executable, first_line=first_line, exit_status=exit_status, sleep_s=sleep_s)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fake)
    path.chmod(0o755)
    return path


def a_script(tmp_path: Path) -> Path:
    script = tmp_path / "scripts" / "job.py"
    script.parent.mkdir()
    script.write_text("raise SystemExit('a fake Blender never runs this')\n")
    return script


def test_the_names_other_tasks_use():
    assert BLENDER_SERIES == "4.5"
    assert BLENDER_SIDE == Path(blender.__file__).parent / "blender_side"
    assert issubclass(BlenderMissing, RuntimeError)
    assert inspect.signature(run_script).parameters["timeout_s"].default == 1800.0
    assert not inspect.signature(binary_available).parameters and not inspect.signature(available).parameters


def test_a_missing_binary_is_reported_with_the_command_that_fetches_it(monkeypatch, tmp_path):
    monkeypatch.delenv("SGI_BLENDER", raising=False)
    monkeypatch.setattr(blender, "REPO_ROOT", tmp_path / "checkout")       # a checkout without .tools
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))                      # and no Blender installed
    with pytest.raises(BlenderMissing) as raised:
        find_blender()
    said = str(raised.value)
    assert FETCH_COMMAND == ".venv/bin/python -m sgi.synth.realistic.assets blender" and FETCH_COMMAND in said
    assert "SGI_BLENDER" in said and str(tmp_path / "checkout" / ".tools") in said        # and where it was looked for
    with pytest.raises(BlenderMissing, match="-m sgi.synth.realistic.assets blender"):
        run_script(a_script(tmp_path), [])                                 # nothing is run without it


def test_the_binary_is_the_named_one_then_the_checkouts_then_the_one_on_the_path(monkeypatch, tmp_path):
    named = fake_blender(tmp_path / "named" / "blender-custom")
    bundled = fake_blender(tmp_path / "checkout" / ".tools" / "blender-4.5.14-linux-x64" / "blender")
    installed = fake_blender(tmp_path / "bin" / "blender")
    monkeypatch.setattr(blender, "REPO_ROOT", tmp_path / "checkout")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setenv("SGI_BLENDER", str(named))
    assert find_blender() == named
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SGI_BLENDER", "~/named/blender-custom")           # as a file of settings would write it
    assert find_blender() == named
    monkeypatch.setenv("SGI_BLENDER", str(tmp_path / "named" / "moved-away"))
    with pytest.raises(BlenderMissing, match="moved-away"):                # a wrong name is said, not passed over
        find_blender()
    monkeypatch.delenv("SGI_BLENDER")
    assert find_blender() == bundled
    bundled.unlink()
    assert find_blender() == installed


def test_another_blender_series_is_refused(monkeypatch, tmp_path):
    assert blender_version(fake_blender(tmp_path / "new" / "blender")) == "4.5.14"
    old = fake_blender(tmp_path / "old" / "blender", "Blender 3.6.0")
    with pytest.raises(BlenderMissing) as raised:
        blender_version(old)
    assert "3.6.0" in str(raised.value) and BLENDER_SERIES in str(raised.value) and FETCH_COMMAND in str(raised.value)
    with pytest.raises(BlenderMissing, match="4.50.1"):                    # 4.50 is not the 4.5 series
        blender_version(fake_blender(tmp_path / "later" / "blender", "Blender 4.50.1"))
    with pytest.raises(BlenderMissing, match="not-blender"):
        blender_version(fake_blender(tmp_path / "other" / "not-blender", "usage: something else"))
    with pytest.raises(BlenderMissing, match="nowhere"):
        blender_version(tmp_path / "nowhere" / "blender")
    monkeypatch.setenv("SGI_BLENDER", str(old))                            # and no script is run in it
    with pytest.raises(BlenderMissing, match="3.6.0"):
        run_script(a_script(tmp_path), [])
    assert not Path(f"{old}.json").exists()


def test_available_never_raises(monkeypatch, tmp_path):
    def missing():
        raise BlenderMissing("no Blender here")

    def broken():
        raise ValueError("the manifest is not JSON")

    monkeypatch.setenv("SGI_BLENDER", str(fake_blender(tmp_path / "new" / "blender")))
    monkeypatch.setattr(blender.load_library, "__defaults__", (tmp_path / "no-assets",))      # not this checkout's
    with monkeypatch.context() as prepared:
        prepared.setattr(blender, "is_prepared", lambda library: True)
        assert available() is True                                         # a binary of the series, and prepared assets
        prepared.setenv("SGI_BLENDER", str(fake_blender(tmp_path / "old" / "blender", "Blender 3.6.0")))
        assert available() is False
        prepared.setenv("SGI_BLENDER", str(tmp_path / "moved-away"))
        assert available() is False
        prepared.setattr(blender, "find_blender", missing)
        assert available() is False
    with monkeypatch.context() as unprepared:
        unprepared.setattr(blender, "is_prepared", lambda library: False)  # the binary alone is not enough
        assert available() is False and binary_available() is True
        unprepared.setattr(blender, "load_library", broken)
        assert available() is False and binary_available() is True
    assert available() is False                                            # as on a machine that has fetched nothing


def test_the_binary_alone_can_be_asked_for_whatever_the_assets_and_never_raises(monkeypatch, tmp_path):
    asked = []

    def missing():
        raise BlenderMissing("no Blender here")

    def unforeseen(binary):
        raise KeyError("nothing blender_version is known to raise")

    monkeypatch.setattr(blender, "load_library", lambda: asked.append("the manifest"))
    monkeypatch.setattr(blender, "is_prepared", lambda library: asked.append("the stamp") or False)
    monkeypatch.setenv("SGI_BLENDER", str(fake_blender(tmp_path / "new" / "blender")))
    assert binary_available() is True                                      # a Blender of the series, and no asset
    monkeypatch.setenv("SGI_BLENDER", str(fake_blender(tmp_path / "old" / "blender", "Blender 3.6.0")))
    assert binary_available() is False
    monkeypatch.setenv("SGI_BLENDER", str(fake_blender(tmp_path / "other" / "not-blender", "usage: something else")))
    assert binary_available() is False
    monkeypatch.setenv("SGI_BLENDER", str(tmp_path / "moved-away"))
    assert binary_available() is False
    with monkeypatch.context() as nowhere:
        nowhere.setattr(blender, "find_blender", missing)
        assert binary_available() is False
    monkeypatch.setenv("SGI_BLENDER", str(tmp_path / "new" / "blender"))
    monkeypatch.setattr(blender, "blender_version", unforeseen)
    assert binary_available() is False                                     # whatever goes wrong, it answers
    assert asked == []                                                     # and the assets were never looked at


def test_a_script_runs_with_a_private_user_folder_and_a_clean_pythonpath(monkeypatch, tmp_path):
    binary, script, calls = fake_blender(tmp_path / "new" / "blender"), a_script(tmp_path), []

    def run(command, **options):
        calls.append((command, options))
        answer = "Blender 4.5.14 LTS\n" if "--version" in command else "done\n"
        return subprocess.CompletedProcess(command, 0, stdout=answer)

    monkeypatch.setattr(blender.subprocess, "run", run)
    monkeypatch.setenv("SGI_BLENDER", str(binary))
    monkeypatch.setenv("PYTHONPATH", "/opt/ros/lib/python3/site-packages")
    done = run_script(script, ["--job", tmp_path / "job.json", 3], timeout_s=12.0)
    assert done.stdout == "done\n" and len(calls) == 2
    (asked, asking), (command, options) = calls
    assert asked == [str(binary), "-b", "--factory-startup", "--disable-autoexec", "--version"]
    assert command == [str(binary), "-b", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
                       "-P", str(script), "--", "--job", str(tmp_path / "job.json"), "3"]
    private = str(blender.REPO_ROOT / ".tools" / "blender-user")
    for env in (asking["env"], options["env"]):                            # every call, the version's too
        assert "PYTHONPATH" not in env and env["BLENDER_USER_RESOURCES"] == private
        assert env["PATH"] == os.environ["PATH"]                           # the rest is passed on
    assert options["timeout"] == 12.0
    assert os.environ["PYTHONPATH"] == "/opt/ros/lib/python3/site-packages"   # and the caller's own is left alone


def test_the_binary_is_started_without_the_callers_pythonpath(monkeypatch, tmp_path):
    binary, script = fake_blender(tmp_path / "new" / "blender"), a_script(tmp_path)
    monkeypatch.setenv("SGI_BLENDER", str(binary))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "ros"))
    done = run_script(script, ["--out", "here"])
    assert done.returncode == 0 and done.stdout.splitlines()[-1] == "log line 199"
    assert json.loads(Path(f"{binary}.json").read_text()) == {
        "argv": ["-b", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1", "-P", str(script), "--",
                 "--out", "here"],
        "pythonpath": None, "user": str(blender.REPO_ROOT / ".tools" / "blender-user")}


def test_no_user_folder_that_the_caller_names_reaches_blender(monkeypatch, tmp_path):
    binary, script, calls = fake_blender(tmp_path / "new" / "blender"), a_script(tmp_path), []

    def run(command, **options):
        calls.append((command, options))
        answer = "Blender 4.5.14 LTS\n" if "--version" in command else "done\n"
        return subprocess.CompletedProcess(command, 0, stdout=answer)

    # Blender reads each of these four, when set, in place of the folder of its kind under BLENDER_USER_RESOURCES.
    overriding = ("BLENDER_USER_CONFIG", "BLENDER_USER_SCRIPTS", "BLENDER_USER_EXTENSIONS", "BLENDER_USER_DATAFILES")
    at_home = {name: str(tmp_path / "home" / ".config" / "blender" / name.lower())
               for name in ("BLENDER_USER_RESOURCES", *overriding)}
    monkeypatch.setattr(blender.subprocess, "run", run)
    monkeypatch.setenv("SGI_BLENDER", str(binary))
    for name, folder in at_home.items():
        monkeypatch.setenv(name, folder)
    assert blender_version(binary) == "4.5.14"
    run_script(script, [])
    assert binary_available() is True
    assert ["--version" in command for command, options in calls] == [True, True, False, True]
    for command, options in calls:                                         # every start, those for the version too
        env = options["env"]
        assert [name for name in overriding if name in env] == []
        assert env["BLENDER_USER_RESOURCES"] == str(blender.REPO_ROOT / ".tools" / "blender-user")
        assert env["PATH"] == os.environ["PATH"]                           # the rest is passed on
    assert {name: os.environ[name] for name in at_home} == at_home         # and the caller's own are left alone


def test_the_private_user_folder_is_there_before_every_start(monkeypatch, tmp_path):
    binary, script, there = fake_blender(tmp_path / "new" / "blender"), a_script(tmp_path), []

    def run(command, **options):
        there.append(Path(options["env"]["BLENDER_USER_RESOURCES"]).is_dir())      # as the binary would find it
        answer = "Blender 4.5.14 LTS\n" if "--version" in command else "done\n"
        return subprocess.CompletedProcess(command, 0, stdout=answer)

    # While the folder BLENDER_USER_RESOURCES names is absent, Blender 4.5.14 passes over the variable and reads
    # the caller's home: a checkout that has started no Blender yet must not start its first one without it.
    private = tmp_path / "checkout" / ".tools" / "blender-user"
    monkeypatch.setattr(blender, "REPO_ROOT", tmp_path / "checkout")
    monkeypatch.setattr(blender.subprocess, "run", run)
    monkeypatch.setenv("SGI_BLENDER", str(binary))
    assert not private.exists()
    assert blender_version(binary) == "4.5.14"                             # the question for the version makes it
    assert there == [True] and private.is_dir()
    private.rmdir()
    run_script(script, [])                                                 # and so does a script: both of its starts
    assert there == [True, True, True] and private.is_dir()
    private.rmdir()
    assert binary_available() is True and there == [True] * 4
    run_script(script, [])                                                 # a folder that is there is left as it is
    assert there == [True] * 6 and list(private.iterdir()) == []


def test_neither_a_colour_configuration_nor_a_system_folder_of_the_callers_reaches_blender(monkeypatch, tmp_path):
    binary, script, calls = fake_blender(tmp_path / "new" / "blender"), a_script(tmp_path), []

    def run(command, **options):
        calls.append((command, options))
        answer = "Blender 4.5.14 LTS\n" if "--version" in command else "done\n"
        return subprocess.CompletedProcess(command, 0, stdout=answer)

    # OCIO names a colour configuration that Blender takes in place of its own, so the same scene would give
    # other pictures; each BLENDER_SYSTEM_ variable replaces a folder of the installation itself.
    system = ("BLENDER_SYSTEM_RESOURCES", "BLENDER_SYSTEM_SCRIPTS", "BLENDER_SYSTEM_EXTENSIONS",
              "BLENDER_SYSTEM_DATAFILES", "BLENDER_SYSTEM_PYTHON", "BLENDER_SYSTEM_OF_A_LATER_VERSION")
    theirs = {"OCIO": str(tmp_path / "studio" / "config.ocio"),
              **{name: str(tmp_path / "studio" / name.lower()) for name in system}}
    monkeypatch.setattr(blender, "REPO_ROOT", tmp_path / "checkout")
    monkeypatch.setattr(blender.subprocess, "run", run)
    monkeypatch.setenv("SGI_BLENDER", str(binary))
    for name, value in theirs.items():
        monkeypatch.setenv(name, value)
    assert blender_version(binary) == "4.5.14"
    run_script(script, [])
    assert binary_available() is True
    assert ["--version" in command for command, options in calls] == [True, True, False, True]
    for command, options in calls:                                         # every start, those for the version too
        env = options["env"]
        assert [name for name in theirs if name in env] == []
        assert env["PATH"] == os.environ["PATH"]                           # the rest is passed on
        assert env["BLENDER_USER_RESOURCES"] == str(tmp_path / "checkout" / ".tools" / "blender-user")
    assert {name: os.environ[name] for name in theirs} == theirs           # and the caller's own are left alone


def test_a_script_that_fails_is_reported_with_the_end_of_its_log(monkeypatch, tmp_path):
    script = a_script(tmp_path)
    monkeypatch.setenv("SGI_BLENDER", str(fake_blender(tmp_path / "new" / "blender", exit_status=1)))
    with pytest.raises(RuntimeError) as raised:
        run_script(script, [])
    message = str(raised.value)
    assert not isinstance(raised.value, BlenderMissing) and "job.py" in message and "exit status 1" in message
    assert message.rstrip().endswith("log line 199") and "log line 190" in message
    assert "log line 0\n" not in message                                   # its end, not all of it


def test_a_script_that_does_not_finish_in_time_is_stopped_and_reported(monkeypatch, tmp_path):
    script = a_script(tmp_path)
    monkeypatch.setenv("SGI_BLENDER", str(fake_blender(tmp_path / "new" / "blender", sleep_s=30.0)))
    with pytest.raises(RuntimeError) as raised:
        run_script(script, [], timeout_s=3.0)                              # long enough for the fake to print its log
    assert "job.py" in str(raised.value) and "within 3 s" in str(raised.value) and "log line 199" in str(raised.value)


def test_a_script_that_is_not_there_is_refused_before_blender_starts(monkeypatch, tmp_path):
    binary = fake_blender(tmp_path / "new" / "blender")
    monkeypatch.setenv("SGI_BLENDER", str(binary))
    with pytest.raises(FileNotFoundError, match="prepare_models.py"):
        run_script(BLENDER_SIDE / "no_such_folder" / "prepare_models.py", [])
    assert not Path(f"{binary}.json").exists()
