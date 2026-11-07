"""Finding Blender, checking its version, and running a script inside it.

sgi never imports bpy. The scripts under blender_side/ run in Blender's own Python, started here as a
subprocess, and the two sides exchange files only. Every start of the binary, the question for its version
included, is headless, with factory settings, with no script of a .blend file allowed to run, with
PYTHONPATH taken out of the environment (the shell's would be read by Blender's Python) and with Blender's
user folders set to a private directory of the checkout, so nothing is read from or written to the home
of whoever runs it. BLENDER_USER_RESOURCES names that directory, and the four variables that would each
name one user folder apart from it (BLENDER_USER_CONFIG, _SCRIPTS, _EXTENSIONS, _DATAFILES) are taken out
of the environment with PYTHONPATH. The directory is made before each start: while it is absent Blender
passes over BLENDER_USER_RESOURCES and reads the home after all. Taken out too are OCIO, which names a
colour configuration to use in place of Blender's own, so that the same scene would give other pictures,
and every BLENDER_SYSTEM_ variable, each of which replaces a folder of the installation itself.

Two questions can be asked without anything being raised: binary_available, whether a Blender of the series
can be started, and available, whether the assets are prepared as well.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from sgi.config import REPO_ROOT
from sgi.synth.realistic.library import is_prepared, load_library

BLENDER_SERIES = "4.5"                             # the LTS series the scenes are built and measured with
BLENDER_SIDE = Path(__file__).parent / "blender_side"      # the scripts that run inside Blender; never imported by sgi
FETCH_COMMAND = ".venv/bin/python -m sgi.synth.realistic.assets blender"    # puts the binary under .tools/

_BUNDLED = Path(".tools") / "blender-4.5.14-linux-x64" / "blender"      # under REPO_ROOT: where FETCH_COMMAND puts it
_USER_FOLDER = Path(".tools") / "blender-user"     # under REPO_ROOT too: Blender's user folders, kept out of the home
# Blender takes each of these, when set, in place of the folder of its kind under BLENDER_USER_RESOURCES, so one
# left in the environment would lead it out of the private folder again. They are all there are in the 4.5 series.
_USER_FOLDER_OVERRIDES = ("BLENDER_USER_CONFIG", "BLENDER_USER_SCRIPTS", "BLENDER_USER_EXTENSIONS",
                          "BLENDER_USER_DATAFILES")
_COLOUR_CONFIGURATION = "OCIO"                     # a configuration file Blender takes in place of its own
_SYSTEM_FOLDER_PREFIX = "BLENDER_SYSTEM_"          # _RESOURCES, _SCRIPTS, _EXTENSIONS, _DATAFILES, _PYTHON
_HEADLESS = ("-b", "--factory-startup", "--disable-autoexec")
_SCRIPT_FAILED = 1                                 # the exit status asked of Blender when its script raises
_VERSION_TIMEOUT_S = 60.0
_LOG_TAIL_LINES = 40


class BlenderMissing(RuntimeError):
    """No Blender of the series can be run here. The message says what was looked for and how to get it."""


def _environment() -> dict[str, str]:
    """The environment of every Blender started here: the caller's without PYTHONPATH, without a user folder, a
    system folder or a colour configuration of the caller's own, and with the private user folder, which is
    made here if it is not there: Blender 4.5.14 passes over BLENDER_USER_RESOURCES while the folder it names
    is absent, and reads from the home of whoever runs it."""
    left_out = ("PYTHONPATH", _COLOUR_CONFIGURATION, *_USER_FOLDER_OVERRIDES)
    env = {name: value for name, value in os.environ.items()
           if name not in left_out and not name.startswith(_SYSTEM_FOLDER_PREFIX)}
    private = REPO_ROOT / _USER_FOLDER
    private.mkdir(parents=True, exist_ok=True)
    env["BLENDER_USER_RESOURCES"] = str(private)
    return env


def _fetch_it() -> str:
    """The end of every BlenderMissing message: the command that provides the binary."""
    return (f"Fetch Blender {BLENDER_SERIES} into the checkout with:\n    {FETCH_COMMAND}\n"
            "or name a binary of that series in SGI_BLENDER.")


def find_blender() -> Path:
    """The Blender binary to use: the one SGI_BLENDER names, else the checkout's own, else the one on PATH.

    Its version is not looked at here; blender_version does that. SGI_BLENDER naming no file is an error
    and is not passed over for another binary: whoever set it wants that one."""
    named = os.environ.get("SGI_BLENDER")
    if named:
        if not Path(named).expanduser().is_file():
            raise BlenderMissing(f"SGI_BLENDER names {named}, which is not a file. {_fetch_it()}")
        return Path(named).expanduser()
    bundled = REPO_ROOT / _BUNDLED
    if bundled.is_file():
        return bundled
    installed = shutil.which("blender")
    if installed:
        return Path(installed)
    raise BlenderMissing(f"no Blender found: SGI_BLENDER is not set, there is no {bundled}, and none is on PATH. "
                         + _fetch_it())


def blender_version(binary: Path) -> str:
    """The version the binary reports, as in "4.5.14". BlenderMissing when it cannot be run, does not answer
    as Blender does, or is of another series than BLENDER_SERIES."""
    try:
        done = subprocess.run([str(binary), *_HEADLESS, "--version"], env=_environment(), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                              timeout=_VERSION_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BlenderMissing(f"{binary} could not be run as Blender ({error}). {_fetch_it()}") from error
    found = re.search(r"^Blender (\d+\.\d+(?:\.\d+)*)", done.stdout or "", re.MULTILINE)
    if found is None:
        raise BlenderMissing(f"{binary} did not answer --version as Blender does. {_fetch_it()}")
    version = found.group(1)
    if version != BLENDER_SERIES and not version.startswith(BLENDER_SERIES + "."):
        raise BlenderMissing(f"{binary} is Blender {version}; the realistic scenes need the {BLENDER_SERIES} series. "
                             + _fetch_it())
    return version


def binary_available() -> bool:
    """Whether a Blender of the series answers here. Never raises, and says nothing about the assets.

    Made for the skipif of the tests that start Blender and open no prepared asset: those of the scene
    builder that draw a description without models or people (its depth against the ray caster, the object
    groups, signs, glass). A test that needs anything from the prepared folder (a model, an avatar, a
    texture, an HDRI, a font) is skipped on available() instead; guarded by this alone it would fail, not
    skip, on a machine that has the binary and has prepared nothing."""
    try:
        blender_version(find_blender())
        return True
    except Exception:                              # no binary, one that does not run, one of another series
        return False


def available() -> bool:
    """Whether realistic scenes can be rendered here: the assets are prepared, and a Blender of the series
    answers. Never raises; made for the skipif of the tests that need both, such as a render of a whole
    scene. A test that needs Blender and no prepared asset is skipped on binary_available() instead: this
    is False until the assets are prepared, and would skip it for no reason of its own."""
    try:
        return is_prepared(load_library()) and binary_available()      # the assets first: asking starts nothing
    except Exception:                              # a manifest that does not read
        return False


def _tail(log: str | bytes | None) -> str:
    """The last lines of a log. A process stopped for taking too long leaves its output as bytes."""
    text = log.decode("utf-8", errors="replace") if isinstance(log, bytes) else log or ""
    return "\n".join(text.splitlines()[-_LOG_TAIL_LINES:])


def run_script(script: Path, args: Sequence[str], timeout_s: float = 1800.0) -> subprocess.CompletedProcess:
    """Run a blender_side script headless with a private user folder and a clean PYTHONPATH; raises with the
    tail of the log on a non-zero exit.

    The script finds `args` after "--" in its sys.argv. The process that is returned holds the whole log,
    output and errors together, as `stdout`. BlenderMissing when there is no Blender of the series, before
    anything is run; RuntimeError when the script fails or does not finish within `timeout_s`. Blender by
    itself ends with status 0 when its script raises, so it is asked for a non-zero status then: an
    exception that the script lets through is enough to be reported here."""
    binary = find_blender()
    blender_version(binary)
    if not Path(script).is_file():
        raise FileNotFoundError(f"no script {script} to run in Blender")
    command = [str(binary), *_HEADLESS, "--python-exit-code", str(_SCRIPT_FAILED), "-P", str(script), "--",
               *(str(arg) for arg in args)]
    try:
        done = subprocess.run(command, env=_environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{Path(script).name} did not finish in Blender within {timeout_s:g} s and was stopped; "
                           f"its log ends:\n{_tail(error.stdout)}") from error
    if done.returncode != 0:
        raise RuntimeError(f"{Path(script).name} failed in Blender with exit status {done.returncode}; "
                           f"its log ends:\n{_tail(done.stdout)}")
    return done
