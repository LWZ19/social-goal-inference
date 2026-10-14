import pytest
import yaml
from sgi import run as harness
from sgi import smoke
from sgi.config import load_config
from sgi.vlm.base import QuotaExceededError
from sgi.vlm.mock import centre_responder
from tests.helpers import floor_observation, scene_dict, write_scene
from tests.test_run import records, summary, wire
from tests.test_smoke import scripted, toured

SAID = "429 from made-up:1b: Error code: 429 - {'error': {'message': 'Budget limit reached'}}"    # as a back end raises it
TRIALS, PER_RECORD = 3, 3                                 # one episode run three times: a plan, a context and a goal call each

def spent_at(call):
    """The centre answer, until the provider's allowance runs out: the back end's call number `call` is refused, and
    so is every later one."""
    asked = []
    def responder(parts, json_schema):
        asked.append(parts)
        return QuotaExceededError(SAID) if len(asked) >= call else centre_responder(parts, json_schema)
    return responder

@pytest.fixture
def command(tmp_path, monkeypatch, capsys):
    """sgi.run on one small scene, three trials each, with a cache of its own. Called with a run name and options,
    it gives the exit code, what was printed, the error line and the run folder."""
    monkeypatch.chdir(tmp_path)                           # main writes under ./runs
    write_scene(tmp_path / "scenes", scene_dict(), floor_observation())
    cfg = load_config(overrides={"run": {"trials": TRIALS, "cache_dir": str(tmp_path / "cache")},
                                 "stages": {"goal": {"strategy": "zero_shot"}}})      # a strategy that needs no examples
    (tmp_path / "three.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    def go(name, *options):
        code = harness.main(["--scenes", "scenes", "--config", "three.yaml", "--name", name, *options])
        out, err = capsys.readouterr()
        (folder,) = (tmp_path / "runs").glob(f"*_{name}")
        return code, out, err[err.find("error: "):] if "error: " in err else "", folder
    return go

def test_a_spent_allowance_stops_the_run_at_once_keeps_what_is_done_and_the_run_resumes(command, monkeypatch):
    backend = wire(monkeypatch, spent_at(PER_RECORD + 1))     # the plan call of the second trial
    code, out, line, stopped = command("spent")
    assert (code, out) == (2, "") and len(backend.calls) == PER_RECORD + 1       # no call after the one refused
    assert line.startswith(f"error: {SAID} (") and line.count("\n") == 1 and "Traceback" not in line
    assert "allowance is used up" in line and "the same command run again resumes from the cache" in line
    done = records(stopped)                               # exactly the records finished before the stop
    assert [(r["trial"], r["result"]["status"], len(r["result"]["calls"])) for r in done] == [(0, "ok", PER_RECORD)]
    table = summary(stopped)                              # written over those records
    assert [table[0][name] for name in ("group_by", "records", "vlm_errors", "calls")] == ["overall", "1", "0", "3"]
    doc = yaml.safe_load((stopped / "config.yaml").read_text())
    assert (doc["incomplete"], doc["stopped_by"], doc["vlm_errors"]) == (True, SAID, 0)

    healthy = wire(monkeypatch)                           # the allowance is back: the same command, under another name
    code, out, line, resumed = command("resumed")
    assert (code, line) == (0, "") and len(healthy.calls) == (TRIALS - 1) * PER_RECORD      # only what was not finished
    cached = [[call["cached"] for call in r["result"]["calls"]] for r in records(resumed)]
    assert cached == [[True] * PER_RECORD] + [[False] * PER_RECORD] * (TRIALS - 1)
    assert out.endswith(f"({TRIALS} records, 0 ended in vlm_error; 9 calls, 3 from the cache)\n")
    doc = yaml.safe_load((resumed / "config.yaml").read_text())
    assert doc["incomplete"] is False and "stopped_by" not in doc
    assert yaml.safe_load((stopped / "config.yaml").read_text())["stopped_by"] == SAID     # the first folder is left as it was

def test_a_run_stopped_at_its_first_call_writes_an_empty_summary_and_promises_no_resume_without_the_cache(command, monkeypatch):
    backend = wire(monkeypatch, spent_at(1))
    code, out, line, stopped = command("at_once", "--no-cache")
    assert (code, out, len(backend.calls)) == (2, "", 1) and (stopped / "results.jsonl").read_text() == ""
    assert [(row["group_by"], row["records"], row["success_rate"]) for row in summary(stopped)] == [("overall", "0", "")]
    assert yaml.safe_load((stopped / "config.yaml").read_text())["stopped_by"] == SAID
    assert line.startswith(f"error: {SAID} (") and line.count("\n") == 1 and "allowance is used up" in line
    assert "resumes" not in line and "starts over" in line    # nothing was stored: a second run asks every call again

def test_run_raises_the_quota_error_once_the_outputs_are_written(command, tmp_path, monkeypatch):
    wire(monkeypatch, spent_at(PER_RECORD + 1))
    cfg = load_config(tmp_path / "three.yaml")
    with pytest.raises(QuotaExceededError, match="Budget limit reached"):
        harness.run(tmp_path / "scenes", cfg, "called", tmp_path / "runs")
    (out,) = (tmp_path / "runs").iterdir()
    assert len(records(out)) == 1 and summary(out)[0]["records"] == "1"

@pytest.mark.parametrize("extra", [[], ["--calibrate"]], ids=["smoke", "calibrate"])
def test_smoke_reports_a_spent_allowance_in_one_line_and_exits_2(tmp_path, monkeypatch, capsys, extra):
    monkeypatch.chdir(tmp_path)
    scripted(monkeypatch, spent_at(1))
    scene = write_scene(tmp_path, scene_dict(), floor_observation())
    assert smoke.main(["--model", "mock", "--scene", str(scene), *extra]) == 2
    out, err = capsys.readouterr()
    assert out == "" and err.startswith(f"error: {SAID} (") and err.count("\n") == 1
    assert "allowance is used up" in err and "Traceback" not in err

def test_smoke_does_not_take_a_spent_allowance_on_the_tour_call_for_a_refusal_of_that_call(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    scripted(monkeypatch, spent_at(3))                    # the plan and the goal are answered; the plan on the tour frame is not
    assert smoke.main(["--model", "mock", "--scene", str(toured(tmp_path))]) == 2
    out, err = capsys.readouterr()
    assert "result: ok" in out and "[tour]" not in out
    assert err.startswith(f"error: {SAID} (") and err.count("\n") == 1 and "was refused" not in err
