"""Collect the evidence of T1-T7 from my folders into results.json (no driver run, no calls)."""
import collections
import glob
import json
import os
import re
from pathlib import Path

import yaml

V = Path(__file__).resolve().parent
R = {}


def jl(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def fake(tagglob):
    return [r for f in glob.glob(str(V / "fake" / f"fake_{tagglob}_*.jsonl")) for r in jl(f)]


def wrap(tag):
    return json.loads((V / "wrap" / f"wrap_{tag}.json").read_text())


def folders(p):
    out = {}
    for f in sorted(glob.glob(str(p) + "/*_con*")):
        d = yaml.safe_load(open(f + "/config.yaml"))
        r = f + "/results.jsonl"
        out[os.path.basename(f)] = {"incomplete": d["incomplete"], "stopped_by": d.get("stopped_by"),
                                    "records": len(open(r).read().splitlines()) if os.path.exists(r) else None,
                                    "driver_sha256": d["i09_live"]["driver_sha256"]}
    return out


# audit totals over every wrapped run, the fill, dryrun, t2a, expected_counts
aud = collections.Counter()
for f in glob.glob(str(V / "audit" / "audit_*.json")):
    a = json.loads(open(f).read())
    for k in ("env_hits", "main_cache_hits", "network_hits", "main_write_hits", "key_file_hits"):
        aud[k] += a[k]
    aud["processes"] += 1
    aud["writes_outside_V"] += len(a["writes_outside_V"])
R["audit_totals"] = dict(aud)

# T1
tr = (V / "t1" / "trace.txt").read_text()
led = jl(V / "t1" / "out" / "ledger.jsonl")
pa = re.search(r"A pid (\d+)", tr).group(1)
R["T1"] = {"trace": tr.splitlines(), "A_pid": pa,
           "B_exit": wrap("t1_B")["exit"], "C_exit": wrap("t1_C")["exit"], "A_exit": wrap("t1_A")["exit"],
           "lock_after_B_is_A": f"lock after B: {pa} " in tr, "lock_after_C_is_A": f"lock after C: {pa} " in tr,
           "lock_changed_while_A_alive": "LOCK CHANGED" in tr, "lock_after_A": "lock after A: NONE" in tr,
           "sent_rows": sum(r["event"] == "sent" for r in led), "fake_A": len(fake("t1_A")),
           "fake_B": len(fake("t1_B")), "fake_C": len(fake("t1_C")), "folders": folders(V / "t1" / "out")}
t = R["T1"]
t["pass"] = (t["B_exit"] == 2 and t["C_exit"] == 2 and t["lock_after_B_is_A"] and t["lock_after_C_is_A"]
             and not t["lock_changed_while_A_alive"] and t["lock_after_A"] and t["sent_rows"] <= 60
             and t["sent_rows"] == t["fake_A"] + t["fake_B"] + t["fake_C"] and t["fake_B"] == t["fake_C"] == 0)

# T2
t2a = json.loads((V / "t2a" / "t2a_result.json").read_text())
t2c = json.loads((V / "t2a_control_oldbuild" / "t2a_result.json").read_text())
R["T2"] = {"t2a_checks": t2a["checks"],
           "t2a": {v: {k: t2a[v][k] for k in ("caps", "workers", "prefill", "sent_by_model", "sent_total", "events",
                                               "invalid_json_lines", "totals_are_1_to_N", "model_totals_are_1_to_n",
                                               "reserved_by_workers")} for v in ("v1", "v2", "v3")},
           "t2a_control_first_build": {v: {"sent_by_model": t2c[v]["sent_by_model"], "sent_total": t2c[v]["sent_total"]}
                                       for v in ("v1", "v2", "v3")}}
for v, exp in (("v1", (120, 60, 60)), ("v2", (90, 60, 60))):
    lines = open(V / "t2b" / v / "shared" / "ledger.jsonl").read().splitlines()
    rows, bad = [], 0
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except ValueError:
            bad += 1
    sent = [r for r in rows if r["event"] == "sent"]
    by = collections.Counter(r["model"] for r in sent)
    fk = fake(f"t2b_{v}")
    exits = (V / "t2b" / v / "exits.txt").read_text().splitlines()
    R["T2"][f"t2b_{v}"] = {"lines": len(lines), "invalid_json_lines": bad, "sent_by_model": dict(by),
                          "sent_total": len(sent), "fake_total": len(fk),
                          "fake_by_pid": dict(collections.Counter(r["pid"] for r in fk)),
                          "totals_are_1_to_N": sorted(r["total"] for r in sent) == list(range(1, len(sent) + 1)),
                          "exits": exits}
t = R["T2"]
t["pass"] = (all(t["t2a_checks"].values())
             and t["t2b_v1"]["sent_by_model"] == {"glm53": 60, "dsv41": 60} and t["t2b_v1"]["sent_total"] == 120
             and t["t2b_v1"]["fake_total"] == 120 and t["t2b_v1"]["invalid_json_lines"] == 0
             and t["t2b_v2"]["sent_total"] == 90 and max(t["t2b_v2"]["sent_by_model"].values()) <= 60
             and t["t2b_v2"]["fake_total"] == 90 and t["t2b_v2"]["invalid_json_lines"] == 0
             and t["t2b_v1"]["totals_are_1_to_N"] and t["t2b_v2"]["totals_are_1_to_N"])

# T3
led = jl(V / "t3" / "out" / "ledger.jsonl")
R["T3"] = {"exits": (V / "t3" / "exits.txt").read_text().splitlines(),
           "ledger_events": {f"{e}:{m}": n for (e, m), n in collections.Counter((r["event"], r["model"]) for r in led).items()},
           "records_with_2_sent_rows": [k for k, n in collections.Counter((r["scene"], r["episode"]) for r in led
                                                                          if r["event"] == "sent").items() if n > 1],
           "fake": {c: len(fake(f"t3_{c}")) for c in ("c1", "c2", "c3")},
           "fake_outcomes_c3": dict(collections.Counter(r["outcome"] for r in fake("t3_c3"))),
           "folders": folders(V / "t3" / "out")}
ex = {ln.split()[0]: int(ln.split()[2]) for ln in R["T3"]["exits"] if ln.startswith("c")}
t = R["T3"]
t["pass"] = (ex == {"c1": 3, "c2": 3, "c3": 0, "c4a": 2, "c4b": 2} and t["fake"] == {"c1": 5, "c2": 0, "c3": 25}
             and t["ledger_events"].get("sent:glm53") == 30 and "sent:dsv41" not in t["ledger_events"]
             and len(t["records_with_2_sent_rows"]) == 2 and all("lock left: none" in ln for ln in t["exits"][:3]))

# T4
R["T4"] = {"exits": (V / "t4" / "exits.txt").read_text().splitlines(),
           "removed_context_key": json.loads((V / "t4" / "c_removed_context_key.json").read_text()),
           "fake_titles_b": dict(collections.Counter(r["schema_title"] for r in fake("t4_b"))),
           "fake_titles_c": dict(collections.Counter(r["schema_title"] for r in fake("t4_c"))),
           "a_folders": folders(V / "t4" / "a_out"), "c_folders": folders(V / "t4" / "c_out"),
           "a_check_files": [os.path.basename(p) for p in glob.glob(str(V / "t4" / "a_out" / "check_*"))],
           "c_check_files": [os.path.basename(p) for p in glob.glob(str(V / "t4" / "c_out" / "check_*"))],
           "b_check": json.loads(open(glob.glob(str(V / "t4" / "b_out" / "check_*.json"))[0]).read())["cache_total"]}
t = R["T4"]
t["pass"] = (t["exits"][0].startswith("t4a exit 2 ; lock left: none; ledger sent 0") and "fake 0" in t["exits"][0]
             and t["exits"][1].startswith("t4b exit 0 ; lock left: none; ledger sent 28; fake 28")
             and t["fake_titles_b"] == {"ConstraintsWire": 28} and "refused" not in json.dumps(t["b_check"])
             and t["exits"][2].startswith("t4c exit 2 ; lock left: none; ledger sent 4; fake 4"))

# T5
R["T5"] = {"exits": (V / "t5" / "exits.txt").read_text().splitlines(),
           "sigint_rerun": (V / "t5" / "sigint2" / "exits.txt").read_text().splitlines(),
           "unknown_live_folders": folders(V / "t5" / "unk_live_out"),
           "vlm_error_folders": folders(V / "t5" / "ve_out")}

# T6
caps = {}
for tag in ("t6_mock_glm", "t6_mock_dsv"):
    w = wrap(tag)
    caps[tag] = {"titles": dict(collections.Counter(c["title"] for c in w["captured"])),
                 "constraints_person_lines": [sum(len(re.findall(r"^Person \d+", tx, flags=re.M)) for tx in c["texts"])
                                              for c in w["captured"] if c["title"] == "ConstraintsWire"]}
recs = [json.loads(line) for f in glob.glob(str(V / "t6" / "mock_out" / "*_mock" / "results.jsonl")) for line in open(f)]
chk = {os.path.basename(p): json.loads(open(p).read()) for p in glob.glob(str(V / "t6" / "mock_out" / "check_*.json"))}
R["T6"] = {"cli_mock": caps,
           "cli_mock_people_calls_in_records": sum(c["stage"] == "people" for r in recs for c in r["result"]["calls"] or []),
           "cli_mock_people_source": sorted({r["config"]["people_source"] for r in recs}),
           "cli_mock_detector_people": {k: {s: v["detector_people"] for a in c["arms"] for s, v in a["scenes"].items()}
                                        for k, c in chk.items()},
           "cli_mock_no_ledger_no_lock": not (V / "t6" / "mock_out" / "ledger.jsonl").exists()
           and not (V / "t6" / "mock_out" / "ledger.lock").exists(),
           "dryrun": {p: {"checks": [(c["check"].split(" ")[0], c["pass"]) for c in v["checks"]],
                          "raised": v["raised"]["type"] if v["raised"] else None}
                      for p, v in json.loads((V / "dryrun_out" / "run_dryrun_result.json").read_text())["parts"].items()}}
cnt = {}
for d in ("cnt_full", "cnt_pc"):
    for m in ("glm53", "dsv41"):
        rep = json.loads(open(sorted(glob.glob(str(V / d / f"count_only_{m}_*.json")))[-1]).read())
        cnt[f"{d}:{m}"] = {"cache_total": rep["cache_total"], "bounds": rep["bounds"],
                           "records": sum(a["records"] for a in rep["arms"])}
    cnt[f"{d}:files"] = sorted(os.listdir(V / d))
R["T6"]["count_only_on_my_caches"] = cnt
R["T7"] = (V / "t7_hashes.txt").read_text().splitlines()
(V / "results.json").write_text(json.dumps(R, indent=1, default=str))
for k in ("T1", "T2", "T3", "T4"):
    print(k, "PASS" if R[k]["pass"] else "FAIL")
print(json.dumps({"audit_totals": R["audit_totals"], "T6_cli_mock": R["T6"]["cli_mock"],
                  "people_calls": R["T6"]["cli_mock_people_calls_in_records"],
                  "det_people": R["T6"]["cli_mock_detector_people"], "no_ledger": R["T6"]["cli_mock_no_ledger_no_lock"]},
                 indent=1))
