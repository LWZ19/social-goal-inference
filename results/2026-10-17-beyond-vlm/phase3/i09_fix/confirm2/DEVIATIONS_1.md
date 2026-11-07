# I09 second confirmation: deviation 1 (lead, 18 Oct 2026, 20:05 +08:00)

At writing:
- `confirm2/` did not exist; this file and its sha256 are its first contents.
- A names-only search of `~`, `/tmp` and `/dev/shm` found no folder named `seed` followed by a number from 213 up. (It walked directory names, including the names under `phase2/seeds/`, and opened no file.)

## 1. The text-search exposure (verifier finding 2; accepted)

**What happened.** While writing the frozen note, the protocol author ran one command at 19:18:09 +08:00 (from `verify/calls_freeze.jsonl`):

```text
cd .-scenes; timeout 120 grep -rIl -E 'seed ?(21[3-9]|2[2-9][0-9])\b|seeds 2(1[3-9]|[2-9][0-9])' --include='*.md' --include='*.py' --include='*.json' --include='*.txt' --include='*.yaml' .superpowers src configs docs tests 2>/dev/null | grep -v '/phase2/seeds/' | head -20
```

- **What it opened.** By the verifier's count, 364 files under `phase2/seeds/realistic/` (seeds 200–212, the spent test data of the first confirmation). It also opened `phase2/i09/confirm/posthoc_false.txt`, `raw/meta.json`, `summary.json` and `tables.txt`, and `phase1d/i04/i04_heldout.py` (a script).

- **Seeds 200–212 are spent.** Their aggregates are published, and they serve no later test.
- **No content was seen.** The pattern asks only for seed numbers from 213 up, and the output held file names only, none of them under `phase2/seeds/`.
- **The method was already fixed.** It is the fix step's choice, hashed at 19:10:11, and the criteria are copied from the first protocol. So nothing the search could have shown could shape them.

**Correction.** In the frozen note's "Exposure (disclosed)" section, the sentence "No per-image or per-person row of the first confirmation: no `rows.jsonl`, `people_rows.jsonl`, `raw/` or `posthoc_false.txt`. I opened no file under `phase2/seeds/realistic/` and nothing of seeds 200–219." is to be read as follows:

> "No per-image or per-person row of the first confirmation was read. A process of mine (the text search above) opened the files listed in `confirm2/DEVIATIONS_1.md` section 1, including 364 files under `phase2/seeds/realistic/` and `posthoc_false.txt`, but printed only the names of matching files outside `phase2/seeds/`. I read no content of any of them."

## 2. Section 12's premise (verifier minor point (c))

Section 12 says that one of C07's two fit-validate false people is a detection near hidden-by-rule staff. By the verifier's re-score, both are. Both lie beside the same hidden-by-rule staff member:
- one at 0.41 m (class B; the secondary scorer reclassifies it);
- one at 0.61 m (class C; just outside the secondary scorer's 0.6 m radius).

The prediction's numbers stay as written, since they were made before any seed was named. Only this premise is corrected.

## 3. The harness hash (verifier minor point (b))

## 4. Not covered here

- the test seeds;
- any sharing of 213–219 with another step;
- whether and when to run;
- the flat test dropping glass-staff detections.
