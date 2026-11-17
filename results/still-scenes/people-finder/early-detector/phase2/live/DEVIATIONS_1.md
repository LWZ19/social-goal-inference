# I09 live check: deviation 1 (lead, 21 Oct 2026)

Written before any live call of this check. `runs/` holds only the two count-only passes of 18 Oct, which made zero calls. The note is checked against `FROZEN.md`, which still verifies with `sha256sum -c FROZEN.sha256`. It changes none of the following: the driver, the call cap, the records, the reading rule (section 5: consistent if GLM >= 66 and DeepSeek >= 68; contradicts if GLM <= 63 and DeepSeek <= 64) or the analysis code.

## What changed outside the protocol

## Why it matters here

- **DeepSeek's baseline may be another version.** The check's DeepSeek records pair three things:
  - the cached plan, locate and context answers of the s20 runs (13 Oct);
  - the recorded C trial t0 of those runs (65 of 84), which sets the reading's DeepSeek bar;
  - fresh constraints calls, which would come from the current deployment.
- **The records are mixed-version.** A DeepSeek live result therefore mixes two versions within each record. A difference against the s20 baseline may come partly from the model change rather than from the detector's people.
- **GLM is not affected,** as far as is known.

## decision (lead)
- **The reading stays as frozen.** Its verdict ("consistent", "contradicts" or "mixed") is computed exactly as `FROZEN.md` section 5 says.
- **DeepSeek carries a caveat.** The README reports DeepSeek's part of the result with this caveat attached: "possibly confounded by the x-test-2 version change (DEVIATIONS_1.md)". GLM's part is the cleaner half of the evidence.
- **The design for future comparisons.** Every comparison that involves `x-test-2` pairs the arms within one stage, on the same deployment. The propose-and-select study plans this already.
