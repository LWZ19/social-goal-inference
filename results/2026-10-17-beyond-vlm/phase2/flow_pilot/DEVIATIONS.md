# C5 flow pilot: deviations and clarifications

The compute agent wrote this on 18 Oct 2026, after the rasters, labels, development arrays and the four trainings, and **before any evaluation sample was drawn** (validation or development) and before any evaluation metric existed. FROZEN.md (sha256 a60f1d62…767f, verified before work started) is not edited. This note's hash is in `DEVIATIONS.sha256`.

## Deviations

Checks so far:
- Unlabelled units (no free region cell inside the raster): 142 train and 20 validation units, all waiting_line_6.
- Oracle lift failures: 0, so the 3-D-anchor fallback for the seen face and the baselines never arose.
- Label self-check (train, I14 replica on every label pose, n 128,576): 96.5% in region, 96.5% heading within tolerance, 99.1% reachable, 95.7% all three. The misses come from the ±0.05 m jitter crossing a region or freespace boundary.
- Training (one GPU process at a time, each started after nvidia-smi showed at least 7,080 MiB free): each of the four models ran its 30 epochs well inside its cap. Best epochs by validation loss: main 30 (0.24571, 237 s); LOSO waiting_line 30 (0.28978, 179 s); LOSO conversation_group 24 (0.18543, 193 s); LOSO audience 28 (0.25461, 199 s). Peak torch allocation was 1.9–2.0 GB. Epoch 1 of the main model reproduced the smoke test's losses to 5 decimals. The checkpoint hashes are in `CHECKPOINTS.sha256`, written before this note.
- Development arrays: 549 goal-step records (old realistic 143/141/157, watching 36 each). In both settings `_over_floor`'s picks[0] reproduces the recorded place (b) or P's place (a) within 1e-6 m. ρ_room(b) equals I04's within 1e-9, and I04's F size matches, on all 549. P's verdict matches I01's rows_old_* on all 441 realistic records (0 mismatches).

## Clarifications (mechanics the note leaves open, fixed before any evaluation sample)

1. **Labels.** The rng draws the 32 cell indices first (`integers(0, n, 32)`), then the jitter (`uniform(-0.05, 0.05, (32, 2))`). Region cells and the heading rule use the package's `point_in_polygon`, with a bounding-box prefilter.
2. **Seen face at training.** `_seen_face` is given `oracle_target`'s GroundedTarget. The 3-D-anchor fallback would build a GroundedTarget whose pixel is the projected anchor (only `.world` is read); it was never needed.
3. **Floor channel.** Read with the package's own `Floor.index` at each cell centre, one call per cell.
4. **Extended region lattice (6(i) coverage).** Computed at build time for every unit and stored in `data/seeds/seed<k>.npz` (`ext_xy`, raster frame).
5. **Model.** `nn.LayerNorm(256)` with PyTorch's default affine parameters; convolutions with padding 1; time MLP = Linear(128, 256), SiLU, Linear(256, 256); FiLM output split as (γ, β) in that order; sinusoid order [sin, cos].
6. **Training mechanics.**
   - `torch.manual_seed(0)` before the model is built. Shuffling uses `torch.randperm` on the CPU generator; z and t use the CUDA generator.
   - `cudnn.deterministic = True`, `cudnn.benchmark = False`, 2 CPU threads.
   - An epoch is `ceil(N / 256)` steps, the last batch partial. The cosine schedule runs over 30 such epochs.
   - The wall-time cap is checked before every step and after every validation pass. A partial epoch at the cap is not validated.
   - Validation z and t come from a CPU generator seeded 1: z (N × 4) first, then t (N).
   - Checkpoints are written as `.npz` state dicts, because the sidecar writes only `.npz` and `.json`.
7. **Sampling.** z is drawn on a CPU generator seeded with crc32(unit key), so it does not depend on the device. Unit keys: `seed<k>/<split>/<scene>/<episode>` (validation) and `<run folder>/<scene>/<episode>/<trial>` (development, as in I04). The top pose and the hit@5 ordering sort by density descending, then by sample index (ties go to the lowest index). hit@5 spacing is measured between snapped positions.
8. **6(i) package check.** It covers the top pose plus 8 samples drawn without replacement with `default_rng([7, crc32(key)])`, compared on success, in_region, heading_ok and reachable.
9. **B1.** members = the oracle-people numbers (OraclePeople's filter and order) of the visible members of `truth.group_for(episode, target)`; needed = the target person's number. A `solve` that raises (no standard-goal pose) would count as a failure, with the error recorded.
10. **Latency.** The timed span includes drawing z on the CPU, the encoder, 16 Euler steps, the copy back, the world transform, the snap, validity, density and picks. The raw-top density (secondary) is computed after the timer stops. nvidia-smi is polled every 0.5 s during the timed passes. Setting (b) is the latency that S3 reads; setting (a) is reported beside it.
11. **I04 rows.** I04's verification had finished before this pilot started (`phase1d/i04/verify/v_summary.json` and `v_analysis.txt`, 18 Oct 00:45). That counts as the verification summary FROZEN.md 6(iv) waits for, so the rows were used without the 2 h wait.
12. **Order.** The development arrays were built after the seed rasters and before training, as section 8 says.
13. **Smoke test (disclosed).** Before the real runs, the training code ran once for 15 s on the real data (`train ../smoke/t none 0.25`) to check that it executes: 1 epoch, validation loss 0.305. Its output was deleted. Nothing was chosen from it.
