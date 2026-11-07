# Profile angle of people

**Question.** How far from facing the camera do people stand in the realistic scenes?

**Result.** Line members stand almost side-on: the angle between facing and the direction to the camera has median 88.7 degrees (74.1 to 107.3, 42 of 48 within 75 to 107).

**Files.**
- `profile_angles.py / profile_angles.txt`: the angle tables
- `explore.py / explore.txt`: first look
- `verify/`: independent checks (profile, visibility, yaw convention, staff identity)

**Reproduce.** From the repo root: `SGI_CACHE_ONLY=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src python results/still-scenes/speed/profile/profile_angles.py`. The scripts import the harness in `results/still-scenes/analysis-scripts/harness/` and read the development scenes from `data/dev_scenes/` (release dev-scenes-v1).

**Data.** Development scenes in `data/dev_scenes/` (release dev-scenes-v1); `results/logs/still-runs.tar.gz` (unpacks to `runs/` at the repo root).
