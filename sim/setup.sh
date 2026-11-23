#!/usr/bin/env bash
# Everything the simulator needs after a fresh clone, in one go (sim/README.md, "Quick start").
#   sim/setup.sh [DOWNLOADS]
# Creates the conda environments that are missing, the pipeline's test scenes and Habitat's data, builds
# the scenes made from nothing, and builds the Sketchfab scenes whose downloads are in DOWNLOADS (default
# ~/Downloads); for any that are not there, it says what to download. Everything it makes is git-ignored (data/).
# Running it again redoes nothing that is done.
set -euo pipefail
cd "$(dirname "$0")/.."
DL="${1:-$HOME/Downloads}"

base="$(conda info --base 2>/dev/null || true)"
for guess in "$base" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
  [[ -n "$guess" && -f "$guess/etc/profile.d/conda.sh" ]] && { base="$guess"; break; }
done
[[ -f "$base/etc/profile.d/conda.sh" ]] || { echo "conda not found: install Miniconda first" >&2; exit 2; }
source "$base/etc/profile.d/conda.sh"

step() { printf '\n== %s\n' "$*"; }
have_env() { conda env list | awk '{print $1}' | grep -qx "$1"; }

step "conda environments"
for env in sgi sgi-habitat sgi-tools; do
  if have_env "$env"; then echo "$env: already there"; else conda env create -f "envs/$env.yml"; fi
done

step "the pipeline's test scenes (about 60 MB, two minutes)"
if [[ -d data/placeholder/eval ]]; then
  echo "already made"
else
  conda run --no-capture-output -n sgi python -m sgi.synth.generate --out data/placeholder
fi

step "Habitat data: humanoids, Spot, ReplicaCAD (about 370 MB)"
conda run --no-capture-output -n sgi-habitat python -m habitat_sim.utils.datasets_download \
  --uids habitat_humanoids hab_spot_arm replica_cad_dataset --data-path data/habitat --no-replace

step "scenes built from nothing"
conda activate sgi-habitat
[[ -f data/habitat/stages/cafe.glb ]] || python -m sgi_habitat.make_cafe
[[ -f data/habitat/stages/room.glb ]] || python -m sgi_habitat.make_room
[[ -f data/habitat/stages/concert_ground.glb ]] || python -m sgi_habitat.make_room --width 30 --depth 32 --no-walls \
  --floor-rgb 92 92 96 --out data/habitat/stages/concert_ground.glb
python -m sgi_habitat.fetch_models coffee_brewer plant_tall cooler_water

D=data/habitat/downloads
missing=()

step "the Sweet Spot café (Sketchfab download)"
if [[ -f data/habitat/stages/sweet_spot_cafe.glb ]]; then
  echo "already built"
elif [[ -f "$DL/modern_coffee_shop_interior_-_sweet_spot_cafe.zip" ]]; then
  mkdir -p "$D/sweet_spot_cafe"
  unzip -qo "$DL/modern_coffee_shop_interior_-_sweet_spot_cafe.zip" -d "$D/sweet_spot_cafe"
  python -m sgi_habitat.paint_stage "$D/sweet_spot_cafe/scene.gltf" data/habitat/stages/sweet_spot_cafe.glb \
    --floor-y 0.15 --region 7.5 21.0 -0.3 9.3 --counter 11.0 16.7 4.9 8.85
else
  missing+=("Sweet Spot café, 'glTF' download: https://sketchfab.com/3d-models/modern-coffee-shop-interior-sweet-spot-cafe-0d7900fbf8af442ebac526fe3f5b5394")
fi

step "the concert stage and barricade (Sketchfab downloads)"
if [[ -f data/habitat/objects/concert_stage/concert_stage.glb ]]; then
  echo "stage already built"
elif [[ -f "$DL/simple-concert-stage.zip" ]]; then
  S="$D/simple-concert-stage/source/stageSF"
  mkdir -p "$D/simple-concert-stage" "$S"
  unzip -qo "$DL/simple-concert-stage.zip" -d "$D/simple-concert-stage"
  unzip -qo "$D/simple-concert-stage/source/stageSF.zip" -d "$S"
  conda run -n sgi-tools python sim/fbx_to_glb.py "$S/stage-final.fbx" "$S/stage-final.glb"
  python -m sgi_habitat.texture_model "$S/stage-final.glb" data/habitat/objects/concert_stage/concert_stage.glb \
    --base "$S/stage-main_None_BaseColor.png" --roughness "$S/stage-main_None_Roughness.png" \
    --metallic "$S/stage-main_None_Metallic.png" --normal "$S/stage-main_None_Normal.png"
else
  missing+=("concert stage, 'Original format' download: https://sketchfab.com/3d-models/simple-concert-stage-d5c7733d06d24947bf60b3a0fe203f69")
fi
if [[ -f data/habitat/objects/barricade/barricade.glb ]]; then
  echo "barricade already built"
elif [[ -f "$DL/barigate-8-met-besi-hitam.zip" ]]; then
  B="$D/barigate-8-met-besi-hitam"
  mkdir -p "$B/source/model"
  unzip -qo "$DL/barigate-8-met-besi-hitam.zip" -d "$B"
  unzip -qo "$B/source/barigate 8 met besi hitam.zip" -d "$B/source/model"
  python -m sgi_habitat.import_model "$B/source/model/barigate 8 met besi hitam.obj" --name barricade
else
  missing+=("barricade, 'Original format' download: https://sketchfab.com/3d-models/barigate-8-met-besi-hitam-07ba7df39b3a4947bb5f0910b9fbc530")
fi

step "model key"
if [[ -f .env ]]; then
  echo ".env is there"
else
  cp .env.example .env
  echo "made .env: put one key in it, with no spaces or quotes, e.g. ANTHROPIC_API_KEY=sk-ant-..."
fi

step "done"
echo "Try:  sim/demo.sh --model claude_haiku55 -- --scenario sim/scenarios/cafe_line2.yaml"
if (( ${#missing[@]} )); then
  echo
  echo "Not built, for want of a download. Each is free with a Sketchfab account (CC BY 4.0); save the .zip"
  echo "in $DL as it comes, and run sim/setup.sh again:"
  printf '  - %s\n' "${missing[@]}"
  echo "Until then, the sweet_spot_* and concert_* scenarios cannot run; the others can."
fi
