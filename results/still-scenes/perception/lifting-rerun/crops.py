"""Crops (zoom x3) round the model's point for the person and the near-miss place wrong targets: target body mask
outline in green, the model's point red, the snapped pixel yellow. Reads only through research_harness."""
import sys, math, csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, 'results/still-scenes/analysis-scripts/harness')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_harness as H
import lifting as L
OUT = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(OUT / 'lifting_records.tsv'), delimiter='\t'))
want = [r for r in rows if r['has_point'] == 'True' and r['arm'] != 'ph' and
        (r['class'] == 'person' or (r['class'] == 'place' and float(r['d_body_px']) <= 15))]
keyset = {(r['stage'], r['arm'], r['model'], r['scene'], r['episode'], r['trial']) for r in want}
tiles = []
for stage, model in [('s19', 'qwen38_low'), ('s19', 'glm53'), ('s20', 'dsv41')]:
    for arm in L.REAL_ARMS:
        run = H.runs(stage=stage, arm=arm, model=model, method='con')[0]
        for rec in H.records(run):
            k = (stage, arm, model, rec.scene_id, rec.episode_id, str(rec.trial))
            if k not in keyset:
                continue
            sd = L.scene_data(rec)
            th = H.recorded_config(rec.ref)[1].thresholds
            tid = sd['sc'].truths[rec.episode_id].gt.target_ids[0]
            match, body = L.target_masks(sd, tid, th)
            rgb = np.asarray(sd['sc'].observation.rgb).copy()
            edge = body & ~(np.roll(body, 1, 0) & np.roll(body, -1, 0) & np.roll(body, 1, 1) & np.roll(body, -1, 1))
            rgb[edge] = (0, 255, 0)
            u, v = rec.target['pixel']
            q, d = L.nearest_pixel(body, u, v)
            half = 40
            i0, j0 = int(v) - half, int(u) - half
            pad = np.pad(rgb, ((half, half), (half, half), (0, 0)))
            crop = pad[i0 + half:i0 + 3 * half, j0 + half:j0 + 3 * half]
            im = Image.fromarray(crop.astype(np.uint8)).resize((240, 240), Image.NEAREST)
            dr = ImageDraw.Draw(im)
            cu, cv = (u - j0) * 3, (v - i0) * 3
            dr.line([cu - 8, cv, cu + 8, cv], fill=(255, 0, 0), width=2)
            dr.line([cu, cv - 8, cu, cv + 8], fill=(255, 0, 0), width=2)
            dr.rectangle([cu - 7.5, cv - 7.5, cu + 7.5, cv + 7.5], outline=(255, 0, 0))  # 5x5 lift window
            if q is not None:
                qu, qv = (q[1] + 0.5 - j0) * 3, (q[0] + 0.5 - i0) * 3
                dr.ellipse([qu - 4, qv - 4, qu + 4, qv + 4], outline=(255, 255, 0), width=2)
            dr.rectangle([0, 0, 240, 14], fill=(0, 0, 0))
            dr.text((2, 1), f"{model[:4]} {arm[:5]} {rec.scene_id.split('_')[2]} {rec.episode_id}t{rec.trial} d={d:.0f}px", fill=(255, 255, 255))
            tiles.append(im)
cols = 5
rowsn = math.ceil(len(tiles) / cols)
sheet = Image.new('RGB', (cols * 244, rowsn * 244), (40, 40, 40))
for n, t in enumerate(tiles):
    sheet.paste(t, ((n % cols) * 244, (n // cols) * 244))
sheet.save(OUT / 'crops_wrong_targets.png')
print(len(tiles), 'tiles ->', OUT / 'crops_wrong_targets.png')
