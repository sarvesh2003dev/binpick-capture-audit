"""Compare stored depth validity; flag missing calibration and possible range cutoff.

Counts are observations, not calibrated accuracy or causal material diagnostics.
The optional L515 conversion is a stated hypothesis, never inferred metadata.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from audit_depth import font
from rasterizer import read_ply, render_scene

GUIDE = 'https://support.realsenseai.com/hc/en-us/article_attachments/360081846174'


def summarize_depth(raw, ids, cad, erosion_px=3):
    if raw.ndim != 2 or raw.shape != ids.shape or ids.shape != cad.shape:
        raise ValueError('Expected matching 2D raw depth, instance and CAD depth arrays')
    if not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0):
        raise ValueError('Raw depth must be nonnegative integer data')
    inner = np.zeros(raw.shape, bool)
    for instance in np.unique(ids[ids > 0]):
        kernel = np.ones((2 * erosion_px + 1, 2 * erosion_px + 1), np.uint8)
        inner |= cv2.erode((ids == instance).astype(np.uint8), kernel).astype(bool)
    if not inner.any():
        raise ValueError('No object interiors; check CAD, poses and erosion')
    positive = raw[raw > 0]
    z = cad[inner]
    stats = {
        'image_pixels': int(raw.size), 'image_zero_pixels': int((raw == 0).sum()),
        'interior_pixels': int(inner.sum()),
        'interior_zero_pixels': int(((raw == 0) & inner).sum()),
        'outside_pixels': int((ids == 0).sum()),
        'outside_zero_pixels': int(((raw == 0) & (ids == 0)).sum()),
        'positive_raw_min': int(positive.min()) if positive.size else None,
        'positive_raw_median': float(np.median(positive)) if positive.size else None,
        'positive_raw_max': int(positive.max()) if positive.size else None,
        'cad_interior_z_min_mm': float(z.min()),
        'cad_interior_z_max_mm': float(z.max()),
    }
    return stats, inner


def cutoff_hypothesis(frames, assumed_mm_per_unit):
    if assumed_mm_per_unit <= 0 or not np.isfinite(assumed_mm_per_unit):
        raise ValueError('Assumed scale must be positive and finite')
    minima = [f['positive_raw_min'] for f in frames if f['positive_raw_min'] is not None]
    if not minima:
        return {'status': 'unassessable_no_positive_depth'}
    values, counts = np.unique(minima, return_counts=True)
    index = int(np.argmax(counts)); repeated = int(values[index])
    cutoff = repeated * assumed_mm_per_unit
    below = all(f['cad_interior_z_max_mm'] < cutoff for f in frames)
    suspect = len(minima) >= 3 and counts[index] / len(minima) >= .8 and below
    return {'status': 'suspected_range_cutoff' if suspect else 'not_flagged',
            'shared_positive_raw_min': repeated, 'views_at_shared_min': int(counts[index]),
            'views_with_positive_depth': len(minima),
            'assumed_mm_per_raw_unit': assumed_mm_per_unit,
            'hypothesized_cutoff_mm': cutoff,
            'all_cad_interiors_below_hypothesized_cutoff': below,
            'threshold_rule': '>=3 views; >=80% share raw minimum; every CAD interior is nearer than cutoff',
            'interpretation': 'Heuristic, not causation. Stored depth scale and acquisition settings are unverified.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--l515-assumed-mm-per-unit', type=float, default=.25,
                        help='Explicit hypothesis only; NOT a measured dataset scale')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    model_root = args.data / 'dimo_unpacked/dimo/models'
    results = {}; previews = {}; world_labels = {}
    for sensor in ('real_l515', 'real_d415'):
        scene = args.data / 'selected/dimo_sample' / sensor / '000042'
        if not scene.exists():
            continue
        cameras = json.loads((scene / 'scene_camera.json').read_text())
        poses = json.loads((scene / 'scene_gt.json').read_text())
        world_labels[sensor] = json.loads((scene / 'scene_gt_world.json').read_text())
        models = {i: read_ply(model_root / f'obj_{i:06d}.ply')
                  for i in {o['obj_id'] for frame in poses.values() for o in frame}}
        frames = []
        for key in sorted(poses, key=int):
            number = int(key)
            raw = np.asarray(Image.open(scene / f'depth/{number:06d}.png'))
            rgb = np.asarray(Image.open(scene / f'rgb/{number:06d}.png').convert('RGB'))
            if rgb.shape[:2] != raw.shape:
                raise ValueError('RGB/depth dimensions differ; register explicitly before using RGB poses')
            cad, ids = render_scene(models, poses[key], np.array(cameras[key]['cam_K']).reshape(3, 3),
                                    raw.shape[1], raw.shape[0])
            stats, inner = summarize_depth(raw, ids, cad)
            stats['view'] = number; frames.append(stats)
            if number == 0: previews[sensor] = (rgb, raw, ids, inner)
        def ratio(a, b):
            denominator = sum(f[b] for f in frames)
            return sum(f[a] for f in frames) / denominator if denominator else None
        entry = {'frames': frames, 'views': len(frames), 'physical_arrangements': 1,
                 'interior_zero_fraction': ratio('interior_zero_pixels', 'interior_pixels'),
                 'whole_image_zero_fraction': ratio('image_zero_pixels', 'image_pixels'),
                 'outside_zero_fraction': ratio('outside_zero_pixels', 'outside_pixels'),
                 'stored_camera_metadata_keys': sorted(set().union(*(c.keys() for c in cameras.values()))),
                 'calibration_gate': 'HOLD_FOR_SENSOR_METADATA',
                 'gate_reason': 'Scale, acquisition settings and RGB/depth registration not verified; no material model fit.'}
        if sensor == 'real_l515':
            entry['cutoff_hypothesis'] = cutoff_hypothesis(frames, args.l515_assumed_mm_per_unit)
        results[sensor] = entry
        print(sensor, f"objects {entry['interior_zero_fraction']:.2%} zero; whole image {entry['whole_image_zero_fraction']:.2%} zero", flush=True)
    if not results: raise ValueError('No supported scenes found')
    same_world = (world_labels['real_l515'] == world_labels['real_d415']) if len(world_labels) == 2 else None
    report = {'observations': results, 'world_pose_labels_identical_across_cameras': same_world,
              'registration_status': 'CAD/RGB visually inspected; depth registration unverified',
              'limits': ['One arrangement; camera comparisons do not isolate sensor hardware from capture settings.',
                        'Nonzero values can be erroneous; this report counts availability only.',
                        'Suspected cutoff is conditional on the explicitly assumed scale.'],
              'source': 'https://pderoovere.github.io/dimo/', 'license': 'CC BY 4.0',
              'sensor_reference': GUIDE}
    (args.output / 'capture_diagnosis.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    lines = ['# Capture diagnosis', '', 'This report separates stored-data observations from a sensor-configuration hypothesis.', '',
             '| Camera | Views | Zero inside CAD interiors | Zero across image |', '|---|---:|---:|---:|']
    for sensor, r in results.items():
        lines.append(f"| {sensor} | {r['views']} | {r['interior_zero_fraction']:.2%} | {r['whole_image_zero_fraction']:.2%} |")
    h = results.get('real_l515', {}).get('cutoff_hypothesis', {})
    if h.get('status') == 'suspected_range_cutoff':
        lines += ['', f"L515 hypothesis: {h['views_at_shared_min']} views share raw minimum {h['shared_positive_raw_min']}. "
                  f"**If** each stored unit represents {h['assumed_mm_per_raw_unit']} mm, that boundary is {h['hypothesized_cutoff_mm']} mm. "
                  'Every projected object interior is closer than that boundary. A minimum-range setting is a plausible explanation, not a confirmed cause.', '',
                  f'[The L515 guide]({GUIDE}) documents that Min Distance sets nearer values to zero. Actual dataset settings and scale are missing.']
    lines += ['', '**Decision: HOLD_FOR_SENSOR_METADATA.** Do not fit a material-specific dropout model from these captures yet.', '',
              'Required before calibration: stored depth units, capture preset/minimum range, exposure/confidence settings, '
              'verified RGB/depth registration, and multiple independent object arrangements.', '',
              'The previous constant-dropout profile remains useful only as an almost-total-depth-loss software stress test.', '',
              'DIMO: Peter De Roovere, Steven Moonen, Nick Michiels, Francis Wyffels. CC BY 4.0. '
              'https://pderoovere.github.io/dimo/usage/ . Counts, masks, crops and overlays are derived outputs.']
    (args.output / 'CAPTURE_DIAGNOSIS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    canvas = Image.new('RGB', (1400, 170 + 330 * len(previews) + 90), '#101724')
    draw = ImageDraw.Draw(canvas)
    draw.text((24,20), 'Check the camera setup before fitting the simulator', font=font(29), fill='white')
    draw.text((24,65), 'DIMO scene 000042 | real camera captures | CAD-derived masks | one arrangement',font=font(19),fill='#b3c4db')
    draw.text((24,110), 'RED = stored zero depth    GREEN = stored nonzero depth (not certified accurate)',font=font(19),fill='#b3c4db')
    for row, (sensor, (rgb, raw, ids, inner)) in enumerate(previews.items()):
        y = 170 + row * 330
        ys,xs = np.where(ids>0)
        box = (max(0,int(xs.min())-50),max(0,int(ys.min())-50),min(rgb.shape[1],int(xs.max())+50),min(rgb.shape[0],int(ys.max())+50))
        overlay=rgb.copy()
        for instance in np.unique(ids[ids>0]):
            contours,_=cv2.findContours((ids==instance).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,contours,-1,(25,235,175),3)
        validity=(rgb*.25).astype(np.uint8)
        validity[inner&(raw==0)]=(242,79,84);validity[inner&(raw>0)]=(38,202,165)
        for col,arr in enumerate((overlay,validity)):
            panel=Image.fromarray(arr).crop(box);panel.thumbnail((370,280))
            canvas.paste(panel,(24+col*395,y))
        r=results[sensor]
        draw.text((830,y+15),sensor.replace('real_','').upper(),font=font(28),fill='white')
        draw.text((830,y+65),f"Part interiors: {r['interior_zero_fraction']:.2%} zero",font=font(22),fill='white')
        draw.text((830,y+100),f"Whole images: {r['whole_image_zero_fraction']:.2%} zero",font=font(22),fill='white')
        draw.text((830,y+145),f"Pooled over {r['views']} views",font=font(18),fill='#b3c4db')
        draw.text((830,y+180),'Capture settings are unverified.',font=font(18),fill='#b3c4db')
    draw.text((24,canvas.height-75),'Decision: investigate capture settings before material-specific calibration.',font=font(21),fill='#ffd282')
    draw.text((24,canvas.height-38),'Derived from DIMO | De Roovere, Moonen, Michiels & Wyffels | CC BY 4.0 | pderoovere.github.io/dimo',font=font(16),fill='#b3c4db')
    canvas.save(args.output/'capture_comparison.png')


if __name__ == '__main__': main()
