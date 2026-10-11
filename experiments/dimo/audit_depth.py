"""Measure missing depth inside CAD-derived masks on real DIMO captures.

No assumed depth scale: only zero/nonzero validity is evaluated. Pose-derived
masks are approximate geometric labels, not independently hand-annotated truth.
"""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rasterizer import read_ply, render_scene


def font(size):
    for path in ['C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    scene=args.data/'selected/dimo_sample/real_l515/000042'
    model_root=args.data/'dimo_unpacked/dimo/models'
    cameras=json.loads((scene/'scene_camera.json').read_text())
    poses=json.loads((scene/'scene_gt.json').read_text())
    models={i:read_ply(model_root/f'obj_{i:06d}.ply') for i in sorted({o['obj_id'] for a in poses.values() for o in a})}
    frames=[];instance_rows=[];preview=None
    for key in sorted(poses,key=int):
        frame_id=int(key)
        rgb=np.asarray(Image.open(scene/f'rgb/{frame_id:06d}.png').convert('RGB'))
        raw=np.asarray(Image.open(scene/f'depth/{frame_id:06d}.png'))
        h,w=raw.shape
        if rgb.shape[:2]!=(h,w):raise ValueError('RGB and depth dimensions differ')
        k=np.array(cameras[key]['cam_K']).reshape(3,3)
        ideal,ids=render_scene(models,poses[key],k,w,h)
        foreground=ids>0
        # Erode each instance independently: avoid borders and mutual overlap
        # seams where rasterization/calibration error has greatest influence.
        interior=np.zeros((h,w),bool)
        missing=raw==0
        for j,obj in enumerate(poses[key],start=1):
            inner=cv2.erode((ids==j).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
            interior|=inner
            n=int(inner.sum())
            instance_rows.append({'frame':frame_id,'instance':j,'object_id':obj['obj_id'],
                'interior_pixels':n,'missing_pixels':int((missing&inner).sum()),
                'missing_fraction':float(missing[inner].mean()) if n else None})
        n=int(interior.sum());m=int((missing&interior).sum())
        if n==0:raise ValueError('Empty CAD mask; check units/poses')
        frames.append({'frame':frame_id,'object_interior_pixels':n,'missing_depth_pixels':m,
                       'missing_fraction':m/n,'whole_image_missing_fraction':float(missing.mean()),
                       'whole_image_pixels':int(raw.size),'whole_image_missing_pixels':int(missing.sum()),
                       'outside_cad_pixels':int((~foreground).sum()),
                       'outside_cad_missing_pixels':int((missing&~foreground).sum())})
        print(f'View {frame_id:02}: {100*m/n:.2f}% zero depth within eroded object masks ({n:,} pixels)',flush=True)
        np.savez_compressed(args.output/f'view_{frame_id:02d}_labels.npz',
                            cad_instance_ids=ids.astype(np.uint16),cad_depth_mm=ideal.astype(np.float32),
                            object_interior=interior,observed_depth_missing=missing)
        if frame_id==0:preview=(rgb,raw,ideal,ids,interior)

    fit=[r for r in frames if r['frame'] in (0,2,4,6,8,10)]
    held=[r for r in frames if r not in fit]
    calibrated=sum(r['missing_depth_pixels'] for r in fit)/sum(r['object_interior_pixels'] for r in fit)
    def score(rate):
        n=sum(r['object_interior_pixels'] for r in held);m=sum(r['missing_depth_pixels'] for r in held)
        brier=(m*(1-rate)**2+(n-m)*rate**2)/n
        mae=float(np.mean([abs(rate-r['missing_fraction']) for r in held]))
        return {'predicted_missing_fraction':rate,'held_view_fraction_mae':mae,'held_pixel_brier':brier}
    metrics={'dataset':'DIMO','camera':'RealSense L515','scene':'000042','frames':frames,
             'instances':instance_rows,'physical_arrangements':1,'object_instances_per_view':5,
             'mask_source':'CAD rasterization at published object poses; 3 px erosion per instance',
             'mask_alignment_status':'RGB/CAD overlay manually inspected, not independently relabeled',
             'depth_rgb_registration_status':'Stored arrays have matching dimensions; registration not independently calibrated here',
             'invalid_depth_definition':'stored uint16 value equals zero',
             'depth_scale':'not provided in selected metadata; no metric sensor-depth error claimed',
             'fit_views':[r['frame'] for r in fit],'held_views':[r['frame'] for r in held],
             'naive_2pct_dropout':score(.02),'fitted_constant_dropout':score(calibrated),
             'overall_missing_fraction':sum(r['missing_depth_pixels'] for r in frames)/sum(r['object_interior_pixels'] for r in frames),
             'overall_whole_image_missing_fraction':sum(r['whole_image_missing_pixels'] for r in frames)/sum(r['whole_image_pixels'] for r in frames),
             'overall_outside_cad_missing_fraction':sum(r['outside_cad_missing_pixels'] for r in frames)/sum(r['outside_cad_pixels'] for r in frames),
             'limits':['One physical arrangement: view holdout, not scene-level generalization.',
                       'Most pixels outside the CAD objects also have zero depth; missingness is not confined to the metal objects.',
                       'Missing returns do not isolate reflectivity from range, exposure, camera settings or alignment.',
                       'Constant dropout matches aggregate missingness only; it does not recreate spatial sensor failures.',
                       'No neural model trained and no pick-success improvement measured.',
                       'CAD depth is generated by CPU rasterization, not by Isaac Sim or the real depth sensor.']}
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    profile={'schema_version':1,'purpose':'measured object-interior missingness prior for a stress test',
             'material_calibration_status':'NOT_APPROVED: capture settings and saved depth scale unverified; possible minimum-distance cutoff',
             'source':'DIMO real_l515/000042','camera':'RealSense L515',
             'fit_view_ids':metrics['fit_views'],'held_view_ids':metrics['held_views'],
             'missing_probability':calibrated,'sampled_physical_arrangements':1,
             'do_not_generalize_to_other_cameras_or_parts':True,
             'spatial_correlations_modeled':False,'metric_depth_error_modeled':False}
    (args.output/'dropout_profile.json').write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8')
    shutil.copyfile(args.data/'provenance.json',args.output/'provenance.json')
    rgb,raw,ideal,ids,interior=preview
    ys,xs=np.where(ids>0);pad=85
    box=(max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),min(rgb.shape[1],int(xs.max())+pad),min(rgb.shape[0],int(ys.max())+pad))
    overlay=rgb.copy()
    for j in range(1,6):
        contours,_=cv2.findContours((ids==j).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,(25,235,175),3)
    failures=(rgb.astype(float)*.32).astype(np.uint8)
    failures[interior&(raw==0)]=(242,79,84)
    failures[interior&(raw>0)]=(38,202,165)
    rng=np.random.default_rng(7)
    naive=(rgb.astype(float)*.32).astype(np.uint8)
    naivemissing=rng.random(raw.shape)<.02
    naive[interior&naivemissing]=(242,79,84);naive[interior&~naivemissing]=(38,202,165)
    panels=[('Real camera + CAD outline',overlay),('Existing 2% dropout assumption',naive),('Actual recorded missing depth',failures)]
    out=Image.new('RGB',(1500,680),'#101724');draw=ImageDraw.Draw(out)
    draw.text((26,18),'Real data exposes the gap in our synthetic depth',font=font(29),fill='white')
    draw.text((26,59),'DIMO / L515 / scene 000042 / view 00 / five reflective metal blocks',font=font(19),fill='#b3c4db')
    for col,(title,data) in enumerate(panels):
        draw.text((25+col*500,105),title,font=font(21),fill='white')
        image=Image.fromarray(data).crop(box);image.thumbnail((470,360))
        out.paste(image,(15+col*500+(470-image.width)//2,150+(360-image.height)//2))
    draw.text((25,538),'RED: zero depth returned     GREEN: nonzero depth returned',font=font(21),fill='#e1e8f3')
    draw.text((25,574),f"Measured on part interiors: {100*frames[0]['missing_fraction']:.2f}% missing in this view",font=font(23),fill='white')
    draw.text((25,613),'Derived visualization. CAD labels are approximate; one scene does not establish generalization.',font=font(17),fill='#b3c4db')
    draw.text((25,642),'DIMO: De Roovere, Moonen, Michiels & Wyffels · CC BY 4.0 · pderoovere.github.io/dimo',font=font(16),fill='#b3c4db')
    out.save(args.output/'real_depth_audit.png')
    print(json.dumps({k:metrics[k] for k in ['overall_missing_fraction','naive_2pct_dropout','fitted_constant_dropout']},indent=2),flush=True)

if __name__=='__main__':main()
