"""Apply a measured missingness stress prior to a generated depth/instance pair.

This is independent pixel deletion restricted to object IDs supplied by the
caller. It matches only an aggregate missingness rate; not spatial sensor physics.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--depth',type=Path,required=True,help='float32 .npy depth in meters')
    parser.add_argument('--instances',type=Path,required=True,help='integer .npy instance mask')
    parser.add_argument('--object-ids',type=int,nargs='+',required=True,help='explicit foreground IDs; never include background')
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Output already exists')
    if args.output.suffix!='.npy':raise ValueError('Output must use .npy extension')
    profile=json.loads(args.profile.read_text(encoding='utf-8'))
    p=float(profile['missing_probability'])
    if not 0<=p<=1:raise ValueError('Invalid dropout probability')
    depth=np.load(args.depth,allow_pickle=False)
    instances=np.load(args.instances,allow_pickle=False)
    if depth.shape!=instances.shape or depth.ndim!=2:raise ValueError('Expected matching 2D depth and instance arrays')
    if not np.issubdtype(depth.dtype,np.floating):raise ValueError('Depth must be float, in meters')
    if not np.issubdtype(instances.dtype,np.integer):raise ValueError('Instance mask must be integer')
    foreground=np.isin(instances,args.object_ids)&np.isfinite(depth)&(depth>0)
    if not foreground.any():raise ValueError('No valid depth for selected object IDs')
    remove=foreground&(np.random.default_rng(args.seed).random(depth.shape)<p)
    output=depth.astype(np.float32,copy=True);output[remove]=np.nan
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.save(args.output,output)
    metadata={'profile':profile,'seed':args.seed,'object_ids':args.object_ids,
              'valid_object_pixels_before':int(foreground.sum()),'deleted_object_pixels':int(remove.sum()),
              'warning':'Aggregate-rate stress augmentation. Not a calibrated sensor or spatial dropout model.'}
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
