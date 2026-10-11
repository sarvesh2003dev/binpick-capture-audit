"""Render real generated mesh geometry with Matplotlib; NOT an Isaac render."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from assets import build_geometry, generate_specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("geometry_preview.png"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    specs = generate_specs(50, args.seed)
    fig = plt.figure(figsize=(15,10), facecolor="#0e1524")
    light = np.array([.3,-.5,.8]); light /= np.linalg.norm(light)
    for index,spec in enumerate(specs):
        ax = fig.add_subplot(5,10,index+1,projection="3d",facecolor="#0e1524")
        geometry = build_geometry(spec)
        points = np.asarray(geometry["points"])
        triangles = points[np.asarray(geometry["faces"])]
        normals = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        shade = .30 + .70*np.clip(normals @ light,0,1)
        base = np.array([.65,.76,.88]) if spec["density_kg_m3"]==2700 else np.array([.87,.72,.49])
        colors = np.clip(shade[:,None]*base,0,1)
        poly = Poly3DCollection(triangles,facecolors=colors,linewidths=0,rasterized=True)
        ax.add_collection3d(poly)
        extent = .043
        ax.set(xlim=(-extent,extent),ylim=(-extent,extent),zlim=(-extent,extent))
        ax.set_box_aspect((1,1,1)); ax.view_init(elev=32,azim=-50)
        ax.set_axis_off()
        ax.set_title(f"{index:02d}  {spec['mass_kg']*1000:.0f} g",color="#c6d4e8",fontsize=8,pad=-2)
    fig.suptitle("50 generated part variants — actual mesh geometry",color="white",fontsize=19,y=.98)
    fig.text(.5,.935,"Bushings · flanged bushings · simplified gears | Color identifies material density",
             ha="center",color="#bccbe0",fontsize=11)
    fig.text(.5,.025,"CPU geometry preview. No physics, RTX rendering or real-camera validation implied.",
             ha="center",color="#f2c575",fontsize=11)
    fig.subplots_adjust(left=.015,right=.985,bottom=.07,top=.91,wspace=.0,hspace=.15)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=120,facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
