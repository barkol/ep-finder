"""Figure 1: Blender (Cycles) renders of the eigenvalue branches. Input data/fig01_blender.json,
output data/fig01_render_<panel>.png with transparent background. The vertical axis is theta, the horizontal
ones are Re and Im lambda.
Run: blender -b -P render_fig01_blender.py -- [samples]   (default 256 samples; the panels in
../data were rendered with 200; uses the GPU if available)."""
import bpy, json, sys, os, math
DATA=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data'))
from mathutils import Vector
ARG=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
SPP=int(ARG[0]) if ARG else 256
D=json.load(open(os.path.join(DATA,'fig01_blender.json'))); H=2.5
PAL=[(0.0,0.447,0.698),(0.835,0.369,0.0),(0.0,0.62,0.451),(0.8,0.475,0.655)]
def lin(c): return tuple((x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4) for x in c)+(1,)
def material(n,c,r=.32,a=1.):
    m=bpy.data.materials.new(n); m.use_nodes=True; b=m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value=lin(c); b.inputs['Roughness'].default_value=r; b.inputs['Alpha'].default_value=a
    if a<1: m.blend_method='BLEND'
    return m
def tube(pts,m,R=.042):
    cu=bpy.data.curves.new('c','CURVE'); cu.dimensions='3D'; cu.bevel_depth=R; cu.bevel_resolution=6; cu.use_fill_caps=True
    s=cu.splines.new('POLY'); s.points.add(len(pts)-1)
    for i,p in enumerate(pts): s.points[i].co=(*p,1)
    o=bpy.data.objects.new('c',cu); bpy.context.collection.objects.link(o); cu.materials.append(m); return o
def sphere(p,m,R):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=R,location=p,segments=48,ring_count=24); o=bpy.context.object; bpy.ops.object.shade_smooth(); o.data.materials.append(m)
def plate(z,r,m):
    bpy.ops.mesh.primitive_plane_add(size=2*r,location=(0,0,z)); o=bpy.context.object; o.data.materials.append(m)
    # edge of the plate
    k=[(-r,-r,z),(r,-r,z),(r,r,z),(-r,r,z),(-r,-r,z)]; tube(k,MK,.006)
for kl,S in D.items():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc=bpy.context.scene; sc.render.engine='CYCLES'; sc.cycles.samples=SPP; sc.cycles.use_denoising=True
    try:
        pr=bpy.context.preferences.addons['cycles'].preferences; pr.compute_device_type='CUDA' if False else 'OPTIX'; pr.get_devices()
        for d in pr.devices: d.use=True
        sc.cycles.device='GPU'
    except Exception as e: print('CPU',e)
    sc.render.film_transparent=True; sc.render.resolution_x=1260; sc.render.resolution_y=1500; sc.render.image_settings.file_format='PNG'; sc.render.image_settings.color_mode='RGBA'
    sc.view_settings.view_transform='Standard'
    w=bpy.data.worlds.new('w'); sc.world=w; w.use_nodes=True; w.node_tree.nodes['Background'].inputs[0].default_value=(1,1,1,1); w.node_tree.nodes['Background'].inputs[1].default_value=.55
    MK=material('kr',(.45,.45,.45)); MP=material('pl',(.72,.74,.76),.5,.28)
    r=1.12; plate(0,r,MP); plate(H,r,MP)
    for i,st in enumerate(S):
        n=len(st); pts=[(a,b,H*j/(n-1)) for j,(a,b) in enumerate(st)]
        m=material(f's{i}',PAL[i%4]); tube(pts,m); sphere(pts[0],m,.085); sphere(pts[-1],m,.085)
    c=bpy.data.cameras.new('k'); c.type='ORTHO'; c.ortho_scale=4.05; cam=bpy.data.objects.new('k',c); sc.collection.objects.link(cam); sc.camera=cam
    focus=Vector((0,0,H/2)); view_dir=Vector((1.0,-1.55,.62)).normalized(); cam.location=focus+view_dir*12
    cam.rotation_euler=(focus-cam.location).to_track_quat('-Z','Y').to_euler()
    for loc,en,sz in (((3,-3,6),320,5),((-4,-1,4),110,6)):
        l=bpy.data.lights.new('l','AREA'); l.energy=en; l.size=sz; o=bpy.data.objects.new('l',l); sc.collection.objects.link(o); o.location=loc; o.rotation_euler=(focus-Vector(loc)).to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=os.path.join(DATA,f'fig01_render_{kl}.png'); bpy.ops.render.render(write_still=True); print('OK',kl)
