# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Turn a SuperPEEC ``.glb`` into a shaded, camera'd Blender scene.

``sppeec_cli.py --export-glb`` writes the geometry and the field; this
script is the other half -- it opens that file in Blender, wires the
current density into a material that actually reads as a field, frames
a camera, and hands back a ``.blend`` you can orbit (and, with
``--render``, a PNG).

This is a POST-PROCESSING script, like ``studies/dbc_r4_plot.py``: no
solve, seconds to run, and it never touches the numbers. Blender is
needed here and nowhere else in the pipeline.

Run headless (writes ``<stem>.blend``, and a PNG with ``--render``)::

    blender -b --python studies/blender_view.py -- \\
        results/dbc_halfbridge_r3_1e06Hz.glb --render

Or open the finished scene interactively::

    blender --python studies/blender_view.py -- \\
        results/dbc_halfbridge_r3_1e06Hz.glb

Options after the ``--``: ``--view iso|plan|front`` (camera),
``--hide A,B`` (drop objects whose names contain these -- almost
always ``--hide bottom_metal``, because the ground plane is a
40 x 50 mm sheet that occludes the module from above and out-glows it
from everywhere else), ``--render`` (write a PNG), ``--res N``
(long edge in pixels),
``--emit F`` (how much of the image is the field rather than the
lighting; the default 0.92 leaves just enough diffuse to give the
geometry form, 1.0 makes the rendered pixel exactly the colourmap
value, 0 gives an ordinary lit object).

THE MATERIAL. Base colour comes from the baked ``Color`` attribute, so
the scene is right the moment it opens. The raw ``_JMAG`` attribute
(A/m^2 on the surface, A on the wires) is wired into the same node
tree through a Map Range + Color Ramp that reproduces the baked ramp
from the legend JSON -- muted by default, but it is there so you can
drag the ramp against TRUE numbers instead of re-exporting to change
a colour scale. Switch by connecting the ramp output to Base Colour.

THE SCALE BAR PROBLEM. A render has no colour bar, so the legend's
range is burned into the image as render stamp metadata. An image of
a current density with no scale on it is decoration, not a
measurement.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector


def _argv():
    return sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []


def _opt(args, name, default=None, cast=str):
    if name in args:
        return cast(args[args.index(name) + 1])
    return default


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load(path):
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.context.scene.objects if o.type == 'MESH']


def legend_for(path):
    """The colour range the exporter recorded, or {} if absent."""
    side = os.path.splitext(path)[0] + '_legend.json'
    try:
        with open(side) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def field_material(obj, legend, kind, emit=0.75):
    """Vertex-colour material with a live ``_JMAG`` ramp alongside it."""
    mat = bpy.data.materials.new('%s_field' % obj.name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes['Principled BSDF']

    col = nt.nodes.new('ShaderNodeVertexColor')
    col.layer_name = 'Color'
    col.location = (-400, 200)
    nt.links.new(col.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.45
    bsdf.inputs['Metallic'].default_value = 0.0
    if 'Emission Color' in bsdf.inputs:
        nt.links.new(col.outputs['Color'], bsdf.inputs['Emission Color'])
        bsdf.inputs['Emission Strength'].default_value = float(emit)

    # the live path: raw _JMAG -> normalised -> ramp. Left unconnected
    # so the baked colour wins by default; connect ramp -> Base Color
    # to drive the shading from the true numbers instead.
    lo = legend.get('%s_vmin' % kind)
    hi = legend.get('%s_vmax' % kind)
    if lo is not None and hi is not None and '_JMAG' in obj.data.attributes:
        attr = nt.nodes.new('ShaderNodeAttribute')
        attr.attribute_name = '_JMAG'
        attr.location = (-900, -200)
        logscale = legend.get('%s_scale' % kind) == 'log10'
        src = attr.outputs['Fac']
        if logscale:
            lg = nt.nodes.new('ShaderNodeMath')
            lg.operation = 'LOGARITHM'
            lg.location = (-700, -200)
            lg.inputs[1].default_value = 10.0
            nt.links.new(src, lg.inputs[0])
            src = lg.outputs['Value']
            lo, hi = math.log10(max(lo, 1e-300)), math.log10(max(hi, 1e-300))
        rng = nt.nodes.new('ShaderNodeMapRange')
        rng.location = (-520, -200)
        rng.inputs['From Min'].default_value = lo
        rng.inputs['From Max'].default_value = hi
        rng.clamp = True
        nt.links.new(src, rng.inputs['Value'])
        ramp = nt.nodes.new('ShaderNodeValToRGB')
        ramp.location = (-320, -200)
        ramp.label = '%s: %s' % (kind, legend.get('%s_quantity' % kind, ''))
        nt.links.new(rng.outputs['Result'], ramp.inputs['Fac'])

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return mat


def bounds(objs):
    lo = Vector((1e30,)*3)
    hi = Vector((-1e30,)*3)
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    return lo, hi


def frame(objs, view='iso', res=1600, emit=0.85):
    """Camera + lights sized to the model; returns the camera object."""
    lo, hi = bounds(objs)
    centre = (lo + hi)*0.5
    radius = max((hi - lo).length*0.5, 1e-6)

    cam_data = bpy.data.cameras.new('cam')
    cam_data.lens = 50.0
    # Clip planes MUST follow the model, not Blender's defaults. One
    # Blender unit is one millimetre here, so a 40 x 50 mm module puts
    # the camera ~107 units out -- past the default 100-unit far clip,
    # which silently truncates the scene. The near plane matters just
    # as much in the other direction: 0.1 units is 0.1 mm, thick
    # enough to slice into a 0.2 mm trace.
    cam_data.clip_start = max(radius*1e-4, 1e-5)
    cam_data.clip_end = radius*100.0
    cam = bpy.data.objects.new('cam', cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    d = {'iso': Vector((0.62, -1.0, 0.62)),
         'plan': Vector((0.0, -0.001, 1.0)),
         'front': Vector((0.0, -1.0, 0.12))}[view].normalized()
    fov = 2.0*math.atan(0.5*cam_data.sensor_width/cam_data.lens)
    cam.location = centre + d*(radius/math.tan(0.5*fov)*1.18)
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()

    sun = bpy.data.objects.new(
        'sun', bpy.data.lights.new('sun', 'SUN'))
    # Lights are FORM, not brightness: the emission term already
    # carries the field, so lighting is scaled by whatever emission
    # left over. Without this the diffuse response of a 40 mm plate
    # to an area light is a smooth gradient across the board that
    # looks exactly like a physical result and is not one.
    sun.data.energy = 2.6*(1.0 - min(emit, 1.0)) + 0.10
    sun.rotation_euler = (math.radians(48), 0.0, math.radians(-40))
    bpy.context.scene.collection.objects.link(sun)
    fill = bpy.data.objects.new(
        'fill', bpy.data.lights.new('fill', 'AREA'))
    fill.data.energy = (40.0*(1.0 - min(emit, 1.0))
                       + 1.0)*radius*radius
    fill.data.size = radius
    fill.location = centre + Vector((-1.0, -0.6, 0.9))*radius*2.0
    fill.rotation_euler = (Vector((1.0, 0.6, -0.9)).normalized()
                           .to_track_quat('-Z', 'Y').to_euler())
    bpy.context.scene.collection.objects.link(fill)

    # THE COLOUR CORRECTNESS FIX. Blender's default view transform
    # (AgX) is a filmic tone map: it desaturates saturated colour and
    # rolls off highlights, so an inferno ramp renders as washed-out
    # orange and four decades of |J| compress into one. 'Standard' is
    # a straight sRGB encode -- the rendered pixel is the colourmap
    # value, which is the only way a field image is readable AS data.
    vs = bpy.context.scene.view_settings
    vs.view_transform = 'Standard'
    if hasattr(vs, 'look'):
        vs.look = 'None'
    vs.exposure = 0.0
    vs.gamma = 1.0

    world = bpy.data.worlds.new('world')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs[0].default_value = \
        (0.02, 0.02, 0.025, 1.0)
    bpy.context.scene.world = world

    sc = bpy.context.scene
    span = hi - lo
    wide = span.x >= span.y if view == 'plan' else True
    sc.render.resolution_x = res if wide else int(res*0.75)
    sc.render.resolution_y = int(res*0.75) if wide else res
    return cam


def stamp(legend, path):
    """Burn the colour scale into the render -- a plot needs its axis."""
    sc = bpy.context.scene
    bits = []
    for kind in ('surface', 'wire'):
        if '%s_vmin' % kind in legend:
            bits.append('%s %s: %.3g..%.3g (%s)'
                        % (kind, legend.get('%s_quantity' % kind, ''),
                           legend['%s_vmin' % kind],
                           legend['%s_vmax' % kind],
                           legend.get('%s_scale' % kind, 'linear')))
    if legend.get('wire_radius_scale', 1.0) != 1.0:
        bits.append('wire radius x%.3g (NOT to scale)'
                    % legend['wire_radius_scale'])
    if 'frequency_Hz' in legend:
        bits.append('%.4g Hz' % legend['frequency_Hz'])
    bits.append(os.path.basename(path))
    sc.render.use_stamp = True
    sc.render.use_stamp_note = True
    sc.render.stamp_note_text = '  |  '.join(bits)
    for flag in ('use_stamp_date', 'use_stamp_time', 'use_stamp_render_time',
                 'use_stamp_frame', 'use_stamp_scene', 'use_stamp_camera',
                 'use_stamp_filename', 'use_stamp_lens'):
        setattr(sc.render, flag, False)
    sc.render.stamp_font_size = max(14, sc.render.resolution_x//90)


def set_engine():
    """First available real-time engine (the name moved in 4.2)."""
    prop = bpy.types.RenderSettings.bl_rna.properties['engine']
    have = {e.identifier for e in prop.enum_items}
    for cand in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE', 'CYCLES'):
        if cand in have:
            bpy.context.scene.render.engine = cand
            return cand
    return bpy.context.scene.render.engine


def main():
    args = _argv()
    files = [a for a in args if a.endswith('.glb')]
    if not files:
        print(__doc__)
        return 2
    path = os.path.abspath(files[0])
    view = _opt(args, '--view', 'iso')
    res = _opt(args, '--res', 1600, int)
    emit = _opt(args, '--emit', 0.92, float)
    hide = [h for h in (_opt(args, '--hide', '') or '').split(',') if h]

    clear()
    objs = load(path)
    legend = legend_for(path)
    if hide:
        drop = [o for o in objs
                if any(h.lower() in o.name.lower() for h in hide)]
        # names FIRST: bpy.data.objects.remove invalidates the Python
        # wrapper, and touching o.name afterwards raises ReferenceError
        names = [o.name for o in drop]
        keep = [o for o in objs if o not in drop]
        for o in drop:
            bpy.data.objects.remove(o, do_unlink=True)
        objs = keep
        print('hid %d object(s): %s' % (len(names), ', '.join(names)))
        if not objs:
            print('--hide matched everything; nothing left to render')
            return 2
    for o in objs:
        kind = 'wire' if 'wire' in o.name.lower() else 'surface'
        field_material(o, legend, kind, emit=emit)
    frame(objs, view=view, res=res, emit=emit)
    engine = set_engine()
    stamp(legend, path)

    stem = os.path.splitext(path)[0]
    blend = '%s_%s.blend' % (stem, view)
    # framing follows what is VISIBLE, so hiding the ground plane also
    # zooms the camera to the module instead of to the empty sheet
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    print('wrote %s  (%d objects, engine %s)'
          % (blend, len(objs), engine))
    if '--render' in args:
        png = '%s_%s.png' % (stem, view)
        bpy.context.scene.render.filepath = png
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.ops.render.render(write_still=True)
        print('wrote %s' % png)
    return 0


if __name__ == '__main__':
    main()
