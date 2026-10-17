"""Generate a MuJoCo scene from model_config without editing robot assets."""
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET


def _vector(value, length, *, positive=False):
    numbers = tuple(float(x) for x in value)
    if len(numbers) != length or not all(math.isfinite(x) for x in numbers):
        raise ValueError(f"Expected {length} finite values")
    if positive and any(x <= 0. for x in numbers):
        raise ValueError("Dimensions must be positive")
    return ' '.join(format(x, '.12g') for x in numbers)


def camera_quaternion(look_down_deg, roll_deg):
    """Compose a body Y rotation with local Z roll; viewing axis is -Z."""
    if not math.isfinite(look_down_deg) or not 0. <= look_down_deg <= 90. or not math.isfinite(roll_deg):
        raise ValueError("Camera downward angle must be in [0, 90] degrees and roll must be finite")
    pitch = math.radians(look_down_deg - 90.) / 2.
    roll = math.radians(roll_deg) / 2.
    cp, sp, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    return cp * cr, sp * sr, sp * cr, cp * sr


def _stream_values(stream):
    dimensions = [stream[key] for key in ('color_width', 'color_height', 'depth_width', 'depth_height')]
    if any(isinstance(x, bool) or int(x) != x or not 1 <= x <= 4096 for x in dimensions):
        raise ValueError("Camera resolutions must be integers in [1, 4096]")
    for key in ('color_fov_y_deg', 'depth_fov_y_deg'):
        if not 0. < stream[key] < 180.:
            raise ValueError("Camera FOV must be in (0, 180) degrees")
    if not .1 <= stream['fps'] <= 240.:
        raise ValueError("Camera FPS must be in [0.1, 240]")
    if not 0. <= stream['min_distance_m'] < stream['max_distance_m']:
        raise ValueError("Invalid camera distance range")
    return (
        stream['color_width'], stream['color_height'], stream['color_fov_y_deg'],
        stream['depth_width'], stream['depth_height'], stream['depth_fov_y_deg'],
        stream['fps'], int(stream['depth_enabled']), stream['min_distance_m'],
        stream['max_distance_m'], int(stream['preview_enabled']),
    )


def generate_simulation_xml(base_xml, *, camera=None, terrain_steps=(),
                            terrain_rgba=(.6, .55, .45, 1.), terrain_friction=(.8, .01, .001),
                            output_dir=None):
    """Write a scene in the temporary directory; all asset paths stay valid.

    The source MJCF is read only. Content-addressed paths keep concurrent
    launches with different settings from overwriting an active model file.
    """
    source = Path(base_xml).resolve()
    root = ET.parse(source).getroot()
    compiler = root.find('compiler')
    asset_dir = source.parent
    if compiler is not None:
        asset_dir = (source.parent / compiler.get('assetdir', '.')).resolve()
    mesh_dir = (source.parent / compiler.get('meshdir')).resolve() if compiler is not None and compiler.get('meshdir') else asset_dir
    texture_dir = (source.parent / compiler.get('texturedir')).resolve() if compiler is not None and compiler.get('texturedir') else asset_dir
    for item in root.iter():
        file = item.get('file')
        if file:
            directory = (mesh_dir if item.tag in ('mesh', 'skin') else texture_dir if item.tag == 'texture'
                         else source.parent if item.tag == 'include' else asset_dir)
            item.set('file', str((directory / file).resolve()))
    if compiler is not None:
        for key in ('meshdir', 'texturedir', 'assetdir'):
            if compiler.get(key):
                compiler.set(key, str((source.parent / compiler.get(key)).resolve()))
    if camera is not None:
        name = camera['name']
        if re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name) is None:
            raise ValueError("Invalid camera name")
        parents = [body for body in root.iter('body') if body.get('name') == camera['parent']]
        if len(parents) != 1:
            raise ValueError(f"Camera parent body must exist exactly once: {camera['parent']}")
        if any(item.get('name') == name for item in root.iter('camera')):
            raise ValueError(f"Camera already exists in base XML: {name}")
        values = _stream_values(camera['stream'])
        ET.SubElement(parents[0], 'camera', name=name, mode='fixed',
            pos=_vector(camera['position'], 3),
            quat=_vector(camera_quaternion(camera['look_down_deg'], camera['roll_deg']), 4),
            fovy=str(camera['stream']['depth_fov_y_deg']))
        size = root.find('size')
        if size is None:
            size = ET.SubElement(root, 'size')
        size.set('nuser_sensor', str(max(11, int(size.get('nuser_sensor', '0')))))
        sensors = root.find('sensor')
        if sensors is None:
            sensors = ET.SubElement(root, 'sensor')
        if any(item.get('name') == name for item in sensors):
            raise ValueError(f"Sensor already exists in base XML: {name}")
        ET.SubElement(sensors, 'user', name=name, objtype='camera', objname=name, dim='1',
            datatype='real', needstage='pos', user=_vector(values, 11))
    if terrain_steps:
        world = root.find('worldbody')
        if world is None:
            world = ET.SubElement(root, 'worldbody')
        names = {item.get('name') for item in root.iter('geom')}
        for step in terrain_steps:
            if not step['name'] or step['name'] in names:
                raise ValueError(f"Duplicate or empty terrain name: {step['name']}")
            names.add(step['name'])
            ET.SubElement(world, 'geom', name=step['name'], type='box',
                pos=_vector(step['position'], 3), size=_vector(step['size'], 3, positive=True),
                rgba=_vector(terrain_rgba, 4), friction=_vector(terrain_friction, 3))
    ET.indent(root, space='  ')
    payload = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    signature = hashlib.sha256(payload).hexdigest()[:24]
    directory = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir()) / f'bxi_elf3_sim_{os.getuid()}'
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f'elf3_{signature}.xml'
    if not destination.is_file() or destination.read_bytes() != payload:
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.xml', delete=False) as file:
            temporary = Path(file.name)
            file.write(payload)
        try:
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return str(destination)
