import sys
import json
import os, bpy, bmesh
this_dir = os.path.dirname(bpy.data.filepath)
if not this_dir in sys.path:
    sys.path.append(this_dir)
from blender_utils import ArgumentParserForBlender, BLENDERTOOLBOX_PATH
sys.path.append(BLENDERTOOLBOX_PATH)
import BlenderToolBox as bt

# blender --background --python blender_render.py -- -s {path to mesh file} -c {config name}
parser = ArgumentParserForBlender()
parser.add_argument('-s', '--mesh_path', type=str, required=True, help="path to mesh file")
parser.add_argument('-o', '--output_path', type=str, default=None, help="output path")
parser.add_argument('-c', '--config', type=str, default='default', help="saved object/lighting configuration")
parser.add_argument('--image_resolution', nargs=2, type=int, default=(1080, 1080), help="resolution of image plane")
parser.add_argument('--number_of_samples', type=int, default=200, help="number of samples")
parser.add_argument('--shading', type=str, default='smooth', choices=['smooth', 'flat'])
parser.add_argument('--subdivision_iteration', type=int, default=0)
parser.add_argument('--mesh_color', type=str, default='blue', choices=['red', 'blue', 'green'])
args = parser.parse_args()
arguments = args.__dict__
if arguments['output_path'] is None:
    arguments['output_path'] = os.path.splitext(arguments['mesh_path'])[0] + '.png'

with open('render_configs.json', 'r') as fp:
    saved_config = json.load(fp)[arguments['config']]
arguments.update(saved_config)
for k, v in arguments.items():
    print(k, ":", v)

outputPath = arguments['output_path'] # make it abs path for windows

## initialize blender
imgRes_x = arguments['image_resolution'][0] # recommend > 1080 (UI: Scene > Output > Resolution X)
imgRes_y = arguments['image_resolution'][1] # recommend > 1080 
numSamples = arguments['number_of_samples'] # recommend > 200 for paper images
exposure = 1.5 
bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

## read mesh (choose either readPLY or readOBJ)
meshPath = arguments['mesh_path']
location = arguments['mesh_position'] # (UI: click mesh > Transform > Location)
rotation = arguments['mesh_rotation'] # (UI: click mesh > Transform > Rotation)
scale = arguments['mesh_scale'] # (UI: click mesh > Transform > Scale)
mesh = bt.readMesh(meshPath, location, rotation, scale)

## set shading (uncomment one of them)
if arguments['shading'] == 'smooth':
    bpy.ops.object.shade_smooth() # Option1: Gouraud shading
elif arguments['shading'] == 'flat':
    bpy.ops.object.shade_flat() # Option2: Flat shading
else:
    raise NotImplementedError
# bt.edgeNormals(mesh, angle = 10) # Option3: Edge normal shading

## subdivision
if arguments['subdivision_iteration'] > 0:
    bt.subdivision(mesh, level = arguments['subdivision_iteration'])

###########################################
## Set your material here (see other demo scripts)

# colorObj(RGBA, H, S, V, Bright, Contrast)
color_dict = {
    "blue": [152, 199, 255, 255],
    # "green": [186, 221, 173, 255],
    "green": [165, 221, 144, 255],
    # "red": [255, 189, 189, 255],
    "red": [255, 154, 156, 255],
}
RGBA = [x / 255.0 for x in color_dict[arguments['mesh_color']]]
meshColor = bt.colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)
bt.setMat_plastic(mesh, meshColor)

## End material
###########################################

## set invisible plane (shadow catcher)
bt.invisibleGround(shadowBrightness=arguments['ground_shadowBrightness'])

## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
camLocation = arguments['camLocation']
lookAtLocation = (0,0,0.5)
focalLength = 45 # (UI: click camera > Object Data > Focal Length)
cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

## set light
## Option1: Three Point Light System 
# bt.setLight_threePoints(radius=4, height=10, intensity=1700, softness=6, keyLoc='left')
## Option2: simple sun light
lightAngle = arguments['light_angle'] # UI: click Sun > Transform > Rotation
strength = 2
shadowSoftness = 0.3
sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

## set ambient light
bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

## set gray shadow to completely white with a threshold (optional but recommended)
bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

import bpy
bpy.context.scene.world.light_settings.use_ambient_occlusion = True  # turn AO on
bpy.context.scene.world.light_settings.ao_factor = 0.5  # set it to 0.5

## save blender file so that you can adjust parameters in the UI
print('before save .blend')
bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')

## save rendering
print('before renderImage')
bt.renderImage(outputPath, cam)

## remove blender file after rendering
os.remove(os.getcwd() + '/test.blend')
