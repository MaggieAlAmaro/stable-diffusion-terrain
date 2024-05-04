from PIL import Image
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np

import random, os

# from normals import write_obj, obj_normals

# points = np.loadtxt('./Sio020320.csv', skiprows=1, delimiter=',')[:, 1:]

def generateTriangleGrid(grid_size):
    tris = []
    for x in range(grid_size[0]-1):
        for y in range(grid_size[1]-1):
            index = x*grid_size[0]+y
            a = index
            b = index+1
            c = index+grid_size[0]+1
            d = index+grid_size[0]
            tris.append((a, b, c))
            tris.append((a, c, d))
    return tris

def SplitRGBAInput(RGBAImage : Image): #-> Union[Image,Image]:
    r, g, b, a = RGBAImage.split()
    a = a.convert('L')
    rgb = Image.merge('RGB', (r, g, b))
    return rgb, a

class MeshFromHeightmap():
    def __init__(self, heightmapImage : Image) -> None:
        self.data = np.array(heightmapImage)
        self.size = self.data.shape[0] 
        self.base = random.seed
        pass

    ## Courtesy of this https://loady.one/blog/terrain_mesh.html#generating-the-texture
    def generate_vertices(self, heightmap, heightmap_size):
        vertices = []
        uvs = []

        print(heightmap_size)
        print(heightmap_size[0])
        # The origin and size of mesh
        origin = (-1, -0.75, -1)
        size = 256
        max_height = 0.5

        # We need to calculate the step between vertices 
        step_x = size/(heightmap_size[0]-1)
        step_y = size/(heightmap_size[1]-1)

        for x in range(heightmap_size[0]):
            for y in range(heightmap_size[1]):
                # x_coord = self.base[0] + step_x*x 
                # y_coord = self.base[1] + max_height*heightmap[x][y]
                # z_coord = self.base[2] + step_y*y
                x_coord = step_x*x 
                y_coord = max_height*heightmap[x][y]
                z_coord = step_y*y
                vertices.append((x_coord, y_coord, z_coord))
                uvs.append((y / size, x / size))
        return vertices, uvs

    def generate_tris(self,grid_size):
        return generateTriangleGrid(grid_size)
    
	# https://community.khronos.org/t/calculating-vertex-normal-for-obj-faces/74139
    def calculate_normals(self, verts, tris):
        # obj_n = obj_normals(verts, tris)
        # return obj_n
        pass


    def makeMesh(self, heightmap_size, grid_size):
        v, uvs = self.generate_vertices(self.data,heightmap_size)
        t = self.generate_tris(grid_size)
        return v, t, uvs
        # norms = self.calculate_normals(v,t)
        # return v, t, uvs, norms


# Obj format info
# https://en.wikipedia.org/wiki/Wavefront_.obj_file
# Video explanation of Obj https://www.youtube.com/watch?v=iClme2zsg3I
def export_obj(vertices, tris, UVs, filename, normals=None):
    file = open(filename, "w")

    # Write Vertices
    for vertex in vertices:
        file.write("v " + str(vertex[0]) + " " + str(vertex[1]) + " " + str(vertex[2]) + "\n")
      
    # Write UVs
    for uv in UVs:
    #   file.write("vt " + str(uv[0]) + " " + str(uv[1]) + "\n")
        file.write("vt " + str(uv[0]) + " " + str(uv[1]) + " " + str(float(0)) + "\n")


    # Write Normals
    if normals!= None:
        for n in normals:
            vn_line = f'vn {n[0]} {n[1]} {n[2]}\n'
            file.write(vn_line)


    # Groups (g), textCoord name (_something), and smoothing (s)
    file.write("\ng Terrain\n")
    file.write("usemtl _something\n")
    file.write("s 1\n\n")
    
    # Write Triangles
    for tri in tris:
        file.write("f " + str(tri[0]+1) + " " + str(tri[1]+1) + " " + str(tri[2]+1) + "\n")
    file.close()

    print(f"Exported obj to {filename}")
    return



# DOESNT WORK
# Texture is flipped for some reason
def TexturedGLTFFromObj(objFilename,textureFilename):
    # It screams when reading Objs with texture coordiantes
    # Look into the vtkObjReader 
    # https://github.com/pyvista/pyvista/blob/release/0.43/pyvista/core/utilities/reader.py#L1217-L1234
    reader = pv.get_reader(objFilename)
    mesh = reader.read()
    colorArray = np.array(textureFilename)
    # colorArray = np.flip(colorArray, 1)
    # image = np.moveaxis(colorArray, 3, -1)
    tex = pv.numpy_to_texture(colorArray)

    # Create the ground control points for texture mapping
    # Note: Z-coordinate doesn't matter
    # Maybe try switching u and v?
    o = 0.0, 0.0, 0.0 # Bottom Left
    u = 1.0, 0.0, 0.0 # Bottom Right
    v = 0.0, 1.0, 0.0 # Lop left

    # Tried to flip to align
    # flippedMesh = mesh.flip_y()  


    mapped_surf = mesh.texture_map_to_plane()

    pl = pv.Plotter()
    _ = pl.add_mesh(
        mapped_surf,
        # color='blue',
        texture=tex,
        smooth_shading=True,
        show_scalar_bar=False,
    )
    pl.export_gltf('balls.gltf')  
    pl.show()

import math
def InverseCumulativeExponential(value, lmbda):
    #Using  Mathf.Exp((-1 * value) / 0.11f) scaling, the inverse is:
    return -lmbda * math.log(-value + 1)

def scale(value):
        #Downsizing the real maximums by x30
        max = 291; # max is 8729;
        min = -14; # min is -415;

        #Note: No need to normalize, result will be [0,1], if not then normalizedData = (value - min) / (max - min)
        normalizedValue = (value - 0) / (255 - 0)
        lmbda = 0.11 # wsnt it 0.08
        scaledVal = InverseCumulativeExponential(normalizedValue, lmbda)
        rescaledData = scaledVal * (max - min) + min


        return rescaledData


# This helped a little 
# https://docs.pyvista.org/version/stable/examples/02-plot/topo-map#topo-map-example
def TexturedGLTFFromHeightmap(heightmapFilename : str, gltfOutFilename : str, rgbFilename=None):
    
    img = Image.open(heightmapFilename)
    data = np.array(img)
    
    # Deform Height by a scaler
    scalingFunction = np.vectorize(scale)
    rescaledData = scalingFunction(data) #* (maxOriginal - minOriginal) + minOriginal
    print(f"max is: {rescaledData.max()}, min is: {rescaledData.min()}")
    ##GOTTA ADD TO mAKE IT ABOVE 0
    rescaledData += 14
    newImg = Image.fromarray(rescaledData)
    newImg = newImg.convert('L')
    
    newImg.save('newnewnewn.png')
    heightmap = pv.read('newnewnewn.png')


    # reader = pv.get_reader(heightmapFilename)
    # heightmap = reader.read()

    # print("READ")
    

    # Get triangle Mesh Grid 
    triangles = generateTriangleGrid(heightmap.dimensions[0:2])
    triangles = np.array(triangles)
    # print(triangles.shape)
    # triangles = np.rot90(triangles,2)

    # Convert Triangles to pyvista Faces by adding a 3 before each triangle 
    # Info https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.PolyData.faces.html
    faces = np.insert(triangles,[0],[3], axis=1)
    faces = faces.flatten()

    # heightmapPoints = heightmap.warp_by_scalar(factor=0.3)   
    heightmapPoints = heightmap.warp_by_scalar()
    # polyDataHeightmap = heightmapPoints.cast_to_poly_points()

    # print("Info:\n")
    # print(heightmapPoints+"\n\n")
    # print(heightmapPoints.point_data)
    # print(heightmapPoints.point_data['PNGImage'])


    # Create Mesh with faces and heightmap points
    # PolyData doc https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.polydata#
    mesh = pv.PolyData(heightmapPoints.points,faces)
    mesh = mesh.texture_map_to_plane(inplace=True)
    # Still dont know how this works
    # Doc https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.datasetfilters.texture_map_to_plane


    if rgbFilename != None:  
        tex = pv.read_texture(rgbFilename)
		# Get Texture 
		# This gets texture from PIL Image
        # img = Image.open(rgbFilename)
        # colorArray = np.array(img)
        # colorArray = np.rot90(colorArray,1)
        # colorArray = np.flip(colorArray,axis=1)
        # tex = pv.numpy_to_texture(colorArray)

        
        # mesh.plot(texture=tex)    
        pl = pv.Plotter()
        _ = pl.add_mesh(
            mesh,
            texture=tex,
            smooth_shading=True,
            show_scalar_bar=False,
        )
        pl.export_gltf(gltfOutFilename)  
        # pl.show()
    else:
        pl = pv.Plotter()
        _ = pl.add_mesh(
            mesh,
            texture=heightmapPoints.point_data['PNGImage'].reshape(3,-1),
            smooth_shading=True,
            show_scalar_bar=False,
        )
        pl.export_gltf(gltfOutFilename)  
        # pl.show()
    print(f"Exported GLTF mesh to {gltfOutFilename}")
    
    
import time
def returnSeperateRGBA(RGBAfile):
    img = Image.open(RGBAfile)
    rgb, a = SplitRGBAInput(img)
    out_path = os.path.join('outputs','seperated-img',time.strftime("%Y-%m-%d"))
    os.makedirs(out_path,exist_ok=True)
    img_count = len(os.listdir(out_path))
    # {img_count:05}.png
    aOut = os.path.join(out_path,(str(img_count)+'-height.png'))	
    rgbOut = os.path.join(out_path,(str(img_count)+'-color.png'))	
    rgb.save(rgbOut)
    a.save(aOut)
    return rgbOut, aOut


if __name__ == '__main__':
    # filename = "c:\\ee\\data\\rgba256\\6569_8641_768_768.png"
    # filename = "c:\\ee\\data\\rgba256\\6240_8312_0_768.png"
    filename = "D:\\StableDiffusion\\stable-diffusion-terrain\\data\\RGBAv4_NewExpMean_FullData\\3210_5160_512_512.png"
    # rgb,a = returnSeperateRGBA(filename)
    rgb = "D:\\StableDiffusion\\stable-diffusion-terrain\\outputs\\seperated-img\\2024-04-28\\8-color.png"
    a = "D:\\StableDiffusion\\stable-diffusion-terrain\\outputs\\seperated-img\\2024-04-28\\8-height.png"

	# To create GLTF	
    TexturedGLTFFromHeightmap(a,'erere.gltf',rgb)
    # TexturedGLTFFromHeightmap('my-a.png','erere.gltf','my-rgb.png')


	# To create Obj -> No texture
    # mesh = MeshFromHeightmap(a)
    # v, t, uvs, normals = mesh.makeMesh([256,256],[256,256])
    # export_obj(v,t,uvs,normals,'my-new-snow.obj')