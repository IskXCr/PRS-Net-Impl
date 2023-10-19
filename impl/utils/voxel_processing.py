import open3d as o3d
import numpy as np
import argparse
import os


def normalize_mesh(mesh):
    '''
    Normalize mesh in-place.
    '''
    center = mesh.get_center()
    mesh.translate(-center)
    extent = mesh.get_axis_aligned_bounding_box().get_max_extent()
    mesh.scale(2.0 / extent, [0, 0, 0])


def create_data_from_file(file_path):
    '''
    Read a `.obj` file specified by `file_path`. 
    
    Return the normalized mesh and `omap`, a `numpy.matrix` object that stores the resulting `32x32x32` voxelization.
    '''
    mesh = o3d.io.read_triangle_mesh(file_path)
    normalize_mesh(mesh)
    
    # sample 1000 points uniformly, as suggested
    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=float(2/32 + 0.001))
    omap = np.zeros((32, 32, 32), dtype=int)
    
    for voxel in voxel_grid.get_voxels():
        # print(voxel.grid_index)
        omap[tuple(voxel.grid_index)] = 1

    return (mesh, omap)


def preprocess_files(src_path, dst_path):
    '''
    Do preprocessing on all the `.obj` file inside `src_path`, and place the results inside `dst_path`.
    
    No directory hiearchy will be maintained.
    '''
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    files = []
    for dir, _, f in os.walk(src_path):
        for filename in f:
            if '.obj' in filename:
                files.append(os.path.join(dir, filename))
                
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        
    for id, path in enumerate(files):
        print(f'Processing: {path}', end='')
        mesh, omap = create_data_from_file(path)
        dst = os.path.join(dst_path, 'voxel_grid_' + str(id))
        print(f' -> {dst}')
        np.savetxt(dst + '.mat', omap.reshape((1, -1)), fmt='%d', delimiter=' ')
        o3d.io.write_triangle_mesh(dst + '.obj', mesh, write_vertex_colors=False, write_triangle_uvs=False, print_progress=True)
    
    print(f'{len(files)} file(s) have been processed.')


def compute_closest_points_to_grids(mesh, coords):
    '''
    Precompute, for each grid, the closest point on the mesh to that specific grid. 
    
    Return a tensor size of 32x32x32x3.
    
    The mesh itself must has already been normalized.
    '''
    mesh0 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh0)
    ans = scene.compute_closest_points(coords)
    return ans['points'].numpy().reshape(32, 32, 32, 3)


def read_data_from_path(src_path):
    '''
    Read all meshes and voxelized data from a path. They must be produced by `preprocess_files` or the behavior is undefined.
    
    The property of produced files:
    - Meshes are all normalized to $ [-1]^3 $ to $ [1]^3 $
    - Voxels are of size 32x32x32
    
    Return a list of tuples (mesh, mat, points) representing the mesh, its corresponding voxel, and the precomputed closest points.
    For details on the last term, please refer to the paper.
    '''
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    files = []
    for dir, _, f in os.walk(src_path):
        for filename in f:
            if 'voxel_grid_' in filename:
                files.append(os.path.join(dir, filename))

    # tuple (index, type, object)
    #     - type: 0 for mesh, 1 for mat
    data = {}

    for path in files:
        print(f'Reading: {path}')
        path: str
        l_id_index = path.rindex('voxel_grid_') + 11
        r_id_index = path.rindex('.')
        index = int(path[l_id_index:r_id_index])
        if path.endswith('obj'):
            mesh = o3d.io.read_triangle_mesh(path, print_progress=True)
            if index in data:
                data[index] = (mesh, data[index])
            else:
                data[index] = (mesh)
        elif path.endswith('mat'):
            omap = np.loadtxt(path, dtype=int).reshape((32, 32, 32))
            if index in data:
                data[index] = (data[index], omap)
            else:
                data[index] = (omap)
        else:
            print(f'Skipping uncognized format {path}')

    result = []
    for key in list(data.keys()):
        if len(data[key]) < 2:
            print(f'Ignoring incomplete data with index {key}')
            data.pop(key)
        else:
            result.append(data[key])

    # Compute closest points
    primitive_coords = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    coords = np.array(np.meshgrid(primitive_coords, primitive_coords, primitive_coords)).T.reshape(-1, 3)

    for i in range(len(result)):
        mesh = result[i][0]
        omap = result[i][1]
        grid_points = compute_closest_points_to_grids(mesh, coords)
        result[i] = (mesh, omap, grid_points)
    
    print(f'{len(result)} dataset(s) have been processed. `grid_points` have been computed.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='If this file is executed directly, then preprocess object files and convert them to voxelized matrices and store the corresponding mesh and matrix data into files under destionation path.')
    parser.add_argument('src_path', help='Source path for object files. The path specified will be searched recursively.')
    parser.add_argument('dst_path', help='Destination path for txt files that stores the information of voxels and normalized meshes. They will be named in the form "voxel_grid_{id}.[mat/obj]". ')
    args = parser.parse_args()
    preprocess_files(args.src_path, args.dst_path)