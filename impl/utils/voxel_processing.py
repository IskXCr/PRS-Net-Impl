import torch
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


def compute_standard_grid_centers():
    bounds = np.linspace(-1.0, 1.0, 33)
    primitive_coords = (bounds[1:] + bounds[:-1]) / 2
    coord_grid = np.array(np.meshgrid(primitive_coords, primitive_coords, primitive_coords, indexing='ij'))

    x, y, z = coord_grid
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    coords = np.concatenate([x, y, z], axis=1)
    
    return coords


def compute_offset_vector_from_std(std_centers, voxel_grid):
    '''
    Given centers for the standard grid, compute the offset vector from the standard grid to the given voxel grid.
    '''
    voxels = voxel_grid.get_voxels()
    base = np.array([1024, 32, 1])
    n_voxels = len(voxels)

    vcenters = np.array([voxel_grid.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxels])
    std_centers = np.array([std_centers[np.dot(voxel.grid_index, base)] for voxel in voxels])
    
    offset_vec = np.mean(vcenters - std_centers, axis=0)
    return offset_vec


def create_data_from_file(file_path, std_centers=compute_standard_grid_centers()):
    '''
    Read a `.obj` file specified by `file_path`. 
    
    Return the normalized mesh and `omap`, a `numpy.matrix` object that stores the resulting `32x32x32` voxelization, and 
    an offset vector which specifies the deviation from the standard grid to the voxel_grid.
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

    offset_vec = compute_offset_vector_from_std(std_centers, voxel_grid)
    return (mesh, omap, offset_vec)


def compute_closest_points_to_grids(mesh, grid_centers):
    '''
    Precompute, for each grid, the closest point on the mesh to that specific grid. 
    
    Return a tensor size of 32x32x32x3.
    
    The mesh itself must has already been normalized.
    '''
    mesh0 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh0)
    ans = scene.compute_closest_points(np.array(grid_centers, dtype=np.float32))
    return ans['points'].numpy()


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
        
    # Compute closest points
    std_centers = compute_standard_grid_centers()
    
    for id, path in enumerate(files):
        dst = os.path.join(dst_path, 'voxel_grid_' + str(id))
        print(f'Processing: {path} -> {dst}')
        
        mesh, omap, offset_vec = create_data_from_file(path)
        
        o3d.io.write_triangle_mesh(dst + '.obj', mesh, write_vertex_colors=False, write_triangle_uvs=False, print_progress=False)

        np.savetxt(dst + '.omap', omap.reshape((1, -1)), fmt='%d', delimiter=' ')
        
        biased_centers = std_centers + offset_vec
        grid_points = compute_closest_points_to_grids(mesh, biased_centers)[:] - offset_vec
        np.savetxt(dst + '.gridpoints', grid_points.reshape((1, -1)), delimiter=' ')
        
        np.savetxt(dst + '.offsetvec', offset_vec, delimiter=' ')
    
    print(f'{len(files)} file(s) have been processed.')


def read_data_from_path(src_path):
    '''
    Read all meshes and voxelized data from a path. They must be produced by `preprocess_files` or the behavior is undefined.
    
    The property of produced files:
    - Meshes are all normalized to $ [-1]^3 $ to $ [1]^3 $
    - Voxels are of size 32x32x32
    
    Return a list of `data_entries` : `tuple(mesh, mat, points, offset_vector)` representing the mesh, its corresponding voxel, 
    the precomputed closest points, and the offset vector.
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
        if path.endswith('.obj'):
            mesh = o3d.io.read_triangle_mesh(path, print_progress=True)
            if index in data:
                data[index].append((0, mesh))
            else:
                data[index] = [(0, mesh)]
        elif path.endswith('.omap'):
            omap = np.loadtxt(path, dtype=int).reshape((32, 32, 32))
            if index in data:
                data[index].append((1, omap))
            else:
                data[index] = [(1, omap)]
        elif path.endswith('.gridpoints'):
            grid_points = np.loadtxt(path, dtype=np.float32).reshape((32, 32, 32, 3))
            if index in data:
                data[index].append((2, grid_points))
            else:
                data[index] = [(2, grid_points)]
        elif path.endswith('.offsetvec'):
            offset_vec = np.loadtxt(path, dtype=np.float32).reshape((3,))
            if index in data:
                data[index].append((3, offset_vec))
            else:
                data[index] = [(3, offset_vec)]
        else:
            print(f'Skipping uncognized format {path}')

    result = []
    for key in list(data.keys()):
        if len(data[key]) < 4:
            print(f'Ignoring incomplete dataset with index={key}')
            data.pop(key)
        else:
            t0, t1, t2, t3 = sorted(data[key])
            result.append((t0[1], t1[1], t2[1], t3[1]))
    
    print(f'{len(result)} dataset(s) have been processed. ')
    return result


def prepare_data_entry(data_entry):
    '''
    Return `tuple(mesh, omap, grid_points, offset_vector, sample_points)`
    '''
    mesh = data_entry[0]
    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    sample_points = np.asarray(pcd.points)[:] - data_entry[3]
    return (data_entry[0], torch.tensor(data_entry[1]), torch.tensor(data_entry[2]), torch.tensor(data_entry[3]), torch.tensor(sample_points))


def prepare_data(data):
    result = []
    for entry in data:
        result.append(prepare_data_entry(entry))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='If this file is executed directly, then preprocess object files and convert them to voxelized matrices and store the corresponding mesh and matrix data into files under destionation path.')
    parser.add_argument('src_path', help='Source path for object files. The path specified will be searched recursively.')
    parser.add_argument('dst_path', help='Destination path for txt files that stores the information of voxels and normalized meshes. They will be named in the form "voxel_grid_{id}.[mat/obj]". ')
    args = parser.parse_args()
    preprocess_files(args.src_path, args.dst_path)