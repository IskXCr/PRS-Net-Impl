import open3d as o3d
import torch
import numpy as np
from torch.utils.data import Dataset

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
    std_centers = np.concatenate([x, y, z], axis=1)
    
    return std_centers


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


def compute_std_grid_indices(query_points):
    '''
    Compute the corresponding grid indices in std space of the given sample points.
    '''
    bounds = torch.linspace(-1.0, 1.0, 33)
    primitive_coords = (bounds[1:] + bounds[:-1]) / 2
    
    spt = query_points.T
    xs = spt[0]
    ys = spt[1]
    zs = spt[2]
    
    x = torch.searchsorted(primitive_coords, xs) - 1
    y = torch.searchsorted(primitive_coords, ys) - 1
    z = torch.searchsorted(primitive_coords, zs) - 1
    x[x < 0] = 0
    y[y < 0] = 0
    z[z < 0] = 0
    
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    return torch.cat([x, y, z], dim=1)


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


def read_dataset_from_path(src_path):
    '''
    Read all meshes and voxelized data from a path. They must be produced by `preprocess_files` or the behavior is undefined.
    
    The property of produced files:
    - Meshes are all normalized to $ [-1]^3 $ to $ [1]^3 $
    - Voxels are of size 32x32x32
    
    Return a list of `data_entries` : `tuple(path_prefix, index)`
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
        print(f'Reading: "{path}"', end='')
        path: str
        l_id_index = path.rindex('voxel_grid_') + 11
        r_id_index = path.rindex('.')
        index = int(path[l_id_index:r_id_index])
        
        if index not in data:
            data[index] = [path[0:path.rindex('.')], index, 0, 0, 0, 0]
        
        if path.endswith('.obj'):
            data[index][2] = 1
        elif path.endswith('.omap'):
            data[index][3] = 1
        elif path.endswith('.gridpoints'):
            data[index][4] = 1
        elif path.endswith('.offsetvec'):
            data[index][5] = 1
        else:
            print(' - ignored', end='')
        print()
        
    result = []
    for key in list(data.keys()):
        if data[key][2] == 1 and data[key][3] == 1 and data[key][4] == 1 and data[key][5] == 1:
            result.append((data[key][0], data[key][1]))
        else:
            print(f"Ignoring incomplete dataset: {data[key][0]}")
    
    print(f'Detected {len(result)} dataset(s). ')
    return result


def sample(mesh, offset_vec, num):
    '''
    Return sampled points.
    '''
    pcd = mesh.sample_points_uniformly(number_of_points=num)
    sample_points = np.asarray(pcd.points)[:] - offset_vec.numpy()
    return sample_points


def prepare_single_data(data_entry, sample_num=1000):
    '''
    Accepts a single `data_entry` : `tuple(path_prefix, index)`.
    '''
    
    index = data_entry[1]
    mesh = o3d.io.read_triangle_mesh(data_entry[0] + '.obj', print_progress=True)
    omap = torch.tensor((np.loadtxt(data_entry[0] + '.omap', dtype=np.int32)).reshape(1, 32, 32, 32), dtype=torch.float64)
    grid_points = torch.tensor(np.loadtxt(data_entry[0] + '.gridpoints', dtype=np.float64)).reshape(32, 32, 32, 3)
    offset_vector = torch.tensor(np.loadtxt(data_entry[0] + '.offsetvec', dtype=np.float64)).reshape(1, 3)
    sample_points = torch.tensor(sample(mesh, offset_vector, sample_num).reshape(-1, 3))
    
    return (index, mesh, omap, grid_points, offset_vector, sample_points)


@DeprecationWarning
def prepare_dataset(raw_dataset, sample_num=1000):
    '''
    This method is deprecated, as it prepares all data at once.
    
    Input: `tuple(mesh, omap, grid_points, offset_vector`
    
    Return: `tuple(mesh, omap, grid_points, offset_vector, sample_points)`
    '''
    result_lst = []
    
    for entry in raw_dataset:
        mesh = entry[0]
        omap = torch.tensor(entry[1].reshape(1, 32, 32, 32), dtype=torch.float64)
        grid_points = torch.tensor(entry[2].reshape(32, 32, 32, 3))
        offset_vector = torch.tensor(entry[3].reshape(1, 3))
        sample_points = torch.tensor(sample(entry[0], entry[3], sample_num).reshape(-1, 3))
        result_lst.append((mesh, omap, grid_points, offset_vector, sample_points))
    
    print(f'{len(result_lst)} dataset(s) have been prepared. ')
    
    return result_lst


class CustomVoxelDataset(Dataset):
    def __init__(self, dataset_lst, sample_num=1000):
        self.dataset_lst = dataset_lst
        self.sample_num = sample_num
    
    def __len__(self):
        return len(self.dataset_lst)
    
    def __getitem__(self, index):
        return prepare_single_data(self.dataset_lst[index], self.sample_num)


def collate_data_list(raw_dataset):
    data_index_lst = []
    mesh_lst = []
    omap_lst = []
    grid_points_lst = []
    offset_vector_lst = []
    sample_points_lst = []
    
    for entry in raw_dataset:
        data_index_lst.append(entry[0])
        mesh_lst.append(entry[1])
        omap_lst.append(entry[2].reshape(1, 1, 32, 32, 32))
        grid_points_lst.append(entry[3].reshape(1, 32, 32, 32, 3))
        offset_vector_lst.append(entry[4].reshape(1, 1, 3))
        sample_points_lst.append(entry[5].reshape(1, -1, 3))
        
    # batch_mesh = torch.concat(mesh_lst, dim=0)
    batch_omap = torch.concat(omap_lst, dim=0)
    batch_grid_points = torch.concat(grid_points_lst, dim=0)
    batch_offset_vector = torch.concat(offset_vector_lst, dim=0)
    batch_sample_points = torch.concat(sample_points_lst, dim=0)
    return data_index_lst, mesh_lst, batch_omap, batch_grid_points, batch_offset_vector, batch_sample_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='If this file is executed directly, then preprocess object files and convert them to voxelized matrices and store the corresponding mesh and matrix data into files under destionation path.')
    parser.add_argument('src_path', help='Source path for object files. The path specified will be searched recursively.')
    parser.add_argument('dst_path', help='Destination path for txt files that stores the information of voxels and normalized meshes. They will be named in the form "voxel_grid_{id}.[mat/obj]". ')
    args = parser.parse_args()
    preprocess_files(args.src_path, args.dst_path)