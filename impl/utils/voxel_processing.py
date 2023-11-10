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
    if extent == 0.0:
        raise ValueError("Invalid extent.")
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
    
    return torch.tensor(std_centers)


def compute_offset_vector_from_std(std_centers, voxel_grid):
    '''
    Given centers for the standard grid, compute the offset vector from the standard grid to the given voxel grid.
    '''
    voxels = voxel_grid.get_voxels()
    base = torch.tensor([1024, 32, 1])
    n_voxels = len(voxels)
    
    vcenters = torch.tensor(np.array([voxel_grid.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxels]))
    orig_centers = torch.tensor(np.array([std_centers[np.dot(voxel.grid_index, base)] for voxel in voxels]))
    
    offset_vec = torch.mean(vcenters - orig_centers, dim=0)
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
    return torch.tensor(ans['points'].numpy())


def sample(mesh, offset_vec, sample_num=1000):
    '''
    Return sampled points.
    '''
    pcd = mesh.sample_points_uniformly(number_of_points=sample_num)
    sample_points = np.asarray(pcd.points)[:] - offset_vec.numpy()
    return sample_points


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
    omap = torch.zeros((32, 32, 32), dtype=torch.int)
    
    for voxel in voxel_grid.get_voxels():
        # print(voxel.grid_index)
        omap[tuple(voxel.grid_index)] = 1
    
    std_centers = compute_standard_grid_centers()
    offset_vec = compute_offset_vector_from_std(std_centers, voxel_grid)
    
    biased_centers = std_centers + offset_vec
    grid_points = compute_closest_points_to_grids(mesh, biased_centers)[:] - offset_vec
    sample_points = torch.tensor(sample(mesh, offset_vec).reshape(-1, 3))
    
    return mesh, omap, grid_points, offset_vec, sample_points


def preprocess_files(src_path, dst_path):
    '''
    Do preprocessing on all the `.obj` file inside `src_path`, and place the results inside `dst_path`.
    
    No directory hiearchy will be maintained.
    '''
    print(f'Please wait while this utility is gathering object files...')
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    files = []
    for dir, _, f in os.walk(src_path):
        for filename in f:
            if filename.endswith('.obj'):
                files.append(os.path.join(dir, filename))
                
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        
    identifier = 'prs_dat_'
    custom_obj_ext = '.objdat'
    custom_dat_ext = '.voxdat'
    
    print(f'{len(files)} file(s) have been detected by the preprocessor.')
    
    for id, path in enumerate(files):
        dst = os.path.join(dst_path, identifier + str(id))
        print(f'Processing: {path} -> {dst}')
        try:
            mesh, omap, grid_points, offset_vec, sample_points = create_data_from_file(path)
            torch.save({
                        'triangles': np.asarray(mesh.triangles),
                        'vertices': np.asarray(mesh.vertices),
                        }, dst + custom_obj_ext)
            torch.save({
                        'omap': omap,
                        'grid_points': grid_points,
                        'offset_vec': offset_vec,
                        'sample_points': sample_points
                        }, dst + custom_dat_ext)
            
        except ValueError as v:
            print(f'\tAn error has occurred during processing "{dst}": {v}. Skipping.')
    
    print(f'{len(files)} file(s) have been processed.')
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)


def read_mesh_from_file(src_path):
    '''
    Read mesh from file with `custom_obj_ext` as extension.
    '''
    mesh_dict = torch.load(src_path)
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = mesh_dict['triangles']
    mesh.vertices = mesh_dict['vertices']
    mesh.compute_triangle_normals()
    return mesh


def read_dataset_from_path(src_path):
    '''
    Read all meshes and voxelized data from a path. They must be produced by `preprocess_files` or the behavior is undefined.
    
    The property of produced files:
    - Meshes are all normalized to $ [-1]^3 $ to $ [1]^3 $
    - Voxels are of size 32x32x32
    
    Return a list of `data_entries` : `tuple(path_prefix, index)`
    '''
    
    custom_ext = '.voxdat'
    files = []
    for dir, _, f in os.walk(src_path):
        for filename in f:
            if filename.endswith(custom_ext):
                files.append(os.path.join(dir, filename))

    # tuple (index, type, object)
    #     - type: 0 for mesh, 1 for mat
    result = []

    for path in files:
        # print(f'Reading: "{path}"', end='')
        path: str
        r_id_index = path.rindex('.')
        index = path[0:r_id_index]
        result.append((path, index))
        
    print(f'Detected {len(result)} dataset(s). ')
    return result


def prepare_single_data(data_entry):
    '''
    Accepts a single `data_entry` : `tuple(path, index)`.
    '''
    # print(f'Preparing: "voxel_data_{data_entry[1]}"')
    path, index = data_entry
    data = torch.load(path)
    
    omap = data['omap']
    grid_points = data['grid_points']
    offset_vec = data['offset_vec']
    sample_points = data['sample_points']
    
    return (index, omap, grid_points, offset_vec, sample_points)


class CustomVoxelDataset(Dataset):
    def __init__(self, dataset_lst):
        self.dataset_lst = dataset_lst
    
    def __len__(self):
        return len(self.dataset_lst)
    
    def __getitem__(self, index):
        return prepare_single_data(self.dataset_lst[index])


def collate_data_list(raw_dataset):
    data_index_lst = []
    omap_lst = []
    grid_points_lst = []
    offset_vector_lst = []
    sample_points_lst = []
    
    for entry in raw_dataset:
        data_index_lst.append(entry[0])
        omap_lst.append(entry[1].reshape(1, 1, 32, 32, 32))
        grid_points_lst.append(entry[2].reshape(1, 32, 32, 32, 3))
        offset_vector_lst.append(entry[3].reshape(1, 1, 3))
        sample_points_lst.append(entry[4].reshape(1, -1, 3))
        
    # batch_mesh = torch.concat(mesh_lst, dim=0)
    batch_omap = torch.concat(omap_lst, dim=0)
    batch_grid_points = torch.concat(grid_points_lst, dim=0)
    batch_offset_vector = torch.concat(offset_vector_lst, dim=0)
    batch_sample_points = torch.concat(sample_points_lst, dim=0)
    return data_index_lst, batch_omap, batch_grid_points, batch_offset_vector, batch_sample_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='If this file is executed directly, then preprocess object files and convert them to voxelized matrices and store the corresponding mesh and matrix data into files under destionation path.')
    parser.add_argument('src_path', help='Source path for object files. The path specified will be searched recursively.')
    parser.add_argument('dst_path', help='Destination path for txt files that stores the information of voxels and normalized meshes. They will be named in the form "voxel_grid_{id}.[mat/obj]". ')
    args = parser.parse_args()
    preprocess_files(args.src_path, args.dst_path)