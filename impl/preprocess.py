import open3d as o3d
import numpy as np
import argparse
import os

def normalize(mesh):
    '''
    Normalize mesh in-place.
    '''
    center = mesh.get_center()
    mesh.translate(-center)
    extent = mesh.get_axis_aligned_bounding_box().get_max_extent()
    mesh.scale(2.0 / extent, [0, 0, 0])


def voxelization(file_path):
    '''
    Read a `.obj` file specified by `file_path`. 
    
    Return a `numpy.matrix` object that stores the resulting `32x32x32` voxelization.
    '''
    mesh = o3d.io.read_triangle_mesh(file_path)
    normalize(mesh)
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=float(2/32 + 0.001))
    res = np.zeros((32, 32, 32), dtype=np.int32)
    
    for voxel in voxel_grid.get_voxels():
        # print(voxel.grid_index)
        res[tuple(voxel.grid_index)] = 1

    return res
    

def preprocess_path(src_path, dst_path):
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
        print(f'Processing {path}', end='')
        res = voxelization(path)
        dst = os.path.join(dst_path, 'voxel_grid_' + str(id) + '.txt')
        print(f' -> {dst}')
        np.savetxt(dst, res.reshape((1, -1)), fmt='%d', delimiter=' ')
    
    print(f'{len(files)} file(s) have been processed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python3 preprocess.py',
        description='Preprocess object files and convert them to voxelized matrices.')
    parser.add_argument('src_path', help='Source path for object files. The path specified will be searched recursively.')
    parser.add_argument('dst_path', help='Destination path for txt files that stores the information of voxels. They will be named in the form "voxel_grid_{id}.txt". ')
    args = parser.parse_args()
    preprocess_path(args.src_path, args.dst_path)