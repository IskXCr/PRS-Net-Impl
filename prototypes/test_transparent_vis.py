# Test Transparent Visualization

import open3d as o3d
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import sys
print(f'cwd={os.getcwd()}')
sys.path.insert(0, 'impl/utils/')
import voxel_processing as vp

# device will determine whether to run the training on GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load datasets
data = vp.read_dataset_from_path('data/voxel_data/')
data = vp.prepare_dataset(data)

mesh = data[0][0]
plane = np.array([1, 1, 1, 0])
A = plane[0]
B = plane[1]
C = plane[2]
D = plane[3]

coords_mat = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])

zs = (-coords_mat @ plane[0:2] - plane[3]) / plane[2]
zs = zs.reshape(-1, 1)

plane_vertices = np.concatenate([coords_mat, zs], axis=1)

plane_triangles = np.array([[0, 1, 2], [2, 1, 3], [0, 2, 1], [1, 2, 3]])

normal = plane[0:3] / np.linalg.norm(plane[0:3])
plane_mesh = o3d.geometry.TriangleMesh()
plane_mesh.vertices = o3d.utility.Vector3dVector(plane_vertices)
plane_mesh.triangles = o3d.utility.Vector3iVector(plane_triangles)
plane_mesh.compute_triangle_normals()
mesh.compute_vertex_normals()
plane_mesh.paint_uniform_color([0.3, 0.3, 0.6])

import open3d.visualization as vis
# vis.draw_geometries([plane_mesh, mesh])

# Visualize the plane
mat_plane = vis.rendering.MaterialRecord()
mat_plane.shader = 'defaultLitSSR'
mat_plane.base_color = [0.5, 1., 0.5, 1.0]
mat_plane.base_roughness = 0.
mat_plane.base_reflectance = 0.
mat_plane.base_clearcoat = 1.
mat_plane.thickness = 1.
mat_plane.transmission = 1.
mat_plane.absorption_distance = 10
mat_plane.absorption_color = [0.3, 0.7, 0.3]

geoms = [{'name': 'plane', 'geometry': plane_mesh, 'material': mat_plane}, {'name': 'object', 'geometry': mesh}]

vis.draw(geoms, title='Hi')