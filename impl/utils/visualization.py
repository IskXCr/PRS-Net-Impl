import open3d as o3d
import numpy as np
import torch
import open3d.visualization as vis
from torch.utils.data import Dataset

def get_trans_mat(normal):
    z = normal
    y = z - np.array([0., 0., 0.3])
    y = y - np.dot(y, z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    result = np.eye(4, 4)
    result[0:3, 0] = x
    result[0:3, 1] = y
    result[0:3, 2] = z
    return result

def visualize_planar_symmetry(mesh, plane):
    '''
    `plane` should be a (4, ) numpy array, describing the equation Ax + By + Cz + D = 0.
    '''
    coords_mat = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    if plane[2] == 0.:
        zs = ((np.random.random_sample(4) - 0.5) * 2.0).reshape(-1, 1)
    else:
        zs = ((-coords_mat @ plane[0:2] - plane[3]) / plane[2]).reshape(-1, 1)
        
    mesh.compute_vertex_normals()
    
    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=0.01)
    plane_mesh.paint_uniform_color(np.array([0.1, 0.1, 0.4]))
    plane_mesh.translate(np.array([-.5, -.5, 0.])) 
    normal = plane[0:3] / np.linalg.norm(plane[0:3])
    plane_mesh.transform(get_trans_mat(normal))
    plane_mesh.translate(plane[3] * normal)
    
    print(f'[visualization.py] Drawing symmetry plane {plane}')
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.draw_geometries([plane_mesh, mesh, frame])
    
    
def visualize_quat_symmetry(mesh, quaternion):
    '''
    `quaternion` should be a (4, ) numpy array, describing the quaternion `a + bi + cj + dk`.
    '''
    normal = np.array(quaternion[1:4])
    if quaternion[0] == 1.:
        normal = np.zeros((3,))
    else:
        normal = normal / np.sin(np.arccos(normal[0]))
        normal = normal / np.linalg.norm(normal)
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=.2, cone_radius=.25, cylinder_height=.8, cone_height=.2)
    print(f'[visualization.py] Drawing symmetry axis {normal}')
    vis.draw_geometries([arrow, mesh])