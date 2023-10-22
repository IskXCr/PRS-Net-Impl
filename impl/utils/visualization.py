import open3d as o3d
import numpy as np
import torch
import open3d.visualization as vis
from torch.utils.data import Dataset

def sliceplane(mesh, axis, value, direction):
    # axis can be 0,1,2 (which corresponds to x,y,z)
    # value where the plane is on that axis
    # direction can be True or False (True means remove everything that is
    # greater, False means less
    # than)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    new_vertices = list(vertices)
    new_triangles = []

    # (a, b) -> c
    # c refers to index of new vertex that sits at the intersection between a,b
    # and the boundingbox edge
    # a is always inside and b is always outside
    intersection_edges = dict()

    # find axes to compute
    axes_compute = [0,1,2]
    # remove axis that the plane is on
    axes_compute.remove(axis)

    def compute_intersection(vertex_in_index, vertex_out_index):
        vertex_in = vertices[vertex_in_index]
        vertex_out = vertices[vertex_out_index]
        if (vertex_in_index, vertex_out_index) in intersection_edges:
            intersection_index = intersection_edges[(vertex_in_index, vertex_out_index)]
            intersection = new_vertices[intersection_index]
        else:
            intersection = [None, None, None]
            intersection[axis] = value
            const_1 = (value - vertex_in[axis])/(vertex_out[axis] - vertex_in[axis])
            c = axes_compute[0]
            intersection[c] = (const_1 * (vertex_out[c] - vertex_in[c])) + vertex_in[c]
            c = axes_compute[1]
            intersection[c] = (const_1 * (vertex_out[c] - vertex_in[c])) + vertex_in[c]
            assert not (None in intersection)
            # save new vertice and remember that this intersection already added an edge
            new_vertices.append(intersection)
            intersection_index = len(new_vertices) - 1
            intersection_edges[(vertex_in_index, vertex_out_index)] = intersection_index

        return intersection_index

    for t in triangles:
        v1, v2, v3 = t
        if direction:
            v1_out = vertices[v1][axis] > value
            v2_out = vertices[v2][axis] > value
            v3_out = vertices[v3][axis] > value
        else: 
            v1_out = vertices[v1][axis] < value
            v2_out = vertices[v2][axis] < value
            v3_out = vertices[v3][axis] < value

        bool_sum = sum([v1_out, v2_out, v3_out])
        # print(f"{v1_out=}, {v2_out=}, {v3_out=}, {bool_sum=}")

        if bool_sum == 0:
            # triangle completely inside --> add and continue
            new_triangles.append(t)
        elif bool_sum == 3:
            # triangle completely outside --> skip
            continue
        elif bool_sum == 2:
            # two vertices outside 
            # add triangle using both intersections
            vertex_in_index = v1 if (not v1_out) else (v2 if (not v2_out) else v3)
            vertex_out_1_index = v1 if v1_out else (v2 if v2_out else v3)
            vertex_out_2_index = v3 if v3_out else (v2 if v2_out else v1)
            # print(f"{vertex_in_index=}, {vertex_out_1_index=}, {vertex_out_2_index=}")
            # small sanity check if indices sum matches
            assert sum([vertex_in_index, vertex_out_1_index, vertex_out_2_index]) == sum([v1,v2,v3])

            # add new triangle 
            new_triangles.append([vertex_in_index, compute_intersection(vertex_in_index, vertex_out_1_index), 
                compute_intersection(vertex_in_index, vertex_out_2_index)])

        elif bool_sum == 1:
            # one vertice outside
            # add three triangles
            vertex_out_index = v1 if v1_out else (v2 if v2_out else v3)
            vertex_in_1_index = v1 if (not v1_out) else (v2 if (not v2_out) else v3)
            vertex_in_2_index = v3 if (not v3_out) else (v2 if (not v2_out) else v1)
            # print(f"{vertex_out_index=}, {vertex_in_1_index=}, {vertex_in_2_index=}")
            # small sanity check if outdices sum matches
            assert sum([vertex_out_index, vertex_in_1_index, vertex_in_2_index]) == sum([v1,v2,v3])

            new_triangles.append([vertex_in_1_index, compute_intersection(vertex_in_1_index, vertex_out_index), vertex_in_2_index])
            new_triangles.append([compute_intersection(vertex_in_1_index, vertex_out_index), 
                compute_intersection(vertex_in_2_index, vertex_out_index), vertex_in_2_index])

        else:
            assert False

    # TODO remap indices and remove unused 

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))
    return mesh


def clean_crop_xy(mesh, min_corner, max_corner):
    min_x = min(min_corner[0], max_corner[0])
    min_y = min(min_corner[1], max_corner[1])
    max_x = max(min_corner[0], max_corner[0])
    max_y = max(min_corner[1], max_corner[1])

    # mesh = sliceplane(mesh, 0, min_x, False)
    mesh_sliced = sliceplane(mesh, 0, max_x, True)
    mesh_sliced = sliceplane(mesh_sliced, 0, min_x, False)
    mesh_sliced = sliceplane(mesh_sliced, 1, max_y, True)
    mesh_sliced = sliceplane(mesh_sliced, 1, min_y, False)
    # mesh_sliced = mesh_sliced.paint_uniform_color([0,0,1])

    return mesh_sliced


def get_standard_AABB():
    points0 = np.array([[-1., -1., -1.], [1., 1., 1.]])
    points = o3d.utility.Vector3dVector(points0)
    return o3d.geometry.AxisAlignedBoundingBox.create_from_points(points)


def visualize_planar_symmetry(mesh, plane):
    '''
    `plane` should be a (4, ) numpy array, describing the equation Ax + By + Cz + D = 0.
    '''
    coords_mat = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    if plane[2] == 0.:
        zs = ((np.random.random_sample(4) - 0.5) * 2.0).reshape(-1, 1)
    else:
        zs = ((-coords_mat @ plane[0:2] - plane[3]) / plane[2]).reshape(-1, 1)
    
    plane_vertices = np.concatenate([coords_mat, zs], axis=1)
    plane_triangles = np.array([[0, 1, 2], [2, 1, 3], [0, 2, 1], [1, 2, 3]])
    
    plane_mesh0 = o3d.geometry.TriangleMesh()
    plane_mesh0.vertices = o3d.utility.Vector3dVector(plane_vertices)
    plane_mesh0.triangles = o3d.utility.Vector3iVector(plane_triangles)
    plane_mesh0.compute_triangle_normals()
    plane_mesh0.paint_uniform_color([0.3, 0.3, 0.6])
    # plane_mesh = clean_crop_xy(plane_mesh0, (-1, -1), (1, 1))
    
    mesh.compute_vertex_normals()
    print(f'standard_AABB={get_standard_AABB()}')
    print(f'[visualization.py] Drawing symmetry plane {plane}')
    vis.draw_geometries([plane_mesh0, mesh])
    
    
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
    line_set = o3d.geometry.LineSet()
    print(f'[visualization.py] Drawing symmetry axis {normal}')
    vis.draw_geometries([line_set, mesh])