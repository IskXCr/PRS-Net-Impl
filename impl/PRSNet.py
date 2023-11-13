import torch
import torch.nn as nn

class PRSNet_Encoder(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        
        leaky_ReLU_slope = 0.2
        
        # 32^3x1
        self.conv_layer0 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.max_pool0 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.leaky_relu0 = nn.LeakyReLU(negative_slope=leaky_ReLU_slope)
        
        # 16^3x4
        self.conv_layer1 = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.max_pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=leaky_ReLU_slope)
        
        # 8^3x8
        self.conv_layer2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.max_pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=leaky_ReLU_slope)
        
        # 4^3x16
        self.conv_layer3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.max_pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=leaky_ReLU_slope)
        
        # 2^3x32
        self.conv_layer4 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.max_pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=leaky_ReLU_slope)
        # 1^3x64
    
    def forward(self, voxels):
        out = self.conv_layer0(voxels)
        out = self.max_pool0(out)
        out = self.leaky_relu0(out)
        
        out = self.conv_layer1(out)
        out = self.max_pool1(out)
        out = self.leaky_relu1(out)
        
        out = self.conv_layer2(out)
        out = self.max_pool2(out)
        out = self.leaky_relu2(out)
        
        out = self.conv_layer3(out)
        out = self.max_pool3(out)
        out = self.leaky_relu3(out)
        
        out = self.conv_layer4(out)
        out = self.max_pool4(out)
        out = self.leaky_relu4(out)
        
        return out


class PRSNet_Plane_Predictor(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        
        leaky_ReLU_slope = 0.2
        
        # implicit symmetry planes: 4 features, aX + bY + cZ + d = 0
        self.fc0 = nn.Linear(64, 32)
        self.relu0 = nn.LeakyReLU(negative_slope=leaky_ReLU_slope)
        self.fc1 = nn.Linear(32, 16)
        self.relu1 = nn.LeakyReLU(negative_slope=leaky_ReLU_slope)
        self.fc2 = nn.Linear(16, 4)
    
    def set_initial_bias(self, feature):
        self.fc2.bias.data = feature.clone().detach()
        
    def set_initial_weight(self, value=0.000):
        self.fc2.bias.data.fill_(value)
        
    def forward(self, features):
        out = self.fc0(features)
        out = self.relu0(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        return out
    

class PRSNet_Quaternion_Predictor(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        
        leaky_ReLU_slope = 0.2
        
        # quaterion rotation: 4 features, a + bi + cj + dk
        self.fc0 = nn.Linear(64, 32)
        self.relu0 = nn.LeakyReLU(negative_slope=leaky_ReLU_slope)
        self.fc1 = nn.Linear(32, 16)
        self.relu1 = nn.LeakyReLU(negative_slope=leaky_ReLU_slope)
        self.fc2 = nn.Linear(16, 4)
    
    def set_initial_bias(self, feature):
        self.fc2.bias.data = feature.clone().detach()
        
    def forward(self, features):
        out = self.fc0(features)
        out = self.relu0(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        return out


class PRSNet(nn.Module):
    def __init__(self, ) -> None:
        super(PRSNet, self).__init__()
        
        self.encoder = PRSNet_Encoder()
        
        self.plane_predictor0 = PRSNet_Plane_Predictor()
        self.plane_predictor1 = PRSNet_Plane_Predictor()
        self.plane_predictor2 = PRSNet_Plane_Predictor()
        
        self.plane_predictor0.set_initial_bias(torch.tensor([1., 0., 0., 0.]))
        self.plane_predictor1.set_initial_bias(torch.tensor([0., 1., 0., 0.]))
        self.plane_predictor2.set_initial_bias(torch.tensor([0., 0., 1., 0.]))
        # self.plane_predictor0.set_initial_weight()
        # self.plane_predictor1.set_initial_weight()
        # self.plane_predictor2.set_initial_weight()
        
        self.quaternion_predictor0 = PRSNet_Quaternion_Predictor()
        self.quaternion_predictor1 = PRSNet_Quaternion_Predictor()
        self.quaternion_predictor2 = PRSNet_Quaternion_Predictor()
        
        cos_theta = torch.cos(torch.tensor(torch.pi / 2))
        sin_theta = torch.sin(torch.tensor(torch.pi / 2))
        
        self.quaternion_predictor0.set_initial_bias(torch.tensor([cos_theta, sin_theta, 0., 0.]))
        self.quaternion_predictor1.set_initial_bias(torch.tensor([cos_theta, 0., sin_theta, 0.]))
        self.quaternion_predictor2.set_initial_bias(torch.tensor([cos_theta, 0., 0., sin_theta]))
        # self.quaternion_predictor0.set_initial_weight(torch.tensor([0., 0., 0., 0.]))
        # self.quaternion_predictor1.set_initial_weight(torch.tensor([0., 0., 0., 0.]))
        # self.quaternion_predictor2.set_initial_weight(torch.tensor([0., 0., 0., 0.]))
    
    def normalize(self, batch_single_feature):
        ut = batch_single_feature.transpose(0, 1) / torch.norm(batch_single_feature, dim=1)
        return ut.transpose(0, 1)
        
    def forward(self, batch_voxels):
        if batch_voxels.dim == 4:
            batch_voxels = batch_voxels.unsqueeze(0)
        out0 = self.encoder(batch_voxels)
        
        out0 = out0.reshape(-1, 64)
        M = out0.shape[0]
        
        plane0 = self.plane_predictor0(out0)
        plane1 = self.plane_predictor1(out0)
        plane2 = self.plane_predictor2(out0)
        
        plane0 = self.normalize(plane0.reshape(M, 4))
        plane1 = self.normalize(plane0.reshape(M, 4))
        plane2 = self.normalize(plane0.reshape(M, 4))
        
        quat0 = self.quaternion_predictor0(out0)
        quat1 = self.quaternion_predictor1(out0)
        quat2 = self.quaternion_predictor2(out0)
        
        quat0 = self.normalize(quat0.reshape(M, 4))
        quat1 = self.normalize(quat1.reshape(M, 4))
        quat2 = self.normalize(quat2.reshape(M, 4))
        
        return torch.stack([plane0, plane1, plane2], dim=1), torch.stack([quat0, quat1, quat2], dim=1)
    
    
class PRSNet_Symm_Dist_Loss(nn.Module):
    '''
    PRSNet Symmetry Distance Loss
    '''
    def __init__(self, ) -> None:
        super().__init__()

    def compute_batch_std_grid_indices(self, batch_query_points: torch.Tensor):
        '''
        `batch_query_points` should be of shape `(M, N, 3)`.
        
        Return a tensor of shape `(M, 3, N)`, where
        - `M` is the number of samples inside the batch,
        - `N` is the number of queries inside a single sample.
        
        Should return a tensor of shape `(4, M * N)
        '''
        M = batch_query_points.shape[0]
        N = batch_query_points.shape[1]
        
        device = batch_query_points.device
        dtype = batch_query_points.dtype
        
        tmp0 = batch_query_points.transpose(1, 2).contiguous() * 16.0 + 16.0
        
        tmp0 = tmp0.floor().clamp(0, 31).to(torch.int)
        
        tmp = tmp0.permute(1, 0, 2).reshape(3, -1)
        
        midx = torch.arange(0, M, device=device, dtype=int).reshape(M, 1).repeat(1, N).reshape(1, -1)
        
        return torch.cat([midx, tmp], dim=0)

    def compute_batch_dist_sum(self, batch_grid_points, batch_query_points):
        '''
        Compute, for each sample, the sum of 'shortest distances' between points 
        and the closest point on the mesh's surface in their corresponding grids.
        
        Return summed_distance of shape (M, 1), where
        - `M` is batch size.
        
        '''
        M = batch_query_points.shape[0]
        N = batch_query_points.shape[1]
        m, x, y, z = self.compute_batch_std_grid_indices(batch_query_points)
        
        g = batch_grid_points[m, x, y, z].reshape(M, N, 3)
        q = batch_query_points
        batch_displacements = g - q
        vector_norm = torch.linalg.vector_norm(batch_displacements, dim=2)
        result = torch.sum(vector_norm, dim=1)
        
        return result

    def apply_planar_transform(self, batch_plane, batch_sample_points):
        '''
        Compute, for each sample, sample points after planar reflective transformation.
        
        The formula is given by `q' = q - 2 <q - r, n> n`, where:
        - q is the target point, q' is the point after symmetric transformation,
        - r is the orthogonal displacement vector of the plane, and
        - n is the normalized normal vector of the plane.
        
        Return a tensor of shape `(M, N, D, 3)`
        '''
        M = batch_sample_points.shape[0]
        N = batch_sample_points.shape[1]
        D = batch_plane.shape[1]
        
        q = batch_sample_points
        
        n_norm = torch.norm(batch_plane[:, :, 0:3], dim=2).reshape(M, D, 1)
        n = (batch_plane[:, :, 0:3] / n_norm).transpose(1, 2)
        d = batch_plane[:, :, 3].reshape(M, 1, D).repeat([1, N, 1])
        coeff = (torch.einsum('bij,bjk->bik', q, n) + d) * 2
        
        coeff1 = coeff.reshape(M, N, D, 1).repeat([1, 1, 1, 3])
        n0 = n.transpose(1, 2).reshape(M, 1, D, 3).repeat([1, N, 1, 1])
        tmp = coeff1 * n0
        
        q0 = q.reshape(M, N, 1, 3).repeat([1, 1, D, 1])
        result = q0 - tmp
        
        return result
    
    def batch_quat_normalize(self, batch_quaternions):
        M = batch_quaternions.shape[0]
        D = batch_quaternions.shape[1]
        
        q_norm = torch.norm(batch_quaternions, dim=2).reshape(M, D, 1)
        q = batch_quaternions / q_norm
        return q
    
    def batch_multiply_quaternion(self, r, s):
        M = r.shape[0]
        N = r.shape[1]
        D = r.shape[2]
        
        result_r = torch.zeros((M, N, D), requires_grad=True)
        result_i = torch.zeros((M, N, D), requires_grad=True)
        result_j = torch.zeros((M, N, D), requires_grad=True)
        result_k = torch.zeros((M, N, D), requires_grad=True)
        result_r = r[:, :, :, 0] * s[:, :, :, 0] - r[:, :, :, 1] * s[:, :, :, 1] \
                    - r[:, :, :, 2] * s[:, :, :, 2] - r[:, :, :, 3] * s[:, :, :, 3]
        result_i = r[:, :, :, 1] * s[:, :, :, 0] + r[:, :, :, 0] * s[:, :, :, 1] \
                    + r[:, :, :, 2] * s[:, :, :, 3] - r[:, :, :, 3] * s[:, :, :, 2]
        result_j = r[:, :, :, 2] * s[:, :, :, 0] + r[:, :, :, 0] * s[:, :, :, 2] \
                    + r[:, :, :, 3] * s[:, :, :, 1] - r[:, :, :, 1] * s[:, :, :, 3]
        result_k = r[:, :, :, 3] * s[:, :, :, 0] + r[:, :, :, 0] * s[:, :, :, 3] \
                    + r[:, :, :, 1] * s[:, :, :, 2] - r[:, :, :, 2] * s[:, :, :, 1]
                    
        result_r = result_r.reshape(M, N, D, 1)
        result_i = result_i.reshape(M, N, D, 1)
        result_j = result_j.reshape(M, N, D, 1)
        result_k = result_k.reshape(M, N, D, 1)
        return torch.cat([result_r, result_i, result_j, result_k], dim=3)
    
    def apply_quaternion_rotation(self, batch_quaternions, batch_sample_points):
        '''
        Compute, for each sample, sample points after rotation using quaternions.
        '''
        M = batch_sample_points.shape[0]
        N = batch_sample_points.shape[1]
        D = batch_quaternions.shape[1]
        
        device = batch_quaternions.device
        
        p = torch.cat([torch.zeros((M, N, 1), device=device), batch_sample_points], dim=2).reshape(M, N, 1, 4).repeat([1, 1, D, 1])
        # normalized the quaternion first
        q = self.batch_quat_normalize(batch_quaternions)
        # prepare for quaternion multiplication
        q0 = q.reshape(M, 1, D, 4).repeat([1, N, 1, 1])
        q0p_im = -q0[:, :, :, 1:4].clone()
        q0p_re = q0[:, :, :, 0].clone().reshape(M, N, D, 1)
        q0p = torch.cat([q0p_re, q0p_im], dim=3)
        
        tmp0 = self.batch_multiply_quaternion(p, q0)
        tmp0 = self.batch_multiply_quaternion(q0p, tmp0)
        
        return tmp0[:, :, :, 0:3].clone()
    
    def forward(self, batch_planar_features, batch_quat_features, batch_grid_points, batch_sample_points):
        M = batch_sample_points.shape[0]
        N = batch_sample_points.shape[1] # N samples
        D0 = batch_planar_features.shape[1] # number of planes predicted
        D1 = batch_quat_features.shape[1] # number of quaternions predicted
        
        assert batch_grid_points.shape == (M, 32, 32, 32, 3)
        assert batch_sample_points.shape == (M, N, 3)
        
        p_trans_points = self.apply_planar_transform(batch_planar_features, batch_sample_points)
        p_losses = self.compute_batch_dist_sum(batch_grid_points, p_trans_points.reshape(M, -1, 3))
        planar_loss = torch.sum(p_losses)
        
        q_trans_points = self.apply_quaternion_rotation(batch_quat_features, batch_sample_points)
        q_losses = self.compute_batch_dist_sum(batch_grid_points, q_trans_points.reshape(M, -1, 3))
        quat_loss = torch.sum(q_losses)
        
        return (planar_loss + quat_loss).mean()
    

class PRSNet_Reg_Loss(nn.Module):
    '''
    PRS Regularization Loss
    '''
    def __init__(self, ) -> None:
        super().__init__()
    
    def forward(self, batch_planar_features, batch_quat_features):
        '''
        `batch_planar_features` should have a shape of `(M, D, 4)`
        '''
        # TODO: Compute Regularization Loss
        M = batch_planar_features.shape[0]
        D1 = batch_planar_features.shape[1]
        D2 = batch_quat_features.shape[1]
        device = batch_planar_features.device
        
        # m1_norm = torch.norm(batch_planar_features[:, :, 0:3], dim=2).reshape(M, D1, 1)
        # m1 = (batch_planar_features[:, :, 0:3] / m1_norm)
        m1 = batch_planar_features[:, :, 0:3]
        
        m1_m1t = torch.einsum('bij, bjk->bik', m1, m1.transpose(1, 2).contiguous())
        m1_id_mat = torch.eye(D1, device=device).reshape(1, D1, D1).repeat([M, 1, 1])
        A = m1_m1t - m1_id_mat
        
        m2_norm = torch.norm(batch_planar_features[:, :, 1:4], dim=2).reshape(M, D1, 1)
        m2 = (batch_planar_features[:, :, 1:4] / m2_norm)
        
        m2_m2t = torch.einsum('bij, bjk->bik', m2, m2.transpose(1, 2).contiguous())
        m2_id_mat = torch.eye(D2, device=device).reshape(1, D1, D1).repeat([M, 1, 1])
        B = m2_m2t - m2_id_mat
        
        loss = torch.norm(A, dim=(1, 2)) + torch.norm(B, dim=(1, 2))
        
        return loss.mean()


class PRSNet_Loss(nn.Module):
    def __init__(self, w_r=25) -> None:
        super().__init__()
        self.symmetry_loss = PRSNet_Symm_Dist_Loss()
        self.reg_loss = PRSNet_Reg_Loss()
        self.w_param_penalty = 1
        self.w_r = w_r
    
    def forward(self, batch_planar_features, batch_quat_features, batch_grid_points, batch_sample_points):
        '''
        `batch_grid_points`: of size `Mx(32**3)x3`
        `batch_sample_points`: of size `MxNx3`
        '''
        if batch_planar_features.dim == 3:
            batch_planar_features = batch_planar_features.unsqueeze(0)
        if batch_quat_features.dim == 3:
            batch_quat_features = batch_quat_features.unsqueeze(0)
        if batch_grid_points.dim == 4:
            batch_grid_points = batch_grid_points.unsqueeze(0)
        if batch_sample_points.dim == 2:
            batch_sample_points = batch_sample_points.unsqueeze(0)
        M = batch_planar_features.shape[0]
            
        symm_loss = self.symmetry_loss(batch_planar_features, batch_quat_features, batch_grid_points, batch_sample_points)
        reg_loss = self.reg_loss(batch_planar_features, batch_quat_features)
        
            #    + self.w_param_penalty * torch.mean(torch.norm(batch_planar_features, dim=2)) \
            #    + self.w_param_penalty * torch.mean(torch.norm(batch_planar_features, dim=2)) \
        return symm_loss \
               + self.w_r * reg_loss