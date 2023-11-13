import argparse
import os

import torch
from impl.options import Options

parser = argparse.ArgumentParser(description='Train the PRSNet.')
parser.add_argument('--checkpoint_src', required=True, help='Load PRSNet from checkpoint file. The file must be created by the previous training using this program.')
parser.add_argument('--data_path', required=True, help='The path for preprocessed datasets. If the data have not been preprocessed, please use "impl/utils/voxel_processing.py" to convert them.')

parser.add_argument('--dtype', dest='dtype', default='float32', help='The target fp type to use. Allowed: float16, float32, float64')

parser.add_argument('--weight_reg', dest='w_r', default=25, type=float, help='The default w_r for summing symmetry distance loss and regularization loss. If not specified, use 25.')

args = parser.parse_args()

test_options = Options()
test_options.checkpoint_src = args.checkpoint_src
test_options.data_path = args.data_path
test_options.w_r = args.w_r
if args.dtype == 'float16':
    test_options.dtype = torch.float16
elif args.dtype == 'float32':
    test_options.dtype = torch.float32
elif args.dtype == 'float64':
    test_options.dtype = torch.float64
else:
    raise ValueError(f'Invalid data type: {args.dtype}')

# Place imports here to speed up the execution a little
import open3d as o3d
import torch
from torch.utils.data import DataLoader
from impl.PRSNet import *
import impl.utils.voxel_processing as vp
from impl.utils.voxel_processing import CustomVoxelDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = PRSNet()
model.to(device=device, dtype=test_options.dtype)

criterion = PRSNet_Loss(test_options.w_r)

checkpoint = torch.load(test_options.checkpoint_src, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
test_options.w_r = checkpoint['w_r']
criterion.w_r = checkpoint['w_r']
model.eval()

print(f'The model has been loaded from: {test_options.checkpoint_src}')

print('========Model Info========')
print(model)
print('=====End of Model Info====')

print(test_options)

print('Loading data...')
raw_data = vp.read_dataset_from_path(test_options.data_path)

test_dataset = vp.CustomVoxelDataset(raw_data)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=vp.collate_data_list)

# Inference
with torch.no_grad():
    for i, (data_indices, omap, grid_points, sample_points) in enumerate(test_loader):
        omap = omap.to(device=device, dtype=test_options.dtype)
        grid_points = grid_points.to(device, dtype=test_options.dtype)
        sample_points = sample_points.to(device, dtype=test_options.dtype)
        
        p_features, q_features = model(omap)
        loss = criterion(p_features, q_features, grid_points, sample_points)
        
        print(f'Data_Index:{data_indices}; Inference loss: {loss.item()}; p_features: {p_features / torch.norm(p_features)}; q_features: {q_features / torch.norm(q_features)}')