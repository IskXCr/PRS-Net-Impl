import argparse
import os

import torch
from impl.options import Options

parser = argparse.ArgumentParser(description='Train the PRSNet.')
parser.add_argument('--checkpoint_src', default=None, help='Load PRSNet from checkpoint file. The file must be created by the previous training using this program.')
parser.add_argument('--checkpoint_dst', default=os.path.join('checkpoints', 'prsnet.pt'), help='The path to save checkpoint. If not specified, defaults to "checkpoints/prsnet.pt"')
parser.add_argument('--data_path', required=True, help='The path for preprocessed datasets. If the data have not been preprocessed, please use "impl/utils/voxel_processing.py" to convert them.')

parser.add_argument('--batch_size', default=25, type=int, help='Batch size.')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs that will be used.')
parser.add_argument('--dtype', dest='dtype', default='float32', help='The target fp type to use. Allowed: float16, float32, float64')

parser.add_argument('--weight_reg', dest='w_r', default=25, type=float, help='The default w_r for summing symmetry distance loss and regularization loss. If not specified, use 25.')

args = parser.parse_args()

train_options = Options()
train_options.checkpoint_src = args.checkpoint_src
train_options.checkpoint_dst = args.checkpoint_dst
train_options.data_path = args.data_path
train_options.batch_size = args.batch_size
train_options.num_epochs = args.num_epochs
train_options.w_r = args.w_r
if args.dtype == 'float16':
    train_options.dtype = torch.float16
elif args.dtype == 'float32':
    train_options.dtype = torch.float32
elif args.dtype == 'float64':
    train_options.dtype = torch.float64
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

raw_data = vp.read_dataset_from_path(train_options.data_path)

train_dataset = vp.CustomVoxelDataset(raw_data, train_options.sample_num)
train_loader = DataLoader(train_dataset, batch_size=train_options.batch_size, shuffle=True, collate_fn=vp.collate_data_list)

model = PRSNet()
model.to(device=device, dtype=train_options.dtype)

criterion = PRSNet_Loss(train_options.w_r)
optimizer = torch.optim.Adam(model.parameters(), lr=train_options.learning_rate, weight_decay=train_options.weight_decay)
start_epoch = 0
if train_options.checkpoint_src:
    checkpoint = torch.load(train_options.checkpoint_src)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    train_options.w_r = checkpoint['w_r']
    criterion.w_r = checkpoint['w_r']
    
    print(f'The model has been loaded from: {train_options.checkpoint_src}')
else:
    print(f"A new model has been created.")

print('========Model Info========')
print(model)
print('=====End of Model Info====')

# Test if the destination path exists.
if os.path.exists(train_options.checkpoint_dst):
    print(f'The destination path already has a checkpoint file: {train_options.checkpoint_dst}')
    ans = input(f'Continuing this operation would overwrite this checkpoint. Are you sure [Y/n]? ')
    if ans == 'Y' or ans == 'y':
        pass
    elif ans == 'N' or ans == 'n':
        print(f'Operation has been cancelled by the user.')
        exit(1)
    else:
        print(f'Unrecognized option: {ans}. Exiting.')
        exit(1)

# Training
for epoch in range(start_epoch, train_options.num_epochs):
    for i, (data_indices, mesh, omap, grid_points, offset_vector, sample_points) in enumerate(train_loader):
        omap = omap.to(device=device, dtype=train_options.dtype)
        grid_points = grid_points.to(device, dtype=train_options.dtype)
        offset_vector = offset_vector.to(device, dtype=train_options.dtype)
        sample_points = sample_points.to(device, dtype=train_options.dtype)
        
        optimizer.zero_grad()
        p_features, q_features = model(omap)
        loss = criterion(p_features, q_features, grid_points, sample_points)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch + 1}/{train_options.num_epochs}, Loss: {"{:.4f}".format(loss.item())}]')
    torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'w_r': criterion.w_r
                }, train_options.checkpoint_dst)

if os.path.exists(train_options.checkpoint_dst):
    print(f'The model has been saved to: {train_options.checkpoint_dst}')