import torch
import os

class Options:
    def __init__(self) -> None:
        self.checkpoint_src = None
        self.checkpoint_dst = os.path.join('checkpoints', 'prsnet.pt')
        self.data_path = 'data/voxel_data'
        self.dtype = torch.float32
        self.sample_num = 1000
        
        self.batch_size = 32
        self.num_epochs = 100
        
        self.learning_rate = 0.01
        self.weight_decay = 0.005
        self.w_r = 25