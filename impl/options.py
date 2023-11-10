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
    
    def __str__(self) -> str:
        return f"Options:\n" + \
               f"  checkpoint_src = {self.checkpoint_src}\n" + \
               f"  checkpoint_dst = {self.checkpoint_dst}\n" + \
               f"  data_path = {self.data_path}\n" + \
               f"  dtype = {self.dtype}\n" + \
               f"  sample_num = {self.sample_num}\n" + \
               f"  batch_size = {self.batch_size}\n" + \
               f"  num_epochs = {self.num_epochs}\n" + \
               f"  learning_rate = {self.learning_rate}\n" + \
               f"  weight_decay = {self.weight_decay}\n" + \
               f"  weight_regular_loss = {self.w_r}\n"