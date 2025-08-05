import warnings
warnings.filterwarnings('ignore')

import os

ctrl_folder = r'D:\DL\data_train'
spr_folder = r'D:\DL\data_spread'

suffix = 'grb2'
files = os.listdir(ctrl_folder)
files_000 = [f for f in files if f.endswith('_f000.grb2')]
files_006 = [f for f in files if f.endswith('_f006.grb2')]
file_ctrl_00 = [os.path.join(ctrl_folder, name) for name in files_000]
file_ctrl_06 = [os.path.join(ctrl_folder, name) for name in files_006]
# print(file_ctrl_00)

files = os.listdir(spr_folder)
file_spr = [f for f in files if f.endswith(suffix)]
file_spr = [os.path.join(spr_folder, name) for name in file_spr]
# print(file_spr)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
print(device)

import torch
import numpy as np
from grbdata import grbdata
from tsf2sfm import spharm_transform, spectral_to_grid
from torch.utils.data import Dataset

class WeatherDataset(Dataset):
    def __init__(self, ctrl_filepaths_00, ctrl_filepaths_06, spread_filepaths, layer, X_mean=None, X_std=None, Y_mean=None, Y_std=None):
        assert len(ctrl_filepaths_06) == len(spread_filepaths), "文件数量不一致"
        self.ctrl_filepaths_00 = ctrl_filepaths_00
        self.ctrl_filepaths_06 = ctrl_filepaths_06
        self.spread_filepaths = spread_filepaths
        self.layer = layer
        
        self.X_mean = X_mean
        self.X_std = X_std
        self.Y_mean = Y_mean
        self.Y_std = Y_std

        # 控制变量顺序（不包含lon和lat，因为spharm_transform会删除它们）
        self.ctrl_vars = ['gh','t','r','u','v',
                          'gh_diff','t_diff','r_diff','u_diff','v_diff',
                          'gh_grad','t_grad','r_grad','u_grad','v_grad',
                          'div_ctrl','vor_ctrl']
        self.spread_vars = ['gh', 't', 'r', 'u', 'v']

    def __len__(self):
        return len(self.ctrl_filepaths_00)

    def __getitem__(self, idx):
        ctrl_path_00 = self.ctrl_filepaths_00[idx]
        ctrl_path_06 = self.ctrl_filepaths_06[idx]
        spread_path = self.spread_filepaths[idx]

        ctrl_data, spread_data = grbdata(ctrl_path_00, ctrl_path_06, spread_path, self.layer, zoomin=1, ds=1)
        # # 对数据进行球谐变换
        # ctrl_data, spread_data = spharm_transform(ctrl_data), spharm_transform(spread_data)

        # 安全地构建数组，只使用存在的变量
        try:
            X = np.stack([ctrl_data[k] for k in self.ctrl_vars], axis=0)
            Y = np.stack([spread_data[k] for k in self.spread_vars], axis=0)
        except KeyError as e:
            print(f"缺少变量 {e}")
            print(f"ctrl_data可用变量: {list(ctrl_data.keys())}")
            print(f"spread_data可用变量: {list(spread_data.keys())}")
            raise
        
        # print(X.shape, Y.shape)
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# dataset = WeatherDataset(file_ctrl_00, file_ctrl_06, file_spr, layer=500)
# # 获取第一个 batch
# data_iter = iter(dataset)
# X_batch, Y_batch = next(data_iter)
# print(f"Batch: X.shape = {X_batch.shape}, Y.shape = {Y_batch.shape}")
