# %% [code]
# %% [code]
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio.transforms as T
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchaudio
import torch
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = './musclassifier.pth'
melspec = T.MelSpectrogram(n_mels = 32)
melspec = melspec
batch_size = 256
num_epochs = 8
last_epoch = 0 
features = 64
classes = ['bass','brass','flute','guitar','keyboard','mallet','organ','reed','string','synth_lead','vocal']
criterion = nn.CrossEntropyLoss()


class NSynthDataset(Dataset):
    def __init__(self, inputs_dir, labels_dir, transform = None):
        self.inputs_dir = inputs_dir
        self.labels_dir = labels_dir
        self.transform = transform

        with open(self.labels_dir, 'r') as f:
            self.labels = json.load(f)

        self.labels_list = list(self.labels.items())
       
    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        filename, data = self.labels_list[idx]
        
        path = os.path.join(self.inputs_dir,filename +'.wav')
        
        waveform, sample_rate = torchaudio.load(path)
        waveform = F.pad(waveform, (0, max(0, 64000 - waveform.shape[-1])))[:, :64000]
        new_spec = melspec(waveform.mean(dim=0))
        new_spec = torch.log(new_spec + 1e-10)
        new_spec = new_spec.unsqueeze(0)

        instrument = data["instrument_family"]

        return new_spec, instrument

dataloaders = {
    'train': DataLoader(NSynthDataset('/kaggle/input/nsynth-train/nsynth-train.jsonwav/nsynth-train/audio', 
                                      '/kaggle/input/nsynth-train/nsynth-train.jsonwav/nsynth-train/examples.json'), 
                        batch_size=batch_size, shuffle=True,num_workers=4,prefetch_factor=2,pin_memory=True),

    'valid': DataLoader(NSynthDataset('/kaggle/input/nsynth-train/nsynth-valid.jsonwav/nsynth-valid/audio', 
                                      '/kaggle/input/nsynth-train/nsynth-valid.jsonwav/nsynth-valid/examples.json'), 
                        batch_size=batch_size, shuffle=True,num_workers=4,prefetch_factor=2,pin_memory=True),

    'test': DataLoader(NSynthDataset('/kaggle/input/nsynth-train/nsynth-test.jsonwav/nsynth-test/audio', 
                                     '/kaggle/input/nsynth-train/nsynth-test.jsonwav/nsynth-test/examples.json'), 
                       batch_size=batch_size, shuffle=True,num_workers=4,prefetch_factor=2,pin_memory=True)
}

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads = 4, qkv_bias = False, proj_bias = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q,k,v, mask)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, features, kernel_size = 3, stride = 1, padding=1)
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.attn1 = SelfAttention(128)
        self.attn2 = SelfAttention(128)
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, len(classes))
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        x = self.attn1(x)
        x = self.attn2(x)
        
        x = x.mean(dim=1)  # shape: (B, C)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN()
optimizer = optim.AdamW(model.parameters(),lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,num_epochs, eta_min=0, last_epoch = -1)
