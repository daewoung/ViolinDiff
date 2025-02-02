import torch
import numpy as np
import random
import wandb

from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from data_preprocess.dataset import ViolinDataset, ValidViolinDataset
from data_preprocess.bend_dataset import ViolinDataset as BendViolinDataset
from data_preprocess.bend_dataset import ValidViolinDataset as BendValidViolinDataset


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
  
def seeding(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  
def run_wandb(cfg):
  config_dict = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)

  wandb.init(project=cfg.train.rum_entity, config=config_dict)
  wandb.run.name = cfg.train.run_name 
  wandb.run.save()
  
def get_path(save_dir):
  Path(save_dir + '/save_pt').mkdir(parents=True, exist_ok=True)
  Path(save_dir + '/output').mkdir(parents=True, exist_ok=True)
  return save_dir + '/save_pt', save_dir + '/output'

def get_dataset(cfg : DictConfig):
  sample_rate = cfg.mel_soundstream.sample_rate
  hop_length = cfg.mel_soundstream.hop_length
  mel_len = cfg.mel_soundstream.mel_len

  train_dataset = ViolinDataset(
                      cfg.datasets.midi_pth,
                      ['Kayser', 'Paganini', 'Wohlfahrt'],
                      cfg.datasets.segment_size,
                      cfg.datasets.on_memory,
                      sample_rate,
                      hop_length,
                      mel_len,
                      cfg.datasets.audio_norm
  )
  
  valid_dataset = ValidViolinDataset(
                      cfg.datasets.midi_pth,
                      cfg.datasets.valid_pth,
                      ['Kayser', 'Paganini', 'Wohlfahrt'],
                      cfg.datasets.segment_size,
                      cfg.datasets.on_memory,
                      sample_rate,
                      hop_length,
                      mel_len,
                      cfg.datasets.audio_norm
                      )
  

  print(f'train dataset size: {len(train_dataset)}')
  print(f'valid dataset size: {len(valid_dataset)}')
  
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.train_batch_size, shuffle=True, num_workers=0, drop_last=False)
  valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.train.eval_batch_size, shuffle=False, num_workers=0, drop_last=False)
  
  print('final_step will be:', len(train_loader) * cfg.train.num_epoch)
  
  return train_loader, valid_loader

def get_bend_dataset(cfg : DictConfig):
  sample_rate = cfg.mel_soundstream.sample_rate
  hop_length = cfg.mel_soundstream.hop_length
  mel_len = cfg.mel_soundstream.mel_len

  train_dataset = BendViolinDataset(
                      cfg.datasets.midi_pth,
                      ['Kayser', 'Paganini', 'Wohlfahrt'],
                      cfg.datasets.segment_size,
                      cfg.datasets.on_memory,
                      sample_rate,
                      hop_length,
                      mel_len,
  )
 
    
  valid_dataset = BendValidViolinDataset(
                      cfg.datasets.midi_pth,
                      cfg.datasets.valid_pth,
                      ['Kayser', 'Paganini', 'Wohlfahrt'],
                      cfg.datasets.segment_size,
                      cfg.datasets.on_memory,
                      sample_rate,
                      hop_length,
                      mel_len,
                      )
  

  print(f'train dataset size: {len(train_dataset)}')
  print(f'valid dataset size: {len(valid_dataset)}')
  
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.train_batch_size, shuffle=True, num_workers=0, drop_last=False)
  valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.train.eval_batch_size, shuffle=False, num_workers=0, drop_last=False)
  
  print('final_step will be:', len(train_loader) * cfg.train.num_epoch)
  
  return train_loader, valid_loader