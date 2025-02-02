import torch
import hydra

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

from trainer.trainer_utils import count_parameters, get_bend_dataset, get_path, run_wandb, seeding
from trainer.bend_trainer import Trainer

from model.bends.model import get_model

@hydra.main(config_path='config/', config_name='bend')
def main(cfg : DictConfig):
  import sys
  print(sys.executable)
  print(sys.version)
  from pathlib import Path
  print("Notebook working dir =", Path.cwd())

  seed = cfg.seed
  seeding(seed)

  save_dir = str(HydraConfig.get().run.dir)
  
  save_pt_dir, output_dir = get_path(save_dir)
  
  source_pth = Path(save_dir + '/src')
  source_pth.mkdir(parents=True, exist_ok=True)
  
  if cfg.train.wandb_log:
    run_wandb(cfg)
    
  if cfg.vocoder =='soundstream':
    mel_config = cfg.mel_soundstream

  model = get_model(cfg, mel_config) 
  total_params, trainable_params = count_parameters(model)
  print(f"Total Parameters: {total_params}")
  print(f"Trainable Parameters: {trainable_params}")
  
  train_loader, valid_loader = get_bend_dataset(cfg)
  
  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

  
  trainer = Trainer(model, 
                  optimizer, 
                  None, 
                  train_loader, 
                  valid_loader,
                  cfg.train.num_epoch, 
                  cfg.train.save_epoch, 
                  cfg.train.infer_epoch,
                  'cuda', 
                  cfg.train.wandb_log, 
                  save_pt_dir, 
                  output_dir, 
                  cfg.train.fp_16)
        
  trainer.train()

if __name__ == "__main__":
  main()
  
