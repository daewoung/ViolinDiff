import torch
import hydra
import tensorflow as tf

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

from trainer.trainer_utils import count_parameters, get_dataset, get_path, run_wandb, seeding
from trainer.synth_trainer import Trainer

from model.synth.model import get_model
from model.module.wav2mel import MelConverter

@hydra.main(config_path='config/', config_name='synth')
def main(cfg : DictConfig):
  seed = cfg.seed
  seeding(seed)
  gpus = tf.config.experimental.list_physical_devices('GPU')

  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
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
  
  train_loader, valid_loader = get_dataset(cfg)
  
  mel = MelConverter(cfg.mel_soundstream.sample_rate,
                    cfg.mel_soundstream.win_length,
                    cfg.mel_soundstream.hop_length,
                    cfg.mel_soundstream.n_fft,
                    cfg.mel_soundstream.f_min,
                    cfg.mel_soundstream.f_max,
                    cfg.mel_soundstream.n_mels,
                    cfg.mel_soundstream.clip_value_min,
                    cfg.mel_soundstream.clip_value_max
  ) 

  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
  
  if cfg.diff.use_enc_mel_train:
    print('--'*20)
    print('Use with encoder training')
    print('--'*20)
    
    trainer = Trainer(model, 
                    mel,
                    cfg.vocoder,
                    optimizer, 
                    None, 
                    train_loader, 
                    valid_loader,
                    seed, 
                    cfg.train.num_epoch, 
                    cfg.train.save_epoch, 
                    cfg.train.infer_epoch,
                    'cuda', 
                    cfg.train.wandb_log, 
                    save_pt_dir, 
                    output_dir, 
                    cfg.train.fp_16, 
                    1.25)
    
  trainer.train()

if __name__ == "__main__":
  main()
  
