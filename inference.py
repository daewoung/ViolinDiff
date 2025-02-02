import argparse
import torch
import torchaudio
import tensorflow as tf


from hydra.experimental import compose, initialize
from omegaconf import DictConfig
from einops import repeat


from inference_preprocess import long_midi_processor
from model.bends.model import get_model as get_bends_model
from model.synth.model import get_model as get_synth_model
from model.module.mel2wav import get_mel_inverse_converter, mel_to_wav_soundstream

def get_all(synth_cfg : DictConfig,
            bend_cfg : DictConfig):
  
  bend_mel_config = bend_cfg.mel_soundstream
  synth_mel_config = synth_cfg.mel_soundstream
  
  bends_model = get_bends_model(bend_cfg, bend_mel_config)
  synth_model = get_synth_model(synth_cfg, synth_mel_config)
  
  return bends_model, synth_model, bend_mel_config, synth_mel_config


def load_checkpoint(model, ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def main(args, synth_cfg, bend_cfg):
  bends_model, synth_model, bend_mel_config, synth_mel_config = get_all(synth_cfg, bend_cfg)
  
  synth_model = load_checkpoint(
      synth_model,
      args.synth_pth,
      args.device
    )
  bends_model = load_checkpoint(
      bends_model,
      args.bend_pth,
      args.device
  )
  
  performer_idx = torch.tensor([args.performer])
  overlap_seq_len = 32 
  
  
  pitch_data, onset_data, bend_data, velocity, offset, pitch_dict, _ = long_midi_processor(bend_mel_config, args.midi_pth, overlap_seq_len, pred_bend = None, return_gt_bend = False)


  performer = repeat(performer_idx, '1 -> b', b=pitch_data.shape[0])
  
  condition = [pitch_data.to('cuda'), onset_data.to('cuda'), bend_data.to('cuda'), velocity.to('cuda'), performer.to('cuda'), offset.to('cuda')]
  pred_bend, concat_bend = bends_model.long_sampling(condition, overlap_seq_len, cfg_scale=args.bend_cfg, mask= pitch_data.to('cuda'))
  
  pitch_data, onset_data, pred_bend_data, velocity, offset, pitch_dict, _  = long_midi_processor(synth_mel_config, args.midi_pth, overlap_seq_len, pred_bend = concat_bend.detach().cpu(), return_gt_bend = False)
  performer = repeat(performer_idx, '1 -> b', b=pitch_data.shape[0])

  condition = [pitch_data.to('cuda'), onset_data.to('cuda'), pred_bend_data.to('cuda'), velocity.to('cuda'), performer.to('cuda'), offset.to('cuda')]
  pred_mel, concat_pred_mel = synth_model.long_sampling(condition, overlap_seq_len, args.synth_cfg)

  inverter = get_mel_inverse_converter()
  pred_audio = mel_to_wav_soundstream(concat_pred_mel.unsqueeze(0), inverter)
  pred_audio = torch.tensor(pred_audio)

  torchaudio.save(args.save_pth, pred_audio, synth_mel_config.sample_rate)

if __name__ == "__main__":
  paser = argparse.ArgumentParser()
  paser.add_argument('--synth_pth', type=str, default='synth.pt')
  paser.add_argument('--bend_pth', type=str, default='bend.pt') # norm
  paser.add_argument('--bend_cfg', type=float, default=3.0) # norm
  paser.add_argument('--synth_cfg', type=float, default=1.25) # norm

  paser.add_argument('--midi_pth', type=str, default='thais.mid')
  paser.add_argument('--save_pth', type=str, default='thais.wav')
  paser.add_argument('--performer', type=int, default=0)
  paser.add_argument('--device', type=str, default='cuda')
  args = paser.parse_args()

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
      for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

  with initialize(config_path='config/'):
    synth_cfg = compose(config_name='synth')
  
  with initialize(config_path='config/'):
    bend_cfg = compose(config_name='bend')
    
  main(args, synth_cfg, bend_cfg)
