import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf


def get_mel_inverse_converter():
  module = hub.KerasLayer('https://www.kaggle.com/models/google/soundstream/frameworks/TensorFlow2/variations/mel-decoder-music/versions/1')
  return module

def mel_to_wav_soundstream(mel, module):
  
  if len(mel.shape) == 2:
    mel = mel.unsqueeze(1)
  mel = mel.transpose(-2, -1) # (batch, channel, time -> b, t, c)
  
  mel = mel.cpu().detach().numpy()
  mel = tf.convert_to_tensor(mel)  
  
  return module(mel).numpy()

def save_audio(save_path, audio, sr= 22050):
  if type(audio) == np.ndarray:
    audio = torch.tensor(audio)
    
  if len(audio.shape) == 1:
    audio = audio.unsqueeze(0)
  torchaudio.save(save_path, audio, sample_rate=sr)



def save_image_diff(org_mel, pred_mel, cfg_pred_mel, enc_mel, condition_roll, save_path):
  fig, axs = plt.subplots(5, 1, figsize=(20, 10))
  axs[0].set_title('org_mel')
  axs[0].imshow(org_mel.cpu(), origin='lower', aspect='auto')
  axs[1].set_title('pred_mel')
  axs[1].imshow(pred_mel.cpu(), origin='lower', aspect='auto')
  axs[2].set_title('cfg_pred_mel')
  axs[2].imshow(cfg_pred_mel.cpu(), origin='lower', aspect='auto')
  axs[3].set_title('enc_mel')
  axs[3].imshow(enc_mel.cpu(), origin='lower', aspect='auto')
  axs[4].set_title('condition_roll')
  axs[4].imshow(condition_roll.cpu(), origin='lower', aspect='auto')
  plt.savefig(save_path)
  plt.close()


def save_image_diff_no_enc_mel(org_mel, pred_mel, cfg_pred_mel, condition_roll, save_path):
  fig, axs = plt.subplots(4, 1, figsize=(20, 10))
  axs[0].set_title('org_mel')
  axs[0].imshow(org_mel.cpu(), origin='lower', aspect='auto')
  axs[1].set_title('pred_mel')
  axs[1].imshow(pred_mel.cpu(), origin='lower', aspect='auto')
  axs[2].set_title('cfg_pred_mel')
  axs[2].imshow(cfg_pred_mel.cpu(), origin='lower', aspect='auto')
  axs[3].set_title('condition_roll')
  axs[3].imshow(condition_roll.cpu(), origin='lower', aspect='auto')
  plt.savefig(save_path)
  plt.close()
  
  
def save_image_enc(org_mel, pred_mel, condition_roll, save_path):
  fig, axs = plt.subplots(3, 1, figsize=(20, 10))
  axs[0].set_title('org_mel')
  axs[0].imshow(org_mel.cpu(), origin='lower', aspect='auto')
  axs[1].set_title('pred_mel')
  axs[1].imshow(pred_mel.cpu(), origin='lower', aspect='auto')
  axs[2].set_title('condition_roll')
  axs[2].imshow(condition_roll.cpu(), origin='lower', aspect='auto')
  plt.savefig(save_path)
  plt.close()
  

@torch.no_grad()
def inference_for_diff_train(
                            vocoder_name,
                            org_mel, 
                            pred_mel, 
                            cfg_pred_mel = None, 
                            enc_pred_mel = None,
                            condition_pitch_roll = None,
                            vocoder_pth = None,
                            save_pth = None,
                            current_epoch = None,
                            save_audio_ = False,
                            ):
      
  if len(org_mel.shape) == 4:
    org_mel = org_mel.squeeze(1)
    pred_mel = pred_mel.squeeze(1)
    if cfg_pred_mel is not None:
      cfg_pred_mel = cfg_pred_mel.squeeze(1)
    if enc_pred_mel is not None:
      enc_pred_mel = enc_pred_mel.squeeze(1)
      
  if vocoder_name == 'soundstream':
    mel_to_wav = mel_to_wav_soundstream
    vocoder_pth = get_mel_inverse_converter()
    sr = 16000
    
  org_pred_audio = mel_to_wav(org_mel, vocoder_pth)
  pred_audio = mel_to_wav(pred_mel, vocoder_pth)
  
  if cfg_pred_mel is not None:
    cfg_pred_audio = mel_to_wav(cfg_pred_mel, vocoder_pth)

  if save_pth is not None:  
    org_pred_audio_pth_list = []
    pred_audio_pth_list = []
    cfg_pred_audio_pth_list = []
    img_pth_list = []
    
    for i in range(org_mel.shape[0]):
      org_audio_pth = save_pth + f'/epoch_{current_epoch}_{i}_org.wav'
      pred_audio_pth = save_pth + f'/epoch_{current_epoch}_{i}_pred.wav'
      cfg_pred_audio_pth = save_pth + f'/epoch_{current_epoch}_{i}_cfg_pred.wav'
      img_pth = save_pth + f'/epoch_{current_epoch}_{i}.png'
      
      if save_audio_:
        save_audio(org_audio_pth, org_pred_audio[i], sr)
        save_audio(pred_audio_pth, pred_audio[i], sr)
        if cfg_pred_mel is not None:
          save_audio(cfg_pred_audio_pth, cfg_pred_audio[i], sr)
        org_pred_audio_pth_list.append(org_audio_pth)
        pred_audio_pth_list.append(pred_audio_pth)
        cfg_pred_audio_pth_list.append(cfg_pred_audio_pth)
      else:
        org_pred_audio_pth_list.append(org_pred_audio[i])
        pred_audio_pth_list.append(pred_audio[i])
        cfg_pred_audio_pth_list.append(cfg_pred_audio[i])
      if enc_pred_mel is not None: 
        save_image_diff(org_mel[i], pred_mel[i], cfg_pred_mel[i], enc_pred_mel[i], condition_pitch_roll[i], img_pth)
      else:
        save_image_diff_no_enc_mel(org_mel[i], pred_mel[i], cfg_pred_mel[i], condition_pitch_roll[i], img_pth)
      img_pth_list.append(img_pth)
      
  return org_pred_audio_pth_list, pred_audio_pth_list, cfg_pred_audio_pth_list, img_pth_list