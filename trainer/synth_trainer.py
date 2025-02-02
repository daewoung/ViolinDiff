import wandb
import torch 
import torch.nn as nn 

from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from model.module.mel2wav import inference_for_diff_train

def init_seeding(seed):
  cuda_rng_state_original = torch.cuda.get_rng_state()
  rng_state_original = torch.get_rng_state()
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  return cuda_rng_state_original, rng_state_original

def seeding(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


def seeding_back(cuda_rng_state_original, rng_state_original):
  torch.set_rng_state(rng_state_original)
  torch.cuda.set_rng_state(cuda_rng_state_original)

  
class Trainer:
  def __init__(self, 
              model,
              mel_converter,
              vocoder_name,
              optimizer,
              scheduler,
              train_loader,
              valid_loader,
              seed,
              num_epoch,
              save_epoch,
              infer_epoch,
              device,
              wandb_log,
              pt_save_dir,
              out_save_dir,
              fp_16,
              cfg_scale
              ):
    
    self.wandb_log = wandb_log
    self.mixed_precision = fp_16
    self.device = device
    self.vocoder_name = vocoder_name
    if vocoder_name == 'soundstream':
      self.sr = 16000
    else:
      self.sr = 22050

    self.model = model
    self.optimizer = optimizer
    self.cfg_scale = cfg_scale
    
    if scheduler is not None:
      self.scheduler = scheduler
    else:
      self.scheduler = None 
    
    self.pt_save_dir = pt_save_dir
    self.out_save_dir = out_save_dir
    self.num_epoch = num_epoch
    self.save_epoch = save_epoch
    self.infer_epoch = infer_epoch
    if self.mixed_precision:
      print('mixed precision ON')
      self.grad_scaler = GradScaler()    

    self.train_loader, self.valid_loader = train_loader, valid_loader

    self.model.to(self.device)
    
    self.seed = seed
    self.valid_seed = seed + 1
    
    self.mel_converter = mel_converter
    self.mel_converter.to(self.device)
    
  def save_model(self, path):
    torch.save({'model': self.model.state_dict(), 'optim':self.optimizer.state_dict()}, path)

  def wb_table(self, 
              img_pth, 
              org_voc_audio_pth, 
              pred_audio_pth,
              cfg_pred_audio_pth,
              org_audio_list, 
              current_step):
    sample_rate = self.sr
    
    for i in range(len(img_pth)):
      print(i)
      vocoder_audio = wandb.Audio(org_voc_audio_pth[i], sample_rate=sample_rate)
      org_audio = wandb.Audio(org_audio_list[i], sample_rate=sample_rate)
      
      pred_audio = wandb.Audio(pred_audio_pth[i], sample_rate=sample_rate)
      cfg_pred_audio = wandb.Audio(cfg_pred_audio_pth[i], sample_rate=sample_rate)
      
      image = wandb.Image(img_pth[i])
      
      log_data = {
        f'test/{i}_Org_Audio': org_audio,
        f'test/{i}_Vocoder_Audio': vocoder_audio,
        f'test/{i}_Pred_Audio': pred_audio,
        f'test/{i}_Cfg_Pred_Audio': cfg_pred_audio,
        f'test/{i}_Mel_Image': image
      }
      wandb.log(log_data, step=current_step)
      

  def train(self):
      train_step = 0 
      print('start training')
      for epoch in tqdm(range(1, self.num_epoch+1)):
        train_loss = 0
        train_total = 0
        train_enc_loss = 0
        self.model.train()
        #seeding(random.randint(0, 9999))
        
        for i, batch in enumerate(tqdm(self.train_loader)):
          audio, pitch, onset, bend, vel, perf, offset = batch 

          train_step += 1
          
          condition = [pitch.to(self.device), onset.to(self.device),
                      bend.to(self.device), vel.to(self.device), perf.to(self.device), offset.to(self.device)]
          spec = self.mel_converter(audio.to(self.device))
          spec = spec.to(self.device)
          
          self.optimizer.zero_grad()

          if self.mixed_precision:
            with autocast(enabled=True):
              eps_loss, enc_loss = self.model(spec, condition, valid_mode=False)
              loss = eps_loss + enc_loss
              
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            
            if self.scheduler is not None:
              self.scheduler.step()

          else:
            eps_loss, enc_loss = self.model(spec, condition, valid_mode=False)
            loss = eps_loss + enc_loss

            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
              self.scheduler.step()

          train_loss += eps_loss.item() * spec.shape[0]
          train_enc_loss += enc_loss.item() * spec.shape[0]
          train_total += spec.shape[0]
          
          if self.wandb_log:
            wandb.log({'train/step_loss': eps_loss.item()}, step=train_step)
            wandb.log({'train/step_loss_enc': enc_loss.item()}, step=train_step)
            wandb.log({'train/lr': self.optimizer.param_groups[0]['lr']}, step = train_step)

        print(f'Epoch {epoch} total train loss: {train_loss / train_total}')

        
        if self.wandb_log:
          wandb.log({'train/epoch_loss': train_loss / train_total}, step = train_step)
          wandb.log({'train/epoch_enc_loss': train_enc_loss / train_total}, step = train_step)

        
        self.model.eval()
        self.validate(epoch, train_step)

      
      if self.wandb_log:
        wandb.finish()
  
  @torch.no_grad()
  def validate(self, current_epoch, current_step):
    
    val_loss = 0
    val_enc_loss =0 
    val_total = 0
    
    # seeding 하기
    
    infer_pred_spec = []
    
    cond_list = []
    # seeding(self.seed)   
    
    infer_pitch_list = []
    infer_onset_list = []
    infer_bend_list = []
    infer_vel_list = []
    infer_perf_list = []
    infer_org_spec = []
    org_audio_list = []

    for i, batch in enumerate(tqdm(self.valid_loader)):
      audio, pitch, onset, bend, vel, perf, offset = batch 

      condition = [pitch.to(self.device), onset.to(self.device),
                  bend.to(self.device), vel.to(self.device), perf.to(self.device), offset.to(self.device)]
      
      spec = self.mel_converter(audio.to(self.device))
      spec = spec.to(self.device)

      if self.mixed_precision:
        with autocast():
          eps_loss, enc_loss = self.model(spec, condition, valid_mode=True) # no dropout
          
      else:
        eps_loss, enc_loss = self.model(spec, condition, valid_mode=True)

      val_loss += eps_loss.item() * spec.shape[0]
      val_enc_loss = enc_loss.item() * spec.shape[0]
      val_total += spec.shape[0]
      
    print(f'Epoch {current_epoch} val total loss: {val_loss / val_total}')

    
    if self.wandb_log:
      wandb.log({'val/loss': val_loss / val_total}, step = current_step)
      wandb.log({'val/enc_loss': val_enc_loss / val_total}, step = current_step)
      
    if current_epoch % self.infer_epoch == 0:
      self.save_model(self.pt_save_dir + f'/{current_epoch}_{current_step}.pt')
      
      L2_loss = nn.MSELoss()
      cuda_rng_state_original, rng_state_original = init_seeding(self.valid_seed)
      
      total_loss = 0
      total_cfg_loss = 0
      
      lowest_loss = float('inf')
      lower_loss_infer_out = None
      lower_loss_cfg_infer_out = None
      
      
      for i in range(1, 4):
        pred_mel_loss = 0
        cfg_pred_mel_loss = 0
        val_total = 0
        
        infer_list = [] 
        cfg_infer_list = []
        org_spec_list = []
        infer_pitch = []
        for batch in tqdm(self.valid_loader):
          audio, pitch, onset, bend, vel, perf, offset = batch 

          condition = [pitch.to(self.device), onset.to(self.device),
                      bend.to(self.device), vel.to(self.device), perf.to(self.device), offset.to(self.device)]
          
          spec = self.mel_converter(audio.to(self.device))
          spec = spec.to(self.device)

          infer_out = self.model.p_sample_loop(condition, num_batches = condition[0].shape[0], cfg_scale=1)
          cfg_infer_out = self.model.p_sample_loop(condition, num_batches = condition[0].shape[0], cfg_scale = self.cfg_scale)

          batch_pred_mel_loss = L2_loss(infer_out, spec)
          batch_cfg_pred_mel_loss = L2_loss(cfg_infer_out, spec)
          
          pred_mel_loss += batch_pred_mel_loss.item() * spec.shape[0]
          cfg_pred_mel_loss = batch_cfg_pred_mel_loss.item() * spec.shape[0]
          val_total += spec.shape[0]
          
          if len(infer_list) <= 10:
            for j in range(5):
              infer_list.append(infer_out[j])
              cfg_infer_list.append(cfg_infer_out[j])
              org_spec_list.append(spec[j])
              infer_pitch.append(pitch[j])
              org_audio_list.append(audio[j])
              
        if (pred_mel_loss / val_total) < lowest_loss:
          lowest_loss = pred_mel_loss / val_total
          lower_loss_infer_out = infer_list
          lower_loss_cfg_infer_out = cfg_infer_list

        seeding(self.valid_seed + i)
        total_loss += pred_mel_loss / val_total
        total_cfg_loss += cfg_pred_mel_loss / val_total
        print(f'{i}th infer loss  : {pred_mel_loss / val_total}')
        print(f'{i}th cfg infer loss  : {cfg_pred_mel_loss / val_total}')
        
      if self.wandb_log:
        wandb.log({f'val/mel_loss': total_loss / 3}, step = current_step)
        wandb.log({f'val/cfg_mel_loss': total_cfg_loss / 3}, step = current_step)

      seeding_back(cuda_rng_state_original, rng_state_original)

      infer_org_spec = torch.stack(org_spec_list, dim=0)
      infer_pred_spec = torch.stack(lower_loss_infer_out, dim=0)
      infer_cfg_pred_spec = torch.stack(lower_loss_cfg_infer_out, dim=0)
      infer_pitch = torch.stack(infer_pitch, dim=0)
      print('infer_org_spec', infer_org_spec.shape, infer_pred_spec.shape, infer_cfg_pred_spec.shape, infer_pitch.shape)
      
      # org_pred_audio_pth_list, pred_audio_pth_list, cfg_pred_audio_pth_list, img_pth_list = inference_for_diff_train(self.vocoder_name,
      #                                                                                                               infer_org_spec,
      #                                                                                                               infer_pred_spec,
      #                                                                                                               infer_cfg_pred_spec,
      #                                                                                                               None,
      #                                                                                                               infer_pitch,
      #                                                                                                               '/home/daewoong/userdata/diffwave/src/weights-124000.pt',
      #                                                                                                               self.out_save_dir,
      #                                                                                                               current_epoch,
      #                                                                                                               save_audio_ = False
      #                                                                                                             )
      
      # print('org_pred_audio_pth_list', len(org_pred_audio_pth_list),
      #       'pred_audio_pth_list', len(pred_audio_pth_list),
      #       'cfg_pred_audio_pth_list', len(cfg_pred_audio_pth_list),
      #       'img_pth_list', len(img_pth_list))

      # if self.wandb_log:
      #   #self.wb_table_cfg(img_pth_list, org_pred_audio_pth_list, pred_audio_pth_list, cfg_pred_audio_pth_list, current_epoch, current_step)
      #   self.wb_table(img_pth_list, org_pred_audio_pth_list, 
      #               pred_audio_pth_list, cfg_pred_audio_pth_list, 
      #               org_audio_list, current_step)