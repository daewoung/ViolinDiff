
import wandb
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
from .bend_eval import evaluate_pb, evaluate_pb_get_acc, make_dict, add_dict, wb_evaluate_log



def get_acc(pred, target, pitch, threshold_bin):
    pred_pb = pred
    target_pb = target
    non_zero_indices = (pitch != 0)
    pred_pb_filtered = pred_pb[non_zero_indices]
    target_pb_filtered = target_pb[non_zero_indices]
    correct = torch.abs(pred_pb_filtered - target_pb_filtered) <= threshold_bin
    total_correct = correct.sum().item()
    total_predictions = len(correct)
    return (total_correct, total_predictions)


def get_plot_data(pitch, bend):
    value_lists = []
    pitch = rearrange(pitch, 'c t -> t c')
    bend = rearrange(bend, 'c t -> t c')
    for p, b in zip(pitch, bend):
        nonzero_p = torch.nonzero(p)
        if len(nonzero_p) == 0:
            value = np.nan
        else:
            max_nonzero = torch.max(nonzero_p)
            bend_value = b[max_nonzero]
            value = bend_value + max_nonzero
        value_lists.append(value)
    return value_lists

def plot_pb_data(org_pitch, plot_data, pred_plot_data, save_path):

    plt.figure(figsize=(10, 90))
    plt.subplot(2, 1, 1)
    plt.title('Org')
    plt.plot(plot_data, 'r', label = 'GT')
    plt.xlim(0, 255)
    plt.ylim(0, 54)
    plt.title('Pred')
    plt.plot(pred_plot_data, 'b', label = 'Pred')
    plt.xlim(0, 255)
    plt.ylim(0, 54)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.imshow(org_pitch, aspect='auto', origin='lower')
    plt.savefig(save_path)
    plt.close()


@torch.no_grad()
def inference_plot_data(org_pitch, org_bend, pred_bend, save_path, current_epoch):
    save_pth_lists = []
    zero_indices = org_pitch == 0
    pred_bend[zero_indices] = 0
    org_pitch = org_pitch.cpu()
    org_bend = org_bend.cpu()
    pred_bend = pred_bend.cpu()
    for i in range(org_pitch.shape[0]):
        img_save_pth = save_path + f'/epoch_{current_epoch}_{i}.png'
        pitch = org_pitch[i]
        bend = org_bend[i]
        pred = pred_bend[i]
        org_plot_pitch = get_plot_data(pitch, bend)
        pred_plot_pitch = get_plot_data(pitch, pred)
        plot_pb_data(pitch, org_plot_pitch, pred_plot_pitch, img_save_pth)
        save_pth_lists.append(img_save_pth)
    return save_pth_lists




class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, valid_loader, num_epoch, save_epoch, infer_epoch, device, wandb_log, pt_save_dir, out_save_dir, fp_16):
        self.wandb_log = wandb_log
        self.mixed_precision = fp_16
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.infer_epoch = infer_epoch
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = None
        self.pt_save_dir = pt_save_dir
        self.out_save_dir = out_save_dir
        self.num_epoch = num_epoch
        self.save_epoch = save_epoch
        if self.mixed_precision:
            print('mixed precision ON')
            self.grad_scaler = GradScaler()
        self.train_loader, self.valid_loader = (train_loader, valid_loader)
        self.model.to(self.device)
        self.loss = nn.MSELoss()
        self.loss.to(self.device)

        self.mask = False
        self.tf_ratio = 1.0

    def save_model(self, path):
        torch.save({'model': self.model.state_dict(), 'optim': self.optimizer.state_dict()}, path)

    def wb_log(self, img_lists, current_step):
        for i in range(len(img_lists)):
            image = wandb.Image(img_lists[i])
            log_data = {f'test/{i}_plot': image}
            wandb.log(log_data, step=current_step)

    def train(self):
        train_step = 0
        print('start training')
        for epoch in tqdm(range(1, self.num_epoch + 1)):
            
            train_loss = 0
            train_total = 0
          

            self.model.train()
            for i, batch in enumerate(tqdm(self.train_loader)):
                pitch, onset, bend, vel, perf, offset = batch
                nonzero_mask = (pitch != 0)
                train_step += 1
  
                condition = [pitch.to(self.device), onset.to(self.device), bend.to(self.device), 
                            vel.to(self.device), perf.to(self.device), offset.to(self.device)]

                self.optimizer.zero_grad()
                if self.mixed_precision:
                    with autocast(enabled=True):
                        loss = self.model(bend.to(self.device), condition, masking= pitch.to(self.device))

                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    if self.scheduler is not None:
                        self.scheduler.step()
                else:
                    pred_pb = self.model(condition, ar_mode=True)
                    if self.mask:
                        loss = self.loss(pred_pb[nonzero_mask], bend[nonzero_mask].to(self.device))
                    else:
                        loss = self.loss(pred_pb, bend.to(self.device))
                    loss.backward()
                    self.optimizer.step()
                    
                train_loss += loss.item() * pitch.shape[0]
                train_total += pitch.shape[0]
                
     
                if self.wandb_log:
                    wandb.log({'train/step_loss': loss.item()}, step=train_step)
                    wandb.log({'train/lr': self.optimizer.param_groups[0]['lr']}, step=train_step)
                    
            print(f'Epoch {epoch} total train loss: {train_loss / train_total}')
            
            if self.wandb_log:
                wandb.log({'train/epoch_loss': train_loss / train_total}, step=train_step)

            if epoch % 1000 == 0:
                self.model.eval()
                self.validate(epoch, train_step)
                
            
        wandb.finish()

    @torch.no_grad()
    def validate(self, current_epoch, current_step):
        val_loss = 0
        val_total = 0
      
    
        infer_org_pitch = []
        infer_org_bend = []
        infer_pred_bend = []
        
        total_thres_100 = make_dict()
        total_zeros_thres_100 = make_dict()
        total_thres_200 = make_dict()
        total_zeros_thres_200 = make_dict()
        total_thres_400 = make_dict()
        total_zeros_thres_400 = make_dict()
    
    
        for i, batch in enumerate(tqdm(self.valid_loader)):
            pitch, onset, bend, vel, perf, offset = batch
            nonzero_mask = (pitch != 0)
            condition = [pitch.to(self.device), onset.to(self.device), bend.to(self.device), 
                         vel.to(self.device), perf.to(self.device), offset.to(self.device)]

            if self.mixed_precision:
                with autocast(enabled=True):
                    loss = self.model(bend.to(self.device), condition, valid_mode = True, masking= pitch.to(self.device))

            else:
                pred_pb = self.model(condition, tf_ratio=0)
                if self.mask:
                    loss = self.loss(pred_pb[nonzero_mask], bend[nonzero_mask].to(self.device))
                else:
                    loss = self.loss(pred_pb, bend.to(self.device))

                    
            val_loss += loss.item() * pitch.shape[0]
            val_total += pitch.shape[0]
      

            pred_pb = self.model.p_sample_loop(condition, num_batches = condition[0].shape[0], cfg_scale=1, mask = pitch.to(self.device))


            thres_100, pred_out, max_length_batch = evaluate_pb(onset.to(self.device), offset.to(self.device), pitch.to(self.device), pred_pb, bend.to(self.device), 0.025, None, return_out=True)
            zero_thres_100, zero_out, _ = evaluate_pb(onset.to(self.device), offset.to(self.device), pitch.to(self.device), torch.zeros_like(pitch).to(self.device), bend.to(self.device), 0.025, None, return_out=True)
            thres_200 = evaluate_pb(onset.to(self.device), offset.to(self.device), pitch.to(self.device), pred_pb, bend.to(self.device), 0.05, pred_out)
            zero_thres_200 = evaluate_pb(onset.to(self.device), offset.to(self.device), pitch.to(self.device), torch.zeros_like(pitch).to(self.device), bend.to(self.device), 0.05, zero_out)
            thres_400 = evaluate_pb(onset.to(self.device), offset.to(self.device), pitch.to(self.device), pred_pb, bend.to(self.device), 0.1, pred_out)
            zero_thres_400 = evaluate_pb(onset.to(self.device), offset.to(self.device), pitch.to(self.device), torch.zeros_like(pitch).to(self.device), bend.to(self.device), 0.1, zero_out)

            total_thres_100 = add_dict(total_thres_100, thres_100)
            total_zeros_thres_100 = add_dict(total_zeros_thres_100, zero_thres_100)

            total_thres_200 = add_dict(total_thres_200, thres_200)
            total_zeros_thres_200 = add_dict(total_zeros_thres_200, zero_thres_200)

            total_thres_400 = add_dict(total_thres_400, thres_400)
            total_zeros_thres_400 = add_dict(total_zeros_thres_400, zero_thres_400)
            
            if current_epoch % self.infer_epoch == 0:
                infer_org_pitch.append(pitch.to(self.device)[max_length_batch])
                infer_org_bend.append(bend.to(self.device)[max_length_batch])
                infer_pred_bend.append(pred_pb.to(self.device)[max_length_batch])

            
        print(f'Epoch {current_epoch} val total loss: {val_loss / val_total}')
        
        if self.wandb_log:
            wandb.log({'val/loss': val_loss / val_total}, step=current_step)
            wb_evaluate_log(total_thres_100, 0.025, current_step)
            wb_evaluate_log(total_zeros_thres_100, 0.025, current_step, True)
            wb_evaluate_log(total_thres_200, 0.05, current_step)
            wb_evaluate_log(total_zeros_thres_200, 0.05, current_step, True)
            wb_evaluate_log(total_thres_400, 0.1, current_step)
            wb_evaluate_log(total_zeros_thres_400, 0.1, current_step, True)
            
        if current_epoch % self.save_epoch == 0:
            self.save_model(self.pt_save_dir + f'/{current_epoch}_{current_step}.pt')
            
        if current_epoch % self.infer_epoch == 0:
            infer_org_pitch = torch.cat(infer_org_pitch, dim=0)
            infer_org_bend = torch.cat(infer_org_bend, dim=0)
            infer_pred_bend = torch.cat(infer_pred_bend, dim=0)
            
            if self.wandb_log:
                save_pth_lists = inference_plot_data(infer_org_pitch, infer_org_bend, infer_pred_bend, self.out_save_dir, current_epoch)
                self.wb_log(save_pth_lists, current_step)