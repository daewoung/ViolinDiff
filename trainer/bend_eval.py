import torch
import numpy as np
import time
import wandb

def get_acc(pred_bend, gt_bend, threshold):
  correct = torch.abs(pred_bend - gt_bend) <= threshold
  total_correct = correct.sum().item()
  total_predictions = correct.numel()
  return (total_correct, total_predictions)


def get_note_acc_feature(gt_bend_note, pred_bend_note):
  gt_mean = []
  pred_mean = []  
  
  gt_std = []
  pred_std = []
  
  gt_std_1 = []
  gt_std_3 = []
  gt_std_5 = []
  pred_std_1 = []
  pred_std_3 = []
  pred_std_5 = []
  
  gt_max = []
  gt_min = []
  pred_max = []
  pred_min = []
  
  
  for gt_bend, pred_bend in zip(gt_bend_note, pred_bend_note):

    g_mean = gt_bend.mean()
    
   
    g_max = gt_bend.max()
    g_min = gt_bend.min()
    
    p_mean = pred_bend.mean()

    p_max = pred_bend.max()
    p_min = pred_bend.min()


    if len(gt_bend) == 1:
      g_std = 0
      g_std_1 = g_std_3 = g_std_5 = 0

    else:
      g_std = gt_bend.std()
      g_std_1 = gt_bend[1:-1].std() if len(gt_bend[1:-1]) > 0 else 0
      g_std_3 = gt_bend[3:-3].std() if len(gt_bend[3:-3]) > 0 else 0
      g_std_5 = gt_bend[5:-5].std() if len(gt_bend[5:-5]) > 0 else 0

    if len(pred_bend) == 1:
      p_std = 0
      p_std_1 = p_std_3 = p_std_5 = 0

    else:
      p_std = pred_bend.std()
      p_std_1 = pred_bend[1:-1].std() if len(pred_bend[1:-1]) > 0 else 0
      p_std_3 = pred_bend[3:-3].std() if len(pred_bend[3:-3]) > 0 else 0
      p_std_5 = pred_bend[5:-5].std() if len(pred_bend[5:-5]) > 0 else 0


    gt_mean.append(g_mean)
    gt_std.append(g_std)
    gt_std_1.append(g_std_1) 
    gt_std_3.append(g_std_3) 
    gt_std_5.append(g_std_5)
    gt_max.append(g_max)
    gt_min.append(g_min)
    
    pred_mean.append(p_mean)
    pred_std.append(p_std)
    pred_std_1.append(p_std_1) 
    pred_std_3.append(p_std_3)
    pred_std_5.append(p_std_5)
    
    pred_max.append(p_max)
    pred_min.append(p_min)
  
  gt_mean = torch.tensor(gt_mean)
  gt_std = torch.tensor(gt_std)
  gt_max = torch.tensor(gt_max)
  gt_min = torch.tensor(gt_min)
  
  pred_mean = torch.tensor(pred_mean)
  pred_std = torch.tensor(pred_std)
  pred_max = torch.tensor(pred_max)
  pred_min = torch.tensor(pred_min)
  
  gt_std_1 = torch.tensor(gt_std_1)
  gt_std_3 = torch.tensor(gt_std_3)
  gt_std_5 = torch.tensor(gt_std_5)
  
  pred_std_1 = torch.tensor(pred_std_1)
  pred_std_3 = torch.tensor(pred_std_3)
  pred_std_5 = torch.tensor(pred_std_5)
  
  out = {'gt_mean' : gt_mean,
         'pred_mean' : pred_mean,
         'gt_max' : gt_max,
         'pred_max' : pred_max,
         'gt_min' : gt_min,
         'pred_min' : pred_min,
         'gt_std' : gt_std,
         'pred_std' : pred_std,
         
         'gt_std_1' : gt_std_1,
         'pred_std_1' : pred_std_1,
         'gt_std_3' : gt_std_3,
         'pred_std_3' : pred_std_3, 
         'gt_std_5' : gt_std_5,
         'pred_std_5' : pred_std_5}
           
  return out
  

def get_note(pitch_data):
  zeros_pitch = torch.zeros([pitch_data.size(0), pitch_data.size(1), 1]).to(pitch_data.device)
  pitch_for_diff = torch.cat([zeros_pitch, pitch_data, zeros_pitch], dim=-1)
  indicies = pitch_for_diff.diff(dim=-1).nonzero()
  start_indices = indicies[::2]
  end_indices = indicies[1::2]

  assert len(start_indices) == len(end_indices)
  return start_indices, end_indices

def get_frame_max_lengh_batch(start_indices, end_indices, num_batch=5):
  frame_length = end_indices[:, 2] - start_indices[:, 2]
  batch_ids = start_indices[:, 0]
  _, sort_indicies = frame_length.sort(descending=True)
  max_length_batch = batch_ids[sort_indicies].unique(False)
  max_length_batch = max_length_batch.flip(0)[:num_batch]
  return max_length_batch

def get_per_note_bend(start_indicies, end_indices, bends):
  total_bend_list = []
  for start, end in zip(start_indicies, end_indices):
    assert start[0] == end[0]
    assert start[1] == end[1]
    total_bend_list.append(bends[start[0], start[1], start[2]:end[2]])
  return total_bend_list

def evaluate_pb(onset_data, offset_data, pitch_data, pred_bend, gt_bend, threshold, out = None,
                return_out = False, get_batch_num = 5):
  onset_data_none_zero = (onset_data != 0)
  offset_data_none_zero = (offset_data != 0)
  pitch_data_none_zero = (pitch_data != 0)
  
  onset_correct, onset_total = get_acc(pred_bend[onset_data_none_zero], gt_bend[onset_data_none_zero], threshold)
  offset_correct, offset_total = get_acc(pred_bend[offset_data_none_zero], gt_bend[offset_data_none_zero], threshold)
  frame_correct, frame_total = get_acc(pred_bend[pitch_data_none_zero], gt_bend[pitch_data_none_zero], threshold)
  
  if out is None:

    start_indices, end_indices = get_note(pitch_data)
    max_length_batch = get_frame_max_lengh_batch(start_indices, end_indices, num_batch=get_batch_num)
    gt_note_bends = get_per_note_bend(start_indices, end_indices, gt_bend)
      
    pred_note_bends = get_per_note_bend(start_indices, end_indices, pred_bend)


    out = get_note_acc_feature(gt_note_bends, pred_note_bends)
    
  else:
    out = out

  mean_correct, mean_total = get_acc(out['gt_mean'], out['pred_mean'], threshold) 
  std_correct, std_total =  get_acc(out['gt_std'], out['pred_std'], threshold) 
  max_correct, max_total = get_acc(out['gt_max'], out['pred_max'], threshold) 
  min_correct, min_total = get_acc(out['gt_min'], out['pred_min'], threshold) 
  
  std_1_correct, std_1_total = get_acc(out['gt_std_1'], out['pred_std_1'], threshold)
  std_3_correct, std_3_total = get_acc(out['gt_std_3'], out['pred_std_3'], threshold)
  std_5_correct, std_5_total = get_acc(out['gt_std_5'], out['pred_std_5'], threshold)
  
  evaluate_out = {'onset_correct' : onset_correct,
                  'onset_total' : onset_total,
                  'offset_correct' : offset_correct,
                  'offset_total' : offset_total,
                  'frame_correct' : frame_correct,
                  'frame_total' : frame_total,
                  'mean_correct' : mean_correct,
                  'mean_total' : mean_total,
                  'std_correct' : std_correct,
                  'std_total' : std_total,
                  'max_correct' : max_correct,
                  'max_total' : max_total,
                  'min_correct': min_correct,
                  'min_total' : min_total,
                  'std_1_correct' : std_1_correct,
                  'std_1_total' : std_1_total,
                  'std_3_correct' : std_3_correct,
                  'std_3_total' : std_3_total,
                  'std_5_correct' : std_5_correct,
                  'std_5_total' : std_5_total
                  }
  if return_out:
    return evaluate_out, out, max_length_batch
  else:
    return evaluate_out
  
def evaluate_pb_get_acc(evaluate_out):
  
  onset_acc = evaluate_out['onset_correct'] / evaluate_out['onset_total']
  offset_acc = evaluate_out['offset_correct'] / evaluate_out['offset_total']
  frame_acc = evaluate_out['frame_correct'] / evaluate_out['frame_total']
  mean_acc = evaluate_out['mean_correct'] / evaluate_out['mean_total']
  std_acc = evaluate_out['std_correct'] / evaluate_out['std_total']
  max_acc = evaluate_out['max_correct'] / evaluate_out['max_total']
  min_acc = evaluate_out['min_correct'] / evaluate_out['min_total']
  
  std_1_acc = evaluate_out['std_1_correct'] / evaluate_out['std_1_total']
  std_3_acc = evaluate_out['std_3_correct'] / evaluate_out['std_3_total']
  std_5_acc = evaluate_out['std_5_correct'] / evaluate_out['std_5_total']
  
  acc_dict = {'onset' : onset_acc, 'offset' : offset_acc,
              'frame' : frame_acc, 'mean' : mean_acc,
              'std' : std_acc, 'max' : max_acc, 'min': min_acc,
              'std_1' : std_1_acc, 'std_3' : std_3_acc, 'std_5' : std_5_acc}

  return acc_dict


def make_dict():
    return {'onset_correct' : 0, 'onset_total' : 0,  'offset_correct' : 0,  'offset_total' : 0,
            'frame_correct' : 0, 'frame_total' : 0,  'mean_correct' : 0,'mean_total' : 0, 
            'std_correct' : 0,'std_total' : 0,  'max_correct' : 0, 'max_total' : 0, 'min_correct': 0, 'min_total' : 0,
            'std_1_correct' : 0, 'std_1_total' : 0, 'std_3_correct' : 0, 'std_3_total' : 0, 'std_5_correct' : 0, 'std_5_total' : 0}

def add_dict(org, target):
    for key in org.keys(): 
        org[key] += target[key]   
    return org 




def wb_evaluate_log(dt, threshold, train_step, zero_out=False):
    acc_dict = evaluate_pb_get_acc(dt)

    if zero_out:
        wandb.log({f'val/threshold_{threshold}_onset_zero': acc_dict['onset']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_offset_zero': acc_dict['offset']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_frame_zero': acc_dict['frame']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_mean_zero': acc_dict['mean']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_std_zero': acc_dict['std']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_max_zero': acc_dict['max']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_min_zero': acc_dict['min']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_std_1_zero': acc_dict['std_1']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_std_3_zero': acc_dict['std_3']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_std_5_zero': acc_dict['std_5']}, step=train_step)

    else:
        wandb.log({f'val/threshold_{threshold}_onset': acc_dict['onset']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_offset': acc_dict['offset']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_frame': acc_dict['frame']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_mean': acc_dict['mean']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_std': acc_dict['std']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_max': acc_dict['max']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_min': acc_dict['min']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_std_1': acc_dict['std_1']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_std_3': acc_dict['std_3']}, step=train_step)
        wandb.log({f'val/threshold_{threshold}_std_5': acc_dict['std_5']}, step=train_step)

