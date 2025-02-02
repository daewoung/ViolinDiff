import pickle
import torch
import matplotlib.pyplot as plt
import pretty_midi
import numpy as np
import time

from matplotlib.ticker import FuncFormatter, MultipleLocator

from data_preprocess.vibrato_func import get_vibrato

'''
FOR UTILS
'''

PITCH_MIN = 55
PITCH_MAX = 108

def custom_round(number, digits=6):
  multiplier = 10 ** digits
  return int(number * multiplier) / multiplier


def load_pkl(pkl_path):
  with open(pkl_path, 'rb') as f:
      data = pickle.load(f)
  return data

def load_midi(midi_path):
  pm = pretty_midi.PrettyMIDI(midi_path)
  return pm

def pb_chunking(pitch_bends_array, start_time, clip_start, clip_end):
  pitch_bends_array[:, 0] = pitch_bends_array[:, 0] - start_time
  mask = (pitch_bends_array[:, 0] >= clip_start) & (pitch_bends_array[:, 0] <= clip_end)
  chunk_pb_array = pitch_bends_array[mask]  
  return chunk_pb_array

def get_pb_sustain_duration(chunk_pb_array, clip_end_array, mel_per_time_step):
  cut_dur = np.isin(chunk_pb_array[:, 0], clip_end_array)
  dur = np.diff(chunk_pb_array[:, 0])
  dur = np.append(dur, 0)
  dur[cut_dur] = 0
  
  start_mel_idx = np.floor((chunk_pb_array[:, 0] + 1e-9) / mel_per_time_step).astype(int)
  end_mel_idx = np.floor((chunk_pb_array[:, 0] + dur) / mel_per_time_step).astype(int)
  
  chunk_pb_array = np.stack([chunk_pb_array[:, 0], chunk_pb_array[:, 0] + dur, dur, start_mel_idx, end_mel_idx, chunk_pb_array[:, 1]], axis=1)  
  
  start_times = []  
  sustain_end_times = []
  split_dur = []
  mel_idxs =[]
  sustain = []
  pitch_semitones = []
  
  for arr in chunk_pb_array:
    semitone = pretty_midi.pitch_bend_to_semitones(arr[5])
    start_mel_idx = int(arr[3])
    end_mel_idx = int(arr[4])

    if start_mel_idx == end_mel_idx:
      split_dur.append(arr[2])
      mel_idxs.append(start_mel_idx)
      sustain.append(False)
      start_times.append(arr[0])
      sustain_end_times.append(arr[1])
      pitch_semitones.append(semitone)
      
    else:
      for mel_idx in range(start_mel_idx, end_mel_idx+1):

        if mel_idx == start_mel_idx:
          chunk_mel = (mel_idx+1) * mel_per_time_step
          dur = chunk_mel - arr[0]
          pitch_start_time = arr[0]
          pitch_end_time = arr[0] + dur

        elif mel_idx == end_mel_idx:
          chunk_mel = (mel_idx * mel_per_time_step)
          dur = arr[1] - chunk_mel
          pitch_start_time = chunk_mel
          pitch_end_time = pitch_start_time + dur
        else:
          dur = mel_per_time_step #(0.02)
          pitch_start_time = mel_idx * mel_per_time_step
          pitch_end_time = pitch_start_time + dur

        split_dur.append(dur)
        mel_idxs.append(mel_idx)
        sustain.append(True)        
        start_times.append(pitch_start_time)
        sustain_end_times.append(pitch_end_time)
        pitch_semitones.append(semitone)

  chunk_pb_sus_array = np.stack([start_times, sustain_end_times, split_dur, mel_idxs, pitch_semitones, sustain], axis=1)

  return chunk_pb_sus_array

def get_weighted_pitch_bend(chunk_pb_array, clip_end_array, mel_time_bins, mel_per_time_step):
  chunk_pb_sus_array = get_pb_sustain_duration(chunk_pb_array, clip_end_array, mel_per_time_step)
  weighted_pitch_bend = np.zeros([mel_time_bins])
  
  total_duration = 0

  unique_mel_idxs = np.unique(chunk_pb_sus_array[:, 3])

  for mel_idx in unique_mel_idxs:
    mask = (chunk_pb_sus_array[:, 3] == mel_idx)
    select_mel_array = chunk_pb_sus_array[mask]
    total_duration = np.sum(select_mel_array[:, 2])
    total_weight_bends = np.sum(select_mel_array[:, 2] * select_mel_array[:, 4])
    
    if total_duration > 0:
      weighted_pitch_bend[int(mel_idx)] = total_weight_bends / total_duration
    else:
      weighted_pitch_bend[int(mel_idx)] = 0
  return weighted_pitch_bend, chunk_pb_sus_array
        

def vectorize_midi(pitch_bends, notes, start_time, end_time):
  
  if len(pitch_bends) == 0:
    pitch_bends_array = None
    print('No pitch_bends')
  else:
    pitch_bends_array = np.array([[pb.time, pb.pitch] for pb in pitch_bends])
    pitch_bends_array[:, 0] = np.around(pitch_bends_array[:, 0], 6)
      
    
  notes_array = np.array([[note.start, note.end, note.pitch, note.velocity, note.end-note.start] for note in notes])
  notes_array[:, :2] = np.around(notes_array[:, :2], 6)

  return pitch_bends_array, notes_array
  
def notes_chunking_allgin_mel(notes_array, start_time, end_time, mel_per_time_step):

  mask = (notes_array[:, 1] >= start_time) & (notes_array[:, 0] < end_time)
  chunk_notes = notes_array[mask]
  chunk_notes[:, 0] = np.maximum(chunk_notes[:, 0], start_time)
  chunk_notes[:, 1] = np.minimum(chunk_notes[:, 1], end_time)
  chunk_notes[:, 0] = (chunk_notes[:, 0] - start_time) 
  chunk_notes[:, 1] = (chunk_notes[:, 1] - start_time)
   
  
  mel_start_idx = np.floor((chunk_notes[:, 0] +1e-9) / mel_per_time_step).astype(int)
  mel_end_idx = chunk_notes[:, 1] / mel_per_time_step
  is_inegrer = (mel_end_idx % 1 ==0) & (mel_end_idx > 0)
  mel_end_idx = np.floor(mel_end_idx).astype(int)
  mel_end_idx[~is_inegrer] = mel_end_idx[~is_inegrer] + 1
  
  chunk_notes = np.concatenate([chunk_notes, mel_start_idx[:, None], mel_end_idx[:, None]], axis=1)
  # return [clip_start, clip_end, pitch, velocity, start_mel_idx, end_mel_idx]
  return chunk_notes  
  
  
  
def chunk_midi_roll(pm, start_time, end_time,
                    sr, hop_length, mel_time_bins, return_pitch_dict=False):
  pitch_data = torch.zeros(PITCH_MAX - PITCH_MIN + 1, mel_time_bins)
  onset_data = torch.zeros(PITCH_MAX - PITCH_MIN + 1, mel_time_bins)
  offset_data = torch.zeros(PITCH_MAX - PITCH_MIN + 1, mel_time_bins)
  mel_per_time_step = (hop_length / sr)
  
  velocity_data = torch.zeros(PITCH_MAX - PITCH_MIN + 1, mel_time_bins)
  
  bend_data = torch.zeros(PITCH_MAX - PITCH_MIN + 1, mel_time_bins, dtype=torch.float32)
  eps = 1e-9
  pitch_dict = []
  for inst in pm.instruments:
    
    pitch = inst.notes[0].pitch
    pitch_idx = pitch - PITCH_MIN
    
    pitch_bends_array, notes_array = vectorize_midi(inst.pitch_bends, inst.notes, start_time, end_time)

    # print(notes_array)
    if len(notes_array) == 0:
      continue
    
    chunk_notes_array = notes_chunking_allgin_mel(notes_array, start_time, end_time, mel_per_time_step)
    # [clip_start, clip_end, pitch, velocity, start_mel_idx, end_mel_idx]
    # print(chunk_notes_array)
    np.set_printoptions(precision=6, suppress=True)
    if len(chunk_notes_array) == 0:
      continue
    for i in chunk_notes_array:
      start_mel_idx, end_mel_idx = int(i[5]), int(i[6])
      # print(start_mel_idx, end_mel_idx)
      pitch_data[pitch_idx, start_mel_idx:end_mel_idx] = 1
      velocity_data[pitch_idx, start_mel_idx:end_mel_idx] = i[3] / 127
      onset_data[pitch_idx, start_mel_idx] = 1
      offset_data[pitch_idx, end_mel_idx-1] = 1
      pitch_dict.append({'pitch' : pitch, 'dur' : i[4] ,
                         'start' : start_mel_idx, 'end' : end_mel_idx,
                         'start_time' : i[0], 'end_time' : i[1]})

    if pitch_bends_array is not None:
      chunk_pb_array = pb_chunking(pitch_bends_array, start_time, chunk_notes_array[0][0], chunk_notes_array[-1][1])
      weighted_pitch_bends, chunk_pb_sus_array = get_weighted_pitch_bend(chunk_pb_array, chunk_notes_array[:, 1], mel_time_bins, mel_per_time_step)
      bend_data[pitch - PITCH_MIN] = torch.from_numpy(weighted_pitch_bends)
    
  if torch.any(bend_data < -1) or torch.any(bend_data > 1):
      print('bend_data out of range, clipping')
      bend_data = torch.clamp(bend_data, min=-1, max=1)

  
  sorted_pitch_dict = sorted(pitch_dict, key=lambda x: (x['start_time'], x['pitch']))
  if return_pitch_dict:
    vibrato_rolls = get_vibrato_from_midi(sorted_pitch_dict, bend_data)
    return pitch_data, onset_data, bend_data, velocity_data, offset_data, sorted_pitch_dict, vibrato_rolls
    
  else:
    return pitch_data, onset_data, bend_data, velocity_data, offset_data #, chunk_pb_array, chunk_pb_sus_array




def pitch_to_freq(pitch):
    return 440.0 * 2 ** ((pitch - 69) / 12.0)
  
def get_vibrato_from_midi(pitch_dict, bends_roll):
  vibrato_rolls = torch.zeros(PITCH_MAX - PITCH_MIN + 1, bends_roll.shape[1])
  
  for i in pitch_dict:
    start = i['start']
    end = i['end']
    pitch = i['pitch']
    if i['dur'] >= 0.2:
      pitch_bend = bends_roll[int(pitch) - PITCH_MIN, start:end]
      gt_f0 = pitch_bend + pitch
      gt_f0 = pitch_to_freq(gt_f0)
      gt_vibrato, _, _ = get_vibrato(int(pitch), gt_f0, sampling_interval= 0.02, min_note_length= 10)
      vibrato_rolls[int(pitch) - PITCH_MIN, start:end] = gt_vibrato
    else:
      vibrato_rolls[int(pitch) - PITCH_MIN, start:end] = 0
    vibrato_rolls = torch.clamp(vibrato_rolls, min=0, max=1)
  return vibrato_rolls
    

'''
For ANALYSIS 
'''

def custom_formatter(y, pos):
  return f"{int(y + PITCH_MIN)}"

def get_piano_roll(pm):
  #plt.figure(figsize=(12, 6))
  plt.xlabel('Time')
  plt.ylabel('Pitch')
  
  plt.imshow(pm, aspect='auto', origin='lower')

  plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_formatter))
  plt.gca().yaxis.set_major_locator(MultipleLocator(2))
  
def plot_midi_mel(note_seq, mel):
  plt.figure(figsize=(12, 6))
  plt.subplot(2, 1, 1)
  get_piano_roll(note_seq)
  plt.subplot(2, 1, 2)
  plt.imshow(mel, aspect='auto', origin='lower')
  plt.tight_layout()
  
