import torch
import pretty_midi
import numpy as np

from data_preprocess.midi_process import chunk_midi_roll

# def long_midi_processor(mel_config, midi_pth, overlap_seq_len):
#   sr = mel_config.sample_rate
#   hop_length = mel_config.hop_length
#   segment_mel_length = mel_config.mel_len
#   overlap_seq_len = overlap_seq_len

#   midi_data = pretty_midi.PrettyMIDI(midi_pth)
  
#   end_time = midi_data.get_end_time()
#   total_mel_length = (end_time * sr) / hop_length
  
#   if int(total_mel_length) != total_mel_length:
#     total_mel_length = int(total_mel_length) + 1
#   else:
#     total_mel_length = int(total_mel_length) + 1 

#   pitch_data, onset_data, bend_data, velocity_data, offset_data = chunk_midi_roll(midi_data, 0, end_time, 16000, 320, total_mel_length)
#   condition_data = torch.stack([pitch_data, onset_data, bend_data, velocity_data, offset_data])

#   total_loop = (1 + (total_mel_length - segment_mel_length) / (segment_mel_length - overlap_seq_len))
#   if int(total_loop) != total_loop:
#     total_loop = int(total_loop) + 1
#   else:
#     total_loop = int(total_loop)
    
#   overlaping_list = []

#   for i in range(total_loop):
#     if i == 0:
#       start_idx = i * segment_mel_length
#       end_idx = start_idx + segment_mel_length
#     else:
#       start_idx = end_idx - overlap_seq_len
#       end_idx = min(start_idx + segment_mel_length, total_mel_length)
      
#     if condition_data[:, :, start_idx:end_idx].shape[-1] != segment_mel_length:
#       padding_length = segment_mel_length - (end_idx - start_idx) 
#       feauture, pitch, _ = condition_data.shape
#       pad_data = torch.zeros(feauture, pitch, padding_length)
#       overlaping_list.append(torch.cat([condition_data[:, :, start_idx:end_idx], pad_data], dim=-1))      
#     else:
#       overlaping_list.append(condition_data[:, :, start_idx:end_idx])
    
#   condition = torch.stack(overlaping_list)
#   pitch_data, onset_data, bend_data, velocity_data, offset_data = condition[:, 0, :, :], condition[:, 1, :, :], condition[:, 2, :, :], condition[:, 3, :, :], condition[:, 4, :, :]  
  
#   return pitch_data, onset_data, bend_data, velocity_data, offset_data


def split_midi_by_pitch(input_pth):
    pm = pretty_midi.PrettyMIDI(input_pth)

    instruments = {}

    for instrument in pm.instruments:
        for note in instrument.notes:
            pitch = note.pitch
            if pitch not in instruments:
                instruments[pitch] = pretty_midi.Instrument(program=40)  # 새로운 인스트루먼트 생성
            instruments[pitch].notes.append(note)

    new_pm = pretty_midi.PrettyMIDI()

    for pitch, instrument in instruments.items():
        new_pm.instruments.append(instrument)

    return new_pm



def long_midi_processor(mel_config, midi_pth, overlap_seq_len, pred_bend = None, return_gt_bend = False):
  sr = mel_config.sample_rate
  hop_length = mel_config.hop_length
  segment_mel_length = mel_config.mel_len
  overlap_seq_len = overlap_seq_len

  midi_data = split_midi_by_pitch(midi_pth)
  end_time = midi_data.get_end_time()
  total_mel_length = (end_time * sr) / hop_length + 1
  
  if int(total_mel_length) != total_mel_length:
    total_mel_length = int(total_mel_length) + 1
  total_mel_length = int(total_mel_length)

  pitch_data, onset_data, bend_data, velocity_data, offset_data, pitch_dict, _ = chunk_midi_roll(midi_data, 0, end_time, 16000, 320, int(total_mel_length), True)
  gt_bend = bend_data.clone()
  org_pitch_data = pitch_data.clone()
  if pred_bend is not None:
    bend_data = pred_bend[:, :gt_bend.shape[-1]]
  
  condition_data = torch.stack([pitch_data, onset_data, bend_data, velocity_data, offset_data])
  total_loop = int(np.ceil((total_mel_length - segment_mel_length) / (segment_mel_length - overlap_seq_len) + 1))

  overlaping_list = []
  for i in range(total_loop):
      start_idx = max(0, i * (segment_mel_length - overlap_seq_len))
      end_idx = min(start_idx + segment_mel_length, total_mel_length)
      
      segment = condition_data[:, :, start_idx:end_idx]
      if segment.shape[-1] < segment_mel_length:
          padding_length = segment_mel_length - segment.shape[-1]
          pad_data = torch.zeros((condition_data.shape[0], condition_data.shape[1], padding_length))
          segment = torch.cat([segment, pad_data], dim=-1)
      
      overlaping_list.append(segment)

  condition = torch.stack(overlaping_list)
  pitch_data, onset_data, bend_data, velocity_data, offset_data = tuple(condition[:, k, :, :] for k in range(5))

  
  if return_gt_bend:
    return pitch_data, onset_data, bend_data, velocity_data, offset_data, pitch_dict, gt_bend
  else:
    return pitch_data, onset_data, bend_data, velocity_data, offset_data, pitch_dict, org_pitch_data
