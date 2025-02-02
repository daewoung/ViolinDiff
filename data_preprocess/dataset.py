import re 
import random

from pathlib import Path
from tqdm.auto import tqdm

from .data_utils import *
from .midi_process import chunk_midi_roll

class ViolinDataset():
  def __init__(self, 
              midi_pth, 
              composers=['Kayser', 'Paganini', 'Wohlfahrt'],
              segment_size = 5.12,
              on_memory = True,
              sample_rate = 16000,
              hop_length = 320,
              mel_time_bins = 256,
              audio_normalize = False):
    
    self.segment_size = segment_size
    self.hop_length = hop_length
    self.mel_time_bins = mel_time_bins

    self.per_midi_time = (hop_length / sample_rate)
    self.audio_normalize = audio_normalize
    self.on_memory = on_memory

    self.sr = sample_rate
    self.midi_pth, self.audio_pth, self.perform = get_file_pth(midi_pth, composers)
    assert len(self.midi_pth) == len(self.audio_pth), 'Number of midi and audio files should be same'

    unique_performers = sorted(set(self.perform))
    self.perform = {idx: name for idx, name in enumerate(unique_performers)}
    self.rev_perform_dict = {name: idx for idx, name in enumerate(unique_performers)}    

    if on_memory:
      print('Loading all data to memory')
      self.audio_list = [load_audio(pth, sample_rate) for pth in tqdm(self.audio_pth)]
      print('All audio loaded to memory')
      self.midi_list = [load_midi(str(pth)) for pth in tqdm(self.midi_pth)]
      print('All midi loaded to memory')
      self.midi_end_time_list = [midi.get_end_time() for midi in self.midi_list]
      print('All midi end time loaded to memory')
      self.midi_roll_list = [self.preprocess_midi(midi) for midi in (tqdm(self.midi_list))]
      print('All data loaded to memory')

  def __len__(self):
    return len(self.midi_pth)
  
  def preprocess_midi(self, midi_data):
    max_end_time = midi_data.get_end_time()
    max_mel_length = int(max_end_time / self.per_midi_time) + 2
    
    all_time_midi_roll = chunk_midi_roll(midi_data, 0, max_end_time, self.sr, self.hop_length, max_mel_length)
    return all_time_midi_roll
  
  def audio_normalizing(self, audio):
    audio_max = audio.max()
    audio_min = audio.min()
    audio = (audio - audio_min) / (audio_max - audio_min) * 2 - 1  # Normalize to [-1, 1]
    return audio
  
  def random_chunk(self, audio, midi_end_time):
    max_audio_start = int(midi_end_time - self.segment_size) ## sec
    audio_start_sec = random.randint(0, max_audio_start - 5)
    audio_end_sec = round(audio_start_sec + self.segment_size, 2)    
    audio_start = int(audio_start_sec * self.sr)
    audio_end = audio_start + int(self.segment_size * self.sr)

    return audio[audio_start:audio_end], audio_start_sec, audio_end_sec

  def __getitem__(self, idx):
    if self.on_memory:
      audio = self.audio_list[idx]
      midi = self.midi_list[idx]
      chunk_audio, audio_start, audio_end = self.random_chunk(audio, self.midi_end_time_list[idx])
      mel_start_idx = int(audio_start / self.per_midi_time)
      mel_end_idx = mel_start_idx + self.mel_time_bins
      pitch, onset, bend, velocity, offset = self.midi_roll_list[idx]

    else:
      audio = load_audio(self.audio_pth[idx], self.sr)
      midi = load_midi(self.midi_pth[idx])
      chunk_audio, audio_start, audio_end = self.random_chunk(audio, midi.get_end_time())  
      chunk_pitch, chunk_onset, chunk_bend, chunk_velocity, chunk_offset = chunk_midi_roll(midi, audio_start, audio_end, self.sr, self.hop_length, self.mel_time_bins)

    performer = self.rev_perform_dict[self.perform[idx]]
    chunk_pitch = pitch[:, mel_start_idx:mel_end_idx]
    chunk_onset = onset[:, mel_start_idx:mel_end_idx]
    chunk_bend = bend[:, mel_start_idx:mel_end_idx]
    chunk_velocity = velocity[:, mel_start_idx:mel_end_idx]
    chunk_offset = offset[:, mel_start_idx:mel_end_idx]
    
    if self.audio_normalize:
      chunk_audio = self.audio_normalizing(chunk_audio)
    
    return chunk_audio, chunk_pitch, chunk_onset, chunk_bend, chunk_velocity, performer, chunk_offset

    
class ValidViolinDataset(ViolinDataset):
  def __init__(self, 
              midi_pth, 
              valid_pth,
              composers=['Kayser', 'Paganini', 'Wohlfahrt'],
              segment_size = 5.12,
              on_memory = True,
              sample_rate = 16000,
              hop_length = 320,
              mel_time_bins = 256,
              audio_normalize = False):
    
    
    super().__init__(midi_pth, composers, segment_size, 
                    on_memory=False, sample_rate = sample_rate,
                    hop_length = hop_length,
                    mel_time_bins = mel_time_bins,
                    audio_normalize = audio_normalize)
    
    print('Loading validation data')
    self.audio_normalize = audio_normalize
    
    self.midi_pth, self.audio_pth, self.perform = get_file_pth(valid_pth, composers)
    
    
    print('Loading all data to memory')
    self.audio_list = [load_audio(pth, sample_rate) for pth in tqdm(self.audio_pth)]
    print('All audio loaded to memory')
    self.midi_list = [load_midi(str(pth)) for pth in tqdm(self.midi_pth)]
    print('All midi loaded to memory')
    self.midi_roll_list = [self.preprocess_midi(midi) for midi in (tqdm(self.midi_list))]
    print('All data loaded to memory')
    self.chunk_audio, self.chunk_pitch, self.chunk_onset, self.chunk_bend_list, self.chunk_vel_list, self.chunk_perf, self.chunk_offset, self.chunk_audio_pth, self.chunk_sec = self.chunk_all_audio()

  def chunk_all_audio(self):
    all_chunk_audio = []
    all_chunk_pitch = []
    all_chunk_onset = []
    all_chunk_bend = []
    all_chunk_vel = []
    all_chunk_perf = []
    all_chunk_offset = []
    
    all_chunk_audio_pth = []
    all_chunk_sec = []
    
    for i in tqdm(range(len(self.audio_list))):
      audio = self.audio_list[i]
      audio_pth = self.audio_pth[i]
      midi_roll = self.midi_roll_list[i]
      midi_data = self.midi_list[i]
      perforormer = self.perform[i]
        
      chunk_audio, chunk_pitch, chunk_onset, chunk_bend, chunk_velocity, chunk_perform_list, chunk_offset_list, chunk_sec, chunk_audio_pth= self.chunk_data_roll(audio, audio_pth, midi_data, midi_roll, perforormer, self.segment_size)
        
      all_chunk_audio.extend(chunk_audio)
      all_chunk_pitch.extend(chunk_pitch)
      all_chunk_onset.extend(chunk_onset)
      all_chunk_bend.extend(chunk_bend)
      all_chunk_perf.extend(chunk_perform_list)
      all_chunk_audio_pth.extend(chunk_audio_pth)
      all_chunk_sec.extend(chunk_sec)
      all_chunk_vel.extend(chunk_velocity)
      all_chunk_offset.extend(chunk_offset_list)
    return all_chunk_audio, all_chunk_pitch, all_chunk_onset, all_chunk_bend, all_chunk_vel, all_chunk_perf, all_chunk_offset, all_chunk_audio_pth, all_chunk_sec
  

  def chunk_data_roll(self, waveform, audio_pth, midi_data, mid_roll, perforormer, segment_size):
    chunk_audio_list = []
    chunk_pitch_list = []
    chunk_onset_list = []
    chunk_bend_list = []
    chunk_perform_list = []
    chunk_offset_list = [] 
    chunk_audio_pth = []
    chunk_midi_pth = []
    chunk_velocity_list = []
    chunk_sec = []
    
    segment_length = int(segment_size * self.sr)

    end_midi = midi_data.get_end_time()

    num_chunks = int(end_midi // self.segment_size)
    
    pitch, onset, bend, velocity, offset = mid_roll
    
    for i in range(num_chunks):
        chunk_audio = waveform[i*segment_length:(i+1)*segment_length]
        if self.audio_normalize:
          chunk_audio = self.audio_normalizing(chunk_audio)

        start_time = (i * segment_length) / self.sr
        end_time = ((i+1) * segment_length) / self.sr
        
        mel_start_idx = int(start_time / self.per_midi_time)
        mel_end_idx = mel_start_idx + self.mel_time_bins
        chunk_pitch = pitch[:, mel_start_idx:mel_end_idx]
      
        chunk_onset = onset[:, mel_start_idx:mel_end_idx]
        chunk_bend = bend[:, mel_start_idx:mel_end_idx]
        chunk_veloicty = velocity[:, mel_start_idx:mel_end_idx]
        chunk_offset = offset[:, mel_start_idx:mel_end_idx]
        
        chunk_audio_list.append(chunk_audio)
        chunk_pitch_list.append(chunk_pitch)
        chunk_onset_list.append(chunk_onset)
        chunk_bend_list.append(chunk_bend)
        chunk_velocity_list.append(chunk_veloicty)
        chunk_perform_list.append(perforormer)
        chunk_offset_list.append(chunk_offset)
        chunk_sec.append((start_time, end_time))
        chunk_audio_pth.append(audio_pth)

    return chunk_audio_list, chunk_pitch_list, chunk_onset_list, chunk_bend_list, chunk_velocity_list, chunk_perform_list, chunk_offset_list, chunk_sec, chunk_audio_pth

  def __len__(self):
    return len(self.chunk_audio)
  
  def __getitem__(self, idx):
    performer = self.chunk_perf[idx]
    performer = self.rev_perform_dict[performer]
    
    return self.chunk_audio[idx], self.chunk_pitch[idx], self.chunk_onset[idx], self.chunk_bend_list[idx], self.chunk_vel_list[idx], performer, self.chunk_offset[idx]