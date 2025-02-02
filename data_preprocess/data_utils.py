import re 
import torchaudio
import pretty_midi
from pathlib import Path

def get_performer(pth):
  pth = Path(pth)
  split_pth = (pth.stem.split('_'))
  return split_pth[2]

def sort_files_by_opus_composer(file_paths):
  def extract_details(file_path):
    # Use regex to extract the Opus number and Composer name
    match = re.search(r'([A-Za-z]+)_Op(\d+)-(\d+)_([A-Za-z]+).+-(\d+)-(\d+)', file_path.name)
    if match:
      
      opus_number = int(match.group(3))
      composer_name = match.group(4)
      start_number = int(match.group(5))  # 시작 번호
      return (opus_number, composer_name, start_number)
    return (0, "", 0)
  
  return sorted(file_paths, key=extract_details)

def get_file_pth(mid_pth, composers):
  all_midi_pth = []
  all_audio_pth = []
  all_perform_pth = []
  
  for composer in composers:
    print(mid_pth + '/' + composer)
    print(list(Path(mid_pth + '/' + composer).glob('*.mid')))
    midi_pth = Path(mid_pth + '/' + composer).rglob('*.mid')
    midi_pth = list(midi_pth)
    
    print(f'{composer} Total Midi : {len(midi_pth)}')
    
    midi_pth = sort_files_by_opus_composer(midi_pth)
    performer_pth = [get_performer(pth) for pth in midi_pth]
    
    audio_pth  = [m.with_suffix(".wav") for m in midi_pth]

    all_midi_pth.extend(midi_pth)
    all_audio_pth.extend(audio_pth)
    all_perform_pth.extend(performer_pth)
    
  return all_midi_pth, all_audio_pth, all_perform_pth


def get_midi_pth(mid_pth, composers):
  all_midi_pth = []
  all_perform_pth = []
  
  for composer in composers:
    midi_pth = Path(mid_pth + '/' + composer).rglob('*.mid')
    midi_pth = list(midi_pth)
    
    print(f'{composer} Total Midi : {len(midi_pth)}')
    
    midi_pth = sort_files_by_opus_composer(midi_pth)
    performer_pth = [get_performer(pth) for pth in midi_pth]
    
    all_midi_pth.extend(midi_pth)
    all_perform_pth.extend(performer_pth)
    
  return all_midi_pth, all_perform_pth


def load_audio(audio_pth, new_freq=16000):
  y, sr = torchaudio.load(audio_pth)  
  y = y.mean(0)
  
  if sr != new_freq:
    y = torchaudio.functional.resample(y, sr, new_freq)
  return y

def load_midi(midi_path):
  pm = pretty_midi.PrettyMIDI(midi_path)
  return pm