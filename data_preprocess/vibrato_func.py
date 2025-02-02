import numpy as np
# from evaluation_funcition import get_midi_notes
from scipy.interpolate import interp1d


def pitch_to_freq(pitch):
    return 440.0 * 2 ** ((pitch - 69) / 12.0)
  
    
def freq_to_pitch(freq):
    return 69 + 12 * np.log2(freq / 440.0)

def get_normal_window(t, alpha=0.5):
    # Cosine 윈도우 생성
    w = 0.5 * (1 - np.cos(2 * np.pi * t / alpha))
    
    # 마스크 생성
    mask = (t > 0.5 * alpha) & (t < (1 - 0.5 * alpha))
    
    # 마스크를 더하여 강조된 부분 생성
    w = np.clip(w + mask, 0, 1)
    return w
  
def get_vibrato(midi_pitch, f0, target_len = 1000, vibrato_rate_min = 3, vibrato_rate_max = 9, sampling_interval = 0.004, min_note_length = 50):

    zero_indicies = np.where(f0 == 0)[0]
    
    midi_pitch_f0 = freq_to_pitch(f0)
    pitch_deviation = midi_pitch - midi_pitch_f0
    pitch_deviation = np.where(abs(pitch_deviation) > 2.0, 0.0, pitch_deviation)

    # pitch_deviation = f0
    
    note_mask = np.concatenate([np.ones(len(pitch_deviation)), np.zeros(target_len - len(pitch_deviation))])
    note_mask[zero_indicies] = 0
    
    pad_pitch_deviation = np.pad(pitch_deviation, (0, target_len - len(pitch_deviation)), 'constant', constant_values=(0, 0))
    pitch_deviation_mean = np.mean(pad_pitch_deviation[note_mask == 1])


    pad_pitch_deviation = (pad_pitch_deviation - pitch_deviation_mean) * note_mask


    each_note_idx = np.cumsum(note_mask) * (note_mask != 0)
    each_note_len = np.sum(note_mask)
    each_note_time_ratio = each_note_idx / each_note_len
    # print(note_mask)
    window = get_normal_window(each_note_time_ratio)

    pitch_deviation_masked = pad_pitch_deviation * window
    
    f = np.linspace(0, int(1 / sampling_interval), 1000)
    # print(f)
    s_vibrato = np.abs(np.fft.rfft(pitch_deviation_masked))
    vibrato_rate_idx = np.argmax(s_vibrato)
    s_vibrato = s_vibrato / each_note_len
    vibrato_rate_idx = np.argmax(s_vibrato)

    vibrato_rate = f[vibrato_rate_idx]
    vibrato_extend = s_vibrato[vibrato_rate_idx]

    # nan을 0으로 대체
    if np.isnan(vibrato_extend):
        vibrato_extend = 0

    vibrato_mask = (vibrato_rate >= vibrato_rate_min) & (vibrato_rate <= vibrato_rate_max)

    # 노트 길이가 최소 길이보다 큰지 확인
    vibrato_mask &= (each_note_len > min_note_length)

    # 진동수가 한 주기 이상인지 확인
    more_than_one_cycle_mask = vibrato_rate > (1. / (each_note_len * sampling_interval))
    vibrato_mask &= more_than_one_cycle_mask

    if not vibrato_mask:
        vibrato_rate = 0
        vibrato_extend = 0
        
    vibrato_rate, vibrato_extend
    if vibrato_extend < 0:
      print(vibrato_extend)
    final_vibrato = vibrato_extend * 10
    return final_vibrato, s_vibrato, pad_pitch_deviation



def zero_indicies_change(f0):
  zero_indices = np.where(f0 == 0)[0]
  for idx in zero_indices:
    if idx < len(f0) - 1:
        if f0[idx + 1] != 0:
            f0[idx] = f0[idx + 1]
        else:
            # 다음 값도 0인 경우 연쇄적으로 다음 값 찾기
            next_non_zero_idx = idx + 1
            while next_non_zero_idx < len(f0) and f0[next_non_zero_idx] == 0:
                next_non_zero_idx += 1
            if next_non_zero_idx < len(f0):
                f0[idx] = f0[next_non_zero_idx]
            else:
                f0[idx] = f0[idx - 1]  # 전 값으로 교체
    else:
        f0[idx] = f0[idx - 1]  # 전 값으로 교체
  return f0


def find_nearest_index(array, value, point='start'):
    array = np.asarray(array)
    if point == 'start':
      valid_indices = np.where(array <= value)[0]
    elif point == 'end':
      valid_indices = np.where(array >= value)[0]    
      
    valid_array = array[valid_indices]
    idx_within_valid = (np.abs(valid_array - value)).argmin()
    return valid_indices[idx_within_valid]
  
  

def get_vibrato_e2e(f0_df, note_pth, target='freq'):
  
  note_df = get_midi_notes(note_pth)

  new_df = note_df.copy()
  new_df['vibrato'] = 0.0

  for idx in range(len(note_df)):
    start_sec = np.round(note_df.loc[idx, 'start'], 3)
    end_sec = np.round(note_df.loc[idx, 'end'], 3)
    midi_note = note_df.loc[idx, 'midi_note']
    dur = np.round(note_df.loc[idx, 'dur'], 3)
    
    if dur < 0.2:
      new_df.loc[idx, 'vibrato'] = 0.0
      continue
    
    start_idx = find_nearest_index(f0_df['time'].values, start_sec, point='start')
    end_idx = find_nearest_index(f0_df['time'].values, end_sec, point='end') + 1

    f0 = f0_df[target].values[start_idx:end_idx]
    f0_time = f0_df['time'].values[start_idx:end_idx]

    zero_indicies = np.where(f0 == 0)[0]
    f0 = zero_indicies_change(f0)
    target_time = np.linspace(start_sec, end_sec, int(dur / 0.004))
    f = interp1d(f0_time, f0, kind='linear', fill_value='extrapolate')

    interpol_f0 = f(target_time)
    
    final_vibrato, s_vibrato, pad_pitch_deviation = get_vibrato(midi_note, interpol_f0, sampling_interval=0.004, min_note_length=50)
    
    
    new_df.loc[idx, 'vibrato'] = final_vibrato
    
  return new_df