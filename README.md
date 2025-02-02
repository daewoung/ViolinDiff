# ViolinDiff
This model is provided for non-commercial, research use only.

Official **PyTorch implementation** of  **"ViolinDiff: Enhancing Expressive Violin Synthesis with Pitch Bend Conditioning"**.
**Keywords**: Violin Synthesis, Neural Audio Synthesis, Pitch Bend Modeling, Expressive Performance, Diffusion Models
This work has been accepted at **ICASSP 2025**.



[![arXiv](https://img.shields.io/badge/arXiv-2408.11915-brightgreen.svg?style=flat-square)](https://arxiv.org/pdf/2409.12477)  [![githubio](https://img.shields.io/badge/GitHub.io-Demo_page-blue?logo=Github&style=flat-square)](https://daewoung.github.io/ViolinDiff-Demo/)  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/dawokim/ViolinDiff)

<table>
  <tr>
    <td><img src="./static/model.png" alt="VioliDiff" width="800"/></td>
   </a></td>
  </tr>
</table>


## Overview

This repository provides the official PyTorch codebase for **ViolinDiff**, a diffusion-based model that focuses on generating expressive violin performances via **pitch bend modeling**. 

ViolinDiff is divided into two main modules:
1. **Bend Module** : Predict the **pitch bend roll** from MIDI.

2. **Synth (Synthesis) Module** : Converts pitch and bend information, along with other performance controls, into the final violin audio signal.  

## Running on Google Colab

If you prefer not to install everything locally, you can **run ViolinDiff directly in Google Colab**:

- [**ViolinDiff on Colab**](https://colab.research.google.com/drive/12CpNd3gjGVGJYaALrwYJdvrg7hFOHMy4?usp=sharing)

Just open the link, make sure to **enable GPU** (`Runtime` → `Change runtime type` → `Hardware accelerator: GPU`), and execute the provided cells in order.


## Getting Started

### Installation
1. Clone this repository.
   ```bash
   git clone https://github.com/daewoung/ViolinDiff.git
   cd ViolinDiff
   ```

2. Create a new Conda environment.
   ```bash
   conda create -n VD python=3.10
   conda activate VD
   ```

3. Install [PyTorch](https://pytorch.org/get-started/previous-versions) 
   ```bash
   conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
4. Install other dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Download Pre-trained Models
Pretrained checkpoints (`bend.pt`, `synth.pt`) are available on **Hugging Face**:[dawokim/ViolinDiff](https://huggingface.co/dawokim/ViolinDiff)

### 1) Using Git + Git LFS
Make sure you have [Git LFS](https://git-lfs.github.com/) installed:

   ```bash
   git lfs install
   git clone https://huggingface.co/dawokim/ViolinDiff
   ```

### 2) Using wget
   ```bash
   wget https://huggingface.co/dawokim/ViolinDiff/resolve/main/bend.pt
   wget https://huggingface.co/dawokim/ViolinDiff/resolve/main/synth.pt
   ```


## Inference

We provide a script called `inference.py` to generate violin audio (`.wav`) from a given MIDI file.  
By default, it expects the following arguments:

### Example Usage

  ```bash
    python3 inference.py \
    --synth_pth synth.pt \
    --bend_pth bend.pt \
    --midi_pth example.mid \
    --save_pth example_out.wav \
    --performer 13 \
    --device cuda
  ```

- `--synth_pth`: Path to the **Synth** checkpoint (default: `synth.pt`)
- `--bend_pth`: Path to the **Bend** checkpoint (default: `bend.pt`)
- `--bend_cfg`: CFG scale for the bend model (default: `3.0`)
- `--synth_cfg`: CFG scale for the synth model (default: `1.25`)
- `--midi_pth`: Path to the input MIDI file (default: `thais.mid`)
- `--save_pth`: Path to save the output WAV file (default: `thais.wav`)
- `--performer`: Performer ID (int), default: `0` (currently up to 21 performers supported)
- `--device`: Device to run on (`cuda` or `cpu`), default: `cuda`


## Training

### Data Preparation
- **MIDI Files**: We recommend downloading violin MIDI files from [MUSC_violin](https://github.com/nctamer/MUSC_violin). This repository provides various violin pieces in MIDI format.  
- **Audio Files**: You will need to obtain corresponding audio recordings separately, as they are not provided in the above repo.  
- **Directory Structure**: Organize your data such that each composer (or dataset split) resides in a folder. 
For example:
```text
/data/train/
├── Kayser/
│   ├── piece1.mid
│   ├── piece1.wav
│   ├── piece2.mid
│   ├── piece2.wav
│   └── ...
```
Ensure that each `.mid` file has a matching `.wav` file of the same piece.  

### Model Configuration

All training hyperparameters, file paths, and other settings are defined in the `config/` folder. Each `.yaml` file corresponds to different modules or training configurations (e.g., `synth.yaml`, `bend.yaml`).
  ```bash
    python3 bend_train.py
    python3 synth_train.py
  ```

- **`bend_train.py`**: Trains the **Bend** module (to predict pitch bend envelopes).
- **`synth_train.py`**: Trains the **Synthesis** module (to generate mel spectrogram, conditioned on pitch/bend).


## Citation

If you use **ViolinDiff** in your research, please cite:

```bibtex
@article{kim2024violindiff,
  title={ViolinDiff: Enhancing Expressive Violin Synthesis with Pitch Bend Conditioning},
  author={Kim, Daewoong and Dong, Hao-Wen and Jeong, Dasaem},
  journal={arXiv preprint arXiv:2409.12477},
  year={2024}
}
```

### References

- [DDPM](https://github.com/lucidrains/denoising-diffusion-pytorch)

- [MIDI-DDSP](https://github.com/magenta/midi-ddsp)

