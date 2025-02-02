# ViolinDiff

Official **PyTorch implementation** of  **"ViolinDiff: Enhancing Expressive Violin Synthesis with Pitch Bend Conditioning"**
**Keywords**: Violin Synthesis, Neural Audio Synthesis, Pitch Bend Modeling, Expressive Performance, Diffusion Models



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
1. **Bend Module**  
   - Predict the **pitch bend roll** from MIDI.

2. **Synth (Synthesis) Module**  
   - Converts pitch and bend information, along with other performance controls, into the final violin audio signal.  


This model is provided for non-commercial, research use only.
