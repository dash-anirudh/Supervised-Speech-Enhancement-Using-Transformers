# Supervised Speech Enhancement using Transformers

This project implements supervised speech denoising using deep learning. It compares:
- A **CNN-based convolutional autoencoder**
- A hybrid **CNN + Transformer** architecture

The models operate in the **time-frequency (STFT) domain**, applying learnable masks on spectrograms to recover clean speech from noisy input.

---

## Motivation

Traditional methods like spectral subtraction and Wiener filtering fail in low-SNR environments. Deep learning allows us to learn robust, nonlinear mappings from noisy to clean speech, with recent advances in transformers offering global attention mechanisms for long-range modeling.

## Dataset

**[Valentini Speech Enhancement Dataset](https://datashare.ed.ac.uk/handle/10283/1942)**  
- 10,000+ clean-noisy speech pairs (16 kHz)
- Contains various noise types: white noise, babble, street noise, etc.


