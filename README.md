# Single Image Dehazing: AOD-Net vs. FFA-Net

This project compares two deep-learning models for single-image dehazing:

- **AOD-Net** — All-in-One Dehazing Network
- **FFA-Net** — Feature Fusion Attention Network

The models were evaluated on five different indoor hazy images from the SOTS/RESIDE dataset. The evaluation includes visual comparisons, PSNR, SSIM, and average runtime measurements.

---

## Project Information

- **Student:** Hala Sbea
- **Student ID:** 322732140
- **Project Topic:** Single Image Dehazing
- **Models:** AOD-Net and FFA-Net
- **Programming Language:** Python
- **Framework:** PyTorch

---

## Project Objectives

The main goals of this project are:

1. Run two pretrained image-dehazing models.
2. Compare their output images visually.
3. Measure image quality using PSNR and SSIM.
4. Compare the average inference runtime.
5. Analyze the trade-off between image quality and execution speed.

---

## Models

### AOD-Net

AOD-Net is a lightweight end-to-end convolutional neural network for image dehazing. It combines atmospheric-light estimation and transmission-map estimation into a unified model.

Original paper:

> B. Li, X. Peng, Z. Wang, J. Xu, and D. Feng,  
> “AOD-Net: All-in-One Dehazing Network,” ICCV, 2017.

PyTorch implementation used in this project:

```text
https://github.com/walsvid/AOD-Net-PyTorch