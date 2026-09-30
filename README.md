<hr>
<h1 align="center">
  D3Bridge <br>
  <sub>Dynamic Dual-Domain Diffusion Bridges for Cross-Modal Brain MRI Synthesis</sub>
</h1>

<div align="center">
  <a href="https://github.com/yufengxiecs" target="_blank">Yufeng&nbsp;Xie</a> &ensp; <b>&middot;</b> &ensp;
  Sanuwani&nbsp;Dayarathna &ensp; <b>&middot;</b> &ensp;
  <a href="mailto:zhaolin.chen@monash.edu" target="_blank">Zhaolin&nbsp;Chen</a>

  <span></span>

  Department of Data Science &amp; AI, Monash University <br>
</div>
<hr>

Official PyTorch implementation of **D3Bridge**, a Dynamic Dual-Domain Diffusion Bridge for cross-modal brain MRI synthesis. D3Bridge formulates spatial- and frequency-domain diffusion bridges as two domain-specific experts, and introduces a step-wise dynamic gating that coordinates their predictions and enables cross-domain exchange throughout recursive diffusion refinement. An adaptive fusion network further integrates the complementary representations from the two domains for the final synthesis.

<p align="center">
  <img src="figures/architecture.png" alt="architecture" style="width: 650px; height: auto;">
</p>


## ⚙️ Installation

This repository has been developed and tested with `CUDA 11.7` and `Python 3.8`. Below commands create a conda environment with required packages. Make sure conda is installed.

```
conda env create --file requirements.yaml
conda activate d3bridge
```

## 🗂️ Prepare dataset

The default data set class `NumpyDataset` requires the following folder structure to organize the data set. Modalities (T1, T2, etc.) are separated by folders, splits (train, val, test) are organized as subfolders which include 2D images: `slice_0.npy`, `slice_1.npy`, ... To use your custom data set class, set `dataset_class` to your own implementation in `dataset.py` by inheriting from the `BaseDataset` class.

> Images should be scaled to have pixel values in the range [0,1].

```
<dataset>/
├── <modality_a>/
│   ├── train/
│   │   ├── slice_0.npy
│   │   ├── slice_1.npy
│   │   └── ...
│   ├── test/
│   │   ├── slice_0.npy
│   │   └── ...
│   └── val/
│       ├── slice_0.npy
│       └── ...
├── <modality_b>/
│   ├── train/
│   ├── test/
│   └── val/
├── ...
  
```

For BraTS 2021, a preparation script is provided in [`datasets/create_brats_dataset.py`](datasets/create_brats_dataset.py) to build the above structure from the raw NIfTI files.

## 🏃 Training

Run the following command to start/resume training. Model checkpoints are saved under `logs/$EXP_NAME/version_x/checkpoints` directory, and sample validation images are saved under `logs/$EXP_NAME/version_x/val_samples`. The script supports both single and multi-GPU training. By default, it runs on a single GPU. To enable multi-GPU training, set `--trainer.devices` argument to the list of devices, e.g. `0,1,2,3`.

```
python main.py fit \
    --config config.yaml \
    --trainer.logger.name $EXP_NAME \
    --data.dataset_dir $DATA_DIR \
    --data.source_modality $SOURCE \
    --data.target_modality $TARGET \
    --data.train_batch_size $BS_TRAIN \
    --data.val_batch_size $BS_VAL \
    [--trainer.max_epoch $N_EPOCHS] \
    [--ckpt_path $CKPT_PATH] \
    [--trainer.devices $DEVICES]

```

### Argument descriptions

| Argument                    | Description                                                                                                                    |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `--config`                  | Config file path.                                                                                                              |
| `--trainer.logger.name`     | Experiment name.                                                                                                               |
| `--data.dataset_dir`        | Data set directory.                                                                                                            |
| `--data.source_modality`    | Source modality, e.g. 'T1', 'T2', 'PD'. Should match the folder name for that modality.                                        |
| `--data.train_batch_size`   | Train set batch size.                                                                                                          |
| `--data.val_batch_size`     | Validation set batch size.                                                                                                     |
| `--trainer.max_epoch`       | [Optional] Number of training epochs (default: 50).                                                                            |
| `--ckpt_path`               | [Optional] Model checkpoint path to resume training.                                                                           |
| `--trainer.devices`         | [Optional] Device or list of devices. For multi-GPU set to the list of device ids, e.g `0,1,2,3` (default: `[0]`).             |


## 🧪 Testing

Run the following command to start testing. The predicted images are saved under `logs/$EXP_NAME/version_x/test_samples` directory. By default, the script runs on a single GPU. To enable multi-GPU testing, set `--trainer.devices` argument to the list of devices, e.g. `0,1,2,3`.

```
python main.py test \
    --config config.yaml \
    --data.dataset_dir $DATA_DIR \
    --data.source_modality $SOURCE \
    --data.target_modality $TARGET \
    --data.test_batch_size $BS_TEST \
    --ckpt_path $CKPT_PATH
```

### Argument descriptions

Some arguments are common to both training and testing and are not listed here. For details on those arguments, please refer to the training section.

| Argument                    | Description                                |
|-----------------------------|--------------------------------------------|
| `--data.test_batch_size`    | Test set batch size.                       |
| `--ckpt_path`               | Model checkpoint path.                     |

## 🦁 Results

Quantitative comparison (PSNR (dB) / SSIM (%)) across eight cross-modal brain MRI synthesis tasks on the IXI and BraTS 2021 datasets:

| Dataset | Task       | PSNR / SSIM        |
|---------|------------|--------------------|
| IXI     | T1→T2      | 29.47 / 93.46      |
| IXI     | T2→T1      | 29.06 / 92.99      |
| IXI     | T1→PD      | 29.92 / 93.93      |
| IXI     | PD→T1      | 30.28 / 94.95      |
| BraTS   | T1→T2      | 26.87 / 91.10      |
| BraTS   | T2→T1      | 27.29 / 92.60      |
| BraTS   | T2→FLAIR   | 26.00 / 89.57      |
| BraTS   | FLAIR→T2   | 26.91 / 90.59      |

## ✒️ Citation

You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.

```
@inproceedings{xie2027d3bridge,
  title={D3Bridge: Dynamic Dual-Domain Diffusion Bridges for Cross-Modal Brain MRI Synthesis},
  author={Xie, Yufeng and Dayarathna, Sanuwani and Chen, Zhaolin},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2027}
}
```


## 🙏 Acknowledgements

This project is built on top of [SelfRDB](https://github.com/icon-lab/SelfRDB). We thank the authors for releasing their code. The backbone network implementations under [`backbones/`](backbones/) are adapted from [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch), [Denoising-Diffusion-GAN](https://github.com/NVlabs/Denoising-Diffusion-GAN), and [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch); see the license headers in individual files for details.


<hr>

Copyright © 2027, Yufeng Xie, Monash University.
