# Quantized Densely Connected U-Nets for Efficient Landmark Localization

## Overview
Traditional random augmentation has two limitations. It doesn't consider the individual difference of training samples when doing augmentation. And it is also independent of the training status of the target network. To tackle these problems, we design an agent to learn more effective data augmentation. 
<p align="center"><img src="figures/overview-1.jpg" alt="Adversarial Data Augmentation in Human Pose Estimation" width="400"></p>
We model the training process as an adversarial learning problem. The agent (generator), conditioning on the individual samples and network status, tries to generate ''hard'' augmentations for the target network. The target network (discriminator), on the other hand, tries to learn better from the augmentations.

<p align="center"><img src="figures/overview-2.jpg" alt="Adversarial Data Augmentation in Human Pose Estimation" width="600"></p>

### Prerequisites

This package has the following requirements:

* `Python 2.7`
* `Pytorch 0.4.0`

### Training

```
python cu-net.py --gpu_id 0 --exp_id cu-net-2 --is_train true --bs 24
```

### Validation

```
python cu-net.py --gpu_id 0 --exp_id cu-net-2 --is_train false --bs 24 --resume_prefix your_pretrained_model.pth.tar
```

## Citation
If you find this code useful in your research, please consider citing:

```
@inproceedings{tang2018quantized,
  title={Quantized densely connected U-Nets for efficient landmark localization},
  author={Tang, Zhiqiang and Peng, Xi and Geng, Shijie and Wu, Lingfei and Zhang, Shaoting and Metaxas, Dimitris},
  booktitle={ECCV},
  year={2018}
}
@inproceedings{tang2018cu,
  title={CU-Net: Coupled U-Nets},
  author={Tang, Zhiqiang and Peng, Xi and Geng, Shijie and Zhu, Yizhe and Metaxas, Dimitris},
  booktitle={BMVC},
  year={2018}
}
```

