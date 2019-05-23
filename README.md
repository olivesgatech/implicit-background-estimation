# Implicit Background Estimation for Semantic Segmentation

Charles Lehman, Dogancan Temel, [Ghassan AlRegib](http://www.ghassanalregib.com)

This repository includes a stand-alone package for implementing Implicit Background Estimation (IBE) on existing PyTorch semantic segmentation models. Further information is available from our paper '**Implicit Background Estimation for Semantic Segmentation**' that was accepted to the International Conference on Image Processing, 22-25 September 2019 in Taipei, Taiwan. [preprint](https://arxiv.org/abs/xxxx)

--------

<p align="center">
<img src="https://github.com/olivesgatech/implicit-background-estimation/raw/master/resources/semseg.png" alt="Semantic Segmentation">
</p>


## Abstract
Scene understanding and semantic segmentation are at the core of many computer vision tasks, many of which, involve interacting with humans in potentially dangerous ways.  It is therefore paramount that techniques for principled design of robust models be developed.  In this paper, we provide analytic and empirical evidence that correcting potentially errant non-distinct mappings that result from the softmax function can result in improving robustness characteristics on a state-of-the-art semantic segmentation model with minimal impact to performance and minimal changes to the code base.

## Getting Started
Install dependencies and the IBE package.

```bash
pip install git+https:/github.com/olivesgatech/implicit-background-estimation.git
```

## Implicit Background Usage

### To use ImplicitBackground as a stand-alone nn.Module

VOC has 20 classes non-background classes the ImplicitBackground nn.Module extends the number of classes by one.

```python
import torch.nn as nn
from IBE import ImplicitBackground
from somewhere import SomeSemanticSegmentationModel as sssm

model = nn.Sequential(sssm(num_classes=20),
                      ImplicitBackground(dim=1))
```
--------

### To modify an existing Semantic Segmentation model:

This is a modified excerpt taken from a PyTorch implementation of DeepLabV3+ from [https://github.com/jfzhang95/pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception):

#### Optional: To improve clarity when instantiating the model.

```diff
-  def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
+  def __init__(self, backbone='resnet', output_stride=16, num_non_bg_classes=20,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
-       self.decoder = build_decoder(num_classes, backbone, BatchNorm)
+       self.decoder = build_decoder(num_non_bg_classes, backbone, BatchNorm)
```

#### Required

```diff
def forward(self, input):
  x, low_level_feat = self.backbone(input)
  x = self.aspp(x)
  x = self.decoder(x, low_level_feat)
- x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
+ _x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
+ non_bg_x = torch.logsumexp(_x, dim=1, keepdim=True)
+ x = torch.cat([-non_bg_x, _x],1)
  return x
```

--------

### Under the hood:

<p align="center">
<img src="https://github.com/olivesgatech/implicit-background-estimation/raw/master/resources/ibe.png" alt="Implicit Background Estimation">
</p>

The IBE model predicts background only when all non-background classes are negative.  This is accomplished by employing a negative log sum exponential on the non-background predictions to extend the prediction vectors by one. A more detailed explanation is available in our [paper](https://arxiv.org/abs/xxxx).

## Expected Non-Distinctiveness Metric

<p align="center">
<img src="https://github.com/olivesgatech/clehman31/implicit-background-estimation/raw/master/resources/end.png" alt="Semantic Segmentation">
</p>

```python
from IBE import ExpectedNonDistinctiveness as END

output = model(input)
end = END(output, background_class_index=0)
```

## Citation: 
If you have found our code useful, we kindly ask you to cite our work. You can cite the arXiv preprint for now: 
```tex
@INPROCEEDINGS{Lehman2019, 
author={C. Lehman and D. Temel and G. AIRegib}, 
booktitle={IEEE International Conference on Image Processing (ICIP)}, 
title={Implicit Background Estimation for Semantic Segmentation},
year={2019},}
```
