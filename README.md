# Age and Gender Recognition with CBAM-CNN Multi-Task Learning Model

### Requirements

The "requirements.txt" file required to configure the environment is in the folder of the corresponding method.

```
pip install -r requirements.txt
```



### Datasets

[UTKFace](https://susanqq.github.io/UTKFace/)

```
@inproceedings{zhifei2017cvpr,
  title={Age Progression/Regression by Conditional Adversarial Autoencoder},
  author={Zhang, Zhifei, Song, Yang, and Qi, Hairong},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017},
  organization={IEEE}
}
```

```
data
└──UTKFace
    ├── train
    ├── val
    └── test
```

### Loss Functions

```python
# Train
# the data path
parser.add_argument('-d', '--data_root', default='./data',
                         type=str, help='data root')
# sex branch loss
parser.add_argument('-sex_loss', '--sex_loss_function', default='CE',
                        type=str, choices=['CE', 'LDAM','Focal','LogitAdjust'],help='loss function')
# age branch loss
parser.add_argument('-age_loss', '--age_loss_function', default='CE',
                        type=str, choices=['CE', 'LDAM', 'Focal', 'LogitAdjust'], help='loss function')
# model
parser.add_argument('-m', '--model', default='resnet50-cbam',
                         type=str, help='resnet50-cbam or resnet50')
# mini batch size
parser.add_argument('--batch_size', default=32,
                         type=int, help='model train batch size')
```



### Technology

* [Label-Distribution-Aware Margin Loss](https://github.com/kaidic/LDAM-DRW)

* [Focal Loss](https://github.com/clcarwin/focal_loss_pytorch/)
* [Logit Adjustment Loss](https://github.com/FlamieZhu/Balanced-Contrastive-Learning)
* [ResNet](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
