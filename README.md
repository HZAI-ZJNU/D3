# D3
This repo is the official of implementation of "[Attention-modulated frequency-aware pooling via spatial guidance](https://www.sciencedirect.com/science/article/abs/pii/S0925231225001791)".

The implementation code of D3 is located at [here](https://github.com/HZAI-ZJNU/D3/blob/main/mmpretrain/models/pools/D3.py).

## Running

### Install

We implement D3 using `MMPretrain V1.2.0`, `MMDetection V3.3.0`, `MMSegmentation V1.2.2` and `MMCV V2.1.0`.  
We train and test our models under `python=3.10`, `pytorch=2.1.1`, `cuda=11.8`.


```shell
# First, create a virtual environment and activate it.
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install -e .
```

```shell
# Next, install and activate the virtual environment.
pip install virtualenv
cd D3
virtualenv venv
source venv/bin/activate
```

### Data preparation

The ImageNet dataset should be prepared as follows:

```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

### Training
Our work employs 8X GPUs for training on classification and object detection.     
Here is an example: train D3 + ResNet-50 with an single GPU:
```shell
python ./tools/train.py work_dirs/resnet50_8xb32_in1k_D3.py --work-dir path_to_exp --amp 
```

Here is an example: train D3 + ResNet-50 with an multiple GPUs:
```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PORT=54321
bash ./tools/dist_train.sh work_dirs/resnet50_8xb32_in1k_D3.py 8 --work-dir path_to_exp --amp 
```

### Testing
Test D3 + ResNet-50 with an single GPU:
```shell
python .tools/test.py work_dirs/resnet50_8xb32_in1k_D3. path_to_checkpoint --work-dir path_to_exp
```

## Results
We will open source the relevant model weights later.

## Acknowledgement
The code in this repository is developed based on the [MMPretrain](https://github.com/open-mmlab/mmpretrain). Furthermore, the detection and segmentation tasks involved in this work are implemented based on the [MMDetection](https://github.com/open-mmlab/mmdetection) and [Ultralytics](https://github.com/ultralytics/ultralytics).

## Cite D3
If you find this repository useful, please use the following BibTeX entry for citation.
```latex
@article{si2025attention,
  title={Attention-modulated frequency-aware pooling via spatial guidance},
  author={Si, Yunzhong and Xu, Huiying and Zhu, Xinzhong and Liu, Rihao and Li, Hongbo},
  journal={Neurocomputing},
  pages={129507},
  year={2025},
  publisher={Elsevier}
}
```

## Concat
If you have any questions, please feel free to contact the authors.

Yunzhong Si: 
[siyunzhong@zjnu.edu.cn](mailto:iyunzhong@zjnu.edu.cn)
