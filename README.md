# Ucolor-Reimplementation

Reimplementation of Underwater Image Enhancement via Medium Transmission-Guided Multi-Color Space Embedding

Edited from https://github.com/59Kkk/pytorch_Ucolor_lcy

## Fixed Problems

1. Use Kornia library for color space conversion
2. Use accelerate to implement distributed training
3. Fix the problem in depth map concatenation
4. LeakyReLU over ReLU to prevent nan loss

## Dataset Structure

The dataset should be formatted like below

```
dataset/
├─ train/
│  ├─ input/
│  │  ├─ 1.jpg
│  │  ├─ ...
│  ├─ depth/
│  │  ├─ 1.jpg
│  │  ├─ ...
│  └─ target/
│     ├─ 1.jpg
│     ├─ ...
└─ test/
   ├─ input/
   │  ├─ 1.jpg
   │  ├─ ...
   ├─ depth/
   │  ├─ 1.jpg
   │  ├─ ...
   └─ target/
      ├─ 1.jpg
      ├─ ...

```

input folder contains underwater image

depth folder contains transmission map

target folder contains ground truth

each triplet should have the exact same name and extension

## Training

You may download the dataset first, and then specify TRAIN_DIR, VAL_DIR and SAVE_DIR in the section TRAINING in `config.yml`.

For single GPU training:

```bash
python train.py
```

For multiple GPUs training:

```bash
accelerate config
accelerate launch train.py
```

If you have difficulties with the usage of `accelerate`, please refer to [Accelerate](https://github.com/huggingface/accelerate).

## Inference

Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in `config.yml`.

```bash
python infer.py
```