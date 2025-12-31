# Towards Personalized Multi-Modal MRI Synthesis across Heterogeneous Datasets

## Dataset preparation
The imaging data were stored in a slice-wise manner, following the storage architecture described below:

```text
BraTS/
├── train/
├── validation/
└── test/

ISLES22/
├── train/
│   ├── t1/
│   ├── t2/
│   ├── t1ce/
│   ├── flair/
│   ├── dwi/
│   └── adc/
│       ├── sub-strokecase0001_0.png
│       ├── sub-strokecase0001_1.png
│       └── sub-strokecase0001_3.png
├── validation/
└── test/
```

## Train
1. Use util/batch_scheduler to generate the file list, and load the list in models/dataload.py
2. python train.py --training --augmentation --gpu_ids 0 --checkpoints_dir ckpt --batch_size 16 --save_latest_freq 100000

## Test
python test.py --gpu_ids 0 --checkpoints_dir ckpt --which_epoch best
