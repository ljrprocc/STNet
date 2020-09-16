## STNet

Implementation of STNet. Paper No.5325.

### Proposed Synthetic Dataset

Our proposed synthetic datasets are in the formula **base+Syn$t$x**, Here the available datasets includes **ICDAR2015+Syn1x**([Google Drive](https://drive.google.com/drive/folders/1Sq-b9cYIE6dLdl6YPNAN_8jgo04CFGtl?usp=sharing)), **ICDAR2015+Syn2x**([Google Drive](https://drive.google.com/drive/folders/1PjgJLBdfnLq4x3qB12CClrT_zeoq0qz0?usp=sharing)), **ICDAR2015+Syn3x**([Google Drive](https://drive.google.com/drive/folders/1NwqPgynvJVAW6d3CH4JsaB7Vqrq_fgCZ?usp=sharing)), **ICDAR2015+Syn4x**([Google Drive](https://drive.google.com/drive/folders/11qWj1yd6qViWcoOO81L-gq9F1nOCjWyc?usp=sharing)) and **MSRATD500+Syn2x**([Google Drive](https://drive.google.com/drive/folders/1odgS9YvmK-o8Y2uk12KhWZN7P4XectWw?usp=sharing)). Dataset COCO+Syn2x is not available yet.

Both datasets are organized as

```
|--ds_root
|------train  # Original images
|------train_syn # Train images with synthetic text instances
|------train_syn_gt # The ground truth of synthetic text instances.
|------val  # The validation dataset
|------val_syn
|------val_syn_gt
```

### Train STNet

1.  Get the files of STNet.
2.  Change the parameter defined in `train/train_main.py`, like `net_path`, `train_tag`  with and `batch_size`, `patch_size` . Furthermore, set the epoch and print frequency properly.
3. Run the command `CUDA_VISIBILE_DEVICES=0 python train/train_main.py` to train STNet.



### Evaluate STNet(Not available yet)

1. Change the parameter defined in `train/eval.py`, such as `net_path`, `train_tag` for specifying the path of synthetic datasets.
2. Check whether the checkpoint exists in `net_path`.
3. Run the command `CUDA_VISIBLE_DEVICES=0 python train/eval.py`.