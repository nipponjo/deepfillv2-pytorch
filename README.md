# deepfillv2-pytorch
A PyTorch reimplementation of the paper **Free-Form Image Inpainting with Gated Convolution** (DeepFillv2) (https://arxiv.org/abs/1806.03589) based on the [original TensorFlow implementation](https://github.com/JiahuiYu/generative_inpainting/tree/v2.0.0).

Example images (raw | masked | inpainted):

<div align="center">
  <img src="https://github.com/nipponjo/deepfillv2-pytorch/blob/master/examples/inpaint/case1.png" width="30%"> <img src="https://github.com/nipponjo/deepfillv2-pytorch/blob/master/examples/inpaint/case1_masked.png" width="30%"> <img src="https://github.com/nipponjo/deepfillv2-pytorch/blob/master/examples/inpaint/case1_out.png" width="30%">
  <img src="https://github.com/nipponjo/deepfillv2-pytorch/blob/master/examples/inpaint/case2.png" width="30%"> <img src="https://github.com/nipponjo/deepfillv2-pytorch/blob/master/examples/inpaint/case2_masked.png" width="30%"> <img src="https://github.com/nipponjo/deepfillv2-pytorch/blob/master/examples/inpaint/case2_out.png" width="30%">
  <img src="https://github.com/nipponjo/deepfillv2-pytorch/blob/master/examples/inpaint/case3.png" width="30%"> <img src="https://github.com/nipponjo/deepfillv2-pytorch/blob/master/examples/inpaint/case3_masked.png" width="30%"> <img src="https://github.com/nipponjo/deepfillv2-pytorch/blob/master/examples/inpaint/case3_out.png" width="30%">
</div>


## Pretrained models
The models in `networks_tf.py` can be used with the weights from the [official repository](https://github.com/JiahuiYu/generative_inpainting/tree/v2.0.0#pretrained-models), which I have converted to PyTorch state dicts.

Download converted weights: [Places2](https://drive.google.com/u/0/uc?id=1tvdQRmkphJK7FYveNAKSMWC6K09hJoyt&export=download) | [CelebA-HQ](https://drive.google.com/u/0/uc?id=1fTQVSKWwWcKYnmeemxKWImhVtFQpESmm&export=download)

## Test the model
Before running the following commands make sure to put the downloaded weights file into the `pretrained` folder.
```bash
python test.py --image examples/inpaint/case1.png --mask examples/inpaint/case1_mask.png --out examples/inpaint/case1_out_test.png --checkpoint pretrained/states_places2.pth
```
Include the `--tfmodel` flag to test with the converted TensorFlow weights.
```bash
python test.py --tfmodel --image examples/inpaint/case1.png --mask examples/inpaint/case1_mask.png --out examples/inpaint/case1_out_test.png --checkpoint pretrained/states_tf_places2.pth
```
The [Jupyter](https://jupyter.org/) notebook `test.ipynb` shows how the model can be used.

## Train the model
Train with options from a config file:
```bash
python train.py --config configs/train.yaml
```

Run `tensorboard --logdir <your_log_dir>` to see the [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html) logging.

## Requirements
  + python3
  + pytorch
  + torchvision
  + numpy
  + Pillow
  + tensorboard
  + pyyaml
