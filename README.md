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

Download converted weights: [Places2](https://drive.google.com/u/0/uc?id=1tvdQRmkphJK7FYveNAKSMWC6K09hJoyt&export=download) | [CelebA-HQ](https://drive.google.com/u/0/uc?id=1fTQVSKWwWcKYnmeemxKWImhVtFQpESmm&export=download) (for `networks_tf.py`)

The networks in `networks_tf.py` use TensorFlow-compatibility functions (padding, down-sampling), while the networks in `networks.py` do not. In order to adjust the weights to the different settings, the model was trained on Places2/CelebA-HQ for some time using the pretrained weights as initialization.

Download fine-tuned weights: [Places2](https://drive.google.com/u/0/uc?id=1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa&export=download) | [CelebA-HQ](https://drive.google.com/u/0/uc?id=17oJ1dJ9O3hkl2pnl8l2PtNVf2WhSDtB7&export=download) (for `networks.py`)


## Test the model
Before running the following commands make sure to put the downloaded weights file into the `pretrained` folder.
```bash
python test.py --image examples/inpaint/case1.png --mask examples/inpaint/case1_mask.png --out examples/inpaint/case1_out_test.png --checkpoint pretrained/states_tf_places2.pth
```

The [Jupyter](https://jupyter.org/) notebook `test.ipynb` shows how the model can be used.

## Train the model
Train with options from a config file:
```bash
python train.py --config configs/train.yaml
```

Run `tensorboard --logdir <your_log_dir>` to see the [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html) logging.

  
## Demo web app
The web app uses a JS/React frontend and a FastAPI backend. To run it you need the following packages:
  + fastapi, python-multipart: for the backend api
  + uvicorn: for serving the app
 
 Install with:
  `pip install fastapi python-multipart "uvicorn[standard]"`
 
Run with:
 `python app.py`
 
New models can be added in `app/models.yaml`
  
  <div align="center">
    <img src="https://user-images.githubusercontent.com/28433296/179576283-bff8a7ef-b594-4a5e-8389-1bbd70fe6c1f.png" width="66%">  
  </div>


## Requirements
  + python3
  + pytorch
  + torchvision
  + numpy
  + Pillow
  + tensorboard
  + pyyaml
