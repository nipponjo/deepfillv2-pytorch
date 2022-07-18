## Pretrained models
The models in `networks_tf.py` can be used with the weights from the [official repository](https://github.com/JiahuiYu/generative_inpainting/tree/v2.0.0#pretrained-models), which I have converted to PyTorch state dicts. 

Download converted weights: [Places2](https://drive.google.com/u/0/uc?id=1tvdQRmkphJK7FYveNAKSMWC6K09hJoyt&export=download) | [CelebA-HQ](https://drive.google.com/u/0/uc?id=1fTQVSKWwWcKYnmeemxKWImhVtFQpESmm&export=download) (for `networks_tf.py`)

The networks in `networks_tf.py` use TensorFlow-compatibility functions (padding, down-sampling), while the networks in `networks.py` do not. In order to adjust the weights to the different settings, the model was trained on Places2/CelebA-HQ for some time using the pretrained weights as initialization.

Download fine-tuned weights: [Places2](https://drive.google.com/u/0/uc?id=1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa&export=download) | [CelebA-HQ](https://drive.google.com/u/0/uc?id=17oJ1dJ9O3hkl2pnl8l2PtNVf2WhSDtB7&export=download) (for `networks.py`)