# IPU-Net
<p>
We propose an identity-preserving end-to-end image-to-image translation deep neural network which is capable of super-resolving very low-resolution faces to their high-resolution counterparts while preserving identity-related information. We achieved this by training a very deep convolutional encoder-decoder network with a symmetric contracting path between corresponding layers. This network was trained with a combination of a reconstruction and an identity-preserving loss, on multi-scale low-resolution conditions.
</p>


## Important Links
[Paper](https://arxiv.org/abs/2010.12249) </br>
[FaceNet model](https://drive.google.com/drive/folders/12aMYASGCKvDdkygSv1yQq8ns03AStDO_) </br>
[VggFace2 dataset](https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b) </br>

## Model Architecture
<p>
Our proposed method includes two deep neural networks which are designed to receive low-resolution faces and produce the super-resolution counterparts which are optimized for both visual quality and face recognition. The first network is a deep convolutional encoder-decoder architecture with symmetric skip-connections between the corresponding layers (UNet-based architecture) which helps to map the low-resolution and high-resolution face pairs more accurately through learning richer semantic representations and sharing low-level information across the network and the other network is a pre-trained face recognition model with an Inception-ResNet  architecture. </br>

Our identity-preserveing encoder-decoder network consists of 7 downsampling and 7 upsampling blocks with symmetric contracting paths (skip connections) for better localization of pixels between low-resolution and high-resolution pairs. In each downsampling block, we applied strided convolution with stride 2 followed by batch normalization and leaky relu layers. In each upsampling block, we used transposed convolution for doubling the spatial dimensions and also batch normalization, dropout, and relu activation functions. We added skip connections between each layer i in the encoder and layer (n - i + 1) in the decoder, where n is the total number of layers. Each skip connection simply concatenates all channels at layer i with those at layer (n - i + 1). These skip-connections helped to localize the feature maps better through sharing information between the encoder and the decoder.

Below figure depicts the proposed model architecture. It consists of an identity-preserving face hallucination image-to-image translation model and a pre-trained face recognition network. The deep convolutional face hallucination model translates the the bicubic upsampled low-resolution faces to super-resolved version during training (i.e. image-to-image translation). The pre-trained face recognition model produces discriminative, low-dimensional embeddings for both super-resolved and high-resolution faces in order to compute the identity loss.
</p>

<p align="center">
  <img src="figures/model.png">
</p>


## Requirements: 
```
Python 3.7
TensorFlow 2.0+
Pillow
Scikit-learn
OpenCV
```

## How to train
To Train the model, run this:
```
python train.py
```

## Result

<p align="center">
  <img src="figures/ar_result.png">
</p>
