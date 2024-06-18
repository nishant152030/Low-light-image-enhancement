# Low-Light-Image Enhancement
Low light image enhancement is a vital field of study that addresses the challenges posed by poor lighting conditions in image capture. Many photos are often captured under suboptimal lighting conditions due to inevitable environmental and/or technical constraints. These include inadequate and unbalanced lighting conditions in the environment, incorrect placement of objects against extreme back light, and under-exposure during image capturing.

## Project Overview
This project is an implementation of paper [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement](https://arxiv.org/pdf/2001.06826v2). It uses Deep Curve Estimation (DCE) network, a convolutional neural network (CNN), which is designed to enhance low-light images by learning to adjust the illumination in a data-driven manner. A unique advantage of this deep learning-based method is zero-reference, i.e., it does not require any paired or even unpaired data in the training process as in existing CNN-based and GAN-based methods.

## How to Use
### Clone the repository.
 ```
 git clone https://github.com/nishant152030/Low-light-image-enhancement.git
 ```
### Train Model
   On running main.py, the model is trained using the low-light images in the low directory. After training, it processes the test images and produces enhanced images. Finally, it outputs the original and enhanced images along with the PSNR value.
