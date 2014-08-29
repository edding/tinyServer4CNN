tinyServer4CNN
==============

A tiny server that receives images &amp; return a classification result by using Convolutional Neural Network. This project is based on Caffe framework from UCB, and uses the pre-trained ImageNet model, an ILSVRC12 image classifier. More functions will be added later.

For using this porject, be sure to install Caffe and set up the environment properly. You can check this site: http://caffe.berkeleyvision.org/installation.html
for more informations on how to install Caffe on your devices.

Codes are tested on OSX Mavericks with python 2.7 (anaconda), and Ubuntu 12.04.4 LTS with python 2.7.3

### Attention
If you are using this server to provide services for iOSClassifier, please leave `oversample` as `False`, as the path of the input images is irregular, which means you may only get some blank crop from the corner.
