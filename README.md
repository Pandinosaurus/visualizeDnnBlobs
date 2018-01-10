# visualizeDnnBlobs
A sample code to visualize with OpenCV the deep learning blobs generated with the OpenCV dnn module.

## Purpose :
This program provides a way to visualize the outputs of each layer of a Deep Convolutional Neural Network with OpenCV (https://github.com/opencv/opencv).

<img src="./results/output1.PNG" align="center" height="25%" width="25%">

## Why :
OpenCV (>= 3.2) already provides the dnn module. 
The dnn module allows you to load a network trained with a dedicated framework/library (e.g. caffe, tensorflow) in OpenCV. It also allows you to catch each output of each layer when you perform a forward pass through the forward() method.
The ouptut of each layer is returned in a data structure named "blob".
A blob is a 4 dimensional array stored in an OpenCV Mat objet (cv::Mat). 
You can't display a blob trivialy with cv::imshow() except for the output of the "prob" layer (at least with the Googlenet network). 
If you want to see the result of a given layer, you need to extract the images stored in its output the blob(s).
This functionality has not already been integrated in the DNN module. Note that the inverse function, image to blob, already exist (demo in code below).

## What it does :
In essence, this program demonstrates how to extract the images from the "blobs". 
It generates one cv::Mat per filtered image per layer (e.g. per convolutional filter in a convolutional layer).  
You can use the resulting cv::Mat images as you have the habit to do in OpenCV. 
It allows you to display/store/save/study/understand the results of each layer direclty in OpenCV.
In this code sample, an example is provided with the GoogleNet model from the Caffe Model Zoo.

## What it does not :
It does not demosntrate how to classify an image in OpenCV with a trained network.
See the offical OpenCV documentation for the DNN module for a classification tutorial : https://docs.opencv.org/trunk/d5/de7/tutorial_dnn_googlenet.html.

## Requirements :
### Libraries :
	- OpenCV (version >= 3.2)
	
### OS :
	- Windows 10 (you may need to change the relative paths on Unix)

### Data :
	- The /data/ folder provides you with almost everything you need to get started but the trained model (bvlc_googlenet.caffemodel). 
	You can download it from : https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet.
	Once downloaded, put it in the /data/ folder.
