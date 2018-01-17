/*
	File : main.cpp
	Author : Rémi Ratajczak
	E-mail : Remi.Ratjczak@gmail.com
	License : MIT

	Purpose :
	This program provides a way to visualize the outputs of each layer of a Deep Convolutional Neural Network with OpenCV.
	
	Why? :
	OpenCV (>= 3.4) already provides the dnn module.
	The dnn module allows you to load a network trained with a dedicated framework/library (e.g. caffe, tensorflow) in OpenCV. It also allows you to catch each output of each layer when you perform a forward pass through the forward() method.
	The ouptut of each layer is returned in a data structure named "blob".
	A blob is a 4 dimensional array stored in an OpenCV Mat objet (cv::Mat).
	You can't display a blob trivialy with cv::imshow() except for the output of the "prob" layer (at least with the Googlenet network).
	If you want to see the result of a given layer, you need to extract the images stored in the blob.
	This functionality has not already been integrated in the DNN module. Note that the inverse function, image to blob, already exist (demo in code below).

	What it does :
	In essence, this program demonstrates how to extract the images from the "blobs".
	It generates one cv::Mat per filtered image per layer (e.g. per convolutional filter in a convolutional layer).
	You can use the resulting cv::Mat images as you have the habit to do in OpenCV.
	It allows you to display/store/save/study/understand the results of each layer direclty in OpenCV.
	In this code sample, an example is provided with the GoogleNet model from the Caffe Model Zoo.

	What it does not :
	It does not demosntrate how to classify an image in OpenCV with a trained network.
	See the offical OpenCV documentation for the DNN module for a classification tutorial : https://docs.opencv.org/trunk/d5/de7/tutorial_dnn_googlenet.html.
 */


/* OpenCV things */
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
/* Standard things */
#include <fstream>
#include <iostream>
#include <cstdlib>
/* The function to extract images from a blob */
#include "extractImagesFromABlob.hpp"

using namespace cv; //boo, remove it
using namespace cv::dnn; //boo, remove it
using namespace std; //boo, remove it

int main(int argc, char **argv)
{
	//Load the model parameters paths in memory
	//You will find the caffemodel there: http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
	String modelTxt = "..\\data\\bvlc_googlenet.prototxt"; //definition of the model
	String modelBin = "..\\data\\bvlc_googlenet.caffemodel"; //weights of the model
	String imageFile = "..\\data\\space_shuttle.jpg"; //image to read - you can use your own
	String imageFile2 = "..\\data\\space_shuttle2.jpg"; //image to read - you can use your own
	String classNameFile = "..\\data\\synset_words.txt";//used for classification only - not presented here

	//Try to instantiate the network with its parameters
	Net net;
	try {
		net = dnn::readNetFromCaffe(modelTxt, modelBin);
	}
	catch (const cv::Exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		if (net.empty())
		{
			std::cerr << "Can't load network by using the following files: " << std::endl;
			std::cerr << "prototxt:   " << modelTxt << std::endl;
			std::cerr << "caffemodel: " << modelBin << std::endl;
			std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
			std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
			cv::waitKey(0);
			exit(-1);
		}
	}
	
	//Read the image to process with the network
	Mat img = imread(imageFile);
	Mat img2 = imread(imageFile2);
	if (img.empty())
	{
		std::cerr << "Can't read image from the file: " << imageFile << std::endl;
		exit(-1);
	}
	
	//Input images in a vector
	std::vector< cv::Mat > inputImgs;
	inputImgs.push_back(img);
	inputImgs.push_back(img2);

	//Convert the image into a blob so that we could feed the network with it.
	//The blob will internally store the image in floating point precision (CV_32F).
	Mat inputBlob = blobFromImages(inputImgs, //the image to convert
								   1.0f, //a multiplicative factor, in float due to the conversion realized by blobFromImage
								   Size(224, 224), //the size of the image in the blob / should correspond to the expected input size of the data in the network / either crop/bilinear resizing can be used
								   Scalar(104, 117, 123), //the mean of the images / in practice you should calculate it from your dataset / it is used to mean center the images
								   false); //a boolean to convert a BGR image (default in OpenCV) to a RGB image / the format should correspond to the one used to train you network!

	//For each layer in the network, we are going to perform a forward pass then store
	//the output blobs and extract the images from them
	for (string layer : net.getLayerNames()) //getLayerNames() gives a vector of string containing the names of every layer in the network - see prototxt for the names
	{

		//A container for our blobs
		std::vector<cv::Mat> vectorOfBlobs;
		
		//Perform a forward pass
		net.setInput(inputBlob, "data"); //Set the network input - with GoogleNet, "data" is the input layer
		net.forward(vectorOfBlobs, layer);	//Operate a forward pass, output the result of the selected layer

		//For each blobs in our vectorOfBlobs
		for(cv::Mat blob : vectorOfBlobs )
		{
			//A simple vector that will contain each filtered image (i.e. the result of each operation in the layer)
			std::vector<cv::Mat> vectorOfImages;

			//if the blob is not empty, extract images from it
			//DO NOT CHECK its size  the blob is a cv::Mat in nature, but the data are stored differently (4 dimensions) 
			//than with the images and the size() method will result in an unhandled expection.
			if (!blob.empty()) vectorOfImages = extractImagesFromABlob2(blob); //see extractImagesFromABlob.hpp

			//Quality check is done with CV_8U images and a JET colormap
			//Quality check
			for (auto image : vectorOfImages)
			{
				std::cout << "nbOfImages : " << vectorOfImages.size() << std::endl;
				std::cout << "channels : " << image.channels() << std::endl;
				std::cout << "size : " << image.size() << std::endl;

				//Display each 2D channel contained in the current image
				//The channels are obtained using the split method
				std::vector< cv::Mat > channels;
				cv::split(image, channels);
				for (auto channel : channels)
				{
					channel.convertTo(channel, CV_8UC1); //float to unsigned char for applyColorMap only
					cv::Mat tmpMatColored;
					cv::cvtColor(channel, tmpMatColored, cv::COLOR_GRAY2BGR);
					cv::applyColorMap(tmpMatColored, tmpMatColored, cv::COLORMAP_JET);
					cv::imshow(layer + " : output", tmpMatColored);
					cv::waitKey(0);
				}//for loop on channels
			}//for loop on images

			//Destroy bothering windows from Quality check
			cv::destroyAllWindows();

		} //for loop on blobs
	} //for loop on layers

	return 0;
} //main
