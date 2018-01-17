/*
	File : extractImagesFromABlob.hpp
	Author : Rémi Ratajczak
	E-mail : Remi.Ratjczak@gmail.com
	License : MIT
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

/* Parse a 4D blob and output the 2D images it contains in a std::vector< cv::Mat >. 
   One image is output per image input in the batch. 
   The number of channels for each image corresponds to the size of the layer's output.
   Each channel of each image could be then accessed through the cv::split() method.
   The channels of the returned images are all in floating point precision (CV_32F). 
   They are not normalized.
*/
std::vector<cv::Mat> extractImagesFromABlob(cv::Mat blob)
{
	//A container to store our images
	std::vector<cv::Mat> vectorOfImages;

	//A blob is a 4 dimensional matrix
	if (blob.dims != 4) return vectorOfImages;

	//Store each dimension size - it is ok to hardcode it (for now) since we checked the dimension with a strong prior
	int nbOfImages = blob.size[0]; //= nb of input in the network
	int nbOfChannels = blob.size[1]; //= nb of filtered images
	int height = blob.size[2]; //= the height of the images
	int width = blob.size[3]; //= the width of the images

	//Access the elements of each channel for each images
	//Store the matrix in a vector of Images
	for (int i = 0; i < nbOfImages; i++)
	{
		//store all the channels for each image
		std::vector<cv::Mat> vectorOfChannels;

		for (int c = 0; c < nbOfChannels; c++)
		{
			//store the current channel
			cv::Mat currentChannel(width, height, CV_32F); 

			for (int w = 0; w < width; w++)
			{
				for (int h = 0; h < height; h++)
				{
					int indx[4] = { i, c, h, w };
					currentChannel.at<float>(h, w) = blob.at<float>(indx); //blobs store images in floating point precision
				}//loop on height
			}//loop on width

			 //Sanity check
			if (currentChannel.empty()) {
				std::cerr << "No image retrieved --> quit function extractImagesFromABlob." << std::endl;
				return vectorOfImages;
			}

			//Save the current channel
			//The channel is still in floating point precision
			//Note that the channel has > not < been normalized here
			vectorOfChannels.push_back(currentChannel);

		}//loop on channels

		//Merge all the channels in one mat
		cv::Mat imgToOutput;
		cv::merge(vectorOfChannels, imgToOutput);

		//Store the image with all its channels concatenated 
		//All data are in CV_32F, channels not normalized
		vectorOfImages.push_back(imgToOutput);

	}//loop on images

	return vectorOfImages;

}//extractImagesFromABlob function
