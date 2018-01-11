/*
	File : extractImagesFromABlob.cpp
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

/* Parse a blob and output each image it contains in a cv::Mat.
   All cv::Mat are stored in a simpler data structure (akka std::vector) for latter use.
   The returned images are in floating point precision. They are not normalized.
   The quality check is done with CV_8U images and a JET colormap.
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
		for (int c = 0; c < nbOfChannels; c++)
		{
			cv::Mat tmpMat(width, height, CV_32F);

			for (int w = 0; w < width; w++)
			{
				for (int h = 0; h < height; h++)
				{
					int indx[4] = { 0, c, h, w };
					tmpMat.at<float>(h, w) = blob.at<float>(indx); //blobs store images in floating point precision
				}//loop on height
			}//loop on width

			//Sanity check
			if (tmpMat.empty()) {
				std::cerr << "No image retrieved --> quit function extractImagesFromABlob." << std::endl;
				return vectorOfImages;
			}

			//Store the image(s) - note that the image has > not < been normalized here
			//The image is still int floating point precision
			vectorOfImages.push_back(tmpMat);

			//Quality check
			tmpMat.convertTo(tmpMat, CV_8UC1); //float to unsigned char for applyColorMap only
			cv::Mat tmpMatColored;
			cv::cvtColor(tmpMat, tmpMatColored, cv::COLOR_GRAY2BGR);
			cv::applyColorMap(tmpMatColored, tmpMatColored, cv::COLORMAP_JET);
			cv::imshow("tmpMat_" + std::to_string(c), tmpMatColored);
			cv::waitKey(0);

		}//loop on channels
	}//loop on images

	//It is bothering to see all these windows from the quality check, isn't it?
	cv::destroyAllWindows();
	return vectorOfImages;

}//extractImagesFromABlob function
