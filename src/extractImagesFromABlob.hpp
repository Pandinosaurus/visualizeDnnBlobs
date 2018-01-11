/*
	File : extractImagesFromABlob.cpp
	Author : Rémi Ratajczak
	E-mail : Remi.Ratjczak@gmail.com
	License : GPL 3.0
*/

/* OpenCV things */
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
/* Standard things */
#include <fstream>
#include <iostream>
#include <cstdlib>

/* Parse a blob and output each image it contains in a cv::Mat.
   All cv::Mat are stored in a simpler data structure (akka std::vector) for latter use.
   The returned images are not normalized.
   They displayed images for the quality check are normalized.
*/
std::vector<cv::Mat> extractImagesFromABlob(cv::Mat blob)
{
	//A container to store our images
	std::vector<cv::Mat> vectorOfImages;

	//A blob is a 4 dimensional matrix
	if (blob.dims != 4) return vectorOfImages;

	//Store each dimension size - it is ok to hardcode it (for now) since we checked the dimension with a strong prior on it
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
					tmpMat.at<float>(h, w) = blob.at<float>(indx);
				}//loop on height
			}//loop on width

			//Store the image(s) - note that the image has > not < been normalized here
			vectorOfImages.push_back(tmpMat);

			//Quality check - normalize the image for visualization purpose
			cv::normalize(tmpMat, tmpMat, 0, 1, CV_MINMAX);
			cv::imshow("tmpMat_" + std::to_string(c), tmpMat);
			cv::waitKey(0);

		}//loop on channels
	}//loop on images

	//It is bothering to see all these windows from the quality check, isn't it?
	cv::destroyAllWindows();
	return vectorOfImages;

}//extractImagesFromABlob function
