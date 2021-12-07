// 1612850_BT03.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "cv_algorithms.h"
using namespace std;



int main(int argc, char * argv[])
{
	if (argc < 3)
	{
		cout << "Invalid number of arguments!";
		return 0;
	}
	Mat image = LoadImage(argv[1]);
	if (image.empty())
	{
		cout << "Image not found!";
		return 0;
	}
	string cmd = argv[2];
	if (cmd == "1")
	{
		Mat dummy, dummy2;
		Mat corner = DetectHarris(image, dummy, dummy2);
		Mat result = HighlightCorner(image, corner);
		namedWindow("image", WINDOW_AUTOSIZE);
		imshow("image", result);
		waitKey(0);
	}
	else if (cmd == "2")
	{
		auto blob = DetectBlobColorImage(image);
		Mat result = HighlightBlob(image, blob.first);
		cout << "Processing...\n";
		result = HighlightBlob(result, blob.second);
		namedWindow("image", WINDOW_AUTOSIZE);
		imshow("image", result);
		waitKey(0);
	}
	else if (cmd == "3")
	{
		auto blob = DetectDoGColorImage(image);
		Mat result = HighlightBlob(image, blob.first);
		cout << "Processing...\n";
		result = HighlightBlob(result, blob.second);
		namedWindow("image", WINDOW_AUTOSIZE);
		imshow("image", result);
		waitKey(0);
	}
	else if (cmd == "4")
	{
		if (argc != 4)
		{
			cout << "Invalid number of arguments!";
			return 0;
		}
		Mat matchImage = LoadImage(argv[3]);
		if (matchImage.empty())
		{
			cout << "Image not found!";
			return 0;
		}
		cout << "Processing...\n";		
		Mat result = MatchBySift(image, matchImage, 1);
		namedWindow("image", WINDOW_AUTOSIZE);
		imshow("image", result);
		waitKey(0);
	}
	else
	{
		cout << "Invalid argument!";
		return 0;
	}
	return 0;
}