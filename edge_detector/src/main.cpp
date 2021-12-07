// 1612850_BT02.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "pch.h"
#include "opencv2\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <math.h>
using namespace std;
using namespace cv;
const float PI = (float)3.14159265358979323846;

/// <summary>
/// Load an color image from a file path and normalize its values to the range [0, 1]
/// </summary>
/// <param = "path"> The path to the image </param>
/// <returns> The image contained in the path </returns>
Mat LoadImage(string path)
{
	Mat tmp = imread(path, IMREAD_COLOR); // Read image in RGB.
	Mat res;
	tmp.convertTo(res, CV_32FC3, 1.0 / 255); // Normalize to [0, 1]
	return res;
}

/// <summary>
/// Calculate the correlation of two matrixes
/// </summary>
/// <param = "first"> The first matrix </param>
/// <param = "second"> The second matrix </param>
/// <returns> The correlation of the two matrixes </returns>
float Correlation(Mat first, Mat second)
{	
	// Calculate the pairwise sum	
	return (float)sum(first.mul(second))[0];
}

/// <summary>
/// Calculate the convolution of two matrixes
/// </summmary>
/// <param = "first"> The first matrix </param>
/// <param = "second"> The second matrix </param>
/// <returns> The convolution of the two matrixes </returns>
float Convolve(Mat first, Mat second)
{
	float res = 0;
	for (int i = 0; i < first.rows; i++)
		for (int j = 0; j < first.cols; j++)
		{
			float a = first.at<float>(i, j);
			// Get the value at the corresponding position.
			float b = second.at<float>(second.rows - i - 1, second.cols - j - 1); 
			res += a * b;
		}			
	return res;
}

/// <summary>
/// Apply the given kernel to the given image
/// </summary>
/// <param = "kernel"> The given kernel </param>
/// <param = "image"> The given image to apply the kernel </param>
/// <returns> The image after being applied the kernel </returns>
Mat ApplyKernel(Mat kernel, Mat image)
{
	// Create a matrix to contain the result
	Mat res = Mat(image.rows, image.cols, CV_32F);
	// For each pixel of the image
	int kernelWidth = kernel.rows / 2;
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			float val = 0;
			// If the kernel is completely inside the image then calculate their convolution
			if (i - kernelWidth >= 0 && i < image.rows - kernelWidth 
				&& j - kernelWidth >= 0 && j < image.cols - kernelWidth)
				val = Correlation(kernel, image(Rect(j - kernelWidth, i - kernelWidth, kernelWidth * 2 + 1, kernelWidth * 2 + 1)));
			res.at<float>(i, j) = val;
		}
	}
	return res;
}

/// <summary>
/// Generate a gaussian kernel
/// </summary>
/// <param = "size"> THe size of the kernel (size x size) </param>
/// <param = "sigma"> The sigma of the gaussian kernel </param>
/// <returns> A Mat object that holds the kernel </returns>
Mat GenerateGaussian(int size, int sigma)
{
	// Create the Mat object
	Mat res = Mat(size, size, CV_32F);
	// Fill values into the kernel
	for(int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
		{
			int x = i - size / 2;
			int y = j - size / 2;
			float val = (float)(1.0 / (PI * 2 * sigma * sigma) * exp(-1.0 * (x * x + y * y) / 2 / sigma / sigma));
			res.at<float>(i, j) = val;
		}
	return res;
}

/// <summary>
/// Normalize a given image to make all its values lie inside the range [0, 1]
/// </summary>
/// <param = "image'> The given image </param>
/// <returns> THe image after being normalied </returns>
Mat Normalize(Mat image)
{
	// Get the max and min values
	double maxVal, minVal;
	minMaxLoc(image, &minVal, &maxVal);	
	// Normalize the image
	return (image - minVal) / (maxVal - minVal);
}

/// <summary>
/// Detect edges of an image using Sobel operator
/// </summary>
/// <param = "image"> The given image </param>
/// <param = "dx"> Partial derivative by x </param>
/// <param = "dy"> Partial derivative by y </param>
/// <param = "gradient"> Gradient of the image </param>
void DetectBySobel(Mat image, Mat & dx, Mat & dy, Mat & gradient)
{
	// Convert the image to grayscale
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);		
	// Create the x and y kernels
	float kernelX[] = { -1, 0, 1,
						-2, 0, 2,
						- 1, 0, 1 };
	float kernelY[] = { -1, -2, -1,
						0, 0, 0,
						1, 2, 1 };	
	Mat matKernelX = Mat(3, 3, CV_32F, kernelX) / 8;
	Mat matKernelY = Mat(3, 3, CV_32F, kernelY) / 8;
	Mat xDerivative = ApplyKernel(matKernelX, GenerateGaussian(7, 3));
	Mat yDerivative = ApplyKernel(matKernelY, GenerateGaussian(7, 3));
	// Apply the kernels to the grayscale image
	dx = ApplyKernel(xDerivative, gray);
	dy = ApplyKernel(yDerivative, gray);
	// Calculate the gradient. gradient = sqrt(x^2 + y^2)
	cv::sqrt(dx.mul(dx) + dy.mul(dy), gradient);	
	// Normalize the results
	dx = Normalize(dx);
	dy = Normalize(dy);
	gradient = Normalize(gradient);
}

/// <summary>
/// Detect edges of an image using Prewitt operator
/// </summary>
/// <param = "image"> The given image </param>
/// <param = "dx"> Partial derivative by x </param>
/// <param = "dy"> Partial derivative by y </param>
/// <param = "gradient"> Gradient of the image </param>
void DetectByPrewitt(Mat image, Mat & dx, Mat & dy, Mat & gradient)
{	
	Mat gray;
	// Convert the image to grayscale
	cvtColor(image, gray, COLOR_BGR2GRAY);	
	// Create the x and y kernels
	float kernelX[] = { -1, 0, 1,
						-1, 0, 1,
						-1, 0, 1 };
	float kernelY[] = { -1, -1, -1,
						0, 0, 0,
						1, 1, 1};
	Mat matKernelX = Mat(3, 3, CV_32F, kernelX) / 6;
	Mat matKernelY = Mat(3, 3, CV_32F, kernelY) / 6;
	Mat xDerivative = ApplyKernel(matKernelX, GenerateGaussian(7, 3));
	Mat yDerivative = ApplyKernel(matKernelY, GenerateGaussian(7, 3));
	// Apply the kernels to the grayscale image
	dx = ApplyKernel(xDerivative, gray);
	dy = ApplyKernel(yDerivative, gray);
	// Calculate the gradient. gradient = sqrt(x^2 + y^2)
	cv::sqrt(dx.mul(dx) + dy.mul(dy), gradient);
	// Normalize the results
	dx = Normalize(dx);
	dy = Normalize(dy);
	gradient = Normalize(gradient);
}

/// <summary>
/// Return the sign of a number
/// </summary>
/// <param = "x"> The number </param>
/// <returns> The sign of the number </returns>
int Sign(float x)
{
	return x > 0 ? 1 : (x < 0 ? -1 : 0);
}

/// <summary>
/// Return the square of a number
/// </summary>
/// <param = "x"> The number </param>
/// <returns> 
float sqr(float x)
{
	return x * x;	
}


/// <summary>
/// Calculate the mean and variance of a matrix
/// </summary>
/// <param = "mat"> The given matrix </param>
/// <param = "mean"> The return value of the mean </param>
/// <param = "variance"> The return value of the variance </param>
void GetMeanAndVariance(Mat mat, float & mean, float & variance)
{
	// Get the mean
	mean = sum(mat)[0] / (mat.rows * mat.cols);
	// Create a temp matrix in which each value equals (mat[i,j] - mean)^2
	Mat tmp;
	cv::pow(mat - mean, 2, tmp);
	// The variance equals the sum of its values divided by the size - 1
	variance = sum(tmp)[0] / (mat.rows * mat.cols - 1);	
}

/// <summary> 
/// Check if the gaussian distribution of a sample is flat 
/// Uses Kurt value
/// </summary>
/// <param = "mat"> The matrix that contains the sample </param>
/// <returns> True if the sample is flat, false otherwise </returns>
bool IsFlat(Mat mat)
{
	// Get the mean and variance
	float mean, variance;
	GetMeanAndVariance(mat, mean, variance);
	// Get the standard deviation
	float std = sqrt(variance);
	// Get the Kurt value
	Mat tmp;
	pow((mat - mean) / std, 4, tmp);
	int n = mat.rows * mat.cols;
	float kurt = sum(tmp)[0] * n * (n + 1) / (n - 1) / (n - 2) / (n - 3) - 3 * sqr(n - 1) / (n - 2) / (n - 3);
	// If the Kurt value is less than or equal to zero then the sample is flat
	return kurt <= 0;
}

/// <summary>
/// Find zero-crossing of a derivative image
/// <summary>
/// <param = "image"> The derivative image </param>
/// <returns> The result image. If a pixel is part of an edge, its value will be 1, otherwise 0. </retunrs>
Mat ZeroCrossing(Mat image)
{
	int n = image.rows;
	int m = image.cols;
	Mat res(n, m, CV_32F);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
		{
			res.at<float>(i, j) = 0;
			// Find all neighbor cells that have the different sign to the current cell.
			// If the min absolute values of those cells is greater than the absolute value of the current cell,
			// the current cell will be considered as an edge.
			if (i > 0 && i < n - 1 && j > 0 && j < m - 1)
			{				
				float tmp = 1e9, val = image.at<float>(i, j);
				for (int x = -1; x <= 1; x++)
					for (int y = -1; y <= 1; y++)
						if (x || y)
						{
							if (Sign(image.at<float>(i + x, j + y)) * Sign(val) == -1)
								tmp = min(tmp, fabs(image.at<float>(i + x, j + y)));
						}
				//float variance = GetVariance(image(Rect(j - 1, i - 1, 3, 3)));
				//float std = sqrt(variance);
				if (tmp != 1e9 && tmp > fabs(val) && IsFlat(image(Rect(j - 1, i - 1, 3, 3))))
					res.at<float>(i, j) = 1;
			}
		}
	return res;
}

/// <summary>
/// Detect edges in a given image usign Laplace operator.
/// <summary>
/// <param = "image"> The given image </param>
/// <returns> The image that contains edged </returns>
Mat DetectByLaplace(Mat image)
{
	// Convert the image to grayscale
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	// Reduce noise
	Mat smoothImage = ApplyKernel(GenerateGaussian(7, 3), gray);	
	// Compute the second derivative
	float kernel[] = { -1, -1, -1,
					-1, 8, -1,
					-1, -1, -1 };
	Mat matKernel = Mat(3, 3, CV_32F, kernel);
	Mat gradient = ApplyKernel(matKernel, smoothImage);	
	// Find zero-crossing
	gradient = ZeroCrossing(gradient);	
	return gradient;
}

/// <summary> Convert the given angle to one of four directions 0, 1, 2, 3 </summary>
/// <param = "angle"> The given angle, measured by radian </param>
/// <returns> The converted direction </returns>
uchar GetDirection(float angle)
{
	if (angle < 0) angle += PI * 2;
	if (angle < PI / 4)
		return 0;
	else if (angle < PI / 2)
		return 1;
	else if (angle < PI * 3 / 4)
		return 2;
	else if (angle <= PI)
		return 3;
	return GetDirection(angle - PI);
}

/// <summary> Non-maximum suppression in Canny edge detection </summary>
/// <param = "magnitude"> The magnitude of the gradient </param>
/// <param = "orientation"> The orientation of the gradient </param>
/// <returns> The image after the process </returns>
Mat NonMaximumSuppression(Mat magnitude, Mat orientation)
{
	int n = magnitude.rows;
	int m = magnitude.cols;
	Mat res(n, m, CV_32F);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			if (i > 0 && i < n - 1 && j > 0 && j < m - 1)
			{
				// Get the two values at the coresponding direction.
				int direct = orientation.at<uchar>(i, j);
				float val1, val2;
				if (direct == 0)
				{
					val1 = magnitude.at<float>(i, j - 1);
					val2 = magnitude.at<float>(i, j + 1);
				}
				else if (direct == 1)
				{
					val1 = magnitude.at<float>(i - 1, j + 1);
					val2 = magnitude.at<float>(i + 1, j - 1);
				}
				else if (direct == 2)
				{
					val1 = magnitude.at<float>(i - 1, j);
					val2 = magnitude.at<float>(i + 1, j);
				}
				else
				{
					val1 = magnitude.at<float>(i - 1, j - 1);
					val2 = magnitude.at<float>(i + 1, j + 1);
				}
				// If the current value is greater than both those two values, keep it.
				// Otherwise, suppress the current value.
				if (magnitude.at<float>(i, j) > max(val1, val2))
					res.at<float>(i, j) = magnitude.at<float>(i, j);
				else
					res.at<float>(i, j) = 0;
			}
	return res;
}

/// <summary>
/// Double threshold the image. Mark pixels whose value is above high as 1, below high and above low as 0.5 and the rest as 0.
/// </summary>
/// <param = "image"> The given image </param>
/// <param = "high"> The high threshold </param>
/// <param = "low"> The high threshold </param>
/// <returns> The image after the operation </returns>
Mat DoubleThreshold(Mat image, float high, float low)
{
	int n = image.rows;
	int m = image.cols;
	Mat strong, weak;
	// Get the strong pixels (those whose value is above the high threshold)	
	Mat(image >= high).convertTo(strong, CV_32F);
	// Get the weak pixels (those whose value is between the two thresholds)	
	Mat((image < high) & (image >= low)).convertTo(weak, CV_32F);
	// Convert to [0,1] scale
	strong /= 255;
	weak /= 255 * 2;	
	return strong + weak;
}

/// <summary> The hysteresis operation in Canny edge detection </summary>
/// <param = "image"> The given image (after the non-maxium supression process) </param>
/// <param = "high"> The high threshold </param>
/// <param = "low"> The low threshold </param>
/// <returns> The image after the operation </returns>
Mat Hysteresis(Mat image)
{		
	int n = image.rows;
	int m = image.cols;	
	Mat res(image);
	for(int i = 0; i < n; i++)
		for(int j = 0; j < m; j++)
			// Weak pixels are marked as 0.5
			if(image.at<float>(i, j) == 0.5)
			{
				// Search for strong pixels nearby				
				float tmp = 0;
				for (int x = -1; x <= 1; x++)
					for (int y = -1; y <= 1; y++)
						if (i + x >= 0 && i + x < n && j + y >= 0 && j + y < m)
							tmp = max(tmp, res.at<float>(i + x, j + y));
				// If there is a strong pixel nearby, mark the current pixel as strong				
				if (tmp == 1)
					res.at<float>(i, j) = 1;
				else
					res.at<float>(i, j) = 0;
			}
	return res;
}

/// <summary> Detect edges using Canny operator </summary>
/// <param = "image"> The image to detect edges </param>
/// <returns> The image contains detected edges </returns>
Mat DetectByCanny(Mat image)
{
	// Convert the image to grayscale
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	// Create partial derivative kernels
	float kernelX[] = { -1, 0, 1,
						-2, 0, 2,
						-1, 0, 1 };
	float kernelY[] = { -1, -2, -1,
						0, 0, 0,
						1, 2, 1 };
	Mat matKernelX = Mat(3, 3, CV_32F, kernelX) / 8;
	Mat matKernelY = Mat(3, 3, CV_32F, kernelY) / 8;
	// Apply the derivative kernels to the smooth kernel
	Mat xDerivative = ApplyKernel(matKernelX, GenerateGaussian(7, 3));
	Mat yDerivative = ApplyKernel(matKernelY, GenerateGaussian(7, 3));
	// Apply the smooth-derivative kernels to the grayscale image
	Mat dx = ApplyKernel(xDerivative, gray);
	Mat dy = ApplyKernel(yDerivative, gray);
	int n = image.rows;
	int m = image.cols;
	// Calculate the magnitude and orientation of the gradient
	Mat magnitude(n, m, CV_32F);
	Mat orientation(n, m, CV_8U);
	for(int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
		{
			float x = dx.at<float>(i, j);
			float y = dy.at<float>(i, j);
			magnitude.at<float>(i, j) = sqrt(x * x + y * y);
			orientation.at<uchar>(i, j) = GetDirection(atan2(y, x));
		}
	magnitude = Normalize(magnitude);
	// Apply nonmaximum suppression
	magnitude = NonMaximumSuppression(magnitude, orientation);	
	// Apply double threshold
	magnitude = DoubleThreshold(magnitude, 0.09, 0.05);	
	// Apply hysteresis
	magnitude = Hysteresis(magnitude);				
	return magnitude;
}

int main(int argc, char * argv[])
{	
	if (argc < 3)
	{
		cout << "Not enough arguments!\n";
		return -1;
	}
	string path = argv[1];
	//string path = "C:\\Documents\\TGMT\\BT02\\sample.jpg";	
	Mat image = LoadImage(path);	
	if (!image.data)
	{
		cout << "No image data\n";
		return -1;
	}	
	Mat showImage;	
	if (string(argv[2]) == "1" || string(argv[2]) == "2") // Detect by sobel or prewitt
	{		
		Mat gradient, dx, dy;
		if (string(argv[2]) == "1")
			DetectBySobel(image, dx, dy, gradient);
		else
			DetectByPrewitt(image, dx, dy, gradient);
		if (string(argv[3]) == "gradient")
			showImage = gradient;
		else if (string(argv[3]) == "dx")
			showImage = dx;
		else if (string(argv[3]) == "dy")
			showImage = dy;
		else
		{
			cout << "One or more arguments are not correct!";
			return -1;
		}
	}
	else if (string(argv[2]) == "3") // Detect by laplace
	{
		showImage = DetectByLaplace(image);
	}
	else if (string(argv[2]) == "4") // Detect by canny
	{				
		showImage = DetectByCanny(image);
	}
	else
	{	
		cout << "One or more arguments are not correct!";
		return -1;	
	}		
	namedWindow("original", WINDOW_NORMAL);
	imshow("original", image);
	namedWindow("result", WINDOW_NORMAL);
	imshow("result", showImage);
	waitKey(0);	
	return 0;
}