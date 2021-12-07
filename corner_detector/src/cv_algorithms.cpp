#include "pch.h"
#include "cv_algorithms.h"
#include <cmath>
#include <iostream>
/*
@summary Load an color image from a file path and normalize its values to the range [0, 1]
@param path The path to the image
@returns The image contained in the path
*/
Mat LoadImage(string path)
{
	Mat tmp = imread(path, IMREAD_COLOR); // Read image in RGB.
	Mat res;
	tmp.convertTo(res, CV_32FC3, 1.0 / 255); // Normalize to [0, 1]	
	return res;
}

/*
@summary Calculate the correlation of two matrixes
@param first The first matrix
@param second The second matrix
@returns The correlation of the two matrixes
*/
float Correlation(Mat first, Mat second)
{	
	// Calculate the pairwise sum	
	return (float)sum(first.mul(second))[0];
}

/*
@summary Calculate the convolution of two matrixes
@param first The first matrix
@param second The second matrix
@returns The convolution of the two matrixes
*/
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

/*
@summary Apply the given kernel to the given image
@param kernel The given kernel
@param image The given image to apply the kernel
@returns The image after being applied the kernel
*/
Mat ApplyKernel(Mat kernel, Mat image)
{
	// Create a matrix to contain the result
	Mat res = Mat::zeros(image.rows, image.cols, CV_32F);
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

/*
@summary Normalize a given image to make all its values lie inside the range [0, 1]
@param image The given image
@returns The image after being normalied
*/
Mat Normalize(Mat image)
{
	// Get the max and min values
	double maxVal, minVal;
	minMaxLoc(image, &minVal, &maxVal);
	// Normalize the image
	return (image - minVal) / (maxVal - minVal);
}

/*
@summary Computes the partial derivative with respect to x and y on the image
@param image The given image
@returns dx The partial derivative with respect to x
@returns dy The partial derivative with respect to y
*/
void PartialDerivative(Mat image, Mat & dx, Mat & dy)
{
	// Sobel operators
	float kernelX[] = { -1, 0, 1,
						-2, 0, 2,
						-1, 0, 1 };
	float kernelY[] = { -1, -2, -1,
						0, 0, 0,
						1, 2, 1 };
	Mat matKernelX = Mat(3, 3, CV_32F, kernelX) / 8;
	Mat matKernelY = Mat(3, 3, CV_32F, kernelY) / 8;
	dx = ApplyKernel(matKernelX, image);
	dy = ApplyKernel(matKernelY, image);
}

/*
@summary Generate a gaussian kernel
@param size THe size of the kernel (size x size)
@param sigma The sigma of the gaussian kernel
@returns A Mat object that holds the kernel
*/
Mat GenerateGaussian(int size, float sigma)
{
	// Create the Mat object
	Mat res = Mat(size, size, CV_32F);
	// Fill values into the kernel
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
		{
			int x = i - size / 2;
			int y = j - size / 2;
			float val = (float)(1.0 / (PI * 2 * sigma * sigma) * exp(-1.0 * (x * x + y * y) / 2 / sigma / sigma));
			res.at<float>(i, j) = val;
		}
	return res;
}

/*
@summary Supress all pixels that are not local maximum.
@param image The given image
@returns The image after the operation
*/
Mat NonMaximumSuppression(Mat image)
{
	int n = image.rows, m = image.cols;
	Mat res = Mat::zeros(n, m, CV_32F);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++) 
		{
			// Get max value of neibor cells
			float maxVal = -1e9;
			for (int d = 0; d < 8; d++)
			{
				int x = i + direct8[d].first, y = j + direct8[d].second;
				if (x < 0 || x >= n || y < 0 || y >= m) continue;
				maxVal = max(maxVal, image.at<float>(x, y));
			}
			if (image.at<float>(i, j) > maxVal)
				res.at<float>(i, j) = 1;			
		}
	}
	return res;
}

/*
@summary Highlight the corner pixels on the original image
@param image The original image
@param corner The mat object that contains the corner pixels
@param color The color to highlight the corner pixels
@returns The original image with highlighted corners
*/
Mat HighlightCorner(Mat image, Mat corner, Vec3f color)
{
	int n = image.rows, m = image.cols;
	Mat res;
	image.copyTo(res);
	for(int i = 0; i < n; i++) // For each pixel
		for (int j = 0; j < m; j++)
		{
			if (corner.at<float>(i, j) == 1) // If the pixel is a corner
			{
				res.at<Vec3f>(i, j) = color; // Mark that pixel using the given color
				for (int d = 0; d < 8; d++) // For all its neighbor pixels
				{
					int x = i + direct8[d].first, y = j + direct8[d].second;
					if (x < 0 || x >= n || y < 0 || y >= m) continue;
					res.at<Vec3f>(x, y) = color; // Also mark the neighbor pixels
				}
			}			
		}
	return res;
}

/*
@summary Harris corner detection on an image
@param image The given image
@param k The parameter for calculating the R value
@returns A matrix in which each pixel is either 0 or 1. 1 indicates that pixel is a corner.
*/
Mat DetectHarris(Mat image, Mat & keypoints, Mat & positions, float k)
{
	k = max(k, 0.04f);
	k = min(k, 0.06f);
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	Mat dx, dy;
	PartialDerivative(gray, dx, dy);
	Mat gaussianKernel = GenerateGaussian(5, 2); // Get the gaussian window
	int n = gray.rows, m = gray.cols;
	Mat respond = Mat::zeros(n, m, CV_32F);
	for (int i = 2; i + 2 < n; i++) // Slide the window
	{
		for (int j = 2; j + 2 < m; j++) 
		{
			Mat M = Mat::zeros(2, 2, CV_32F); // The second moment matrix
			for (int x = -2; x <= 2; x++) // For every position in the window
				for (int y = -2; y <= 2; y++)
				{
					// Get the matrix at the current position
					float dxVal = dx.at<float>(i + x, j + y);
					float dyVal = dy.at<float>(i + x, j + y);
					float arr[] = { dxVal * dxVal,
								dxVal * dyVal,
								dyVal * dxVal,
								dyVal * dyVal };
					Mat tmp(2, 2, CV_32F, arr);
					M += tmp * gaussianKernel.at<float>(x + 2, y + 2); // Sum all the weighted matrixes up
				}
			// Approximate the lambdas
			float det = M.at<float>(0, 0) * M.at<float>(1, 1) - M.at<float>(0, 1) * M.at<float>(1, 0);
			float trace = M.at<float>(0, 0) + M.at<float>(1, 1);
			respond.at<float>(i, j) = det - k * trace * trace;
		}
	}
	double minVal, maxVal;
	minMaxLoc(respond, &minVal, &maxVal);
	Mat keep;
	Mat(respond >= maxVal * 0.01).convertTo(keep, CV_32F);
	respond = keep.mul(respond);
	respond = NonMaximumSuppression(respond);		
	keypoints = Mat(0, 128, CV_32F);	
	positions = Mat(0, 2, CV_32F);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			if (respond.at<float>(i, j) == 1)
			{								
				Mat tmp = GetFeatureVector(i, j, gray);				
				if (tmp.empty() == false)
				{
					keypoints.push_back(tmp);
					Mat tmp(1, 2, CV_32F);					
					tmp.at<float>(0, 0) = i;
					tmp.at<float>(0, 1) = j;
					positions.push_back(tmp);
				}					
			}					
	return respond;
}

/*
@summary Generate a laplacian of gaussian kernel
@param size The size of the kernel
@param sigma The sigma of the gaussian
@returns a Mat object that contains the filter
*/
Mat GenerateLoG(int size, float sigma)
{
	// Create the Mat object
	Mat res = Mat::zeros(size, size, CV_32F);
	float var = sigma * sigma;
	// Fill values into the kernel
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
		{
			int x = i - size / 2;
			int y = j - size / 2;
			float tmp = 1.0f * (x * x + y * y) / 2 / var;
			float val = 1.0f / PI / (var * var) * (tmp - 1) * exp(-tmp);
			res.at<float>(i, j) = val;
		}
	return res;
}

/*
@summary Detect blobs on a given gray-scale image
@param image The given gray-scale image
@param nScales The number of scales for detection
@param sigma The initialize scale
@param k A scalar that indicates how the size will stretch
@returns A mat object that contains the radius of each blob
*/
Mat DetectBlob(Mat image, int nScales, float sigma, float k)
{		
	// Create the kernels
	Mat * kernels = new Mat[nScales];
	float * sigmas = new float[nScales];	
	for (int i = 0; i < nScales; i++)
	{
		int size = (int)sigma * 5;
		if (size % 2 == 0) size++;
		kernels[i] = GenerateLoG(size, sigma) * sigma * sigma; // Normalized Laplaceian of gaussian
		sigmas[i] = sigma;
		sigma *= k; // Stretch the scale
	}	
	// Aplly the kernels
	Mat * responds = new Mat[nScales];
	for (int i = 0; i < nScales; i++)	
		responds[i] = ApplyKernel(kernels[i], image);				
	int n = image.rows, m = image.cols;	
	// Threshold
	for (int i = 0; i < nScales; i++)
	{
		double minVal, maxVal;
		minMaxLoc(responds[i], &minVal, &maxVal);
		Mat keep;
		Mat(responds[i] >= maxVal * 0.3).convertTo(keep, CV_32F);
		responds[i] = responds[i].mul(keep);
	}
	
	Mat res = Mat::zeros(n, m, CV_32F);		
	for (int i = 0; i < nScales; i++) // For each scale
	{
		for(int x = 0; x < n; x++) // 
			for (int y = 0; y < m; y++)
			{
				float maxVal = -1e9f;									
				float minVal = 1e9f;
				//for (int j = -1; j <= 1; j++)
				for (int j = -i; j < nScales - i; j++)
					if (i + j >= 0 && i + j < nScales)
					{
						if (j != 0)
						{
							maxVal = max(maxVal, responds[i + j].at<float>(x, y));
							//minVal = min(minVal, responds[i + j].at<float>(x, y));
						}							
						for (int d = 0; d < 8; d++) 
						{
							int newX = x + direct8[d].first, newY = y + direct8[d].second;
							if (newX < 0 || newX >= n || newY < 0 || newY >= m) continue;
							maxVal = max(maxVal, responds[i + j].at<float>(newX, newY));												
							//minVal = min(minVal, responds[i + j].at<float>(newX, newY));
						}
					}				
				if (responds[i].at<float>(x, y) > maxVal)
					res.at<float>(x, y) = sigmas[i];
			}		
	}
	delete[] kernels;
	delete[] responds;
	delete[] sigmas;
	return res;
}

Mat HighlightBlob(Mat image, Mat blob, Vec3f color)
{
	int n = image.rows, m = image.cols;	
	Mat res;
	image.copyTo(res);
	for (int i = 0; i < n; i++) // For each pixel
		for (int j = 0; j < m; j++)
		{
			if (blob.at<float>(i, j) > 0) // If the pixel is a blob
			{
				circle(res, Point(j, i), (int)(blob.at<float>(i, j) * sqrt(2)), color, 1);
			}
		}	
	return res;
}

/*
@summary Detect blobs on a given color image
@param image The given color image
@param nScales The number of scales for detection
@param sigma The initialize scale
@param k A scalar that indicates how the size will stretch
@returns A pair of mat objects that contains the radius of each white blob and black blob
*/
pair<Mat, Mat> DetectBlobColorImage(Mat image, int nScales, float sigma, float k)
{
	// Convert the image to grayscale
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);		
	return pair<Mat, Mat>(DetectBlob(gray, nScales, sigma, k), DetectBlob(1 - gray, nScales, sigma, k));
}

/*
@summary Detect blobs on a given grayscale image.
Appoximate the LoG kernel by difference of gaussians.
@param image The given color image
@param nScales The number of scales for detection
@param sigma The initialize scale
@param k A scalar that indicates how the size will stretch
@returns A mat object that contains the radius of each blob
*/
Mat DetectDoG(Mat image, int nScales, float sigma, float k)
{	
	// Create the kernels
	Mat * kernels = new Mat[nScales];
	float * sigmas = new float[nScales];
	for (int i = 0; i < nScales; i++)
	{
		int size = (int)sigma * 5;
		if (size % 2 == 0) size++;
		kernels[i] = GenerateGaussian(size, sigma * k) - GenerateGaussian(size, sigma); // Difference of gaussians
		sigmas[i] = sigma;
		sigma *= k; // Stretch the scale
	}	
	// Aplly the kernels	
	Mat * responds = new Mat[nScales];
	for (int i = 0; i < nScales; i++)
		responds[i] = ApplyKernel(kernels[i], image);	
	int n = image.rows, m = image.cols;
	// Threshold
	for (int i = 0; i < nScales; i++)
	{
		double minVal, maxVal;
		minMaxLoc(responds[i], &minVal, &maxVal);
		Mat keep;
		Mat(responds[i] >= maxVal * 0.3).convertTo(keep, CV_32F);
		responds[i] = responds[i].mul(keep);
	}

	Mat res = Mat::zeros(n, m, CV_32F);
	for (int i = 0; i < nScales; i++) // For each scale
	{
		for (int x = 0; x < n; x++) // 
			for (int y = 0; y < m; y++)
			{
				float maxVal = -1e9f;
				float minVal = 1e9f;
				//for (int j = -1; j <= 1; j++)
				for (int j = -i; j < nScales - i; j++)
					if (i + j >= 0 && i + j < nScales)
					{
						if (j != 0)
						{
							maxVal = max(maxVal, responds[i + j].at<float>(x, y));
							//minVal = min(minVal, responds[i + j].at<float>(x, y));
						}
						for (int d = 0; d < 8; d++)
						{
							int newX = x + direct8[d].first, newY = y + direct8[d].second;
							if (newX < 0 || newX >= n || newY < 0 || newY >= m) continue;
							maxVal = max(maxVal, responds[i + j].at<float>(newX, newY));
							//minVal = min(minVal, responds[i + j].at<float>(newX, newY));
						}
					}
				if (responds[i].at<float>(x, y) > maxVal)
					res.at<float>(x, y) = sigmas[i];
			}
	}
	delete[] kernels;
	delete[] responds;
	delete[] sigmas;
	return res;
}

/*
@summary Detect blobs on a given color image.
Appoximate the LoG kernel by difference of gaussians.
@param image The given color image
@param nScales The number of scales for detection
@param sigma The initialize scale
@param k A scalar that indicates how the size will stretch
@returns A pair of mat objects that contains the radius of each white blob and black blob
*/
pair<Mat, Mat> DetectDoGColorImage(Mat image, int nScales, float sigma, float k)
{
	// Convert the image to grayscale
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);	
	return pair<Mat, Mat>(DetectDoG(gray, nScales, sigma, k), DetectBlob(1 - gray, nScales, sigma, k));
}

Mat GetOrientHistogram(Mat image, int r, int c, int size, int bins = 8)
{		
	int n = image.rows, m = image.cols;
	Mat buckets = Mat::zeros(1, bins, CV_32F);
	// Sobel operators
	float kernelX[] = { 0, 0, 0,
						-1, 0, 1,
						0, 0, 0 };
	float kernelY[] = { 0, -1, 0,
						0, 0, 0,
						0, 1, 0 };
	Mat matKernelX = Mat(3, 3, CV_32F, kernelX) / 2;
	Mat matKernelY = Mat(3, 3, CV_32F, kernelY) / 2;
	for(int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
		{			
			float dx, dy;
			try
			{
				dx = Correlation(matKernelX, Mat(image, Rect(c + j - 1, r + i - 1, 3, 3)));
				dy = Correlation(matKernelY, Mat(image, Rect(c + j - 1, r + i - 1, 3, 3)));
			}
			catch(exception e){
				return Mat(0, 0, CV_32F);
			}
			float angle = atan2(dy, dx);
			float magnitude = sqrt(dx * dx + dy * dy);
			if (angle < 0)
				angle += PI * 2;
			buckets.at<float>(0, (int)(angle / (PI * 2 / bins))) += magnitude;
		}
	return buckets / sum(buckets)[0];
}

Mat GetFeatureVector(int r, int c, Mat image)
{
	int n = image.rows, m = image.cols;
	if (r - 8 < 0 || r + 8 > n || c - 8 < 0 || c + 8 > m)
		return Mat(0, 0, CV_32F);
	Mat buckets = GetOrientHistogram(image, r - 8, c - 8, 16, 36);
	if (buckets.empty())
		return buckets;
	int maxPos = 0;
	for (int i = 1; i < 36; i++)
		if (buckets.at<float>(0, i) > buckets.at<float>(0, maxPos))
			maxPos = i;
	Mat rotate = getRotationMatrix2D(Point(c, r), PI * 2 / 36 * (2 * maxPos + 1) / 2 * 360 / PI / 2, 1);
	Mat rotated_image;
	warpAffine(image, rotated_image, rotate, Size(m, n));
	Mat tmp;
	/*if (r == 158 && c == 160)
	{
		cvtColor(image, tmp, COLOR_GRAY2BGR);
		circle(tmp, Point(c, r), 3, { 0, 0, 1 });
		namedWindow("image", WINDOW_AUTOSIZE);
		imshow("image", tmp);
		waitKey(0);

		cvtColor(rotated_image, tmp, COLOR_GRAY2BGR);
		circle(tmp, Point(c, r), 3, { 0, 0, 1 });
		namedWindow("image", WINDOW_AUTOSIZE);
		imshow("image", tmp);
		waitKey(0);
	}
	cout << r << ' ' << c << ' ' << -PI * 2 / 36 * (2 * maxPos + 1) / 2 * 360 / PI / 2 << '\n';		*/
	Mat res(1, 128, CV_32F);
	int t = 0;	
	for(int i = -8; i < 8; i += 4)
		for (int j = -8; j < 8; j += 4)
		{
			int newr = r + i, newc = c + j;												
			Mat buckets = GetOrientHistogram(rotated_image, newr, newc, 4, 8);			
			if (buckets.empty())
				return buckets;			
			for(int i = 0; i < buckets.cols; i++)
			{
				res.at<float>(0, t) = buckets.at<float>(0, i);
				t++;
			}				
		}		
	return res;
}

struct cmp
{
	vector<float> data;
	cmp(vector<float> data) : data(data)
	{		
	}
	bool operator()(int i, int j) const
	{
		return data[i] < data[j];
	}
};

Mat MatchBySift(Mat image1, Mat image2, int detector)
{	
	Mat res = Mat::zeros(max(image1.rows, image2.rows), image1.cols + image2.cols, CV_32FC3);
	image1.copyTo(res(Rect(0, 0, image1.cols, image1.rows)));
	image2.copyTo(res(Rect(image1.cols, 0, image2.cols, image2.rows)));	

	Mat keypoints1, keypoints2, pos1, pos2;
	DetectHarris(image1, keypoints1, pos1);
	DetectHarris(image2, keypoints2, pos2);
	Ptr<ml::KNearest> knn(ml::KNearest::create());
	Mat labels = Mat::zeros(keypoints1.rows, 1, CV_32F);
	for (int i = 0; i < labels.rows; i++)
		labels.at<float>(i, 0) = i;	
	knn->train(keypoints1, ml::ROW_SAMPLE, labels);
	vector<float> distance(keypoints2.rows);
	vector<Point> firstPoint(keypoints2.rows), secondPoint(keypoints2.rows);
	vector<int> id(keypoints2.rows);
	for(int i = 0; i < keypoints2.rows; i++)
	{
		id[i] = i;
		Mat respond, dist;				
		knn->findNearest(keypoints2.row(i), 1, noArray(), respond, dist);		
		Point A(pos2.at<float>(i, 1) + image1.cols, pos2.at<float>(i, 0));
		int match = respond.at<float>(0, 0);			
		Point B(pos1.at<float>(match, 1), pos1.at<float>(match, 0));			
		distance[i] = dist.at<float>(0, 0);
		firstPoint[i] = A;
		secondPoint[i] = B;
	}	
	sort(id.begin(), id.end(), cmp{ distance });
	for (int i = 0; i < 20; i++)
	{
		line(res, firstPoint[id[i]], secondPoint[id[i]], { 0,0,1 });
	}
	return res;
}