#pragma once
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/core/operations.hpp"
#include <string>
using namespace cv;
using namespace std;
using namespace ml;
const float PI = (float)3.14159265358979323846;
typedef pair<int, int> ii;
const ii direct8[8] = { ii(-1, -1), ii(-1, 0), ii(-1, 1), ii(0, -1), ii(0, 1), ii(1, -1), ii(1, 0), ii(1, 1) };
Mat LoadImage(string path);
float Correlation(Mat first, Mat second);
float Convolve(Mat first, Mat second);
Mat ApplyKernel(Mat kernel, Mat image);
Mat Normalize(Mat image);
void PartialDerivative(Mat image, Mat & dx, Mat & dy);
Mat GenerateGaussian(int size, float sigma);
Mat DetectHarris(Mat image, Mat & keypoints, Mat & positions, float k = 0.01);
Mat NonMaximumSuppression(Mat image);
Mat HighlightCorner(Mat image, Mat corner, Vec3f color = { 0, 0, 1 });
Mat DetectBlob(Mat image, int nScales = 10, float sigma = 1.6, float k = cbrt(2));
Mat GenerateLoG(int size, float sigma);
Mat HighlightBlob(Mat image, Mat blob, Vec3f color = { 0, 0, 1 });
pair<Mat, Mat> DetectBlobColorImage(Mat image, int nScales = 10, float sigma = 1.6, float k = cbrt(2));
Mat DetectDoG(Mat image, int nScales = 10, float sigma = 1.6, float k = cbrt(2));
pair<Mat, Mat> DetectDoGColorImage(Mat image, int nScales = 10, float sigma = 1.6, float k = cbrt(2));
Mat GetFeatureVector(int r, int c, Mat image);
Mat MatchBySift(Mat image1, Mat image2, int detector);