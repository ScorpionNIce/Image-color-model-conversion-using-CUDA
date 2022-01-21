#include<iostream>
#include<cstdio>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cuda_runtime.h>

// image container variables
cv::Mat image, rgbImage, yuvImage, grayImage;

__global__ void bgr_to_rgb_kernel(uint8_t* input,
	int width,
	int height,
	int colorWidthStep)
{
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Only valid threads perform memory I/O
	if ((xIndex < width) && (yIndex < height))
	{
		// Location of colored pixel in input
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

		// Exchanging B and R channel values (BGR -> RGB)
		const uint8_t t = input[color_tid + 0];
		input[color_tid + 0] = input[color_tid + 2];
		input[color_tid + 2] = t;
	}
}

__global__ void bgr_to_ycrcb_kernel(uint8_t* input,
	int width,
	int height,
	int colorWidthStep)
{
	const float scale = 257.0f / 65535.0f;
	const float offset = 257.0f;

	//2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((xIndex < width) && (yIndex < height))
	{
		//Location of colored pixel in input
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

		auto B = input[color_tid + 0];
		auto G = input[color_tid + 1];
		auto R = input[color_tid + 2];

		// ITU-R BT.601 conversion (RGB to YCbCr conversion formula for Digital media)
		// Source Link : https://en.wikipedia.org/wiki/YCbCr
		auto Y = (R * 65.481f * scale) + (G * 128.553f * scale) + (B * 24.966f * scale) + (16.0f * offset);
		auto Cr = (R * 112.0f * scale) + (G * -93.786f * scale) + (B * -18.214f * scale) + (128.0f * offset);
		auto Cb = (R * -37.797f * scale) + (G * -74.203f * scale) + (B * 112.0f * scale) + (128.0f * offset);

		input[color_tid + 0] = (uint8_t)Y;
		input[color_tid + 1] = (uint8_t)Cr;
		input[color_tid + 2] = (uint8_t)Cb;
	}
}

__global__ void bgr_to_gray_kernel(uint8_t* input1,
	uint8_t* input2,
	int width,
	int height,
	int colorWidthStep1,
	int colorWidthStep2)
{
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Only valid threads perform memory I/O
	if ((xIndex < width) && (yIndex < height))
	{
		// Location of colored pixel in input
		const int color_tid1 = yIndex * colorWidthStep1 + xIndex;
		const int color_tid2 = yIndex * colorWidthStep2 + (3 * xIndex);

		auto B = input2[color_tid2 + 0];
		auto G = input2[color_tid2 + 1];
		auto R = input2[color_tid2 + 2];

		// ITU-R BT.709 standard used for HDTV developed by the ATSC
		// Source Link : https://en.wikipedia.org/wiki/Grayscale
		// Y'=0.2126R'+0.7152G'+0.0722B'
		auto L = 0.2126 * R + 0.7152 * G + 0.0722 * B;
		//auto L = 0.2627 * R + 0.6780 * G + 0.0593 * B;

		input1[color_tid1] = (uint8_t)L;
	}
}

void bgr_to_rgb(cv::Mat& input) {
	// create copy of original image into input image
	input = image.clone();

	// calculating memory allocation size for GPU memory allocation
	const int Bytes = input.step * input.rows;

	// array in GPU to store Mat data
	uint8_t *d_input;

	//allocating size in GPU and copying data from host to GPU
	cudaMalloc((uint8_t  **)&d_input, sizeof(uint8_t) *Bytes);
	cudaMemcpy(d_input, input.data, sizeof(uint8_t) * Bytes, cudaMemcpyHostToDevice);

	// creating 3D block for GPU kernel
	dim3 block(16, 16);
	dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	std::cout << (input.cols + block.x - 1) / block.x << "  " << (input.rows + block.y - 1) / block.y << std::endl;

	// Initiate GPU processing and wait till process completion
	bgr_to_rgb_kernel << < grid, block >> > (d_input, input.cols, input.rows, input.step); //step – Number of bytes each matrix row occupies
	cudaDeviceSynchronize();


	// Copy back calculated data into host and free GPU memory
	cudaMemcpy(input.data, d_input, sizeof(uint8_t) * Bytes, cudaMemcpyDeviceToHost);
	cudaFree(d_input);

	return;
}

void bgr_to_ycrcb(cv::Mat& input) {
	input = image.clone();

	const int Bytes = input.step * input.rows;

	uint8_t *d_input;
	cudaMalloc((uint8_t  **)&d_input, sizeof(uint8_t) * Bytes);
	cudaMemcpy(d_input, input.data, sizeof(uint8_t) * Bytes, cudaMemcpyHostToDevice);

	dim3 block(16, 16);
	dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	std::cout << (input.cols + block.x - 1) / block.x << "  " << (input.rows + block.y - 1) / block.y << std::endl;

	bgr_to_ycrcb_kernel << < grid, block >> > (d_input, input.cols, input.rows, input.step);
	cudaDeviceSynchronize();

	cudaMemcpy(input.data, d_input, sizeof(uint8_t) * Bytes, cudaMemcpyDeviceToHost);
	cudaFree(d_input);

	return;
}

void bgr_to_gray(cv::Mat& input1) {
	input1.create(image.rows, image.cols, CV_8UC1);
	cv::Mat input2 = image.clone();

	const int Bytes1 = input1.step * input1.rows;
	const int Bytes2 = input2.step * input2.rows;

	uint8_t *d_input1;
	uint8_t *d_input2;
	cudaMalloc((uint8_t **)&d_input1, sizeof(uint8_t) * Bytes1);
	cudaMalloc((uint8_t **)&d_input2, sizeof(uint8_t) * Bytes2);
	cudaMemcpy(d_input2, input2.data, sizeof(uint8_t) * Bytes2, cudaMemcpyHostToDevice);

	dim3 block(16, 16);
	dim3 grid((input1.cols + block.x - 1) / block.x, (input1.rows + block.y - 1) / block.y);
	std::cout << (input1.cols + block.x - 1) / block.x << "  " << (input1.rows + block.y - 1) / block.y << std::endl;

	bgr_to_gray_kernel << < grid, block >> > (d_input1, d_input2, input1.cols, input1.rows, input1.step, input2.step);
	cudaDeviceSynchronize();

	cudaMemcpy(input1.data, d_input1, sizeof(uint8_t) * Bytes1, cudaMemcpyDeviceToHost);
	cudaFree(d_input1);
	cudaFree(d_input2);

	return;
}

void showImages() {
	cv::imshow("Original image", image);
	cv::imshow("RGB image", rgbImage);
	cv::imshow("YCrCb image", yuvImage);
	cv::imshow("Gray image", grayImage);

	cv::waitKey();

	return;
}

void saveImages() {
	cv::imwrite("img_rgb.jpg", rgbImage);
	cv::imwrite("img_yuv.jpg", yuvImage);
	cv::imwrite("img_gray.jpg", grayImage);

	return;
}

int main(int argc, char const *argv[]) {
	image = cv::imread("img.jpg");

	// Image Processing by GPU
	bgr_to_rgb(rgbImage);
	bgr_to_ycrcb(yuvImage);
	bgr_to_gray(grayImage);

	//showImages();

	saveImages();

	return 0;
}