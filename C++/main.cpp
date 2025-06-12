//#include <iostream>
//#include <memory>
//#include <stdlib.h>
//#include <stdio.h>
//#include <windows.h>
//#include "highgui/highgui.hpp"
//#include "core/core.hpp"
//#include "imgproc/imgproc.hpp"
//
//#include "cuda_runtime.h"
//
//using namespace std;
//using namespace cv;
//
//extern "C"
//void preprocess(uchar* d_src, uchar* d_tmp1, uchar* d_tmp2, uchar* d_dst, int w, int h);
//
////extern "C"
////void CallDilation(uchar* d_src, uchar* d_dst, int len, int width, int height);
//
//
//
//int main() {
//	Mat src = imread("H:/iPASSR_trainset/Flickr1024/0001_L.png", 1);
//	int w = src.cols;
//	int h = src.rows;
//	Mat srcCopy = Mat::zeros(Size(w, h), CV_8UC3);
//	Mat srcCopy2 = Mat::zeros(Size(w, h), CV_8UC3);
//	cvtColor(src, srcCopy, COLOR_RGB2GRAY);
//
//	unsigned char* d_img_src;
//	unsigned char* d_img_tmp1;
//	unsigned char* d_img_tmp2;
//	unsigned char* d_img_dst;
//
//	cudaMalloc((void**)&d_img_src, w * h * 3);
//	cudaMalloc((void**)&d_img_tmp1, w * h * 3);
//	cudaMalloc((void**)&d_img_tmp2, w * h * 3);
//	cudaMalloc((void**)&d_img_dst, w * h * 3);
//
//	unsigned char* ptr = (unsigned char*)src.ptr();
//	unsigned char* ptr2 = (unsigned char*)srcCopy2.ptr();
//
//	cudaMemcpy(d_img_src, ptr, sizeof(unsigned char) * w * h * 3, cudaMemcpyHostToDevice);
//	preprocess(d_img_src, d_img_tmp1, d_img_tmp2, d_img_dst, w, h);
//	cudaMemcpy(ptr2, d_img_dst, sizeof(unsigned char) * w * h * 3, cudaMemcpyDeviceToHost);
//
//	namedWindow("test", WINDOW_AUTOSIZE);
//	imshow("test", src);
//
//	namedWindow("test2", WINDOW_AUTOSIZE);
//	imshow("test2", srcCopy2);
//	waitKey(0);
//
//	destroyAllWindows();
//	cudaFree(d_img_src);
//	cudaFree(d_img_tmp1);
//	cudaFree(d_img_tmp2);
//	cudaFree(d_img_dst);
//
//	system("pause");
//	return 0;
//}

//#include "torch/script.h" // One-stop header.
//#include "torch/torch.h"
//#include <iostream> 
//#include "opencv2\opencv.hpp"
//#include "opencv2\imgproc\types_c.h"
//#include <time.h>
//#include <windows.h>
//#include "cuda_runtime.h"
//
//using namespace cv;
//using namespace std;
//
//unsigned int WIDTH = 512;
//unsigned int HEIGHT = 512;
//
//int VIRTUAL_NUMBER = 5;
//int INPUT_NUMBER = 2;
//
//void forward_warp(unsigned char* imgl, unsigned char* displ, unsigned char* novel, int height, int width, float delta1) {
//
//	for (int i = 0; i < height; i++) {
//		for (int j = 0; j < width; j++) {
//			int count = i * width;
//			int d = (int)((float)(displ[(count + j)] * delta1));
//			//int d = (int)(displ[(count + j)]);
//			int delta = j + d;
//			if (delta >= 0) {
//				novel[(count + delta) * 3] = imgl[(count + j) * 3];
//				novel[(count + delta) * 3 + 1] = imgl[(count + j) * 3 + 1];
//				novel[(count + delta) * 3 + 2] = imgl[(count + j) * 3 + 2];
//			}
//		}
//	}
//
//}
//
//vector<Mat> grid()
//{
//	string path = "sparse_views/8lu2/";
//	string suffix = ".jpg";
//	vector<Mat> image_vec;	
//
//	for (size_t i = 2; i < INPUT_NUMBER+2; i++) {
//		Mat image_temp, image_resize;
//
//		image_temp = imread(path + to_string(i) + suffix);
//		resize(image_temp, image_resize, Size(960, 512));
//		image_vec.push_back(image_resize);
//	}
//
//	Mat TL = image_vec[0];
//
//	int m = 9;
//	int n = 6;
//
//	Size board_size = Size(m, n);
//
//	Mat imageGray;
//	vector<cv::Point2f> corners;
//	vector<vector<cv::Point2f>> corners_array(8);
//	vector<cv::Point2f> orgcorners(4);
//	vector<cv::Point2f> targetcorners(4);
//
//	for (int i = 0; i < INPUT_NUMBER; i++)
//	{
//		Mat NOW_MAT = image_vec[i];
//
//		Mat AAA = Mat::zeros(TL.rows, TL.cols, CV_8UC3);
//		NOW_MAT.copyTo(AAA);
//		NOW_MAT = AAA;
//   
//		cv::cvtColor(NOW_MAT, imageGray, CV_RGB2GRAY);
//
//		bool patternfound = cv::findChessboardCorners(NOW_MAT, board_size, corners, cv::CALIB_CB_ADAPTIVE_THRESH);
//
//		if (!patternfound)
//		{
//			cout << "please find 4 corner by hand" << endl;
//			//orgcorners = drawcorners(NOW_MAT); //ÊÖ¹€»­µã
//			//drawpts.clear();
//		}
//		else
//		{
//			cv::cornerSubPix(imageGray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
//
//			orgcorners[0] = corners[0];
//			orgcorners[1] = corners[board_size.width - 1];
//			orgcorners[2] = corners[(board_size.width)*(board_size.height) - board_size.width];
//			orgcorners[3] = corners[(board_size.width)*(board_size.height) - 1];
//		}
//		corners_array[i] = orgcorners;
//
//		targetcorners[0] += orgcorners[0];
//		targetcorners[1] += orgcorners[1];
//		targetcorners[2] += orgcorners[2];
//		targetcorners[3] += orgcorners[3];
//	}
//
//	targetcorners[0] /= INPUT_NUMBER;
//	targetcorners[1] /= INPUT_NUMBER;
//	targetcorners[2] /= INPUT_NUMBER;
//	targetcorners[3] /= INPUT_NUMBER;
//
//	vector<Mat> Warp_M_list;
//	for (int i = 0; i < INPUT_NUMBER; i++)
//	{
//		Mat NOW_MAT = image_vec[i];
//		vector<cv::Point2f> orgcorners(4);
//		orgcorners = corners_array[i];
//
//		Mat WarpMat;
//		WarpMat = getPerspectiveTransform(orgcorners, targetcorners);
//
//		Mat warpimg;
//		cv:warpPerspective(NOW_MAT, warpimg, WarpMat, NOW_MAT.size());
//
//		//imshow("warpresult", warpimg);
//		//waitKey(30);
//
//		Warp_M_list.push_back(WarpMat);
//
//		//imwrite("H:/flow/RAFT_flow/build_debug/view_HS0.jpg", warpimg);
//	}
//
//	return Warp_M_list;
//}
//
//void flow_warp(Mat image1, Mat image2, Mat Matrix, torch::jit::script::Module module, int width, int height)
//{
//	//imshow("window1", image1);
//	//imshow("window2", image2);
//	//waitKey(0);
//
//	torch::Tensor tensor_image1 = torch::from_blob(image1.data, { image1.rows, 	image1.cols, image1.channels() }, torch::kByte);
//	tensor_image1 = tensor_image1.permute({ 2, 0, 1 });
//	tensor_image1 = tensor_image1.toType(torch::kFloat);
//	tensor_image1 = tensor_image1.unsqueeze(0);
//	std::vector<torch::jit::IValue> inputs1;
//
//	torch::Tensor tensor_image2 = torch::from_blob(image2.data, { image2.rows, 	 image2.cols, image2.channels() }, torch::kByte);
//	tensor_image2 = tensor_image2.permute({ 2, 0, 1 });
//	tensor_image2 = tensor_image2.toType(torch::kFloat);
//	tensor_image2 = tensor_image2.unsqueeze(0);
//
//	torch::Tensor tensor_image = torch::cat({ tensor_image1, tensor_image2 }, 1);
//	std::vector<torch::jit::IValue> inputs;
//	/*tensor_image = tensor_image.to(torch::kCUDA);*/
//	tensor_image = tensor_image.to(torch::kCPU);
//	inputs.push_back(tensor_image);
//
//	DWORD time_start, time_start2, time_end1 = 0;
//	time_start = GetTickCount();
//	torch::Tensor output = module.forward(inputs).toTensor();
//	time_start2 = GetTickCount();
//	//output = output.to(torch::kCPU);
//	cout << "flow time = " << ((time_start2 - time_start) * 1.0 / 1000) << "s" << endl;
//
//	cout << output.sizes() << endl;
//	torch::Tensor output1 = torch::select(output, 1, 0);
//	cout << output1.sizes() << endl;
//
//	torch::Tensor ten_wrp = output1.detach().permute({ 1, 2, 0 });
//	cout << ten_wrp.sizes() << endl;
//	//ten_wrp = ten_wrp.mul(255).clamp(0, 255).to(torch::kU8);
//	ten_wrp = ten_wrp.clamp(0, 255).to(torch::kU8);
//	cv::Mat resultImg(height, width, CV_8UC1);
//	std::memcpy((void *)resultImg.data, ten_wrp.data_ptr(), sizeof(torch::kU8) * ten_wrp.numel());
//
//	imshow("window1", resultImg*5);
//	waitKey(0);
//
//	Mat flow;
//	warpPerspective(resultImg, flow, Matrix, resultImg.size());
//
//	unsigned char *disp = flow.data;
//	unsigned char *image = image1.data;
//
//	cv::Mat novelview(height, width, CV_8UC3);
//	novelview = novelview.mul(0.);
//	unsigned char *view = novelview.data;
//
//	for (size_t i = 0; i < VIRTUAL_NUMBER; i++) {
//		time_start = GetTickCount();
//		forward_warp(image, disp, view, height, width, (i + 1) / ((VIRTUAL_NUMBER + 1) * 1.0));
//		time_start2 = GetTickCount();
//		time_end1 += ((time_start2 - time_start) * 1.0 / 1000);
//
//		imwrite("synthesis" + to_string(i+1) + ".jpg", novelview);
//	}
//	
//	cout << "warp time = " << time_end1 / (VIRTUAL_NUMBER * 1.0) << "s" << endl;
//
//	//Mat matImg1 = Mat(height, width, CV_8UC3, view1, 0);
//	//Mat matImg2 = Mat(height, width, CV_8UC3, view2, 0);
//	//Mat matImg3 = Mat(height, width, CV_8UC3, view3, 0);
//	//Mat matImg4 = Mat(height, width, CV_8UC3, view4, 0);
//	//Mat matImg5 = Mat(height, width, CV_8UC3, view5, 0);
//	//Mat matImg6 = Mat(height, width, CV_8UC3, view6, 0);
//	imwrite("synthesis0.jpg", image1);
//	//imwrite("H:/flow/RAFT_flow/build_debug/view2.jpg", matImg1);
//	//imwrite("H:/flow/RAFT_flow/build_debug/view3.jpg", matImg2);
//	//imwrite("H:/flow/RAFT_flow/build_debug/view4.jpg", matImg3);
//	//imwrite("H:/flow/RAFT_flow/build_debug/view5.jpg", matImg4);
//	//imwrite("H:/flow/RAFT_flow/build_debug/view6.jpg", matImg5);
//	//imwrite("H:/flow/RAFT_flow/build_debug/view7.jpg", matImg6);
//	imwrite("synthesis6.jpg", image2);
//}
//
//extern "C"
//void preprocess(uchar* d_src, uchar* d_tmp1, uchar* d_dst, int w, int h);
//
//int main() {
//	std::cout << torch::cuda::is_available() << std::endl;
//
//	//Deserialize the ScriptModule from a file
//	torch::jit::script::Module module = torch::jit::load("RAFT_Flow_5_cpu.pt");
//	//module.to(torch::kCUDA);
//	module.to(torch::kCPU);
//
//	vector<Mat> images;
//	//Mat image_temp, image_resize;
//	string path = "sparse_views/8lu2/";
//	string suffix = ".jpg";
//	for (size_t i = 2; i < INPUT_NUMBER+2; i++) {
//		Mat image_temp, image_resize;
//		image_temp = imread(path + to_string(i) + suffix);
//		resize(image_temp, image_resize, Size(WIDTH, HEIGHT));
//
//		images.push_back(image_resize);
//	}
//
//	DWORD time_start, time_end;
//	time_start = GetTickCount();
//	vector<Mat> Warp_M_list = grid();
//	time_end = GetTickCount();
//	cout << "H time = " << ((time_end - time_start)*1.0 / 1000) << "s" << endl;
//
//	for (int i = 0; i < INPUT_NUMBER-1; i++)
//	{
//		flow_warp(images[i+1], images[i], Warp_M_list[i], module, WIDTH, HEIGHT);
//	}
//
//	//float t = 0;
//	//for (int i = 0; i < 10; i++) {
//	//	another_start = GetTickCount();
//	//	module.forward(inputs);
//	//	another_end = GetTickCount();
//	//	std::cout << i << endl;		
//	//	t += ((another_end - another_start) * 1.0 / 1000);
//	//}
//	//std::cout << t / 10.0 << endl;
//
//
//	////imshow("window", disp);
//	////waitKey(0);
//
//	//// close operation
//	//Mat matImg = Mat(256, 256, CV_8UC3, novelview.data, 0);
//
//	//unsigned char* d_img_src;
//	//unsigned char* d_img_tmp1;
//	//unsigned char* d_img_dst;
//	//
//	//cudaMalloc((void**)&d_img_src, WIDTH * HEIGHT * 3);
//	//cudaMalloc((void**)&d_img_tmp1, WIDTH * HEIGHT * 3);
//	//cudaMalloc((void**)&d_img_dst, WIDTH * HEIGHT * 3);
//
//	//another_start = GetTickCount();
//	//cudaMemcpy(d_img_src, (void *)novelview.data, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyHostToDevice);
//	//preprocess(d_img_src, d_img_tmp1, d_img_dst, WIDTH, HEIGHT);
//	//cudaMemcpy((void *)matImg.data, d_img_dst, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);
//	//another_end = GetTickCount();
//	//cout << "close operation time = " << ((another_end - another_start) * 1.0 / 1000.0) << "s" << endl;
//	//// close operation end
//
//	system("pause");
//	return 0;
//}

#include "torch/script.h" // One-stop header. 
#include "torch/torch.h"
#include <iostream> 
#include <opencv2\opencv.hpp> 
#include <opencv2\imgproc\types_c.h> 
#include <time.h>
#include <windows.h>

using namespace cv;
using namespace std;


void forward_warp(unsigned char* imgl, unsigned char* displ, unsigned char* novel, int height, int width) {

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int count = i * width;
			int delta = j + (int)displ[(count + j) * 3];
			if (delta >= 0) {
				novel[(count + delta) * 3] = imgl[(count + j) * 3];
				novel[(count + delta) * 3 + 1] = imgl[(count + j) * 3 + 1];
				novel[(count + delta) * 3 + 2] = imgl[(count + j) * 3 + 2];
			}
		}
	}

}

int main() {
	//Deserialize the ScriptModule from a file
	std::cout << torch::cuda::is_available() << std::endl;
	std::cout << torch::cuda::cudnn_is_available() << endl;
	torch::jit::script::Module module = torch::jit::load("D:/VS Project 2017/Project1/Project1/RAFT_flow_5.pt");
	module.to(torch::kCUDA);
	//assert(module != nullptr);  
	Mat view1 = imread("./sparse_views/4.jpg");
	Mat view2 = imread("./sparse_views/8.jpg");

	Mat view1_resize, view2_resize;
	resize(view1, view1_resize, Size(256, 256));
	resize(view2, view2_resize, Size(256, 256));

	torch::Tensor tensor_view1 = torch::from_blob(view1_resize.data,
		{ view1_resize.rows, view1_resize.cols, view1_resize.channels() }, torch::kByte);
	tensor_view1 = tensor_view1.permute({ 2, 0, 1 });
	tensor_view1 = tensor_view1.toType(torch::kFloat);

	tensor_view1 = tensor_view1.unsqueeze(0);
	std::vector<torch::jit::IValue> inputs1;

	torch::Tensor tensor_view2 = torch::from_blob(view2_resize.data,
		{ view2_resize.rows, view2_resize.cols, view2_resize.channels() }, torch::kByte);
	tensor_view2 = tensor_view2.permute({ 2, 0, 1 });
	tensor_view2 = tensor_view2.toType(torch::kFloat);

	tensor_view2 = tensor_view2.unsqueeze(0);
	std::vector<torch::jit::IValue> inputs2;

	torch::Tensor tensor_views = torch::cat({ tensor_view1,tensor_view2 }, 1);
	std::vector<torch::jit::IValue> inputs;

	tensor_views = tensor_views.to(torch::kCUDA);
	inputs.push_back(tensor_views);

	DWORD time_start1, time_start2, time_end1, another_start, another_end;
	time_start1 = GetTickCount();
	torch::Tensor output = module.forward(inputs).toTensor();
	time_start2 = GetTickCount();
	cout << "First running time = " << ((time_start2 - time_start1) * 1.0 / 1000) << "s" << endl;

	float t = 0.;
	for (int i = 0; i < 100; i++) {
		another_start = GetTickCount();
		//output = module.forward(inputs).toTensor();
		module.forward(inputs);
		another_end = GetTickCount();
		t += ((another_end - another_start) * 1.0 / 1000);

		std::cout << i << std::endl;
	}
	std::cout << t / 100.0 << endl;

	//cout << output.sizes();
	//torch::Tensor output1 = torch::select(output, 1, 0);
	//cout << output1.sizes();
	//torch::Tensor output2 = torch::cat({ output1,output1,output1 }, 0);

	//torch::Tensor ten_wrp = output2.detach().permute({ 1, 2, 0 });
	//cout << ten_wrp.sizes();
	////ten_wrp = ten_wrp.mul(255).clamp(0, 255).to(torch::kU8);
	//ten_wrp = ten_wrp.clamp(0, 255).to(torch::kU8);

	//Mat disp(256, 256, CV_8UC3);

	//std::memcpy((void *)disp.data, ten_wrp.data_ptr(), sizeof(torch::kU8) * ten_wrp.numel());

	//Mat novelview(256, 256, CV_8UC3);
	//novelview = novelview.mul(0.);
	//forward_warp(view1_resize.data, disp.data, novelview.data, 256, 256);

	//time_end1 = GetTickCount();
	//cout << "time = " << ((time_end1 - another_end)*1.0 / 1000) << "s" << endl;

	//Mat matImg = Mat(256, 256, CV_8UC3, novelview.data, 0);
	//imwrite("view1.jpg", view1_resize);
	//imwrite("view2.jpg", matImg);
	//imwrite("view3.jpg", view2_resize);

}

