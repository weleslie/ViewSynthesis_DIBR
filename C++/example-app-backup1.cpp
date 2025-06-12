// #include <torch/script.h>
// #include <torch/torch.h>
// #include <iostream>
// #include <string>
// #include <memory>
// #include <opencv2/opencv.hpp>
// #include <time.h>
// #include <unistd.h>
 
// using namespace std;
// using namespace cv;

// void forward_warp(unsigned char* imgl, unsigned char* displ, unsigned char* novel, int height, int width) {

//     for (int i = 0; i < height; i++) {
//         for (int j = 0; j < width; j++) {
//             int count = i * width;
//             int delta = j + (int)displ[(count + j) * 3];
//             if (delta >= 0) {
//                 novel[(count + delta) * 3] = imgl[(count + j) * 3];
//                 novel[(count + delta) * 3 + 1] = imgl[(count + j) * 3 + 1];
//                 novel[(count + delta) * 3 + 2] = imgl[(count + j) * 3 + 2];
//             }
//         }
//     }

// }

// int main() {
//     int sz1 = 256;
//     int sz2 = 256;

//     std::cout << "CUDA:   " << torch::cuda::is_available() << std::endl;
//     std::cout << "CUDNN:  " << torch::cuda::cudnn_is_available() << std::endl;
//     std::cout << "GPU(s): " << torch::cuda::device_count() << std::endl;

//     //Deserialize the ScriptModule from a file
//     cout << "start!" << endl;
//     torch::jit::script::Module module = torch::jit::load("/home/weleslie/example-app/RAFT_Flow_5.pt");
//     module.to(torch::kCUDA);
//     cout << "loaded!" << endl; 
//     Mat image1 = imread("/home/weleslie/example-app/4.jpg");
//     Mat image2 = imread("/home/weleslie/example-app/5.jpg");

//     Mat img_transfomed1;
//     resize(image1, img_transfomed1, Size(sz1, sz2));

//     Mat img_transfomed2;
//     resize(image2, img_transfomed2, Size(sz1, sz2));

//     torch::Tensor tensor_image1 = torch::from_blob(img_transfomed1.data, 
//         {img_transfomed1.rows, img_transfomed1.cols, img_transfomed1.channels()}, torch::kByte);
//     tensor_image1 = tensor_image1.permute({ 2, 0, 1 });
//     tensor_image1 = tensor_image1.toType(torch::kFloat);
//     tensor_image1 = tensor_image1.unsqueeze(0);
//     std::vector<torch::jit::IValue> inputs1;

//     torch::Tensor tensor_image2 = torch::from_blob(img_transfomed2.data, { img_transfomed2.rows, img_transfomed2.cols, img_transfomed2.channels() }, torch::kByte);
//     tensor_image2 = tensor_image2.permute({ 2, 0, 1 });
//     tensor_image2 = tensor_image2.toType(torch::kFloat);
//     tensor_image2 = tensor_image2.unsqueeze(0);

//     torch::Tensor tensor_image = torch::cat({ tensor_image1,tensor_image2 }, 1);
//     torch::Tensor tensor_group = torch::cat({ tensor_image, tensor_image, tensor_image, tensor_image, tensor_image}, 0);
//     cout << tensor_group.sizes() << endl;

//     std::vector<torch::jit::IValue> inputs;
//     // tensor_image = tensor_image.to(torch::kCUDA);
//     // inputs.push_back(tensor_image);
//     tensor_group = tensor_group.to(torch::kCUDA);
//     inputs.push_back(tensor_group);

//     clock_t time_start = clock();
//     torch::Tensor output = module.forward(inputs).toTensor();
//     clock_t time_start2 = clock();

//     cout << "First inference time = " << ((time_start2 - time_start) * 1.0 / CLOCKS_PER_SEC) << "s" << endl;

//     float ttt[100] = { 0. };
//     float sum = 0.0;
//     for (int i = 0; i < 100; i++) {
//         clock_t another_start = clock();
//         cout << i << endl;
//         output = module.forward(inputs).toTensor();  
//         clock_t another_end = clock();
//         if (i > 0){
//             ttt[i] = ((another_end - another_start) * 1.0 / CLOCKS_PER_SEC);
//             sum += ttt[i];
//             cout << "Average Time: " << sum / (i * 1.0) << endl;
//         }
//     }
    

//     // torch::Tensor output1=torch::select(output, 1, 0);
//     // // cout << output1.sizes();
//     // torch::Tensor output2 = torch::cat({ output1,output1,output1 }, 0);
    
//     // //output1 = output1.squeeze();

//     // torch::Tensor ten_wrp = output2.detach().permute({ 1, 2, 0 });
//     // // cout<<ten_wrp.sizes();
//     // //ten_wrp = ten_wrp.mul(255).clamp(0, 255).to(torch::kU8);
//     // ten_wrp = ten_wrp.clamp(0, 255).to(torch::kU8);
//     // cv::Mat resultImg(sz1, sz2, CV_8UC3);
//     // memcpy((void *)resultImg.data, ten_wrp.data_ptr(), sizeof(torch::kU8) * ten_wrp.numel());

//     // unsigned char *disp = resultImg.data;
//     // unsigned char *image = img_transfomed1.data;

//     // Mat novelview(sz1, sz2, CV_8UC3);
//     // novelview = novelview.mul(0.);
//     // unsigned char *view = novelview.data;
//     // clock_t t1 = clock();
//     // for (int i = 0; i < 100; i++) {
//     //     forward_warp(image, disp, view, sz1, sz2);
//     // }    
//     // clock_t t2 = clock();
//     // cout << "Warp time = " << (t2 - t1) * 1.0 / CLOCKS_PER_SEC / 100. << endl;

//     // Mat matImg = Mat(sz1, sz2, CV_8UC3, view, 0);
//     // imwrite("view1.jpg", img_transfomed1);
//     // imwrite("view2.jpg", matImg);
//     // imwrite("view3.jpg", img_transfomed2);

//     return 0;
// }

#include "torch/script.h" // One-stop header.
#include "torch/torch.h"
#include <iostream> 
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/types_c.h"
#include <time.h>
#include "cuda_runtime.h"
#include <unistd.h>

using namespace cv;
using namespace std;

unsigned int WIDTH = 1024;
unsigned int HEIGHT = 512;

int VIRTUAL_NUMBER = 5;
int INPUT_NUMBER = 6;
int START = 1;

void backward_warp(unsigned char* imgr, signed char* displ, unsigned char* novel, int height, int width, float delta1) {

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int count = i * width;
            float dis = (float)(displ[(count + j)] * -delta1);
            int d_f = (int)(floor(dis));

            float w_f = d_f + 1 - dis;
            float w_c = dis - d_f;

            if (j + d_f >= 0 && j + d_f < width){
//                novel[(count + j) * 3] = w_f * imgr[(count + j + d_f) * 3] + w_c * imgr[(count + j + d_f + 1) * 3];
//                novel[(count + j) * 3 + 1] = w_f * imgr[(count + j + d_f) * 3 + 1] + w_c * imgr[(count + j + d_f + 1) * 3 + 1];
//                novel[(count + j) * 3 + 2] = w_f * imgr[(count + j + d_f) * 3 + 2] + w_c * imgr[(count + j + d_f + 1) * 3 + 2];
//
                novel[(count + j) * 3] = imgr[(count + j + d_f) * 3];
                novel[(count + j) * 3 + 1] = imgr[(count + j + d_f) * 3 + 1];
                novel[(count + j) * 3 + 2] = imgr[(count + j + d_f) * 3 + 2];
            }

        }
    }
}

void flow_warp(vector<Mat> images, torch::jit::script::Module module, int width, int height)
{
    float sum = 0.0;

    std::vector<unsigned char *> d_imgs;
    for (size_t i = 0; i < images.size(); i++){
        unsigned char* d_img_src;
        cudaMalloc((void**)&d_img_src, WIDTH * HEIGHT * 3);
        cudaMemcpy(d_img_src, (void *)images[i].data, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyHostToDevice);

        d_imgs.push_back(d_img_src);
    }

    clock_t s = clock();
    vector<torch::Tensor> tensor_images;
    torch::Tensor tensor_temp, tensor_temp2;
    torch::Tensor tensor_group;
    for (size_t i = 0; i < images.size(); i++){
        tensor_temp = torch::from_blob(d_imgs[i], { images[i].rows,  images[i].cols, images[i].channels() }, torch::kByte);
        tensor_temp = tensor_temp.to(torch::kCUDA);
        tensor_temp = tensor_temp.permute({ 2, 0, 1 });
        tensor_temp = tensor_temp.toType(torch::kFloat);
        tensor_temp = tensor_temp.unsqueeze(0);
        tensor_images.push_back(tensor_temp);
    }

    for (size_t i = 0; i < images.size()-1; i++){
        tensor_temp2 = torch::cat({ tensor_images[i], tensor_images[i+1] }, 1);
        if (i == 0)
            tensor_group = tensor_temp2;
        else
            tensor_group = torch::cat({tensor_group, tensor_temp2}, 0);
    }
    cout << tensor_group.sizes() << endl;
    clock_t e = clock();
    sum += (e - s) * 1.0 / CLOCKS_PER_SEC;
    cout << "memcpy time = " << sum << "s" << endl;

//    vector<torch::Tensor> tensor_images;
//    torch::Tensor tensor_image1, tensor_image2, tensor_temp;
//    for (size_t i = 0; i < images.size()-1; i++){
//        clock_t s = clock();
//        unsigned char* d_img_src;
//        cudaMalloc((void**)&d_img_src, WIDTH * HEIGHT * 3);
//        cudaMemcpy(d_img_src, (void *)images[i].data, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyHostToDevice);
//        clock_t e = clock();
//        sum += (e - s) * 1.0 / CLOCKS_PER_SEC;
//        cout << "memcpy time = " << sum << "s" << endl;
//        tensor_image1 = torch::from_blob(d_img_src, { images[i].rows,  images[i].cols, images[i].channels() }, torch::kByte);
//        std::cout << tensor_image1.sizes() << std::endl;
//
//        tensor_image1 = tensor_image1.to(torch::kCUDA);
//        tensor_image1 = tensor_image1.permute({ 2, 0, 1 });
//        tensor_image1 = tensor_image1.toType(torch::kFloat);
//        tensor_image1 = tensor_image1.unsqueeze(0);
//
//        tensor_image1 = tensor_image1.to(torch::kCPU);
//
//        tensor_image2 = torch::from_blob(images[i+1].data, { images[i+1].rows,  images[i+1].cols, images[i+1].channels() }, torch::kByte);
//        tensor_image2 = tensor_image2.permute({ 2, 0, 1 });
//        tensor_image2 = tensor_image2.toType(torch::kFloat);
//        tensor_image2 = tensor_image2.unsqueeze(0);
//
//        tensor_temp = torch::cat({ tensor_image1, tensor_image2 }, 1);
//
//        tensor_images.push_back(tensor_temp);
//    }
//
//    torch::Tensor tensor_group;
//    for (size_t i = 0; i < tensor_images.size(); i++){
//        if (i == 0)
//            tensor_group = tensor_images[i];
//        else
//            tensor_group = torch::cat({tensor_group, tensor_images[i]}, 0);
//    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_group);
    torch::Tensor output = module.forward(inputs).toTensor();

// --------------------------------------test flow time---------------------------------------------------
    clock_t time_start, time_start2;
    output = module.forward(inputs).toTensor();

    for (size_t i = 0; i < 10; i++){
        time_start = clock();
        output = module.forward(inputs).toTensor();
        time_start2 = clock();
        cout << "flow time = " << ((time_start2 - time_start) * 1.0 / CLOCKS_PER_SEC) << "s" << endl;
    }
// --------------------------------------test flow time end-----------------------------------------------

    time_start = clock();
//    output = output.to(torch::kCPU);        // 330ms

    torch::Tensor output1, ten_wrp;
    cv::Mat resultImg(height, width, CV_8SC1);
    Mat flow;
    cv::Mat novelview(height, width, CV_8UC3);

    signed char* d_img_src;
    cudaMalloc((void**)&d_img_src, WIDTH * HEIGHT * 3);
    for (size_t i = 0; i < output.sizes()[0]; ++i){
        output1 = torch::select(output, 0, i);
        cout << output1.sizes() << endl;
        output1 = torch::select(output1, 0, 0);
        output1 = output1.unsqueeze(0);
        ten_wrp = output1.detach().permute({ 1, 2, 0 });
        ten_wrp = ten_wrp.clamp(-50, 50).to(torch::kInt8);

        cudaMemcpy(d_img_src, (void *)ten_wrp.data_ptr(), sizeof(torch::kInt8) * ten_wrp.numel(), cudaMemcpyDeviceToDevice);
        cudaMemcpy((void *)resultImg.data, d_img_src, sizeof(signed char) * ten_wrp.numel(), cudaMemcpyDeviceToHost);

//        imwrite("disp.png", resultImg);
//        imshow("window1", resultImg);
//        waitKey(0);

        auto *disp = (schar *)resultImg.data;
        unsigned char *image = images[i+1].data;


        for (size_t j = 0; j < VIRTUAL_NUMBER; j++) {
//            novelview = novelview.mul(0.);        // 250ms
            unsigned char *view = novelview.data;

            float temp = (j + 1) / ((VIRTUAL_NUMBER + 1) * 1.0);
            backward_warp(image, disp, view, height, width, temp);

//            imwrite("training" + to_string(i+1) + "_" + to_string(VIRTUAL_NUMBER-j) + ".jpg", novelview);
        }

//        if (i == 0)
//            imwrite("training" + to_string(i+1) + "_0.jpg", images[i]);
//        imwrite("training" + to_string(i+1) + "_6.jpg", images[i+1]);
    }
    time_start2 = clock();
    cout << "warp time = " << ((time_start2 - time_start) * 1.0 / CLOCKS_PER_SEC) << "s" << endl;


//    Mat matImg = Mat(HEIGHT, WIDTH, CV_8UC3, novelview.data, 0);
//
//    unsigned char* d_img_src;
//    unsigned char* d_img_tmp1;
//    unsigned char* d_img_dst;
//
//    cudaMalloc((void**)&d_img_src, WIDTH * HEIGHT * 3);
//    cudaMalloc((void**)&d_img_tmp1, WIDTH * HEIGHT * 3);
//    cudaMalloc((void**)&d_img_dst, WIDTH * HEIGHT * 3);
//
//    clock_t another_start = clock();
//    cudaMemcpy(d_img_src, (void *)novelview.data, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyHostToDevice);
//    cudaMemcpy((void *)matImg.data, d_img_dst, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);
//    clock_t another_end = clock();
//    cout << "memcpy time = " << ((another_end - another_start) * 1.0 / CLOCKS_PER_SEC) << "s" << endl;

}

extern "C"
void preprocess(uchar* d_src, uchar* d_tmp1, uchar* d_dst, int w, int h);

int main() {
    std::cout << torch::cuda::is_available() << std::endl;

    //Deserialize the ScriptModule from a file
    torch::jit::script::Module module = torch::jit::load("/home/weleslie/example-app/model_cpp.pt");

    module.to(torch::kCUDA);

    vector<Mat> images_lr;
    vector<Mat> images;
    //Mat image_temp, image_resize;
    string path = "../training_warp/10/";
    string suffix = ".bmp";

    int idx = START;
    for (int i = START; i <= INPUT_NUMBER + START; i++) {
        Mat image_temp;
        Mat image_resize;

        char a[10];
        sprintf(a, "%02d", idx);
        image_temp = imread(path + a + suffix);

        resize(image_temp, image_resize, Size(WIDTH, HEIGHT));
        images.push_back(image_resize);

        idx += 2;
    }

    flow_warp(images, module, int(WIDTH), int(HEIGHT));

    ////imshow("window", disp);
    ////waitKey(0);

    //// close operation
    //Mat matImg = Mat(256, 256, CV_8UC3, novelview.data, 0);

//    unsigned char* d_img_src;
//    unsigned char* d_img_tmp1;
//    unsigned char* d_img_dst;
//
//    cudaMalloc((void**)&d_img_src, WIDTH * HEIGHT * 3);
//    cudaMalloc((void**)&d_img_tmp1, WIDTH * HEIGHT * 3);
//    cudaMalloc((void**)&d_img_dst, WIDTH * HEIGHT * 3);

    //another_start = GetTickCount();
    //cudaMemcpy(d_img_src, (void *)novelview.data, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyHostToDevice);
    //preprocess(d_img_src, d_img_tmp1, d_img_dst, WIDTH, HEIGHT);
    //cudaMemcpy((void *)matImg.data, d_img_dst, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);
    //another_end = GetTickCount();
    //cout << "close operation time = " << ((another_end - another_start) * 1.0 / 1000.0) << "s" << endl;
    //// close operation end

    return 0;
}