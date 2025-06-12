//#include <GL/glew.h>
//#include <GLFW/glfw3.h>
//
//#include "torch/script.h" // One-stop header.
//#include "torch/torch.h"
//#include <iostream>
//#include "opencv2/opencv.hpp"
//#include "opencv2/imgproc/types_c.h"
//#include <time.h>
//#include "cuda_runtime.h"
//#include <cuda_gl_interop.h>
//#include <unistd.h>
//
//bool SAVE_IMAGE = true;
//bool DISPLAY_TIME = false;
//
//using namespace cv;
//using namespace std;
//
//unsigned int WIDTH = 1024;
//unsigned int HEIGHT = 512;
//
//int OutWidth = 3840;
//int OutHeight = 2160;
//
//int VIRTUAL_NUMBER = 5;
//int INPUT_NUMBER = 10;
//int START = 11;
//
//int ViewNum = 60;
//float LineNum = 26.436;
//float InclinationAngle = 0.1683;
//float MoveValue = 0;
//float ZeroValue = 0;
//
//// OpenGL settings
//unsigned int SCR_WIDTH = 1024;
//unsigned int SCR_HEIGHT = 512;
//int monitorCount = 0;
//GLuint Buffer;
//GLuint Texture;
//struct cudaGraphicsResource *cuda_pbo_rsc;
//
//extern "C" void BackWarp(uchar* d_src, schar* disp, uchar* d_dst, int w, int h, float delta);
//extern "C" void BackEncode(unsigned char* d_src, signed char* disp, unsigned char* d_dst, int w, int h,
//                           int virtualnum,
//                           bool ifReverse,
//                           int outw, int outh,
//                           int viewnum, float LineNum,
//                           float InclinationAngle,
//                           float MoveValue, float ZeroValue);
//void fullScreen(GLFWmonitor* pMonitor, GLFWwindow* window);
//void init();
//void render(unsigned char* d_data);
//
//void flow_warp(vector<Mat>& images, torch::jit::script::Module& module, cv::Mat& novelview,
//               unsigned char *d_img_srcs, signed char *d_flow_src, unsigned char *d_img_dst,
//               unsigned char *d_img_dst2, unsigned char *d_img_code)
//{
//    float sum = 0.0;
//    clock_t time_start, time_end;
//
//
//
//    for (size_t i = 0; i < images.size(); i++){
//        cudaMemcpy(d_img_srcs + WIDTH * HEIGHT * 3 * i, (void *)images[i].data, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyHostToDevice);
//    }
//
//    vector<torch::Tensor> tensor_images;
//    torch::Tensor tensor_temp, tensor_temp2;
//    torch::Tensor tensor_group;
//    for (size_t i = 0; i < images.size(); i++){
//        tensor_temp = torch::from_blob(d_img_srcs + WIDTH * HEIGHT * 3 * i, { images[i].rows,  images[i].cols, images[i].channels() }, torch::kByte);
//        tensor_temp = tensor_temp.to(torch::kCUDA);
//        tensor_temp = tensor_temp.permute({ 2, 0, 1 });
//        tensor_temp = tensor_temp.toType(torch::kFloat);
//        tensor_temp = tensor_temp.unsqueeze(0);
//        tensor_images.push_back(tensor_temp);
//    }
//
//    for (size_t i = 0; i < images.size()-1; i++){
//        tensor_temp2 = torch::cat({ tensor_images[i], tensor_images[i+1] }, 1);
//        if (i == 0)
//            tensor_group = tensor_temp2;
//        else
//            tensor_group = torch::cat({tensor_group, tensor_temp2}, 0);
//    }
//
//    std::vector<torch::jit::IValue> inputs;
//    inputs.push_back(tensor_group);
//
//    torch::Tensor output = module.forward(inputs).toTensor();
//
//    torch::Tensor output1, ten_wrp;
//    cv::Mat resultImg(HEIGHT, WIDTH, CV_8SC1);
//
//    for (size_t i = 0; i < output.sizes()[0]; ++i){
//        output1 = torch::select(output, 0, i);
//        output1 = torch::select(output1, 0, 0);
//        output1 = output1.unsqueeze(0);
//        ten_wrp = output1.detach().permute({ 1, 2, 0 });
//        ten_wrp = ten_wrp.clamp(-50, 50).to(torch::kInt8);
//
//        cudaMemcpy(d_flow_src, (void *)ten_wrp.data_ptr(), sizeof(torch::kInt8) * ten_wrp.numel(), cudaMemcpyDeviceToDevice);
////        vector<float> temp = {0.16, 0.32, 0.48, 0.64, 0.80};
//        for (int j = 0; j < VIRTUAL_NUMBER; j++) {
//            float temp = (j + 1) / ((VIRTUAL_NUMBER + 1) * 1.0);
//
//            cudaMemcpy(d_img_dst + WIDTH * HEIGHT * 3 * i * (VIRTUAL_NUMBER + 1), d_img_srcs + WIDTH * HEIGHT * 3 * i,
//                       sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyDeviceToDevice);
//            BackWarp( d_img_srcs + WIDTH * HEIGHT * 3 * (i + 1), d_flow_src,
//                      d_img_dst + WIDTH * HEIGHT * 3 * (j + 1 + i * (VIRTUAL_NUMBER + 1)) , WIDTH, HEIGHT, temp);
//            if (SAVE_IMAGE){
//                clock_t s = clock();
//                cudaMemcpy((void *)novelview.data, d_img_dst + WIDTH * HEIGHT * 3 * (j + 1 + i * (VIRTUAL_NUMBER + 1)), sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);
//                clock_t e = clock();
//                sum += (e - s) * 1.0 / CLOCKS_PER_SEC;
//                if (DISPLAY_TIME)
//                    cout << "forward time = " << sum << "s" << endl;
//            }
//
//            if (SAVE_IMAGE)
//                imwrite("training" + to_string(i+1) + "_" + to_string(VIRTUAL_NUMBER-j) + ".jpg", novelview);
//        }
//
//        if (SAVE_IMAGE){
//            if (i == 0)
//                imwrite("training" + to_string(i+1) + "_0.jpg", images[i]);
//            imwrite("training" + to_string(i+1) + "_6.jpg", images[i+1]);
//        }
//    }
//
//    time_start = clock();
////    cudaDeviceSynchronize(); //开启设备等待
//    BackEncode(d_img_dst, d_flow_src, d_img_code, WIDTH, HEIGHT, VIRTUAL_NUMBER, 0, OutWidth, OutHeight, ViewNum,
//                               LineNum, InclinationAngle, MoveValue, ZeroValue);
//
//    time_end = clock();
//
//    cout << "warp time = " << ((time_end - time_start) * 1.0 / CLOCKS_PER_SEC) << "s" << endl;
//}
//
////---------------------------------Full Screen-------------------------------
//void fullScreen(GLFWmonitor* pMonitor, GLFWwindow* window)
//{
//    const GLFWvidmode * mode = glfwGetVideoMode(pMonitor);
//    std::cout << "Screen size is X = " << mode->width << ", Y = " << mode->height << std::endl;
//
//    SCR_WIDTH = mode->width;
//    SCR_HEIGHT = mode->height;
//
//    glfwSetWindowMonitor(window, pMonitor, 0, 0, mode->width, mode->height, GLFW_DONT_CARE);
//}
////--------------------------------------------------------------------------
//
////--------------------------------initialize buffer-------------------------
//void init() {
//    //----------------------------initialize Buffer-------------------------
//    glGenBuffers(1, &Buffer);
//    glBindBuffer(GL_ARRAY_BUFFER, Buffer);
////    glBufferData(GL_ARRAY_BUFFER, WIDTH * HEIGHT * sizeof(GLchar) * 3, NULL, GL_STREAM_DRAW);
//    glBufferData(GL_ARRAY_BUFFER, OutWidth * OutHeight * sizeof(GLchar) * 3, NULL, GL_STREAM_DRAW);
//    glBindBuffer(GL_ARRAY_BUFFER, 0);
//
//    //---------------------------link Buffer and CUDA-----------------------
//    cudaGraphicsGLRegisterBuffer(&cuda_pbo_rsc, Buffer, cudaGraphicsMapFlagsWriteDiscard);
//}
////--------------------------------------------------------------------------
//
////----------------------------Process data in CUDA--------------------------
//void render(unsigned char* d_data) {
//    cudaGraphicsMapResources(1, &cuda_pbo_rsc, 0);
//
//    size_t bytes;
//    unsigned char *d_output;
//    cudaGraphicsResourceGetMappedPointer((void **)&d_output, &bytes, cuda_pbo_rsc);
////    cudaMemcpy(d_output, d_data, WIDTH * HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
//    cudaMemcpy(d_output, d_data, OutWidth * OutHeight * 3 * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
//
//    cudaGraphicsUnmapResources(1, &cuda_pbo_rsc, 0);
//}
////------------------------------------------------------------------------
//
//int main() {
//    std::cout << torch::cuda::is_available() << std::endl;
//
//    //Deserialize the ScriptModule from a file
//    torch::jit::script::Module module = torch::jit::load("/home/weleslie/example-app/model_cpp_1.pt");
//    module.to(torch::kCUDA);
//
//    torch::Tensor ten = torch::rand({ 1, 6, 256, 256 }, torch::kFloat).to(torch::kCUDA);
//    std::vector<torch::jit::IValue> inputs;
//    inputs.push_back(ten);
//    torch::Tensor tensor = module.forward(inputs).toTensor();
//    module.forward(inputs);
//
//    vector<Mat> images;
//    string path = "../training_model1_warp_cc/";
//    string suffix = ".jpg";
//
//    int idx = START;
////    for (int i = START; i <= INPUT_NUMBER + START; i++) {
////        Mat image_temp;
////        Mat image_resize;
////
////        char a[10];
////        sprintf(a, "%02d", idx);
////        image_temp = imread(path + a + suffix);
////
////        cvtColor(image_temp, image_temp, CV_BGR2RGB);
////        resize(image_temp, image_resize, Size(WIDTH, HEIGHT));
////        images.push_back(image_resize);
////
////        idx += 2;
////    }
//
////    Mat novelview(OutHeight, OutWidth, CV_8UC3);
//    Mat novelview(HEIGHT, WIDTH, CV_8UC3);
//    unsigned char* d_img_srcs;
//    cudaMalloc((void**)&d_img_srcs, WIDTH * HEIGHT * 3 * (INPUT_NUMBER + 1));
//    signed char* d_flow_src;
//    cudaMalloc((void**)&d_flow_src, WIDTH * HEIGHT);
//    unsigned char* d_img_dst;
//    cudaMalloc((void**)&d_img_dst, WIDTH * HEIGHT * 3 * (VIRTUAL_NUMBER + 1) * INPUT_NUMBER);
//    unsigned char* d_img_dst2;
//    cudaMalloc((void**)&d_img_dst2, WIDTH * HEIGHT * 3 * (VIRTUAL_NUMBER + 1) * INPUT_NUMBER);
//    unsigned char* d_img_code;
//    cudaMalloc((void**)&d_img_code, OutWidth * OutHeight * 3);
//
//    // glfw: initialize and configure
//    // ------------------------------
//    // Initialize GLFW
//    if (!glfwInit())
//        cout << "glfwInit failed." << endl;
//
//    // glfw window creation
//    // --------------------
//    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
//    if (window == NULL)
//    {
//        std::cout << "Failed to create GLFW window" << std::endl;
//        glfwTerminate();
//        return -1;
//    }
//
//    glfwMakeContextCurrent(window);
//
//    glfwSetWindowPos(window, 100, 100);
//    GLFWmonitor** pMonitor = glfwGetMonitors(&monitorCount);
//    // Full Screen
////    fullScreen(pMonitor[1], window);
//
//    // initialize glew
//    GLenum err = glewInit();
//    if (err != GLEW_OK)
//    {
//        std::cerr << "GLEW init failed: " << glewGetErrorString(err) << std::endl;
//        exit(EXIT_FAILURE);
//    }
//
//    // initialize Buffer
//    init();
//
//    clock_t time_start, time_end;
//    while (!glfwWindowShouldClose(window))
//    {
//        for (int k = 0; k <= 373; k++){
//            glfwPollEvents();
//            glfwMakeContextCurrent(window);
//            glMatrixMode(GL_PROJECTION);
//            glLoadIdentity();
//            glOrtho(0, 1, 1, 0, -1, 1);
//            // view port setting
//            glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
//
//            char dir[10];
//            sprintf(dir, "%d", k);
//
//
//            idx = START;
//            for (int i = START; i <= INPUT_NUMBER + START; i++) {
//                Mat image_temp;
//                Mat image_resize;
//
//                char a[10];
//                sprintf(a, "%02d", idx);
//                image_temp = imread(path + dir + "/" + a + suffix);
//                resize(image_temp, image_resize, Size(WIDTH, HEIGHT));
//                images.push_back(image_resize);
//
//                idx += 2;
//            }
//            time_start = clock();
//            flow_warp(images, module, novelview, d_img_srcs, d_flow_src, d_img_dst, d_img_dst2, d_img_code);
//
//            render(d_img_code);
//
//            images.clear();
//
//            // initialize Texture
//            if (!Texture) {
//                glGenTextures(1, &Texture);
//                glBindTexture(GL_TEXTURE_2D, Texture);
//
//                // Change these to GL_LINEAR for super- or sub-sampling
//                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//
//                // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
//                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//
//                // Using TexImage2D to allocate memory to allow employing TexSubImage2D later
////            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
//                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, OutWidth, OutHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
//            }
//
//            glBindTexture(GL_TEXTURE_2D, Texture);
//
//            // send PBO to texture
//            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Buffer);
//
//            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
////        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, 0);
//            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, OutWidth, OutHeight, GL_RGB, GL_UNSIGNED_BYTE, 0);
//
//            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
//
//            glEnable(GL_TEXTURE_2D);
//
//            glBegin(GL_QUADS);
//            glTexCoord2f(0.0f, 0.0f);
//            glVertex2f(0.0f, 0.0f);
//
//            glTexCoord2f(1.0f, 0.0f);
//            glVertex2f(1.0f, 0.0f);
//
//            glTexCoord2f(1.0f, 1.0f);
//            glVertex2f(1.0f, 1.0f);
//
//            glTexCoord2f(0.0f, 1.0f);
//            glVertex2f(0.0f, 1.0f);
//            glEnd();
//
//            glDisable(GL_TEXTURE_2D);
//
//            // glfw: swap buffers
//            glfwSwapBuffers(window);
//
//            time_end = clock();
//
//            cout << "warp time = " << ((time_end - time_start) * 1.0 / CLOCKS_PER_SEC) << "s" << endl;
////            usleep(1000000);
//        }
////        time_start = clock();
////
////        glfwPollEvents();
////        glfwMakeContextCurrent(window);
////        glMatrixMode(GL_PROJECTION);
////        glLoadIdentity();
////        glOrtho(0, 1, 1, 0, -1, 1);
////        // view port setting
////        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
////
////        // CUDA processing
////        flow_warp(images, module, novelview, d_img_srcs, d_flow_src, d_img_dst, d_img_dst2, d_img_code);
////
////        render(d_img_code);
////
////        // initialize Texture
////        if (!Texture) {
////            glGenTextures(1, &Texture);
////            glBindTexture(GL_TEXTURE_2D, Texture);
////
////            // Change these to GL_LINEAR for super- or sub-sampling
////            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
////            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
////
////            // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
////            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
////            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
////
////            // Using TexImage2D to allocate memory to allow employing TexSubImage2D later
//////            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
////            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, OutWidth, OutHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
////        }
////
////        glBindTexture(GL_TEXTURE_2D, Texture);
////
////        // send PBO to texture
////        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Buffer);
////
////        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//////        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, 0);
////        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, OutWidth, OutHeight, GL_RGB, GL_UNSIGNED_BYTE, 0);
////
////        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
////
////        glEnable(GL_TEXTURE_2D);
////
////        glBegin(GL_QUADS);
////        glTexCoord2f(0.0f, 0.0f);
////        glVertex2f(0.0f, 0.0f);
////
////        glTexCoord2f(1.0f, 0.0f);
////        glVertex2f(1.0f, 0.0f);
////
////        glTexCoord2f(1.0f, 1.0f);
////        glVertex2f(1.0f, 1.0f);
////
////        glTexCoord2f(0.0f, 1.0f);
////        glVertex2f(0.0f, 1.0f);
////        glEnd();
////
////        glDisable(GL_TEXTURE_2D);
////
////        // glfw: swap buffers
////        glfwSwapBuffers(window);
////
////        time_end = clock();
////
////        cout << "warp time = " << ((time_end - time_start) * 1.0 / CLOCKS_PER_SEC) << "s" << endl;
//    }
//
//    glfwDestroyWindow(window);
//    glfwTerminate();
//
//    cudaFree(d_img_srcs);
//    cudaFree(d_flow_src);
//    cudaFree(d_img_dst);
//
//    return 0;
//}


#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "torch/script.h" // One-stop header.
#include "torch/torch.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/types_c.h"
#include <time.h>
#include "cuda_runtime.h"
#include <cuda_gl_interop.h>
#include <unistd.h>

bool SAVE_IMAGE = false;
bool DISPLAY_TIME = false;

using namespace cv;
using namespace std;

unsigned int WIDTH = 1024;
unsigned int HEIGHT = 512;

int OutWidth = 7680;
int OutHeight = 4320;

int VIRTUAL_NUMBER = 5;
int INPUT_NUMBER = 10;
int START = 11;
//int START = 1;

int ViewNum = 60;
float LineNum = 26.436;
float InclinationAngle = 0.1683;
//// model 1
float MoveValue = -13;
float ZeroValue = 6;
//// model2
//float MoveValue = -17;
//float ZeroValue = 3;
//// model3
//float MoveValue = -17;
//float ZeroValue = -3.5;

// OpenGL settings
unsigned int SCR_WIDTH = 1024;
unsigned int SCR_HEIGHT = 512;
int monitorCount = 0;
GLuint Buffer;
GLuint Texture;
struct cudaGraphicsResource *cuda_pbo_rsc;

extern "C" void BackWarp(uchar* d_src, schar* disp, uchar* d_dst, int w, int h, float delta);
extern "C" void BackEncode(unsigned char* d_src, signed char* disp, unsigned char* d_dst, int w, int h,
                           int virtualnum,
                           bool ifReverse,
                           int outw, int outh,
                           int viewnum, float LineNum,
                           float InclinationAngle,
                           float MoveValue, float ZeroValue);
void fullScreen(GLFWmonitor* pMonitor, GLFWwindow* window);
void init();
void render(unsigned char* d_data);

void flow_warp(vector<Mat>& images, torch::jit::script::Module& module,
               unsigned char *d_img_srcs, signed char *d_flow_src, unsigned char *d_img_dst)
{
    for (size_t i = 0; i < images.size(); i++){
        cudaMemcpy(d_img_srcs + WIDTH * HEIGHT * 3 * i, (void *)images[i].data, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyHostToDevice);
    }

    vector<torch::Tensor> tensor_images;
    torch::Tensor tensor_temp, tensor_temp2;
    torch::Tensor tensor_group;
    for (size_t i = 0; i < images.size(); i++){
        tensor_temp = torch::from_blob(d_img_srcs + WIDTH * HEIGHT * 3 * i, { images[i].rows,  images[i].cols, images[i].channels() }, torch::kByte);
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

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_group);

    torch::Tensor output = module.forward(inputs).toTensor();

    torch::Tensor output1, ten_wrp;
    cv::Mat resultImg(HEIGHT, WIDTH, CV_8SC1);


    for (size_t i = 0; i < output.sizes()[0]; ++i){
        output1 = torch::select(output, 0, i);
        output1 = torch::select(output1, 0, 0);
        output1 = output1.unsqueeze(0);
        ten_wrp = output1.detach().permute({ 1, 2, 0 });
        ten_wrp = ten_wrp.clamp(-50, 50).to(torch::kInt8);

        cudaMemcpy(d_flow_src + WIDTH * HEIGHT * i, (void *)ten_wrp.data_ptr(), sizeof(torch::kInt8) * ten_wrp.numel(), cudaMemcpyDeviceToDevice);
    }

    BackEncode(d_img_srcs, d_flow_src, d_img_dst, WIDTH, HEIGHT, VIRTUAL_NUMBER, 1, OutWidth, OutHeight,
               ViewNum, LineNum, InclinationAngle, MoveValue, ZeroValue);
}

//---------------------------------Full Screen-------------------------------
void fullScreen(GLFWmonitor* pMonitor, GLFWwindow* window)
{
    const GLFWvidmode * mode = glfwGetVideoMode(pMonitor);
    std::cout << "Screen size is X = " << mode->width << ", Y = " << mode->height << std::endl;

    SCR_WIDTH = mode->width;
    SCR_HEIGHT = mode->height;

    glfwSetWindowMonitor(window, pMonitor, 0, 0, mode->width, mode->height, GLFW_DONT_CARE);
}
//--------------------------------------------------------------------------

//--------------------------------initialize buffer-------------------------
void init() {
    //----------------------------initialize Buffer-------------------------
    glGenBuffers(1, &Buffer);
    glBindBuffer(GL_ARRAY_BUFFER, Buffer);
//    glBufferData(GL_ARRAY_BUFFER, WIDTH * HEIGHT * sizeof(GLchar) * 3, NULL, GL_STREAM_DRAW);
    glBufferData(GL_ARRAY_BUFFER, OutWidth * OutHeight * sizeof(GLchar) * 3, NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //---------------------------link Buffer and CUDA-----------------------
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_rsc, Buffer, cudaGraphicsMapFlagsWriteDiscard);
}
//--------------------------------------------------------------------------

//----------------------------Process data in CUDA--------------------------
void render(unsigned char* d_data) {
    cudaGraphicsMapResources(1, &cuda_pbo_rsc, 0);

    size_t bytes;
    unsigned char *d_output;
    cudaGraphicsResourceGetMappedPointer((void **)&d_output, &bytes, cuda_pbo_rsc);
//    cudaMemcpy(d_output, d_data, WIDTH * HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_output, d_data, OutWidth * OutHeight * 3 * sizeof(unsigned char), cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &cuda_pbo_rsc, 0);
}
//------------------------------------------------------------------------

int main() {
    std::cout << torch::cuda::is_available() << std::endl;

    //Deserialize the ScriptModule from a file
    torch::jit::script::Module module = torch::jit::load("/home/weleslie/example-app/model_cpp_1.pt");
    module.to(torch::kCUDA);

    torch::Tensor ten = torch::rand({ 1, 6, 256, 256 }, torch::kFloat).to(torch::kCUDA);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(ten);
    torch::Tensor tensor = module.forward(inputs).toTensor();
    module.forward(inputs);

    vector<Mat> images;
    string path = "../training_model1_warp_cc/";
    string suffix = ".jpg";

    Mat novelview(OutHeight, OutWidth, CV_8UC3);
    unsigned char* d_img_srcs;
    cudaMalloc((void**)&d_img_srcs, WIDTH * HEIGHT * 3 * (INPUT_NUMBER + 1));
    signed char* d_flow_src;
    cudaMalloc((void**)&d_flow_src, WIDTH * HEIGHT * 3 * INPUT_NUMBER);
    unsigned char* d_img_dst;
    cudaMalloc((void**)&d_img_dst, OutWidth * OutHeight * 3);

    // glfw: initialize and configure
    // ------------------------------
    // Initialize GLFW
    if (!glfwInit())
        cout << "glfwInit failed." << endl;

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glfwSetWindowPos(window, 100, 100);
    GLFWmonitor** pMonitor = glfwGetMonitors(&monitorCount);
    // Full Screen
//    fullScreen(pMonitor[1], window);

    // initialize glew
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        std::cerr << "GLEW init failed: " << glewGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // initialize Buffer
    init();

    int idx;

    clock_t time_start, time_end;

    // For save image
//    for (int k = 0; k <= 373; k++) {
////    for (int k = 0; k <= 423; k++) {
////    for (int k = 0; k <= 299; k++) {
//        char dir[10];
//        sprintf(dir, "%d", k);
//
//        time_start = clock();
//        idx = START;
//        for (int i = START; i <= INPUT_NUMBER + START; i++) {
//            Mat image_temp;
//            Mat image_resize;
//            char a[10];
//
//            sprintf(a, "%02d", idx);
//            image_temp = imread(path + dir + "/" + a + suffix);
//            cvtColor(image_temp, image_temp, COLOR_BGR2RGB);
//            resize(image_temp, image_resize, Size(WIDTH, HEIGHT));
//            images.push_back(image_resize);
//
//            idx += 2;
////                idx += 1;
//        }
//
//        flow_warp(images, module, d_img_srcs, d_flow_src, d_img_dst);
//        cudaMemcpy((void *) novelview.data, d_img_dst, sizeof(unsigned char) * OutWidth * OutHeight * 3,
//                   cudaMemcpyDeviceToHost);
//        imwrite("model1_" + to_string(k) + ".jpg", novelview);
//        images.clear();
//    }

// For display
    while (!glfwWindowShouldClose(window))
    {
        for (int k = 0; k <= 373; k++){
//        for (int k = 0; k <= 423; k++){
//        for (int k = 0; k <= 299; k++){

            glfwPollEvents();
            glfwMakeContextCurrent(window);
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0, 1, 1, 0, -1, 1);
            // view port setting
            glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);

            char dir[10];
            sprintf(dir, "%d", k);

            time_start = clock();
            idx = START;
            for (int i = START; i <= INPUT_NUMBER + START; i++) {
                Mat image_temp;
                Mat image_resize;
                char a[10];

                sprintf(a, "%02d", idx);
                image_temp = imread(path + dir + "/" + a + suffix);
                cvtColor(image_temp, image_temp, COLOR_BGR2RGB);
                resize(image_temp, image_resize, Size(WIDTH, HEIGHT));
                images.push_back(image_resize);

                idx += 2;
//                idx += 1;
            }

            flow_warp(images, module, d_img_srcs, d_flow_src, d_img_dst);
//            cudaMemcpy((void *)novelview.data, d_img_dst, sizeof(unsigned char) * OutWidth * OutHeight * 3, cudaMemcpyDeviceToHost);
//            imwrite("model2_" + to_string(k) + ".jpg", novelview);
            render(d_img_dst);

            images.clear();

            // initialize Texture
            if (!Texture) {
                glGenTextures(1, &Texture);
                glBindTexture(GL_TEXTURE_2D, Texture);

                // Change these to GL_LINEAR for super- or sub-sampling
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

                // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

                // Using TexImage2D to allocate memory to allow employing TexSubImage2D later
//            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, OutWidth, OutHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
            }

            glBindTexture(GL_TEXTURE_2D, Texture);

            // send PBO to texture
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Buffer);

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, 0);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, OutWidth, OutHeight, GL_RGB, GL_UNSIGNED_BYTE, 0);

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            glEnable(GL_TEXTURE_2D);

            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f);
            glVertex2f(0.0f, 0.0f);

            glTexCoord2f(1.0f, 0.0f);
            glVertex2f(1.0f, 0.0f);

            glTexCoord2f(1.0f, 1.0f);
            glVertex2f(1.0f, 1.0f);

            glTexCoord2f(0.0f, 1.0f);
            glVertex2f(0.0f, 1.0f);
            glEnd();

            glDisable(GL_TEXTURE_2D);

            // glfw: swap buffers
            glfwSwapBuffers(window);

            time_end = clock();

            cout << "warp time = " << ((time_end - time_start) * 1.0 / CLOCKS_PER_SEC) << "s" << endl;
//            usleep(1000000);
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    cudaFree(d_img_srcs);
    cudaFree(d_flow_src);
    cudaFree(d_img_dst);

    return 0;
}