#include <GL/glew.h>

#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "../Project3/std_image.h"

#include <iostream>
#include <windows.h>

using namespace std;

void fullScreen(GLFWmonitor* pMonitor, GLFWwindow* window);
void init();
void render(unsigned char* d_output);

// settings
unsigned int SCR_WIDTH = 1024;
unsigned int SCR_HEIGHT = 512;
int monitorCount = 0;
GLuint Buffer;
GLuint Texture;
struct cudaGraphicsResource *cuda_pbo_rsc;
int WIDTH, HEIGHT, CHANNEL;

int main()
{
	unsigned char* data = stbi_load("D:/博二/100views_to_1000views/SimpleViews2Flow1.0/img1.png", &WIDTH, &HEIGHT, &CHANNEL, 0);
	unsigned char* d_data;
	cudaMalloc((void **)&d_data, WIDTH * HEIGHT * sizeof(unsigned char) * CHANNEL);
	cudaMemcpy(d_data, data, WIDTH * HEIGHT * CHANNEL * sizeof(unsigned char), cudaMemcpyHostToDevice);
	
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
	// 全屏
	fullScreen(pMonitor[1], window);

	// 初始化glew，使用gl*系列函数时必须进行初始化
	GLenum err = glewInit();
	if (err != GLEW_OK)
	{
		std::cerr << "GLEW init failed: " << glewGetErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
	
	// 初始化Buffer
	init();
	
	int i = 2;
	char a[100];
	// render loop
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		glfwMakeContextCurrent(window);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, 1, 1, 0, -1, 1);
		// 设置视口
		glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
		
		// CUDA数据处理
		render(d_data);
		
		// 初始化纹理
		if (!Texture) {
			glGenTextures(1, &Texture);
			glBindTexture(GL_TEXTURE_2D, Texture);

			// Change these to GL_LINEAR for super- or sub-sampling
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

			// GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

			// 当下文使用glTexSubImage2D时，需要在这里使用glTexImage2D分配内存
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
		}

		glBindTexture(GL_TEXTURE_2D, Texture);

		// send PBO to texture
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Buffer);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		// glTexImage2D(GL_TEXTURE_2D, 0, 3, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, 0);
		
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

		Sleep(10);

		sprintf(a, "D:/博二/100views_to_1000views/SimpleViews2Flow1.0/img%d.png", i++);
		data = stbi_load(a, &WIDTH, &HEIGHT, &CHANNEL, 0);
		cudaMemcpy(d_data, data, WIDTH * HEIGHT * CHANNEL * sizeof(unsigned char), cudaMemcpyHostToDevice);

		if (i == 12)
			break;
	}

	// 释放内存，关闭窗口，结束程序
	stbi_image_free(data);
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}

//----------------------------------全屏函数---------------------------------
void fullScreen(GLFWmonitor* pMonitor, GLFWwindow* window)
{
	const GLFWvidmode * mode = glfwGetVideoMode(pMonitor);
	std::cout << "Screen size is X = " << mode->width << ", Y = " << mode->height << std::endl;

	SCR_WIDTH = mode->width;
	SCR_HEIGHT = mode->height;

	glfwSetWindowMonitor(window, pMonitor, 0, 0, mode->width, mode->height, GLFW_DONT_CARE);
}
//----------------------------------全屏函数---------------------------------

//--------------------------------初始化buffer--------------------------
void init() {
	//----------------------------初始化Buffer------------------------------
	glGenBuffers(1, &Buffer);
	glBindBuffer(GL_ARRAY_BUFFER, Buffer);
	glBufferData(GL_ARRAY_BUFFER, 1024 * 512 * sizeof(GLubyte) * 3, NULL, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//---------------------------绑定Buffer与CUDA---------------------------
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_rsc, Buffer, cudaGraphicsMapFlagsWriteDiscard);
}
//----------------------------------初始化buffer--------------------------

//----------------------------使用CUDA读取显存中的内容--------------------------
void render(unsigned char* d_data) {
	cudaGraphicsMapResources(1, &cuda_pbo_rsc, 0);

	size_t bytes;
	unsigned char *d_output;
	// 请求一个指向被映射资源的指针 d_output是设备指针	<<<d_output>>>
	cudaGraphicsResourceGetMappedPointer((void **)&d_output, &bytes, cuda_pbo_rsc);
	cudaMemcpy(d_output, d_data, WIDTH * HEIGHT * CHANNEL * sizeof(unsigned char), cudaMemcpyDeviceToDevice);

	cudaGraphicsUnmapResources(1, &cuda_pbo_rsc, 0);
}
//----------------------------使用CUDA读取显存中的内容--------------------------