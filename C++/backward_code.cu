//
// Created by weleslie on 2021/12/17.
//
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

__global__ static void backencode(unsigned char* d_src, signed char* disp, unsigned char* d_dst, int width, int height,
                                  int virtualnum,
                                  bool ifReverse,
                                  int outw, int outh,
                                  int viewnum, float LineNum,
                                  float InclinationAngle,
                                  float MoveValue, float ZeroValue) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = j * outw + i;         //±ê×¼ÏñËØµãË÷Òý

    if (MoveValue < 0)
        MoveValue = viewnum - abs(MoveValue);

    int midnum = viewnum / 2;
    float wid = ZeroValue * width / 100.0;

    if (j < outh && i < outw)
    {
        float step_value = 1.0 / viewnum * LineNum;
        for (int k = 0; k < 3; k++)
        {
            float value_pixel = i * 3 + 3 * j * InclinationAngle + k;
            float judge_value = 0;
            if (LineNum != 0)
                judge_value = value_pixel - int(value_pixel / LineNum) * LineNum;
            else
                judge_value = value_pixel;

            if (judge_value < 0)
            {
                judge_value = judge_value + LineNum;
            }
            int view_point_number = floor(judge_value / step_value);

            //È·¶¨µ±Ç°ÏñËØµã¶ÔÓ¦µÄÊÓµãÐòºÅ
            int thisnum = (view_point_number % viewnum + (viewnum - (int)MoveValue)) % viewnum;

            if (!ifReverse)
            {
                thisnum = viewnum - thisnum - 1;
            }

            float movepixel = wid * (thisnum - midnum) / (float)midnum;

            float orgx = i * width / (outw * 1.0) + movepixel;
            float orgy = j * height / (outh * 1.0);

            int xxx_min = floor(orgx);
            int xxx_max = ceil(orgx);

            int yyy_min = floor(orgy);
            int yyy_max = ceil(orgy);

            float a = xxx_max - orgx;
            float b = yyy_max - orgy;

            //´óÍ¼ÖÐµÄÃ¿¸öÏñËØµã¶ÔÓ¦Ä³¸öÊÓµãÖÐµÄÒ»¸öÏñËØµã
            if ((yyy_min + 1) < height && (xxx_min + 1) < width &&  xxx_min > 0 && yyy_min > 0)
            {
                int sparsenum = floor(thisnum / (virtualnum + 1));
                int modnum = thisnum % (virtualnum + 1);

                float delta = (virtualnum + 1 - modnum) / ((virtualnum + 1) * 1.0);
//                printf("%.4f\n", delta);

                int count_minmin = sparsenum * width * height + yyy_min * width + xxx_min;
                int count_maxmin = sparsenum * width * height + yyy_max * width + xxx_min;
                int count_minmax = sparsenum * width * height + yyy_min * width + xxx_max;
                int count_maxmax = sparsenum * width * height + yyy_max * width + xxx_max;

                int dis_minmin = (int)floor((disp[count_minmin] * -delta));
                int dis_maxmin = (int)floor((disp[count_maxmin] * -delta));
                int dis_minmax = (int)floor((disp[count_minmax] * -delta));
                int dis_maxmax = (int)floor((disp[count_maxmax] * -delta));

                float tempL = 0;

                //**Ë«ÏßÐÔ²åÖµ**//(thisnum - 1) * imgw * imgh * 3 + (yyy_min - 1) * imgw * 3 +
                tempL = d_src[(count_minmin + width * height + dis_minmin) * 3 + k] * a * b +
                        d_src[(count_maxmin + width * height + dis_maxmin) * 3 + k] * a * (1 - b) +
                        d_src[(count_minmax + width * height + dis_minmax) * 3 + k] * (1 - a) * b +
                        d_src[(count_maxmax + width * height + dis_maxmax) * 3 + k] * (1 - a) * (1 - b);

                // For save image
//                d_dst[idx * 3 + 2 - k] = tempL;
                // For display
                d_dst[idx * 3 + k] = tempL;

//                float tempL = 0;
//
//                //**Ë«ÏßÐÔ²åÖµ**//thisnum * imgw * imgh * 3 + (yyy_min - 1) * imgw * 3 +
//                tempL = d_src[thisnum * width * height * 3 + (yyy_min - 1) * width * 3 + (xxx_min - 1) * 3 + k] * a * b +
//                        d_src[thisnum * width * height * 3 + (yyy_max - 1) * width * 3 + (xxx_min - 1) * 3 + k] * a * (1 - b) +
//                        d_src[thisnum * width * height * 3 + (yyy_min - 1) * width * 3 + (xxx_max - 1) * 3 + k] * (1 - a) * b +
//                        d_src[thisnum * width * height * 3 + (yyy_max - 1) * width * 3 + (xxx_max - 1) * 3 + k] * (1 - a) * (1 - b);
//
//                //×ª»»µ½ÊÓµãµÄ³ß´ç
//                d_dst[idx * 3 + 2 - k] = tempL;
            }
            else
            {
                d_dst[idx * 3] = 0;
                d_dst[idx * 3 + 1] = 0;
                d_dst[idx * 3 + 2] = 0;
                break;
            }
        }
    }
}

extern "C" void BackEncode(unsigned char* d_src, signed char* disp, unsigned char* d_dst, int w, int h,
                           int virtualnum,
                           bool ifReverse,
                           int outw, int outh,
                           int viewnum, float LineNum,
                           float InclinationAngle,
                           float MoveValue, float ZeroValue){
    cudaError_t error = cudaSuccess;

    dim3 thread = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 block = dim3((outw + thread.x - 1) / thread.x, (outh + thread.y - 1) / thread.y);

    backencode << <block, thread, 0 >> > (d_src, disp, d_dst, w, h, virtualnum, ifReverse, outw, outh, viewnum, LineNum, InclinationAngle,
                                          MoveValue, ZeroValue);
}


