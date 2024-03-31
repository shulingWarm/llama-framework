#pragma once
#include<cuda_fp16.h>

//feature使用的具体数据类型 这个应该是要对应后面的量化过程的
typedef unsigned char FeatType;
//半精度浮点型
typedef __half HalfType;

//头的个数
const unsigned int HEAD_NUM = 16;
//每个头的特征长度
const unsigned int HEAD_DIM = 64;
//整个的每个向量的长度
const unsigned int FET_LENGTH = HEAD_NUM*HEAD_DIM;
//每个wap的线程数，由于是在kernel里面用的，就直接用宏来表示了
#define WARP_SIZE 32

//角度到数值的转换表的大小
#define ANGLE_CHART_SIZE 256

//PI的大小
#define PI 3.14159265359f