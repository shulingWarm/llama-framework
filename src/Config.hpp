#pragma once
#include<cuda_fp16.h>
#include<stdio.h>

//自定义的debug
#define MY_ASSERT(expression) \
if(!(expression)) \
	printf("error: %s:%d\n",__FILE__,__LINE__);

//debug相关的内容
//#define PRINT_INTERNAL

//PI的大小
#define PI 3.14159265359f

//feature使用的具体数据类型 这个应该是要对应后面的量化过程的
typedef unsigned char FeatType;
//这个type对应的三角函数的周期数
const unsigned int TYPE_SAMPLE_NUM = 256;
//半精度浮点型
typedef __half HalfType;
//attention socre的中间类型
typedef HalfType AttnType;
//int16类型的数值
typedef unsigned short int u16;
typedef unsigned int u32;
typedef short int i16;
typedef char i8;
typedef int TokenId;//注意，一定要把它弄成有符号的类型，不然会出问题的

//头的个数
const unsigned int HEAD_NUM = 32;
//每个头的特征长度
const unsigned int HEAD_DIM = 64;
//整个的每个向量的长度
const unsigned int FET_LENGTH = HEAD_NUM*HEAD_DIM;
//Attention层的个数 写成1只是为了做实验
const unsigned int ATTENTION_NUM = 1;
//旋转位置编码时的基础旋转角
const float BASE_ROTATION = PI / HEAD_DIM;

//每个wap的线程数，由于是在kernel里面用的，就直接用宏来表示了
#define WARP_SIZE 32
//做向量相乘时，每个线程负责的数字个数
#define ATTENTION_THREAD_OP_NUM 8

//角度到数值的转换表的大小
#define ANGLE_CHART_SIZE 256
//角度的90度的位置
#define ANGLE_QUANT_90 64U
//角度的中间值
#define ANGLE_MID_VALUE 128U
//三分位的角度，-90度的地方
#define ANGLE_QUANT_270 192U
//从数值反推角度的分级情况
#define NUM_ANGLE_CHART_SIZE 2048

//训练时数据越界的上界 都是闭界
#define ACCUMULATE_UP_RANGE 255
#define ACCUMULATE_LOW_RANGE 0

//累积满后的初始值
#define ACCUMULATE_INIT 128

//自己定义的二元item,毕竟要给cuda用，所以才自己定义了一个
template<class T1,class T2>
struct PairItem
{
	T1 first;
	T2 second;
};