#pragma once

//feature使用的具体数据类型 这个应该是要对应后面的量化过程的
typedef char FeatType;

//头的个数
const unsigned int HEAD_NUM = 16;
//每个头的特征长度
const unsigned int HEAD_DIM = 64;
//整个的每个向量的长度
const unsigned int FET_LENGTH = HEAD_NUM*HEAD_DIM;
//每个wap的线程数，由于是在kernel里面用的，就直接用宏来表示了
#define WARP_SIZE 32
