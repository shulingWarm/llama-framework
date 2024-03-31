#pragma once
#include"Matrix.hpp"
#include<fstream>
#include"Config.hpp"
#include"TinyCompute.hpp"

//复制scale
__device__ void copyScale(int* tokenData,Matrix* dstFeature,Matrix* featureLib)
{
	//每个线程负责几个数字的计算
	const unsigned int THREAD_RESPONS_NUM = dstFeature->rowNum >= blockDim.x ? divUp(dstFeature->rowNum,blockDim.x) : 1;
	//每个线程负责的数据的起始地址
	const unsigned beginOffset = threadIdx.x * THREAD_RESPONS_NUM;
	//判断是不是整个warp都不用工作
	const bool allWarpNotWork = (threadIdx.x / WARP_SIZE) * WARP_SIZE >= dstFeature->rowNum;
	if(!allWarpNotWork)
	{
		//从每个线程负责的起始位置开始遍历
		for(unsigned id = 0;id<THREAD_RESPONS_NUM;++id)
		{
			//目前实际操作的单元
			const unsigned operationId = beginOffset + id >= dstFeature->rowNum ? dstFeature->rowNum - 1 : beginOffset + id;
			//复制实质正在操作的float数据
			dstFeature->scaleList[operationId] = featureLib->scaleList[tokenData[operationId]];
		}
	}
}

//核函数，选出矩阵里面的对应行
//这里面对feature做完整的复制其实是有多余操作的
//那后面还是应该尽可能对数据做原地运算
//这里用传值的方式传进来，是因为里面的数据指针什么的都是浅拷贝
__global__ void selectRows(int* tokenData,Matrix dstFeature,Matrix featureLib)
{
	//每个线程负责几个数字的计算
	const unsigned int THREAD_RESPONS_NUM = dstFeature.colNum >= blockDim.x ? divUp(dstFeature.colNum,blockDim.x) : 1;
	//每个线程需要复制的数据头
	FeatType* rowHead = getRowHead(&featureLib,tokenData[blockIdx.x]);
	//当前线程块要处理的数据头
	FeatType* dstRowHead = getRowHead(&dstFeature,blockIdx.x);
	//要复制的数据的起始地址 如果超过了访问位置
	const unsigned int beginCopyId = threadIdx.x * THREAD_RESPONS_NUM >= featureLib.colNum ? featureLib.colNum :
		threadIdx.x * THREAD_RESPONS_NUM;
	//计算要复制的数据的长度
	const unsigned int copyLength = beginCopyId + THREAD_RESPONS_NUM >= featureLib.colNum ? featureLib.colNum - beginCopyId : THREAD_RESPONS_NUM;
	//执行数据的复制
	memcpy(dstRowHead + beginCopyId, rowHead + beginCopyId, sizeof(FeatType)*copyLength);
	//如果是第一个线程块，顺便把scale也复制了
	if(blockIdx.x==0)
	{
		copyScale(tokenData,&dstFeature,&featureLib);
	}
}

//把输入的token列表转换成特征列表，每个token对应列表里面的一个feature
class InputFeatureMaker
{
	//用于构造输入feature的权重
	Matrix featureWeight;
public:

	//载入权重数据
	void loadWeight(std::fstream& fileHandle)
	{
		featureWeight = loadMatrix(fileHandle);
	}

	//把每个token的标号转换成特征的标号列表
	//最后得到这个矩阵对应的feature input
	Matrix makeFeatureInput(int* tokenData,unsigned tokenNum)
	{
		//把数据转换到cuda内存
		int* cudaToken = (int*)initFromCpuData(tokenData,tokenNum*sizeof(int));
		//初始化cuda里面转换出来的矩阵
		auto ansMat = initMatrix(tokenNum,FET_LENGTH);
		//从特征矩阵里面选出对应的行
		//释放cuda的token信息
		handleError(cudaFree(cudaToken));
		//返回算出来的矩阵
		return ansMat;
	}
};