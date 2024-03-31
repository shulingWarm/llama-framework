#pragma once
#include"Matrix.hpp"
#include<fstream>
#include"Config.hpp"
#include"TinyCompute.hpp"
#include"FeatureLib.hpp"

//核函数，选出矩阵里面的对应行
//这里面对feature做完整的复制其实是有多余操作的
//那后面还是应该尽可能对数据做原地运算
//这里用传值的方式传进来，是因为里面的数据指针什么的都是浅拷贝
__global__ void selectRows(int* tokenData,FeatureLib dstFeature,FeatureLib featureLib)
{
	//每个线程负责几个数字的计算
	const unsigned int THREAD_RESPONS_NUM = dstFeature.featureSize >= blockDim.x ? divUp(dstFeature.featureSize,blockDim.x) : 1;
	//每个线程需要复制的数据头
	FeatType* rowHead = getFeatHead(&featureLib,tokenData[blockIdx.x]);
	//当前线程块要处理的数据头
	FeatType* dstRowHead = getFeatHead(&dstFeature,blockIdx.x);
	//要复制的数据的起始地址 如果超过了访问位置后面就不再复制了
	const unsigned int beginCopyId = threadIdx.x * THREAD_RESPONS_NUM >= featureLib.featureSize ? featureLib.featureSize :
		threadIdx.x * THREAD_RESPONS_NUM;
	//计算要复制的数据的长度
	const unsigned int copyLength = beginCopyId + THREAD_RESPONS_NUM >= featureLib.featureSize ? featureLib.featureSize - beginCopyId : THREAD_RESPONS_NUM;
	//执行数据的复制
	memcpy(dstRowHead + beginCopyId, rowHead + beginCopyId, sizeof(FeatType)*copyLength);
}

//把输入的token列表转换成特征列表，每个token对应列表里面的一个feature
class InputFeatureMaker
{
	//用于构造输入feature的权重
	//这个地方需要换成feature lib,它不是简单的矩阵，而是一个特征库
	FeatureLib featureWeight;
public:

	//载入权重数据
	void loadWeight(std::fstream& fileHandle)
	{
		featureWeight = loadFeatureLib(fileHandle);
	}

	//把每个token的标号转换成特征的标号列表
	//最后得到这个矩阵对应的feature input
	FeatureLib makeFeatureInput(int* tokenData,unsigned tokenNum)
	{
		//把数据转换到cuda内存
		int* cudaToken = (int*)initFromCpuData((char*)tokenData,tokenNum*sizeof(int));
		//初始化cuda里面转换出来的矩阵
		auto ansMat = initFeatureLib(tokenNum,FET_LENGTH);
		//从特征矩阵里面选出对应的行
		selectRows<<<tokenNum,256>>>(cudaToken,ansMat,featureWeight);
		//释放cuda的token信息
		handleError(cudaFree(cudaToken));
		//返回算出来的矩阵
		return ansMat;
	}
};