#pragma once
#include"CudaHandleError.hpp"


//申请指定类型的cuda数据
char* dataAllocate(unsigned dataSize)
{
	char* tempData;
	handleError(cudaMalloc((void**)&tempData,sizeof(dataSize)));
	return tempData;
}

//从指定的cpu数据里面复制
char* initFromCpuData(char* cpuData,unsigned dataSize)
{
	//初始化cuda数据
	auto gpuData = dataAllocate(dataSize);
	//复制cuda数据
	handleError(cudaMemcpy(gpuData,cpuData,dataSize,cudaMemcpyHostToDevice));
	//返回处理过的数据
	return gpuData;
}