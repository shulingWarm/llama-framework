#pragma once
#include"CudaHandleError.hpp"


//申请指定类型的cuda数据
char* dataAllocate(unsigned dataSize)
{
	char* tempData;
	handleError(cudaMalloc((void**)&tempData,dataSize));
	return tempData;
}

//从指定的cpu数据里面复制
char* initFromCpuData(const char* cpuData,unsigned dataSize)
{
// #ifdef PRINT_INTERNAL
// 	std::cout<<"dataAllocate: "<<dataSize<<std::endl;
// #endif
	//初始化cuda数据
	auto gpuData = dataAllocate(dataSize);
// #ifdef PRINT_INTERNAL
// 	std::cout<<"cudaMemcpy for cpu init"<<std::endl;
// #endif
	//复制cuda数据
	handleError(cudaMemcpy(gpuData,cpuData,dataSize,cudaMemcpyHostToDevice));
	//返回处理过的数据
	return gpuData;
}

void releaseCudaMemory(char* gpuData)
{
	handleError(cudaFree(gpuData));
}

//直接对cuda数据的打印，把所有