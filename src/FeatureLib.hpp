#pragma once
#include"Config.hpp"
#include<fstream>
#include"StreamReader.hpp"
#include<vector>
#include"CudaAllocate.hpp"
#include<iostream>

//一个和featureLib功能重合的代码，但这里引入了泛型，方便后面做扩展
//featureLib相关的操作最好也兼容进去，不过这件事后面再说
template<class T>
struct __FeatureLib
{
	T* data = nullptr;

	unsigned featureNum = 0;
	unsigned featureSize = 0;	
};

//这是贯穿整个推理训练过程的主要类型，甚至可以理解成pytorch里面的tensor
//但这个代码工程属于是完全抛弃了tensor的思想
typedef __FeatureLib<FeatType> FeatureLib;

//特征lib这是一个矩阵，用来表示每个token位置编码出来的特征向量
template<>
struct __FeatureLib<FeatType>
{
public:
	FeatType* data = nullptr;

	unsigned featureNum = 0;
	unsigned featureSize = 0;

	//从其它的cpu的featureLib里面复制，但里面的data需要确保是已经转换成gpu的
	//但需要注意的是，这里面是需要考虑开辟空间的事的
	void deepCopyFrom(const FeatureLib* otherLib)
	{
		if(featureNum != otherLib->featureNum || featureSize != otherLib->featureSize)
		{
			//std::cout<<"init data"<<std::endl;
			//先把本位置的内存释放了
			if(data != nullptr)
			{
				//std::cout<<"release in copy"<<std::endl;
				releaseCudaMemory((char*)data);
			}
			//复制数据
			featureNum = otherLib->featureNum;
			featureSize = otherLib->featureSize;
			//开辟空间
			data = (FeatType*)dataAllocate(featureNum * featureSize * sizeof(FeatType));
		}
		handleError(cudaMemcpy(data,otherLib->data,sizeof(FeatType)*featureNum*featureSize,cudaMemcpyDeviceToDevice));
	}

};

//初始化一个feature的shape
template<class T>
void initFeatureLibShape(__FeatureLib<T>& feature,
	int featureNum,int featureSize
)
{
	//必须假设这是一个没有被初始化过的数据
	MY_ASSERT(feature.data == nullptr);
	feature.featureNum = featureNum;
	feature.featureSize = featureSize;
	//初始化内存空间
	feature.data = (T*)dataAllocate(sizeof(T)*featureSize*featureNum);
}

//能用模板的获得feature头
template<class T>
__device__ T* __getFeatHead(__FeatureLib<T>* lib,unsigned idFeature)
{
	return lib->data + idFeature*lib->featureSize;
}

//获取feature对应的数据头
__device__ FeatType* getFeatHead(FeatureLib* lib,unsigned idFeature)
{
	return lib->data + idFeature*lib->featureSize;
}

__device__ const FeatType* getFeatHead(const FeatureLib* lib,unsigned idFeature)
{
	return lib->data + idFeature*lib->featureSize;
}

//从文件流里面读取特征列表
FeatureLib loadFeatureLib(std::fstream& fileHandle){
	//读取行数和列数
	FeatureLib lib;
	lib.featureNum = StreamReader::read<unsigned>(fileHandle);
	lib.featureSize = StreamReader::read<unsigned>(fileHandle);
	//数据的总数
	auto dataNum = lib.featureNum * lib.featureSize;
	std::vector<FeatType> numData(dataNum);
	fileHandle.read((char*)numData.data(),sizeof(FeatType)*dataNum);
	//从cpu数据里面初始化cuda数据
// #ifdef PRINT_INTERNAL
// 	std::cout<<"initFromCpuData: "<<lib.featureNum<<" "<<lib.featureSize<<std::endl;
// #endif
	lib.data = (FeatType*)initFromCpuData((char*)numData.data(),sizeof(FeatType)*dataNum);
// #ifdef PRINT_INTERNAL
// 	std::cout<<"finish initFromCpuData"<<std::endl;
// #endif
	return lib;
}

//初始化featureLib
FeatureLib initFeatureLib(unsigned featureNum,unsigned featureSize)
{
	FeatureLib lib;
	lib.featureNum = featureNum;
	lib.featureSize = featureSize;
	//初始化data数据
	lib.data = (FeatType*)dataAllocate(sizeof(FeatType)*lib.featureSize*lib.featureNum);
	//返回初始化过的矩阵
	return lib;
}

//打印指定的某个feature,直接在外面复制的话不方便索引
//这是仅使用一个线程的代码，毕竟是为了测试，就不追求速度了
template<unsigned idBegin,unsigned copyLength>
__global__ void copyInternalFeature(FeatureLib lib,unsigned idFeature,FeatType* dstFeature)
{
	FeatType* featureHead = getFeatHead(&lib,idFeature);
	//遍历要处理的数据，直接复制到目标位置
	for(int id=0;id<copyLength;++id)
	{
		int currId = idBegin + id;
		dstFeature[id] = featureHead[currId];
	}
}

//打印部分featureLib
//仅仅是为了看一下它运行过程中会不会有什么问题
template<int idBegin,int copyLength>
void printFeatureLib(FeatureLib* lib,unsigned idFeature)
{
	//初始化用于存储结果的内存
	char* copiedCudaData = dataAllocate(sizeof(FeatType)*copyLength);
	copyInternalFeature<idBegin,copyLength><<<1,1>>>(*lib,idFeature,(FeatType*)copiedCudaData);
	//把数据结果复制到cpu
	FeatType cpuData[copyLength];
	handleError(cudaMemcpy(cpuData,copiedCudaData,sizeof(FeatType)*copyLength,cudaMemcpyDeviceToHost));
	//遍历打印cpu里的数据
	for(int idData=0;idData<copyLength;++idData)
	{
		std::cout<<(int)cpuData[idData]<<" ";
	}
	std::cout<<std::endl;
	//释放内存
	handleError(cudaFree(copiedCudaData));
}