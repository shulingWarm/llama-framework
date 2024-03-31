#pragma once
#include"Config.hpp"
#include<fstream>
#include"StreamReader.hpp"
#include<vector>
#include"CudaAllocate.hpp"

//特征lib这是一个矩阵，用来表示每个token位置编码出来的特征向量
struct FeatureLib
{
public:
	FeatType* data;

	unsigned featureNum;
	unsigned featureSize;
};

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
	lib.data = (FeatType*)initFromCpuData(numData.data(),sizeof(FeatType)*dataNum);
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

//获取feature对应的数据头
__device__ FeatType* getFeatHead(FeatureLib* lib,unsigned idFeature)
{
	return lib->data + idFeature*lib->featureSize;
}