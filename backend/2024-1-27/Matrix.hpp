#pragma once
#include"Config.hpp"
#include<fstream>
#include"StreamReader.hpp"
#include<vector>
#include"CudaAllocate.hpp"

//矩阵，其实就是一个向量的列表
//但特殊的地方是它有多个scale
//注意，每一行是一个feature
//data的数据也是行优先的
struct Matrix
{
public:
	float* scaleList;
	FeatType* data;

	//矩阵的大小
	unsigned rowNum;
	unsigned colNum;
};

//从文件流中读取一个矩阵
Matrix loadMatrix(std::fstream& fileHandle)
{
	//读取行数和列数
	Matrix mat;
	mat.rowNum = StreamReader.read<unsigned>(fileHandle);
	mat.colNum = StreamReader.read<unsigned>(fileHandle);
	//读取scale的列表数据
	std::vector<float> scaleData(mat.rowNum);
	fileHandle.read((char*)scaleData.data(),sizeof(float)*mat.rowNum);
	//矩阵的数据总数
	auto dataNum = mat.rowNum * mat.colNum;
	//读取每一行的数据
	std::vector<FeatType> numData(dataNum);
	fileHandle.read((char*)numData.data(),sizeof(FeatType)*dataNum);
	//初始化矩阵里的两个cuda数据
	mat.scaleList = (float*)initFromCpuData(scaleData.data(),sizeof(float)*mat.rowNum);
	mat.data = (FeatType*)initFromCpuData(numData.data(),sizeof(FeatType)*dataNum);
	//传值的形式，返回已经初始化过的矩阵
	return mat;
}

//构造一个矩阵
Matrix initMatrix(unsigned rowNum,unsigned colNum)
{
	Matrix mat;
	mat.rowNum = rowNum;
	mat.colNum = colNum;
	//初始化矩阵的scale序列
	mat.scaleList = (float*)dataAllocate(sizeof(float)*rowNum);
	//初始化矩阵的所有data数据
	mat.data = (FeatType*)dataAllocate(sizeof(FeatType)*rowNum*colNum);
	//返回初始化过的矩阵
	return mat;
}

//获得一个matrix的某一行的数据头
__device__ FeatType* getRowHead(Matrix* mat,unsigned idRow)
{
	return mat.data + idRow*colNum;
}