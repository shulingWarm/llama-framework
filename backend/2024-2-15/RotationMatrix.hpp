#pragma once
#include"Config.hpp"
#include<fstream>
#include"StreamReader.hpp"
#include"CudaAllocate.hpp"
#include<vector>

//旋转矩阵，里面只有n(n-1)/2个参数
//而且这里面存的都是角度信息
struct RotationMatrix
{
public:
	//对应的数据
	FeatType* data;

	//矩阵的大小
	unsigned matSize;

	//从文件流里面读取这个数据
	void init(std::fstream& fileHanlde)
	{
		//读取矩阵的大小
		matSize = StreamReader::read<unsigned int>(fileHanlde);
		//总的数据个数
		unsigned dataNum = matSize*(matSize-1)/2;
		//开辟cpu端的对应数据
		std::vector<FeatType> tempData(dataNum);
		fileHanlde.read((char*)tempData.data(),sizeof(FeatType)*dataNum);
		//把cpu的数据转换到gpu上
		data = (FeatType*)initFromCpuData((char*)tempData.data(),sizeof(FeatType)*dataNum);
	}

};

