#pragma once
#include "Config.hpp"
#include<cmath>
#include"CudaAllocate.hpp"

//初始化每个三角函数的sin值
//采样数指的是把0~2pi之间分成多少个采样点
void initSinValueChart(HalfType* dstData,int sampleNum)
{
	//计算步长
	float stepLength = 2*PI/sampleNum;
	//目前的数值
	float currAngle = 0;
	//遍历每一步
	for(int idStep=0;idStep<sampleNum;++idStep)
	{
		//计算正弦值
		dstData[idStep] = __float2half(std::sin(currAngle));
		//步长叠加
		currAngle += stepLength;
	}
}

//角度转换器，把中间存储的特征数据类型做各种情况的转换
struct AngleTransformer
{
public:
	//从角度到数值的float转换
	HalfType* angleToNum;

	//初始化角度的transformer
	void init()
	{
		//初始化float形式的角度到数值的转换表
		HalfType cpuChart[ANGLE_CHART_SIZE];
		//在这里面更新sin的查表数据
		initSinValueChart(cpuChart,ANGLE_CHART_SIZE);
		//把数据转换到gpu上
		angleToNum = (HalfType*)initFromCpuData((char*)cpuChart,sizeof(HalfType)*ANGLE_CHART_SIZE);
	}

};

//把输入的数据转换成角度
__device__ HalfType angle2Num(const AngleTransformer* const transformer, const FeatType angle)
{
	return transformer->angleToNum[angle];
}

