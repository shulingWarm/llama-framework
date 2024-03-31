#pragma once
#include"Config.hpp"
#include"CudaAllocate.hpp"
#include"FeatureND.hpp"

struct IntermediateGradient
{
	//主要的梯度信息
	GradFeature mainGradient;

	//对中间数据的初始化
	void init(int tokenNum,int featureSize)
	{
		const int featureNum[] = {tokenNum,featureSize};
		allocateFeatureND<2,HalfType>(mainGradient,featureNum);
	}

	HalfType* getMainPtr()
	{
		return mainGradient.data;
	}
};