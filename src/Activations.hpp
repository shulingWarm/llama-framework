#pragma once
#include"Config.hpp"
#include"AngleTransformer.hpp"

//各种激活函数都会实现在这里面，虽然目前只会使用一种silu

template<class T>
__device__ T activateSiLU(T data)
{

}

//对float的特化实现
template<>
__device__ float activateSiLU(float data)
{
	return data/(1.f+exp(-data));
}

//对half类型的实现
template<>
__device__ HalfType activateSiLU(HalfType data)
{
	return __float2half(activateSiLU<float>(__half2float(data)));
}

//把三角函数作为激活函数
__device__ FeatType sinActivate(FeatType angle,const AngleTransformer* const transformer)
{
	//直接调用正弦值然后再反转换回角度
	return transformer->sinActivate[angle];
}

//对正弦激活的求导
__device__ HalfType getLossDiffOnActivate(
	FeatType angle,const AngleTransformer* const transformer
)
{
	//计算基本的数值
	const HalfType coeff = __float2half(64.f/ANGLE_CHART_SIZE*2.f*PI);
	const HalfType negCoeff = __float2half(-64.f/ANGLE_CHART_SIZE*2.f*PI);
	//计算的梯度
	HalfType baseGradient = angle2CosNum(transformer,angle);
	//判断是不是90~270度
	if(angle >= ANGLE_QUANT_90 && angle<ANGLE_QUANT_270)
	{
		return __hmul(negCoeff,baseGradient);
	}
	//正常情况下返回正的角度
	return __hmul(coeff,baseGradient);
}