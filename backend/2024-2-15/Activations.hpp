#pragma once
#include"Config.hpp"

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