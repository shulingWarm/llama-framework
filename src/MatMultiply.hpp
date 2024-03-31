#pragma once
#include "RotationMatrix.hpp"
#include "FeatureLib.hpp"

__global__ void cudaMatMultiply(const RotationMatrix matrix,FeatureLib inputFeature)
{
	//当前线程块负责处理的向量
	const FeatType* blockFeature = getFeatHead(&inputFeature,blockIdx.x);
	//由多少个线程处理矩阵里面的一行
	
}

//矩阵乘法的实现 这里特别实现的是旋转矩阵的操作，因为这里会保证所有的向量乘完之后的模长都是1
//最后会直接把相乘的结果保存在
void matMultiply(RotationMatrix* matrix,FeatureLib* inputFeature)
{
	assert(matrix->matSize == inputFeature->featureSize);
	//直接放到kernel里面去计算就可以了
	
}