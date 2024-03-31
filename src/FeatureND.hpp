#pragma once
#include"CudaAllocate.hpp"
#include"Config.hpp"

//n维的feature信息
//不过这里还是仅仅存储cuda的版本，完全不考虑扩展性
//做这个是为了保存向量旋转过程中每一步的中间旋转过程
template<unsigned DIM,class T>
struct __FeatureND
{
	T* data;

	int featureNum[DIM];
};

//基本常用的3维feature
typedef __FeatureND<3,FeatType> Feature3D;
//对于存储qkv情况的数据，常用的是4d的操作，第一个维度分别是qkv
typedef __FeatureND<4,FeatType> Feature4D;
//softmax的attention分数的feature
typedef __FeatureND<4,AttnType> AttnFeature4D;
typedef __FeatureND<3,AttnType> AttnFeature3D;
//梯度的feature 目前只是一个2D的
typedef __FeatureND<2,HalfType> GradFeature;

//根据featureNum的信息，计算总的feature个数
int getTotalFeatureSize(int* featureNum,int featureDim)
{
	int size = 1;
	for(int i=0;i<featureDim;++i)
	{
		size *= featureNum[i];
	}
	return size;
}

//后面看情况，有可能会出现步长值超过int表示范围的情况，到时候需要把类型换成u64
//获取featureND的移动步长
template<unsigned DIM,int HEAD_DIM>
__device__ void getHeadStride(const int* featureNum,
	int* dstStride)
{
	int minStride = 1;
	//计算最后一个步长的结果
	for(int i=HEAD_DIM;i<DIM;++i)
	{
		minStride *= featureNum[i];
	}
	//把最小步长保存到最后一位中
	dstStride[HEAD_DIM-1] = minStride;
	//依次处理前面的步长
	for(int i=HEAD_DIM-2;i>=0;--i)
	{
		//把前方的步长乘到自己的位置
		dstStride[i] = dstStride[i+1]*featureNum[i+1];
	}
}

//在不同的维度上索引数据头,这个HEAD_DIM表示自己想要访问哪个层级位置的数据
//例如idDims是2,3时，表示访问第3组的第4行
template<unsigned DIM,class T,int HEAD_DIM>
__device__ T* getFeatHeadND(__FeatureND<DIM,T>* features,const int* idDims)
{
	//获取处理数据用的步长 +1表示当前的函数认为这里指的是目标维度的id
	//而这个获取步长的函数认为HEAD_DIM是一种目标数据的长度
	int headStride[HEAD_DIM];
	getHeadStride<DIM,HEAD_DIM>(features->featureNum,headStride);
	//直接取出对应的位置头
	int idBegin = 0;
	for(int i=0;i<HEAD_DIM;++i)
	{
		idBegin += headStride[i]*idDims[i];
	}
	//返回索引的结果
	return features->data + idBegin;
}

//给N维的特征信息开辟cuda空间
template<unsigned DIM,class T>
void allocateFeatureND(__FeatureND<DIM,T>& features,const int* featureNum)
{
	int totalSize=1;
	for(int i=0;i<DIM;++i)
	{
		totalSize *= featureNum[i];
		features.featureNum[i]=featureNum[i];
	}
	//开辟cuda内存的空间
	features.data = (T*)dataAllocate(sizeof(T)*totalSize);
}

//从featureND里面取出一个子维度的tensor
template<unsigned DIM,class T,int DST_DIM>
__device__ void getSubTensor(__FeatureND<DIM,T>* srcFeature,//tensor的数据来源
	__FeatureND<DST_DIM,T>* dstFeature,const int* idDims
)
{
	//需要索引的数据头的维度，这对应的就是idDims的长度
	const int HEAD_DIM = DIM - DST_DIM;
	//记录子维度
	memcpy(dstFeature->featureNum,srcFeature->featureNum+HEAD_DIM,
		sizeof(int)*DST_DIM);
	//然后软拷贝
	dstFeature->data = getFeatHeadND<DIM,T,HEAD_DIM>(
		srcFeature,idDims
	);
}