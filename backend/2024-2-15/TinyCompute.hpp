#pragma once
#include"Config.hpp"
#include"AngleTransformer.hpp"

//各种小的计算

//向上取整
__device__ int divUp(int num1,int num2)
{
	return (num1 + num2 - 1)/num2;
}

//对两段数据做的局部点乘
//专指的是那种带角度的计算
//这里需要针对不同的输出情况做实例特化的实现
template<unsigned THREAD_OP_NUM,class OutType>
__device__ OutType localDot(const FeatType* vec1,const FeatType* vec2,const AngleTransformer* const transformer)
{
	const unsigned ans = 0;
	return *((OutType*)&ans);
}

//针对输出仍为角度情况下的泛型特化
template<unsigned THREAD_OP_NUM>
__device__ FeatType localDot(const FeatType* vec1,const FeatType* vec2,const AngleTransformer* const transformer)
{
	FeatType ans = 0;
	#pragma unroll
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		//对两个向量进行求和 把两个局部向量相乘的结果加起来
		ans = sinsinAdd(transformer,ans,
			sinsinMul(transformer,vec1[i],vec2[i])
		);
	}
	return ans;
}

//针对输出为half的情况下的泛型特化
template<unsigned THREAD_OP_NUM>
__device__ HalfType localDot(const FeatType* vec1,const FeatType* vec2,const AngleTransformer* const transformer)
{
	HalfType ans = __float2half(0.f);
	#pragma unroll
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		//对两个向量进行求和 把两个局部向量相乘的结果加起来
		ans += angle2Num(transformer,
			sinsinMul(transformer,vec1[i],vec2[i])
		);
	}
	return ans;
}

//把几个连续内存的数字加起来的操作，这是单线程执行的操作
__device__ HalfType localAddup(const HalfType* data,const unsigned THREAD_OP_NUM)
{
	//初始化求和的0数据
	HalfType ans = __float2half(0.f);
	for(int i=0;i<THREAD_OP_NUM;++i)
		ans = __hadd(ans,data[i]);
	return ans;
}

//对连续的几个数依次乘上每个数
__device__ void localScale(HalfType* data,const unsigned THREAD_OP_NUM,HalfType scale)
{
	//遍历每个数，给它除以对应的数
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		data[i] = __hmul(data[i],scale);
	}
}

//对half数据类型做批量的减法操作
//主要是用在减均值的过程中
__device__ void vecMinus(HalfType* data,const HalfType minusData,const unsigned THREAD_OP_NUM)
{
	//遍历每个数，执行对应的减法操作
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		data[i] = __hsub(data[i],minusData);
	}
}

//把连续的几个数乘上scale然后保存到目标向量里面
__device__ void localScaleToDst(const HalfType* srcData,const HalfType scale,FeatType* dst,
	const unsigned THREAD_OP_NUM, //每个线程负责的操作数
	const AngleTransformer* const transformer
)
{
	//遍历需要处理的每个数据
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		dst[i] = num2Angle(transformer,__hmul(srcData[i],scale));
	}
}

//把一个角度形式的数据段加在一个half的数据段上
//先写一个泛型，不过只实现它的一个特化
template<class OutType,class AddType>
__device__ void localVecAdd(OutType* out,const AddType* add,HalfType scale,const unsigned THREAD_OP_NUM,
	const AngleTransformer* const transformer)
{

}

//泛型特化，输出half,输入的是角度变量
//这里面的操作类似于+=
template<>
__device__ void localVecAdd(HalfType* out,const FeatType* add,HalfType scale,const unsigned THREAD_OP_NUM,
	const AngleTransformer* const transformer
)
{
	//遍历每个需要处理的数字
	#pragma unroll
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		out[i] = __hadd(out[i],__hmul(scale,angle2Num(transformer,add[i])));
	}
}

//向量的直接相加，没有scale信息，只是为了维护后面的数据计算方便
__device__ void halfDirectAdd(HalfType* out,HalfType* add,const unsigned THREAD_OP_NUM)
{
	//遍历每个需要处理的数字
	#pragma unroll
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		out[i] += add[i];
	}
}

//把一段half向量转转换成角度
__device__ void halfVecToFeatVec(const HalfType* source,FeatType* dst,const unsigned THREAD_OP_NUM,
	const AngleTransformer* const transformer
)
{
	//遍历每个要处理的数据
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		//把half数据类型转换成角度
		dst[i] = num2Angle<HalfType>(transformer,source[i]);
	}
}

//把两个向量加起来
template<class OutType,class AddType>
__device__ void twoVecAdd(OutType* out,const AddType* add1,const AddType* add2,
	const unsigned THREAD_OP_NUM,const AngleTransformer* const transformer
)
{

}

//两个向量相加情况的特化
template<>
__device__ void twoVecAdd(HalfType* out,const FeatType* add1,const FeatType* add2,
	const unsigned THREAD_OP_NUM,const AngleTransformer* const transformer
)
{
	//遍历每个位置，直接加起来
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		out[i] = __hadd(
			angle2Num(transformer,add1[i]),
			angle2Num(transformer,add2[i])
		);
	}
}

//计算输入数据的局部平方和
__device__ HalfType localSquareAdd(const HalfType* data,const unsigned THREAD_OP_NUM)
{
	HalfType ans = __float2half(0.f);
	//处理中间过程的每个数，求平方和
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		ans = __hadd(ans,__hmul(data[i],data[i]));
	}
	return ans;
}

//对一个half数据先开根号然后再取个倒数
__device__ HalfType invertSqrt(HalfType data)
{
	//临时把数据转换成float
	float tempData = __half2float(data);
	return __float2half(1.f/sqrt(tempData));
}

//对half数据的div操作，会根据各种不同的数据类型做处理
template<class T>
__device__ HalfType halfDiv(HalfType dst,T data)
{
	return dst;
}

//处理float数据类型的特化
template<>
__device__ HalfType halfDiv(HalfType dst,float data)
{
	//把float转换成scale
	HalfType tempScale = __float2half(1.f/data);
	return __hmul(dst,tempScale);
}