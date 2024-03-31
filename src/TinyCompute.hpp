#pragma once
#include"Config.hpp"
#include"AngleTransformer.hpp"

//带有累积权重的数值
//由于采用了角度差分，在叠加数据的时候需要采用累积权重的数字
//两个数据叠加时需要根据累积的权重来取平均中间数
//这样做的结果就是始终取的都是平均值
struct NumWithWeight
{
public:
	float weight;
	float angle;
};

//对浮点型粗糙的取余数
__device__ float cudaFmod(float num,float modNum)
{
	while(num<0)
		num += modNum;
	while(num >= modNum)
		num -= modNum;
	return num;
}

//浮点型版本的计算角度差
//前提是这两个角度都在范围内，这一点不做保证
__device__ float fGetAngleDiff(float angle1,float angle2)
{
	float tempDiff = cudaFmod(angle1-angle2,ANGLE_CHART_SIZE);
	//判断有没有超过一半
	if(tempDiff > ANGLE_MID_VALUE)
	{
		//取另外一半的角度
		return -cudaFmod(angle2-angle1,ANGLE_CHART_SIZE);
	}
	return tempDiff;
}

//考虑上负数的取余
template<class T>
__device__ T cudaMod(T angle,T modNum)
{
	while(angle < 0)
		angle += modNum;
	while(angle >= modNum)
		angle -= modNum;
	return angle;
}

//带符号的角度差，只不过用的时候要取一个绝对值
template<class OutType,class InputType>
__device__ OutType getAngleDiffWithSymbol(InputType angle1,InputType angle2)
{
	//计算临时的角度差
	OutType tempDiff = (OutType)angle1 - (OutType)angle2;
	OutType modedDiff = cudaMod<OutType>(tempDiff,ANGLE_CHART_SIZE);
	//判断是否超过了半角
	if(modedDiff>ANGLE_MID_VALUE)
	{
		//取另一半的角度
		return -cudaMod<OutType>(-tempDiff,ANGLE_CHART_SIZE);
	}
	return modedDiff;
}

//两个NumWithWeight的相加
//注意，需要考虑dst就是num1本身的情况
__device__ void addNumWithWeight(NumWithWeight* dst,
	const NumWithWeight* num1,
	const NumWithWeight* num2
)
{
	//计算两个角度差值的绝对值
	const float angleDiff = fGetAngleDiff(num1->angle,num2->angle);
	//计算角度偏移量
	const float angleOffset = angleDiff * (num1->weight / (num1->weight + num2->weight + 1e-8));
	//计算两个角度的中值
	dst->angle = cudaFmod(num2->angle + angleOffset, ANGLE_CHART_SIZE);
	//把weight的结果加起来
	dst->weight = num1->weight + num2->weight;
}

//各种小的计算

//向上取整
__device__ int divUp(int num1,int num2)
{
	return (num1 + num2 - 1)/num2;
}

//局部的一段数据的angleDis
template<unsigned THREAD_OP_NUM>
__device__ HalfType localAngleDis(const FeatType* vec1,const FeatType* vec2,
	const AngleTransformer* const transformer
)
{
	//初始化返回结果
	HalfType ans = 0;
	//遍历需要处理的每个数
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		FeatType angleDiff = getAngleDiff(vec1[i],vec2[i]);
		ans = __hadd(ans,angle2CosNum(transformer,angleDiff));
	}
	return ans;
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

//求一段连续数据的局部区域最大值
__device__ HalfType getLocalMax(const HalfType* data,const unsigned THREAD_OP_NUM)
{
	//初始化最前的最大值
	HalfType currMax = __float2half(-9999.f);
	//遍历每个数据
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		currMax = __hmax(currMax,data[i]);
	}
	return currMax;
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

//泛型特化，带weight的向量相加
template<>
__device__ void localVecAdd(NumWithWeight* out,const FeatType* add,HalfType scale,
	const unsigned THREAD_OP_NUM, const AngleTransformer* const transformer
)
{
	//把scale临时换成float型
	float fScale = __half2float(scale);
	//遍历每个要处理的数据
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		//临时把当前的数字包装成带权重的weight
		NumWithWeight tempNum = {fScale,(float)add[i]};
		//打印相加之前的数据
		// if(threadIdx.x == 0 && blockIdx.x==0 && blockIdx.y==0)
		// {
		// 	//先打印累积位置的2个数和加上去的两个数
		// 	printf("%f %f %f %f|",out[i].weight,out[i].angle,tempNum.weight,tempNum.angle);
		// }
		//执行两个点的相加
		addNumWithWeight(out+i,out+i,&tempNum);
		// if(threadIdx.x == 0 && blockIdx.x==0 && blockIdx.y==0)
		// {
		// 	//打印相加之后的数据
		// 	printf("%f %f\n",out[i].weight,out[i].angle);
		// }
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

//把两个向量数据加起来
template<class T1,class T2>
__device__ void vecAddOn(T1* dst,const T2* add,const unsigned THREAD_OP_NUM)
{

}

//对带权重数据的特化
template<>
__device__ void vecAddOn(NumWithWeight* dst,const NumWithWeight* add,const unsigned THREAD_OP_NUM)
{
	//遍历每个数据
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		addNumWithWeight(dst+i,dst+i,add+i);
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

//向量数据的集体转换 为了保证能用性还是用泛型吧
template<class T1,class T2>
__device__ void featTypeTransform(T1* dst,const T2* src,const unsigned THREAD_OP_NUM)
{

}

//把数据类型特化，把带权重的角度信息再转换回一般的角度
template<>
__device__ void featTypeTransform(FeatType* dst,const NumWithWeight* src,const unsigned THREAD_OP_NUM)
{
	//遍历每个数据，依次做处理
	//如果要debug的话建议给这里添加一个检测
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		dst[i] = (int)src[i].angle;
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

//使用带权重的数据相加时的特化
template<>
__device__ void twoVecAdd(NumWithWeight* out,const NumWithWeight* add1,
	const NumWithWeight* add2,const unsigned THREAD_OP_NUM,const AngleTransformer* const transformer
)
{
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		//执行两个带weight的数据相加
		addNumWithWeight(out+i,add1+i,add2+i);
	}
}

//处理两个全是feattype的相加过程
//这个实现的时候需要考虑add1就是out本身的情况
template<>
__device__ void twoVecAdd(FeatType* out,const FeatType* add1,const FeatType* add2,
	const unsigned THREAD_OP_NUM,const AngleTransformer* const transformer
)
{
	//遍历每个需要被加的数据
	for(int i=0;i<THREAD_OP_NUM;++i)
	{
		//调用两个角度的中值相加
		out[i] = getMidAngle(add1[i],add2[i]);
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

//分线程执行的数据复制 这里是确保数据数比线程数多的，并且可以确保数据量是线程数的整数倍
//并且这里面不做线程使用入口的判断
template<int THREAD_OP_NUM>
__device__ void dataCopyMultiThread(char* dstData,const char* const srcData)
{
	//直接处理当前线程负责的数据头
	memcpy(dstData + threadIdx.x*THREAD_OP_NUM,
		srcData + threadIdx.x*THREAD_OP_NUM,
		THREAD_OP_NUM);
}

//灵活执行的长度复制
//不保证复制长度是线程数的整数倍
//这里只考虑线程数大于任务量的情况
//这个函数需要保证所有的线程都进入
template<char THREAD_MORE_FLAG>
__device__ void flexibleDataCopy(char* dstData,const char* const srcData,
	unsigned dataNum
)
{
	
}

//泛型特化，对线程数多于数据量情况的实现
//这并不是一个很好的实现
template<>
__device__ void flexibleDataCopy<1>(char* dstData,const char* const srcData,
	unsigned dataNum
)
{
	//判断线程是否在范围内
	if(threadIdx.x < dataNum)
	{
		dstData[threadIdx.x] = srcData[threadIdx.x];
	}
}

//另一种泛型特化，对于数据量大于线程数的实现
template<>
__device__ void flexibleDataCopy<0>(char* dstData,const char* const srcData,
	unsigned dataNum)
{
	//计算每个线程处理的任务量
	int THREAD_OP_NUM = dataNum / blockDim.x;
	//判断自己的处理区间
	int idBegin = threadIdx.x * THREAD_OP_NUM;
	if(idBegin >= dataNum)
	{
		idBegin = 0;
	}
	else if(dataNum - idBegin < THREAD_OP_NUM)
	{
		THREAD_OP_NUM = dataNum - idBegin;
	}
	//执行数据的复制
	memcpy(dstData + idBegin,srcData + idBegin,THREAD_OP_NUM);
}