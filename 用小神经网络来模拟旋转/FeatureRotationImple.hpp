#pragma once
#include"FeatureLib.hpp"
#include"RotationMatrix.hpp"
#include"AngleTransformer.hpp"

//两个向量之间的局部乘法操作
__device__ HalfType localDotProduct(const FeatType* const ptr1,const FeatType* const ptr2,const int dotSize,
	const AngleTransformer* const transAngleTool)
{
	//初始化点乘的结果
	HalfType sum = __float2half(0);
	//遍历要处理的每个特征
	for(int idData=0;idData<dotSize;++idData)
	{
		sum += angle2Num(transAngleTool,ptr1[idData]) * angle2Num(transAngleTool,ptr2[idData]);
	}
	//返回计算结果
	return sum;
}

//计算纵向公共的上半部分
//用于判断是不是计算上半部分公共区域
//这个属于是
template<char UPPER_FLAG,int THREAD_NUM>
__device__ void computeCommonVerticalPart(const FeatureLib* const xyExtractor,const FeatureLib* const srcFeatureLib,
	HalfType* const result,
	const AngleTransformer* const transAngleTool)
{
	//计算起始的位置
	const unsigned idBegin = UPPER_FLAG ? 0 : xyExtractor->featureSize/2;
	//计算每个线程负责的乘法个数
	const unsigned threadTaskNum = xyExtractor->featureSize/2/THREAD_NUM;
	//总共有几个warp参与计算
	const unsigned warpNum = THREAD_NUM/WARP_SIZE;
	//当前的线程所属的warp标号
	const unsigned idWarp = threadIdx.x / warpNum;
	//准备每个warp的点乘计算结果
	__shared__ HalfType warpResult[warpNum];
	//乘法过程的2048个weight数据 这里是专指纵向的数据
	FeatType* ptr1 = xyExtractor->data + idBegin + threadIdx.x*threadTaskNum;
	//原始的输入向量的weight
	FeatType* ptr2 = getFeatHead(srcFeatureLib,blockIdx.x) + idBegin + threadIdx.x*threadTaskNum;
	//执行两个向量的局部乘法操作
	HalfType dotResult = localDotProduct(ptr1,ptr2,threadTaskNum);
	//再下面用蝶式寻址计算每个warp内的相加结果
	for(int id=WARP_SIZE/2;id>=1;id/=2)
		dotResult += __shfl_xor_sync(unsigned(-1), dotResult, id, 32);
	//由第1个线程把各个结果存放到共享内存里面
	if(threadIdx.x % WARP_SIZE == 0)
	{
		warpResult[idWarp] = dotResult;
	}
	//调用线程同步
	__syncthreads();
	//由第1个线程加起来所有数据
	if(threadIdx.x==0)
	{
		result[0] = __float2half(0);
		for(int i=0;i<warpNum;++i)
			result[0] += warpResult[i];
	}
	__syncthreads();
}

//计算横向的公共部分
template<char LEFT_FLAG, int THREAD_NUM>
__device__ void computeCommonHorizontalPart(const FeatureLib* const xyExtractor,
	const AngleTransformer* const transAngleTool
)
{
	//计算起始位置
	const unsigned idBegin = LEFT_FLAG ? 0 : 
}

//核函数，会对输入的每个向量做旋转
//输入和输出都是长度为2048的向量族，其中每个线程块负责一个向量的相乘操作
//这个矩阵的定义，前1024个旋转矩阵填充的是上面，后1024列由旋转矩阵填充下半部分
//其余的空缺部分由输入的向量补齐
template<int THREAD_NUM>
__global__ void rotationKernel(const RotationMatrix matrix,const FeatureLib srcFeatureLib,
	FeatureLib dstFeatureLib, const FeatureLib xyExtractor,const AngleTransformer transAngleTool)
{
	//公共部分的内存，具体的顺序是左右上下
	__shared__ HalfType commonResult[4];
	//计算纵向的公共部分 这里面是纵向的上半部分和纵向的下半部分
	computeCommonVerticalPart<1,THREAD_NUM>(&xyExtractor,&srcFeatureLib,&commonResult[2],&transAngleTool);
	computeCommonVerticalPart<0,THREAD_NUM>(&xyExtractor,&srcFeatureLib,&commonResult[3],&transAngleTool);
	//计算横向的公共部分

}

//这是对特征进行旋转操作的实现
class FeatureRotaryTool
{
	//xy方向上提取特征的向量 是两个长度是2048的向量，第1个负责处理上下的部分 第2个负责处理左右的部分
	FeatureLib xyExtractor;

	//这个是用来存储那个中间层的变量的，这样走到这一层的时候就不用每次都开辟空间了
	//这也是一个4096维度的向量
	FeatureLib midOutput;
public:
	void doRotation(const RotationMatrix* const matrix,
		const FeatureLib* const srcFeatureLib, FeatureLib* const dstFeatureLib,
		const AngleTransformer* const transAngleTool)
	{
		//执行第一层的双向处理，拿出来一个4096的向量。

	}
};