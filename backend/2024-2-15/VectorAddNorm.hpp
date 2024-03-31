#pragma once
#include"FeatureLib.hpp"
#include"Config.hpp"
#include"Activations.hpp"

//对一组数据进行完整叠加的操作
//至于上一层每个线程负责多个数的处理，多个数得到一个数的那个过程，这个函数是不管的
//需要确保每个线程都调用到了这个函数
template<unsigned WARP_NUM_IN_BLOCK>
__inline__ __device__ HalfType blockVectorAddup(HalfType threadSum,HalfType* warpResult)
{
	//进行蝶式寻址，把每个线程的结果求和放到warp的结果里
	for(int idCross=WARP_SIZE/2;idCross>=1;idCross/=2)
	{
		threadSum = __hadd(threadSum,
			__shfl_xor_sync(unsigned(-1), threadSum, idCross, WARP_SIZE));
	}
	//由每个warp的第1个线程把结果保存到对应的warp位置上
	if(threadIdx.x % WARP_SIZE == 0)
		warpResult[threadIdx.x / WARP_SIZE] = threadSum;
	//执行同步，确保每个warp都已经把结果保存到共享内存了
	__syncthreads();
	//执行二分操作，把所有的结果汇集到第一个值上
	for(int idCross=1;idCross<WARP_NUM_IN_BLOCK;++idCross)
	{
		//判断自己是否处于分二位
		if(threadIdx.x % (2*idCross) == 0 && threadIdx.x + idCross < WARP_NUM_IN_BLOCK)
		{
			warpResult[threadIdx.x] = __hadd(
				warpResult[threadIdx.x],warpResult[threadIdx.x + idCross]
			);
		}
		__syncthreads();
	}
	//取第一个位置的结果
	return warpResult[0];
}

//对向量做norm操作 直接传入的就是每个线程负责的向量区段
template<unsigned FEATURE_SIZE,
unsigned WARP_NUM_IN_BLOCK,unsigned THREAD_NUM_TASK
>
__device__ void vectorNorm(HalfType* resultTarget,HalfType* warpResult)
{
	//计算所有数据的均值
	HalfType threadSum = localAddup(resultTarget,THREAD_NUM_TASK);
	//对整个线程里面的数据求和
	threadSum = blockVectorAddup<WARP_NUM_IN_BLOCK>(threadSum,warpResult);
	//每个线程把这个数除以总的向量个数，作为均值，然后准备减去值
	threadSum = halfDiv<float>(threadSum,FEATURE_SIZE);
	//然后开始减均值
	vecMinus(resultTarget,threadSum,THREAD_NUM_TASK);
	//计算数据的局部平方和
	threadSum = localSquareAdd(resultTarget,THREAD_NUM_TASK);
	//再次把整个线程的数据加起来
	threadSum = blockVectorAddup<WARP_NUM_IN_BLOCK>(threadSum,warpResult);
	//由第1个线程把平方和结果开个根号
	if(threadIdx.x==0)
	{
		warpResult[0] = invertSqrt(threadSum);
	}
	__syncthreads();
	//每个线程再取出这个数据
	threadSum = warpResult[0];
	//每个线程把自己负责的区段再除以这个数据 到这里算是做完了norm
	localScale(resultTarget,THREAD_NUM_TASK,threadSum);
}

//把两个特征加起来的cuda函数
//只考虑FEATURE_SIZE能整除线程数的情况
//线程数至少应该是32,并且每个feature都应该有数可算才行
//由于需要复用dstFeature里面保存的那个数据，所以每个线程负责的数据个数至少是两个
//为了方便把kernel融合起来，直接在加完向量之后，norm之前，把那个silu也应用上去
template<unsigned FEATURE_SIZE,unsigned THREAD_PER_BLOCK>
__global__ void cudaFeatureAddAndNorm(FeatureLib dstFeature,const FeatureLib addFeature,
	const AngleTransformer transformer
)
{
	//每个线程负责的数字个数
	const unsigned THREAD_NUM_TASK = FEATURE_SIZE / THREAD_PER_BLOCK;
	//每个线程块总共有多少个warp
	const unsigned WARP_NUM_IN_BLOCK = THREAD_PER_BLOCK/WARP_SIZE;
	//总的共享内存，用于存储中间结果
	__shared__ HalfType addResult[FEATURE_SIZE];
	//共享内存，用于记录每个warp的叠加结果 其实是为了便于把同一个向量里面的各种内容加起来
	__shared__ HalfType warpResult[WARP_NUM_IN_BLOCK];
	//当前线程负责的数据区段
	FeatType* dstThreadTarget = getFeatHead(&dstFeature,blockIdx.x) + threadIdx.x*THREAD_NUM_TASK;
	//叠加数的线程数据区段
	const FeatType* addThreadTarget = getFeatHead(&addFeature,blockIdx.x) + threadIdx.x*THREAD_NUM_TASK;
	//叠加中间结果的数据区段
	HalfType* resultTarget = addResult + threadIdx.x*THREAD_NUM_TASK;
	//处理自己负责的区段
	twoVecAdd<HalfType,FeatType>(resultTarget,dstThreadTarget,addThreadTarget,
		THREAD_NUM_TASK,&transformer);
	//对加起来的数据做norm 注意调用完这个函数之后是还没有同步的，如果后面需要交叉使用数据的话还需要同步一下
	vectorNorm<FEATURE_SIZE,WARP_NUM_IN_BLOCK,THREAD_NUM_TASK>(resultTarget,warpResult);
	//调用silu函数，激活相加的结果
	for(int i=0;i<THREAD_NUM_TASK;++i)
		resultTarget[i]=activateSiLU<HalfType>(resultTarget[i]);
	//再次调用norm
	vectorNorm<FEATURE_SIZE,WARP_NUM_IN_BLOCK,THREAD_NUM_TASK>(resultTarget,warpResult);
	//把计算结果保存到结果中
	for(int i=0;i<THREAD_NUM_TASK;++i)
		dstThreadTarget[i] = num2Angle<HalfType>(&transformer,resultTarget[i]);
}

//把两个featureLib加起来，加完之后当场norm
//这里还是一个基于summarization阶段的操作
void featureAddAndNorm(FeatureLib* dstFeature,const FeatureLib* addFeature,
	const AngleTransformer* const transformer
)
{
	//调用核函数，把待加的向量直接加到dst上
	cudaFeatureAddAndNorm<FET_LENGTH,256><<<dstFeature->featureNum,256>>>(*dstFeature,*addFeature,*transformer);
}