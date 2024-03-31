#pragma once
#include"FeatureLib.hpp"
#include"Config.hpp"
#include"Activations.hpp"
#include"IntermediateDecoderResult.hpp"

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
template<unsigned FEATURE_SIZE,unsigned THREAD_PER_BLOCK,char RECORD_FLAG>
__global__ void cudaFeatureAddAndNorm(FeatureLib dstFeature,const FeatureLib addFeature,
	const AngleTransformer transformer,
	IntermediateDecoderResult intermediateResult
)
{
	//每个线程负责的数字个数
	const unsigned THREAD_NUM_TASK = FEATURE_SIZE / THREAD_PER_BLOCK;
	//当前线程负责的数据区段
	FeatType* dstThreadTarget = getFeatHead(&dstFeature,blockIdx.x) + threadIdx.x*THREAD_NUM_TASK;
	//叠加数的线程数据区段
	const FeatType* addThreadTarget = getFeatHead(&addFeature,blockIdx.x) + threadIdx.x*THREAD_NUM_TASK;
	//处理自己负责的区段 把第二段向量直接加在第一段向量上
	//这里需要换成两个角度相加的情况，或者说是两个角度取中值的情况
	//这种加法操作目前已经不需要再做norm了
	twoVecAdd<FeatType,FeatType>(dstThreadTarget,dstThreadTarget,addThreadTarget,
		THREAD_NUM_TASK,&transformer);
	//判断是否需要记录中间结果 这个函数体里面会自动判断的
	recordMainFeature<RECORD_FLAG,0,ENUM_ID_RES,FEATURE_SIZE>(
		&intermediateResult,(FeatType*)getFeatHead(&dstFeature,blockIdx.x)
	);
	//调用silu函数，激活相加的结果 这里调用的是正弦式的激活函数
	for(int i=0;i<THREAD_NUM_TASK;++i)
		dstThreadTarget[i]=sinActivate(dstThreadTarget[i],&transformer);
}

//把两个featureLib加起来，加完之后当场norm
//这里还是一个基于summarization阶段的操作
//说是加完之后当场做norm,其实现在已经不需要做norm了
//另外就是三角函数的问题，既然要引入非线性，那不如就直接在这里调用三角函数
template<char RECORD_FLAG>
void featureAddAndNorm(FeatureLib* dstFeature,const FeatureLib* addFeature,
	const AngleTransformer* const transformer,
	IntermediateDecoderResult* intermediateResult //用来记录中间结果的结构体
)
{
	//调用核函数，把待加的向量直接加到dst上
	cudaFeatureAddAndNorm<FET_LENGTH,256,RECORD_FLAG><<<dstFeature->featureNum,256>>>(
		*dstFeature,*addFeature,*transformer,*intermediateResult);
}

//计算对输入激活位置的求导，这直接就是一个对应相乘的操作
template<u32 THREAD_PER_BLOCK,u32 FEATURE_SIZE>
static __device__ void getLossDiffOnActInput(
	IntermediateDecoderResult* intermediateResult,//输入输出的历史记录
	IntermediateGradient* lossGradOnOut,
	const AngleTransformer* const transformer
)
{
	//计算每个线程负责的数据量
	constexpr u32 TASK_PER_THREAD = FEATURE_SIZE/THREAD_PER_BLOCK;
	//当前线程负责的起始id
	const u32 idBegin = TASK_PER_THREAD*threadIdx.x;
	//当前线程处理的数据头
	const int inputHeadDims[] = {(int)ENUM_ID_RES,(int)blockIdx.x};
	const FeatType* inputHead = (FeatType*)getFeatHeadND<3,FeatType,2>(
		&intermediateResult->qkvResult,inputHeadDims);
	//中间的求导结果
	HalfType* gradientHead = getFeatHeadND<2,HalfType,1>(
		&lossGradOnOut->mainGradient,inputHeadDims+1);
	//当场计算新的求导结果
	// dL/dx = dL/dy * dy/dx
	//遍历当前线程要处理的每个数字
	for(int i=0;i<TASK_PER_THREAD;++i)
	{
		const int currId = i + idBegin;
		gradientHead[currId] = __hmul(gradientHead[currId],
			getLossDiffOnActivate(inputHead[currId],transformer));
	}
}

//对残差连接块的求导
template<u32 THREAD_PER_BLOCK, //每个线程块的线程数
	u32 FEATURE_SIZE //中间层的特征长度
>
__device__ void backwardKernelLossDiffOnAttnOut(
	//数据大小: [TokenNum,FeatureSize]
	GradFeature* lossDiffOnResdual, //loss对残差连接结果的求导
	//数据大小 [TokenNum,FeatureSize]
	const FeatureLib* const historyResdual, //历史记忆的残差输入
	//数据大小 [TokenNum,FeatureSize]
	const FeatureLib* const attnOutput //历史记录的attention的output
)
{
	//每个线程负责的区段数 这里只处理feature长度大于线程块个数并且可以整除的情况
	constexpr THREAD_NUM_TASK = FEATURE_SIZE / THREAD_PER_BLOCK;
	//每个线程负责的数据区块
	const int lossHeadDims[1] = {blockIdx.x};
	HalfType* lossHead = getFeatHeadND<2,1>(lossDiffOnResdual,lossHeadDims) +
		THREAD_NUM_TASK * threadIdx.x;
	//历史上的残差数据输入
	FeatType* resdualHead = getFeatHead(historyResdual,blockIdx.x) +
		THREAD_NUM_TASK * threadIdx.x;
	//attention层的输出
	FeatType* attnOutHead = getFeatHead(attnOutput,blockIdx.x) + 
		THREAD_NUM_TASK * threadIdx.x;
	//遍历计算当前线程负责的每一组数据
	for(int i=0;i<THREAD_NUM_TASK;++i)
	{
		//取出当前位置负责的数据头
		//准备两个数据简单求和的求导过程
		
	}
}

//获取输入对loss的偏导
//这个层里面不会涉及weight,所以也不会有什么对weight的更新
template<u32 THREAD_PER_BLOCK,u32 FEATURE_SIZE>
__global__ void backwardKernelFeatureAddAndNorm(
	//这里面有原始的decoder输入和当前激活层输入相加的结果
	//但这里面没有激活层的输出结果，激活层的输出结果是直接由上一层传入的
	IntermediateDecoderResult intermediateResult,
	//激活层原本的输出结果，这和下面的偏导信息是不一样的
	FeatureLib historyOutput, //当初历史上记录的这个层的输出结果
	IntermediateGradient lossGradOnOut, //当前层的输出结果的偏导，最后输入的偏导可能也会存在这里面
	const FeatureLib addFeature, //这个decoder数据的原始输入
	const AngleTransformer transformer //角度转换器
)
{
	//对激活层的求导，求导完成后会得到loss对残差连接块的导数
	getLossDiffOnActInput<THREAD_PER_BLOCK,FEATURE_SIZE>(
		&intermediateResult,&lossGradOnOut,&transformer);
}

void backwardFeatureAddAndNorm(
	IntermediateDecoderResult* intermediateResult,//当前层的中间信息
	FeatureLib* lossGradOnOut //loss对输出的tensor的偏导
)
{

}