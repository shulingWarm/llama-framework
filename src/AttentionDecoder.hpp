#pragma once
#include"RotationMatrix.hpp"
#include"FeatureLib.hpp"
#include"LocalRotationImple.hpp"
#include<fstream>
#include"TinyCompute.hpp"
#include"VectorAddNorm.hpp"
#include<iostream>

//用于抓一些中间的cuda数据
struct DebugInfo
{
public:
	FeatType* tempValue;
};

//构造debug信息
void initDebugInfo(DebugInfo* debugInstance)
{
	//开辟对应的空间
	debugInstance->tempValue = (FeatType*)dataAllocate(sizeof(FeatType)*128);
}

//释放debug信息
void releaseDebugInfo(DebugInfo* debugInstance)
{
	//释放对应的空间
	releaseCudaMemory((char*)debugInstance->tempValue);
}

//处理debug信息
void dealwithDebugInfo(DebugInfo* debugInstance)
{
	//把debug信息里面的值拿出来，打印一下value里面的信息，看一下是不是value里面的信息就都是不正常的
	FeatType cpuData[128];
	cudaMemcpy(cpuData,debugInstance->tempValue,sizeof(FeatType)*128,cudaMemcpyDeviceToHost);
	//打印cpu的数据
	std::cout<<"value data"<<std::endl;
	for(int i=0;i<128;++i)
		std::cout<<(int)cpuData[i]<<" ";
	std::cout<<std::endl;
}

//计算当前线程块应该处理哪个query
template<unsigned int HEAD_DIM>
__device__ const FeatType* fetchQuery(const FeatureLib* qFeature)
{
	return getFeatHead(qFeature,blockIdx.x) + blockIdx.y*HEAD_DIM;
}

//把角度的差值转换成attention score的exp形式
__device__ float transAngleDiffToAttentionScore(u16 accumulateAngleDiff)
{
	//把数据转换到float然后调用exp
	return expf(float(accumulateAngleDiff)/256.f);
}

//处理q*k^T
//THREAD_OP_NUM表示每个线程处理多少个数 至少应该保证每一组的处理线程数不要超过32个
//THREAD_OP_NUM,指的是每个线程处理的key里面的数字的个数
//THREAD_OP_NUM只能是2的整数次幂，这样对应的线程数才能是二的整数次幂，这只能是在使用的时候自己去保证它 
template<unsigned int HEAD_DIM,unsigned THREAD_OP_NUM>
__device__ void qDotKTImple(const FeatType* qHeadOfBlock,const FeatureLib* kFeature,HalfType* dstScore,
	const AngleTransformer* const transformer
)
{
	//每个key向量由几个线程来负责
	const unsigned THREAD_NUM_FOR_ONE_KEY = HEAD_DIM/THREAD_OP_NUM;
	//所有的线程一轮可以处理多少个key
	const unsigned KEY_NUM_FOR_ONE_CYCLE = blockDim.x / THREAD_NUM_FOR_ONE_KEY;
	//需要处理的key的任务数
	const unsigned KEY_TASK_NUM = blockIdx.x + 1;
	//每个线程需要分别负责几个key
	const unsigned KEY_NUM_FOR_ONE_THREAD = divUp(KEY_TASK_NUM,KEY_NUM_FOR_ONE_CYCLE);
	//当前线程在每一轮里面负责第几个key
	const unsigned KEY_OFFSET_IN_CYCLE = threadIdx.x/THREAD_NUM_FOR_ONE_KEY;
	//当前的线程在一个key里面负责第几个数字
	const unsigned KEY_OFFSET_IN_KEY = (threadIdx.x%THREAD_NUM_FOR_ONE_KEY) * THREAD_OP_NUM;
	//当前线程负责的q的起始地址
	const FeatType* dotQHead = qHeadOfBlock + KEY_OFFSET_IN_KEY;
	//遍历需要计算的每一层
	for(int idCycle=0;idCycle<KEY_NUM_FOR_ONE_THREAD;++idCycle)
	{
		//计算当前实际访问的keyId 提前准备的idKey 但有可能最后走的不是这个值
		const unsigned idKeyPre = idCycle*KEY_NUM_FOR_ONE_CYCLE + KEY_OFFSET_IN_CYCLE;
		const unsigned idKey = idKeyPre < KEY_TASK_NUM ? idKeyPre : blockIdx.x;
		//当前线程负责的起始地址
		const FeatType* dotHead = getFeatHead(kFeature,idKey) + blockIdx.y*HEAD_DIM + KEY_OFFSET_IN_KEY;
		//计算局部的角度差的余弦的和
		HalfType dotAns = (idKey<KEY_TASK_NUM) ?
			localAngleDis<THREAD_OP_NUM>(dotQHead,dotHead,transformer) : __float2half(0.f);
		//用蝶式寻址把dotAns加起来
		for(int idCross=THREAD_NUM_FOR_ONE_KEY/2;idCross>=1;idCross/=2)
		{
			dotAns = __hadd(dotAns,
				__shfl_xor_sync(unsigned(-1), dotAns, idCross, THREAD_NUM_FOR_ONE_KEY));
			//dotAns += __shfl_xor_sync(unsigned(-1), dotAns, idCross, THREAD_NUM_FOR_ONE_KEY);
		}
		//由第一个线程负责保存Q*K^T的结果 但这里是需要考虑结果过大的问题的
		//这个理论上的最大值是
		if(KEY_OFFSET_IN_KEY == 0 && idKey == idKeyPre)
		{
			//直接在exp里面保存这个数据，后面再把它整体减最大值
			dstScore[idKey] = dotAns;
			//dstScore[idKey] = __float2half(transAngleDiffToAttentionScore(dotAns));
		}
	}
	//调用同步，确保attention分数都算完了
	__syncthreads();
}

//对传入的向量做softmax
//不过需要特别注意的是，这里面传入的数字已经做过exp了，想办法把它们的求和弄成1就行
//现在需要由softmax负责数据均匀处理的问题了，这里仅仅是数值的结果，需要先求最大值，然后把这个最大值减掉再求exp
template<unsigned THREAD_OP_NUM,unsigned THREAD_PER_BLOCK>
__device__ void softmax(HalfType* attentionScore)
{
	//共享内存，用于存储每个warp的局部求和结果
	__shared__ HalfType warpAddupResult[THREAD_PER_BLOCK / WARP_SIZE];
	//总的需要处理的序列长度
	const unsigned SEQ_LENGTH = blockIdx.x+1;
	//计算总共需要多少个warp
	const unsigned WARP_NUM = divUp(divUp(SEQ_LENGTH,THREAD_OP_NUM),WARP_SIZE);
	//计算总共需要的线程数
	const unsigned WORKING_THREAD_NUM = WARP_NUM * WARP_SIZE;
	//当前线程所属的warp
	const unsigned idWarp = threadIdx.x / WARP_SIZE;
	//访问数据时起始位置的id
	const unsigned beginId = THREAD_OP_NUM*threadIdx.x;
	//访问数据的结束位置id
	const unsigned endId = beginId + THREAD_OP_NUM;
	//实际需要叠加的长度
	const unsigned workLength = beginId >= SEQ_LENGTH ? 0 : //如果起始位置就已经超了，那就是0
		(endId <= SEQ_LENGTH ? THREAD_OP_NUM : //如果结束位置也在范围内那就是预定的那个长度
		SEQ_LENGTH - beginId  //否则如果起始位置在范围内，结束位置超了，那就要算出来一个临时的长度
	);
	//当前线程需要处理的局部区域
	HalfType* threadTaskHead = attentionScore + beginId;
	//处理工作范围内的数据，求最大值
	if(threadIdx.x < WORKING_THREAD_NUM)
	{
		//求局部区域的最大值
		HalfType localMax = getLocalMax(threadTaskHead,workLength);
		//使用蝶式寻址求这个warp里面的最大值
		for(int idCross=WARP_SIZE/2;idCross>=1;idCross>>=1)
		{
			localMax = __hmax(localMax,
				__shfl_xor_sync(unsigned(-1), localMax, idCross, WARP_SIZE)
			);
		}
		//把这个warp的最大值记录在这个warp的结果里面
		if(threadIdx.x % WARP_SIZE == 0)
		{
			warpAddupResult[idWarp] = localMax;
		}
	}
	//到这里是每个warp计算了自己的线程块内容
	__syncthreads();
	//在全部的共享内存里面找最大值
	for(int addStep = 1;addStep<WARP_NUM;addStep<<=1)
	{
		//判断记录当前位置和扩展位置都是有效位
		if(threadIdx.x%(addStep<<1) == 0 && threadIdx.x + addStep < WARP_NUM)
		{
			//把当前访问位置和指定位置加起来
			warpAddupResult[threadIdx.x] = __hmax(warpAddupResult[threadIdx.x],
				warpAddupResult[threadIdx.x + addStep]
			);
		}
		__syncthreads();
	}
	//然后让每一段数据减去这个最大值
	if(threadIdx.x < WORKING_THREAD_NUM)
	{
		//对每个位置的数据调用减法操作顺便调用exp
		for(int i=0;i<workLength;++i)
		{
			//把目标数据减掉一个值然后直接调用exp
			threadTaskHead[i] = hexp(__hsub(threadTaskHead[i],warpAddupResult[0]));
		}
	}
	//到这里算是已经把attention score转换完了，可以正常进行之前的exp操作了
	__syncthreads();
	//只有工作范围内的线程才需要走这个分支
	if(threadIdx.x < WORKING_THREAD_NUM)
	{
		//把局部区域加起来
		HalfType localSum = localAddup(threadTaskHead,workLength);
		//使用蝶式寻址把warp里面的数据加起来
		for(int idCross=WARP_SIZE/2;idCross>=1;idCross/=2)
		{
			localSum += __shfl_xor_sync(unsigned(-1), localSum, idCross, WARP_SIZE);
		}
		//把计算结果存在共享内存里面，用于计算attention分数的总和
		if(threadIdx.x % WARP_SIZE == 0)
		{
			warpAddupResult[idWarp] = localSum;
		}
	}
	__syncthreads();
	//尽量用前面的线程，直接用二分查找的方式依次叠加每个分数
	for(int addStep = 1;addStep<WARP_NUM;addStep<<=1)
	{
		//判断记录当前位置和扩展位置都是有效位
		if(threadIdx.x%(addStep<<1) == 0 && threadIdx.x + addStep < WARP_NUM)
		{
			//把当前访问位置和指定位置加起来
			warpAddupResult[threadIdx.x] += warpAddupResult[threadIdx.x + addStep];
		}
		__syncthreads();
	}
	//各个线程找到自己负责的部分，按原来分配的任务区间把这个数除掉
	if(threadIdx.x < WORKING_THREAD_NUM)
	{
		//对局部区域除以算出来的那个数
		localScale(threadTaskHead,workLength,__float2half(1.f)/warpAddupResult[0]);
	}
	//处理完成之后，到这里算是完成了softmax
	__syncthreads();
}

//实现attention_score * V
//虽然attention分数还是那个分数，但叠加数据的时候需要注意使用角度中值叠加
template<unsigned HEAD_DIM, unsigned THREAD_OP_NUM,unsigned THREAD_PER_BLOCK>
__device__ void attnScoreDotValue(HalfType* attentionScore,const FeatureLib* vFeature,FeatureLib* outFeature,
	const AngleTransformer* const transformer
)
{
	//给每个head_dim分配几个线程
	const unsigned THREAD_NUM_FOR_ONE_VALUE = HEAD_DIM/THREAD_OP_NUM;
	//所有的线程一轮可以处理多少个value 最后一共会有64组数据，每一组数据会被分成8个8
	const unsigned VALUE_NUM_FOR_ONE_CYCLE = blockDim.x / THREAD_NUM_FOR_ONE_VALUE;
	//需要处理的value的任务数 其实也就是序列长度
	const unsigned SEQ_LENGTH = blockIdx.x + 1;
	//每个线程需要分别负责几个value 其实就是执行的轮数
	const unsigned VALUE_NUM_FOR_ONE_THREAD = divUp(SEQ_LENGTH,VALUE_NUM_FOR_ONE_CYCLE);
	//当前线程在每一轮里面负责第几个value
	const unsigned VALUE_OFFSET_IN_CYCLE = threadIdx.x / THREAD_NUM_FOR_ONE_VALUE;
	//当前线程在任意一轮的value里面负责第几个数字
	const unsigned VALUE_OFFSET_IN_VALUE = (threadIdx.x % THREAD_NUM_FOR_ONE_VALUE) * THREAD_OP_NUM;
	//最终每个线程块都会有一个属于自己的输出片段，但最开始的时候需要把这个东西放在共享内存上
	//因为最后需要把所有的东西都加起来
	__shared__ NumWithWeight totalOutputSegment[THREAD_OP_NUM * THREAD_PER_BLOCK];
	//自己负责的累加结果
	//这个临时的中间结果还是用Half比较好
	NumWithWeight* outputSegment = totalOutputSegment + threadIdx.x * THREAD_OP_NUM;
	//在初始化阶段，把自己负责的这个片段弄成0
	memset(outputSegment,0,sizeof(NumWithWeight)*THREAD_OP_NUM);
	//处理每一轮的任务
	for(int idCycle=0;idCycle<VALUE_NUM_FOR_ONE_THREAD;++idCycle)
	{
		//当前位置访问的value id
		const unsigned idValue = idCycle*VALUE_NUM_FOR_ONE_CYCLE + VALUE_OFFSET_IN_CYCLE;
		//需要处理的数据头
		const FeatType* valueHead = getFeatHead(vFeature,idValue) + blockIdx.y*HEAD_DIM + VALUE_OFFSET_IN_VALUE;
		//判断value是否在范围内
		if(idValue < SEQ_LENGTH)
		{
			//把结果叠加的输出数据段上 到这里就要当于把数据加到了outputSegment上
			//这里使用的是各个角度按比例相加
			localVecAdd<NumWithWeight,FeatType>(outputSegment,
				valueHead,attentionScore[idValue],THREAD_OP_NUM,transformer);
		}
	}
	__syncthreads();
	//一共有64个线程，这里可以直接用二进制的方式来处理，或者还是按照那种比较远的形式来叠加吧
	for(int idCross=1;idCross<VALUE_NUM_FOR_ONE_CYCLE;idCross<<=1)
	{
		//判断是否应该由这个线程来处理加法
		if((VALUE_OFFSET_IN_CYCLE % (idCross<<1)) == 0 && VALUE_OFFSET_IN_CYCLE + idCross < VALUE_NUM_FOR_ONE_CYCLE)
		{
			//这里要找的是跨越了idCross个output后的同一个位置的输出向量片段
			//直接把两个线程的对应位置加起来
			//这里调用的是中值求和的方案
			vecAddOn(outputSegment,totalOutputSegment+
				(threadIdx.x + idCross*THREAD_NUM_FOR_ONE_VALUE)*THREAD_OP_NUM,
				THREAD_OP_NUM);
		}
		__syncthreads();
	}
	//打印自己的输出位
	// if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
	// {
	// 	for(int i=0;i<THREAD_OP_NUM;++i)
	// 	{
	// 		printf("%f %f |",outputSegment[i].weight,outputSegment[i].angle);
	// 	}
	// 	printf("\n");
	// }
	//到这个地方的时候，所有的向量片段都已经汇聚到了第一个目标数据里面
	//现在开始记录目标数据
	//需要判断自己是不是属于第一个向量片段的那8个数据
	if(VALUE_OFFSET_IN_CYCLE == 0)
	{
		//自己的片段对应到output里面的位置
		FeatType* targetOutFeature = getFeatHead(outFeature,blockIdx.x) + HEAD_DIM*blockIdx.y + VALUE_OFFSET_IN_VALUE;
		//到这里其实就是做个类型转换，转换回正常的角度类型
		featTypeTransform<FeatType,NumWithWeight>(targetOutFeature,outputSegment,THREAD_OP_NUM);
		//把属于自己的那个片段记录到最终的输出向量里面
		//halfVecToFeatVec(outputSegment,targetOutFeature,THREAD_OP_NUM,transformer);
	}
	__syncthreads();
}

//记录softmax的中间结果
//ID_TENSOR可能指的是softmax之前的结果，也可能是softmax之后的结果
template<char RECORD_FLAG,int TENSOR_ID>
inline __device__ void recordSoftmaxResult(AttnType* attentionScore,
	IntermediateDecoderResult* intermediateResult
)
{
	if constexpr (RECORD_FLAG==RECORD_ID_ALL)
	{
		__syncthreads();
		//要取出的数据的维度
		int recordDim[3] = {TENSOR_ID,(int)blockIdx.y,(int)blockIdx.x};
		//取出feature
		AttnType* recordHead = getFeatHeadND<4,AttnType,3>(
			&intermediateResult->attnFeature,recordDim
		);
		//灵活长度的复制
		if((blockIdx.x + 1) > blockDim.x)
		{
			flexibleDataCopy<0>((char*)recordHead,(char*)attentionScore,
				sizeof(AttnType)*(blockIdx.x+1)
			);
		}
		else
		{
			flexibleDataCopy<1>((char*)recordHead,(char*)attentionScore,
				sizeof(AttnType)*(blockIdx.x+1)
			);
		}
		__syncthreads();
	}
}

//对qkv的attention kernel
//到现在来看，这个逻辑应该是只能服务于generation阶段的
//其实这个head_dim本来是可以直接在featureLib里面获取到的，但这样动态的数据不方便开辟共享内存
//所以就用泛型的方式来传入了
//grid的shape是 [TOKEN_NUM, HEAD_NUM]
template<unsigned int HEAD_DIM,unsigned int THREAD_PER_BLOCK,
	char RECORD_FLAG,int FEATURE_LEN
>
__global__ void CUDAAttention(const FeatureLib qFeature,const FeatureLib kFeature,
	const FeatureLib vFeature,FeatureLib outFeature,
	const AngleTransformer transformer,
	IntermediateDecoderResult intermediateResult //用来存储attention中间过程
)
{
	//虽然现在改成了角度差，但毕竟还是要最后结算一个exp的，所以最后难免还是要使用half数据类型
	extern __shared__ HalfType attentionScore[];
	//执行Q*K^T,但是每个q只和自己之前的数据相乘 这里面已经调用过同步了
	qDotKTImple<HEAD_DIM,ATTENTION_THREAD_OP_NUM>(fetchQuery<HEAD_DIM>(&qFeature),
		&kFeature,attentionScore,&transformer
	);
	//记录softmax之前的结果
	recordSoftmaxResult<RECORD_FLAG,BEFORE_SFTMX>(
		attentionScore,&intermediateResult);
	//执行softmax
	softmax<ATTENTION_THREAD_OP_NUM,THREAD_PER_BLOCK>(attentionScore);
	//记录softmax之后的结果
	recordSoftmaxResult<RECORD_FLAG,AFTER_SFTMX>(
		attentionScore,&intermediateResult);
	//准备用attn_score把v加权平均起来
	attnScoreDotValue<HEAD_DIM,ATTENTION_THREAD_OP_NUM,THREAD_PER_BLOCK>(attentionScore,&vFeature,&outFeature,&transformer);
	//记录value的主feature结果
	recordMainFeature<RECORD_FLAG,1,ENUM_ID_ATTN_OUT,FEATURE_LEN>(
		&intermediateResult,getFeatHead(&outFeature,blockIdx.x)
	);
}

//单纯用来测试打印的kernel
__global__ void testKernelPrint()
{
	if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x==0)
	{
		printf("testKernelPrint\n");
	}
}

//attention的运算层
class AttentionDecoder
{
	//本层的历史结果记录
	FeatureLib historyInput;
public:
	//qkv的三个weight 直接用旋转矩阵来表示
	RotationMatrix qkvWeight[3];

	//输出层的weight
	RotationMatrix outputWeight;

	//value的偏置
	FeatureLib valueBias;

	//初始化权重信息
	void init(std::fstream& fileHandle)
	{
		//依次读取qkv的weight
		qkvWeight[0].init(fileHandle);
		qkvWeight[1].init(fileHandle);
		qkvWeight[2].init(fileHandle);
		outputWeight.init(fileHandle);
		//读取bias
		valueBias = loadFeatureLib(fileHandle);
	}

	//获取历史信息的记录
	FeatureLib& getHistoryInput()
	{
		MY_ASSERT(historyInput.data != nullptr);
		return this->historyInput;
	}

	//记录输入的中间数据
	template<char RECORD_FLAG>
	void recordInput(FeatureLib* features)
	{
		if constexpr (RECORD_FLAG == RECORD_ID_TRAIN)
		{
			//调用深拷贝，复制此时的输入feature
			historyInput.deepCopyFrom(features);
		}
	}

	//执行decoder层的前向推导 其实就对应的是pytorch里面写的那个hidden size
	//outFeature是用来保存输出结果的
	template<char RECORD_FLAG>
	void forward(FeatureLib* features,
		const AngleTransformer* const transformer,
		FeatureLib* copyQkv, //用来存储qkv的三个feature 最后还是要走kv cache的
		IntermediateDecoderResult* intermediateResult //不能是空指针
	)
	{
		//记录此时输入的中间结果
		recordInput<RECORD_FLAG>(features);
		//printFeatureLib<0,128>(features,0);
		//先把输入数据复制三份，这是用来存储qkv的结果的
		for(int i=0;i<3;++i)
			copyQkv[i].deepCopyFrom(features);
		//先把qkv乘上去
		qkvRotation<RECORD_FLAG>(qkvWeight,copyQkv,transformer,
			BASE_ROTATION,0,&valueBias,intermediateResult);
		//特征的数量，或者说是序列长度
		const unsigned SEQ_LENGTH = features[0].featureNum;
		//接下来要开始执行正式的attention层操作了
		//注意这里在output这个地方也传入了Query,其实是把输出内存保存到了query上
		//printFeatureLib<0,128>(copyQkv,0);
		CUDAAttention<HEAD_DIM,256,RECORD_FLAG,FET_LENGTH><<<dim3(SEQ_LENGTH,HEAD_DIM,1),256,sizeof(HalfType)*SEQ_LENGTH>>>(
			copyQkv[0],copyQkv[1],copyQkv[2],copyQkv[0],transformer[0],
			intermediateResult[0]
		);

		//输出结果是用query保存的，所以应该乘到query上
		//目前是默认query不会超数值限制，如果超了数据限制那就再议
		localRotation<RECORD_FLAG,ENUM_ID_ROT_OUTPUT>(
			&outputWeight,copyQkv,transformer,intermediateResult);
		//把输入的向量加到这个输出结果上，加完之后当场norm,把结果置换到当时的那个input上面
		featureAddAndNorm<RECORD_FLAG>(features,
			copyQkv,transformer,intermediateResult);
	}

	//反向求导的操作
	//此时的中间结果里是已经有内容的，直接用就行
	void backward(FeatureLib* historyOutput, //这是上次运行时记录下的输出结果
		Optimizer& optInstance, //优化器，里面会存储loss对output里面每个数字的求导
		IntermediateDecoderResult& intermediateResult //自己跑出来的中间结果
	){
		//从后向前遍历每一层的decoder
		
	}

};