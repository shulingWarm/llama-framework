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
		//计算当前实际访问的keyId
		const unsigned idKey = idCycle*KEY_NUM_FOR_ONE_CYCLE + KEY_OFFSET_IN_CYCLE;
		//当前线程负责的起始地址
		const FeatType* dotHead = getFeatHead(kFeature,idKey) + blockIdx.y*HEAD_DIM + KEY_OFFSET_IN_KEY;
		//对两段内存做局部点乘 这里需要考虑idKey越界的情况
		HalfType dotAns = idKey < KEY_TASK_NUM ? 
			localDot<THREAD_OP_NUM,HalfType>(dotQHead,dotHead,transformer) : __float2half(0.f);
		//用蝶式寻址把dotAns加起来 由于送进来的向量的模长都是1,这里理论上是不会加爆的
		for(int idCross=THREAD_NUM_FOR_ONE_KEY/2;idCross>=1;idCross/=2)
		{
			dotAns = __hadd(dotAns,__shfl_xor_sync(unsigned(-1), dotAns, idCross, THREAD_NUM_FOR_ONE_KEY));
		}
		//由第一个线程负责保存Q*K^T的结果
		if(KEY_OFFSET_IN_KEY == 0)
		{
			//直接弄成那个exp的形式 因为所有向量的幅值都是严格控制过的，因此这里不用考虑指数结果过大的问题
			dstScore[idKey] = hexp(dotAns);
		}
	}
	//调用同步，确保attention分数都算完了
	__syncthreads();
}

//对传入的向量做softmax
//不过需要特别注意的是，这里面传入的数字已经做过exp了，想办法把它们的求和弄成1就行
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
	__shared__ HalfType totalOutputSegment[THREAD_OP_NUM * THREAD_PER_BLOCK];
	//自己负责的累加结果
	//这个临时的中间结果还是用Half比较好
	HalfType* outputSegment = totalOutputSegment + threadIdx.x * THREAD_OP_NUM;
	//在初始化阶段，把自己负责的这个片段弄成0
	memset(outputSegment,0,sizeof(HalfType)*THREAD_OP_NUM);
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
			localVecAdd<HalfType,FeatType>(outputSegment,
				valueHead,attentionScore[idValue],THREAD_OP_NUM,transformer);
		}
	}
	__syncthreads();
	//一共有64个线程，这里可以直接用二进制的方式来处理，或者还是按照那种比较远的形式来叠加吧
	for(int idCross=1;idCross<VALUE_NUM_FOR_ONE_CYCLE;idCross<<=1)
	{
		//判断是否应该由这个线程来处理加法
		if(VALUE_OFFSET_IN_CYCLE % (idCross<<1) == 0 && VALUE_OFFSET_IN_CYCLE + idCross < VALUE_NUM_FOR_ONE_CYCLE)
		{
			//这里要找的是跨越了idCross个output后的同一个位置的输出向量片段
			halfDirectAdd(outputSegment,totalOutputSegment+
				(threadIdx.x + idCross*THREAD_NUM_FOR_ONE_VALUE)*THREAD_OP_NUM,
				THREAD_OP_NUM);
		}
		__syncthreads();
	}
	//到这个地方的时候，所有的向量片段都已经汇聚到了第一个目标数据里面
	//现在开始记录目标数据
	//需要判断自己是不是属于第一个向量片段的那8个数据
	if(VALUE_OFFSET_IN_CYCLE == 0)
	{
		//自己的片段对应到output里面的位置
		FeatType* targetOutFeature = getFeatHead(outFeature,blockIdx.x) + HEAD_DIM*blockIdx.y + VALUE_OFFSET_IN_VALUE;
		//把属于自己的那个片段记录到最终的输出向量里面
		halfVecToFeatVec(outputSegment,targetOutFeature,THREAD_OP_NUM,transformer);
	}
	__syncthreads();
}

//对qkv的attention kernel
//到现在来看，这个逻辑应该是只能服务于generation阶段的
//其实这个head_dim本来是可以直接在featureLib里面获取到的，但这样动态的数据不方便开辟共享内存
//所以就用泛型的方式来传入了
//grid的shape是 [TOKEN_NUM, HEAD_NUM]
template<unsigned int HEAD_DIM,unsigned int THREAD_PER_BLOCK>
__global__ void CUDAAttention(const FeatureLib qFeature,const FeatureLib kFeature,
	const FeatureLib vFeature,FeatureLib outFeature,
	const AngleTransformer transformer,
	DebugInfo debugInstance //用于debug的一些中间信息
)
{
	//记录value里面的0~128的信息 这属于debug信息
	if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x<128)
	{
		debugInstance.tempValue[threadIdx.x] = vFeature.data[threadIdx.x];
	}
	//用来保存q*K^T结果的数据 但这里是动态共享内存 实际上共享内存的长度应该是kFeature里面feature的个数
	//对应的也就是feature score的长度 正常情况下这里求和倒是不是超限，但由于后面还要算指数，那就直接在这里改成保存fp16了
	extern __shared__ HalfType attentionScore[];
	//执行Q*K^T,但是每个q只和自己之前的数据相乘 这里面已经调用过同步了
	qDotKTImple<HEAD_DIM,ATTENTION_THREAD_OP_NUM>(fetchQuery<HEAD_DIM>(&qFeature),
		&kFeature,attentionScore,&transformer
	);
	//执行softmax
	softmax<ATTENTION_THREAD_OP_NUM,THREAD_PER_BLOCK>(attentionScore);
	//准备用attn_score把v加权平均起来
	attnScoreDotValue<HEAD_DIM,ATTENTION_THREAD_OP_NUM,THREAD_PER_BLOCK>(attentionScore,&vFeature,&outFeature,&transformer);
}

//attention的运算层
class AttentionDecoder
{
public:
	//qkv的三个weight 直接用旋转矩阵来表示
	RotationMatrix qkvWeight[3];

	//输出层的weight
	RotationMatrix outputWeight;

	//初始化权重信息
	void init(std::fstream& fileHandle)
	{
		//依次读取qkv的weight
		qkvWeight[0].init(fileHandle);
		qkvWeight[1].init(fileHandle);
		qkvWeight[2].init(fileHandle);
		outputWeight.init(fileHandle);
	}

	//执行decoder层的前向推导 其实就对应的是pytorch里面写的那个hidden size
	//outFeature是用来保存输出结果的
	void forward(FeatureLib* features,
		const AngleTransformer* const transformer)
	{
		std::cout<<"deepCopyFrom"<<std::endl;
		//先把输入数据复制三份，这是用来存储qkv的结果的
		FeatureLib copyQkv[3];
		for(int i=0;i<3;++i)
			copyQkv[i].deepCopyFrom(features);
		std::cout<<"just copy query"<<std::endl;
		//打印一下刚刚复制过的value信息
		std::cout<<"just copy value"<<std::endl;
		//先把qkv乘上去
		//乘完之后算是已经获取到了后面需要处理的qkv
		//这里面包含了旋转位置编码的操作，目前只测试生成第一个token的过程
		//等第一个token测试完了再去弄kv cache的问题
		//传入的0表示目前的起始偏移量，现在先用0跑一个summarization阶段出来
		//后面运行generation阶段的时候再改
		std::cout<<"qkvRotation"<<std::endl;
		qkvRotation(qkvWeight,copyQkv,transformer,BASE_ROTATION,0);
		//打印value
		std::cout<<"direct query print"<<std::endl;
		//特征的数量，或者说是序列长度
		const unsigned SEQ_LENGTH = features[0].featureNum;
		//准备一个debug信息用于处理
		DebugInfo debugInstance;
		initDebugInfo(&debugInstance);
		//接下来要开始执行正式的attention层操作了
		//注意这里在output这个地方也传入了Query,其实是把输出内存保存到了query上
		CUDAAttention<HEAD_DIM,256><<<dim3(SEQ_LENGTH,HEAD_DIM,1),256,sizeof(HalfType)*SEQ_LENGTH>>>(
			copyQkv[0],copyQkv[1],copyQkv[2],copyQkv[0],transformer[0],debugInstance
		);
		//处理已经取到数据的debug信息
		dealwithDebugInfo(&debugInstance);
		std::cout<<"query"<<std::endl;
		//释放已经处理过的debug信息
		releaseDebugInfo(&debugInstance);
		//准备给输出结果乘上weight
		//输出结果是用query保存的，所以应该乘到query上
		//目前是默认query不会超数值限制，如果超了数据限制那就再议
		localRotation(&outputWeight,copyQkv,transformer);
		//把输入的向量加到这个输出结果上，加完之后当场norm,把结果置换到当时的那个input上面
		featureAddAndNorm(features,copyQkv,transformer);
	}

};