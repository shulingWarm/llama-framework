#pragma once
#include"FeatureLib.hpp"
#include"FeatureND.hpp"
#include<iostream>

//为了方便索引qkv而做的id
//注意这个0,1,2会直接参与索引计算的
#define ENUM_ID_QUERY 0
#define ENUM_ID_KEY 1
#define ENUM_ID_VALUE 2 
//attention层的输出结果
#define ENUM_ID_ATTN_OUT 3
//输出位置的旋转层的id
//和上面不是同一个体系的
#define ENUM_ID_ROT_OUTPUT 3

//叠加上输入结果后的feature
//RES指的是残差那个单词
#define ENUM_ID_RES 4

//qkv最终结果的维度
#define QKV_RESULT_DIM 5
//qkv的旋转过程的维度，指的是一共有几个
#define QKV_ROT_DIM 4

//softmax维度上的访问
#define BEFORE_SFTMX 0
#define AFTER_SFTMX 1

//什么都不记录的版本
#define RECORD_ID_NONE 0
//中间记录的各种方式 这是反向求导的时候，每次都把每一块重新推理一下，方便做反向求导
#define RECORD_ID_ALL 1
//这个就是训练前的前向推理，记录每个decoder的训练节点
#define RECORD_ID_TRAIN 2

//decoder推理时的中间结果
//当需要进行反向求导时，就先把这个模型存储的临时输入记录一下，这样可以比较方便地去推每一层的真值
struct IntermediateDecoderResult
{

	//qkv的结果 这里直接把qkv的信息合并进一个连续数据里了
	//output那里的旋转结果也会放这里
	Feature4D qkvRotationResult;

	//qkv的结果，其实也会包含激活层的输入相关的内容
	//但为了记录方便，就统一都叫qkv了
	Feature3D qkvResult;

	//attention层的中间结果的feature信息
	AttnFeature4D attnFeature;
};

//对decoder的中间结果的初始化
template<int FEATURE_SIZE>
void initDecoderIntermediateContainer(
	IntermediateDecoderResult& dstResult, //需要被初始化的目标结果
	int tokenNum //目前这一次推理会用到多少token
)
{
	//4D数据的索引
	int idDims[4] = {
		QKV_ROT_DIM, //分别对应qkv
		FEATURE_SIZE-1, //这意味着迭代次数
		tokenNum, //有多少个token就有多少行特征信息
		FEATURE_SIZE //最后是每个feature_size被旋转后的中间结果
	};
	//对qkv数据开辟内存空间
	allocateFeatureND<4,FeatType>(dstResult.qkvRotationResult,
		idDims
	);
	//qkv结果的维度
	int qkvResultDims[3]={QKV_RESULT_DIM,tokenNum,FEATURE_SIZE};
	//对qkv以及attention的output分别开辟空间
	allocateFeatureND<3,FeatType>(dstResult.qkvResult,qkvResultDims);
	//attention层的中间结果
	int attnDims[4] = {
		2,//softmax之前的结果和softmax之后的结果
		tokenNum,//每个token的数据
		HEAD_NUM, //每个头各应有一个匹配结果的数据
		//最后一个维度后面需要考虑再优化
		tokenNum //每个token和其它token的比较，这虽然叫tokenNum,其实是一个变长度的
	};
	allocateFeatureND<4,AttnType>(dstResult.attnFeature,attnDims);
}

//对一般feature结果的记录
//HEAD_FLAG表示这个东西是带注意力头的，并不是要复制完整的数据
//HEAD_FLAG为true的时候，FEATURE_LEN表示的就是一个头的长度
template<char RECORD_FLAG,char HEAD_FLAG,
	int ID_QKV,int FEATURE_LEN>
__device__ void recordMainFeature(IntermediateDecoderResult* intermediateResult,
	FeatType* dataHead //传入的时候需要先把数据取好
)
{
	if constexpr (RECORD_FLAG==RECORD_ID_ALL)
	{
		//上面的旋转位置编码在结尾是没有同步的
		__syncthreads();
		//只拿一部分线程执行这个逻辑
		if(threadIdx.x < WARP_SIZE)
		{
			//根据当前线程决定的取数据的维度
			int idDims[2] = {ID_QKV,(int)blockIdx.x};
			//取出对应位置的数据头，这是直接精确到feature的
			FeatType* recordHead = getFeatHeadND<3,FeatType,2>(
				&intermediateResult->qkvResult,idDims
			);
			//对于有数据头的情况需要再单独处理
			if constexpr (HEAD_FLAG)
			{
				//调用执行数据复制
				dataCopyMultiThread<sizeof(FeatType)*FEATURE_LEN/WARP_SIZE>(
					(char*)(recordHead + blockIdx.y*FEATURE_LEN),
					(const char*)(dataHead + blockIdx.y*FEATURE_LEN)
				);
			}
			else
			{
				//调用执行数据复制
				dataCopyMultiThread<sizeof(FeatType)*FEATURE_LEN/WARP_SIZE>(
					(char*)recordHead,(const char*)dataHead
				);
			}
		}
		__syncthreads();
	}
}

//用于打印中间结果的核函数
static __global__ void kernelPrint(IntermediateDecoderResult srcResult)
{
	if(threadIdx.x == 0)
	{
		//输出位置的数据头
		int outRoteDim[] = {ENUM_ID_RES,5};
		FeatType* dataHead = getFeatHeadND<3,FeatType,2>(
			&srcResult.qkvResult,outRoteDim);
		//打印叠加res之后的部分数据
		for(int i=0;i<128;++i)
		{
			printf("%d ",(int)dataHead[i]);
		}
		printf("\n");
	}
}

//打印已经记录下来的中间结果
void printIntermediateRotationResult(
	IntermediateDecoderResult& srcResult
)
{
	kernelPrint<<<1,32>>>(srcResult);
	cudaDeviceSynchronize();
}

