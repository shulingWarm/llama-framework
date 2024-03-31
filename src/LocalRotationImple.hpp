#pragma once
#include "RotationMatrix.hpp"
#include "FeatureLib.hpp"
#include"FeatureND.hpp"
#include"IntermediateDecoderResult.hpp"
#include"RotationIntermediateCollect.hpp"

//执行旋转时每个线程的状态，包括block的大小，起始点和当前这个线程自己具体访问的位置
struct DimSelectInfo
{
public:
	//目前的block的xy的起始位置
	u16 rowBegin;
	u16 colBegin;
	//当前的block的大小
	u16 blockSize;

	//本次迭代中，本线程负责的两个数
	u16 iterDim[2];
};

//下一个周期的block
__device__ void getNextBlock(DimSelectInfo* const iterInfo)
{
	//计算当前线程在block里面的偏移量
	const u16 offsetInBlock = threadIdx.x % iterInfo->blockSize;
	//判断这个线程是不是属于前半部分
	const bool upperFlag = (offsetInBlock < (iterInfo->blockSize>>1));
	//把block的大小除以二
	iterInfo->blockSize >>= 1;
	//左半边的往左走 右半边的往右走
	iterInfo->colBegin += (upperFlag ? 
		-i16(iterInfo->blockSize) : iterInfo->blockSize
	);
	//行位置的变化 前半部分行起始不变，后半部分行起始到之前的block的下面
	iterInfo->rowBegin += (upperFlag ?
		0 : (iterInfo->blockSize << 1)
	);
}

//计算本次迭代需要旋转哪两个维度
__device__ void initDimForThisIter(DimSelectInfo* const iterInfo,const u16 idTime)
{
	//计算在当前block里面的偏移量
	const u16 offsetInBlock = threadIdx.x % iterInfo->blockSize;
	//计算访问的行位置和列位置
	iterInfo->iterDim[0]=iterInfo->rowBegin + offsetInBlock;
	iterInfo->iterDim[1]=iterInfo->colBegin + (iterInfo->blockSize + idTime - offsetInBlock)%iterInfo->blockSize;
}

//计算带符号位的两个角度的差
//但是专门用前8位来表示符号
//头8位的地方是符号位，0的情况下表示正数，1的情况下表示负数
__device__ i16 getAngleSymbolDiff(const FeatType angle1,const FeatType angle2)
{
	//初始化结果
	i16 ans = 0;
	//按照feattype来处理的指针
	FeatType* ansData = (FeatType*)&ans;
	//求两个角度的绝对值差值
	ansData[1] = angle1 - angle2;
	//判断得到的是不是正常的正数角度
	if(ansData[1] < ANGLE_MID_VALUE)
		ansData[0] = 0U;
	else
	{
		ansData[0] = 1U;
		ansData[1] = angle2 - angle1;
	}
	return ans;
}

//执行角度差缩放
//这个roteAngle虽说是叫旋转角，但其实并不是旋转的概念，而是两个角度之间的相互变换
//其实是把角度沿着某个位置的展开，并且最多也就展开一倍
//并且这东西确实还是加个了bias比较保险,如果要加bias的话，那就后面再加
__device__ FeatType angleDiffScale(const FeatType midAngle,FeatType dstAngle,
	const FeatType roteAngle,const AngleTransformer* const transformer
)
{
	//计算两个角度带符号的差
	i16 angleDiff = getAngleSymbolDiff(dstAngle,midAngle);
	//指向结果的指针
	FeatType* ansData = (FeatType*)&angleDiff;
	//对得到的角度做缩放
	ansData[1] = rotateAngleByScalar(transformer,ansData[1],roteAngle);
	if(ansData[0])
		return midAngle - ansData[1];
	return midAngle + ansData[1];
}

//角度中间展开
__device__ void angularDevelopment(const AngleTransformer* const transformer,
	FeatType* data1,FeatType* data2,const FeatType roteAngle)
{
	//计算中间位
	FeatType tempMid = getMidAngle(data1[0],data2[0]);
	//执行夹角展开
	data1[0] = angleDiffScale(tempMid,data1[0],roteAngle,transformer);
	data1[1] = angleDiffScale(tempMid,data1[1],roteAngle,transformer);
}

//对特征进行乘法操作的kernel
//每个线程块的大小都是1024,这个计算密集程度只能这样处理
//目前这个算法最大也就能处理2048的长度，如果以后要扩展更大的长度，需要实现另外的算法
//这里是完全不考虑通用性的，线程数就是feature的size的一半
//baseRotation是附加旋转角，其实是模仿的旋转位置编码，为了不过分影响原本的向量，只在最后一次旋转的时候才叠加这个旋转矩阵
//RECORD_FLAG表示是否记录旋转的中间结果
//intermediateResult是用来存储中间结果的地方
template<char RECORD_FLAG>
__device__ void cudaMatRotationDevice(const RotationMatrix* matrix,FeatureLib* features,
	const AngleTransformer* const transformer,Feature3D* intermediateResult
)
{
	//当前线程块负责处理的特征
	FeatType* const blockData = getFeatHead(features,blockIdx.x);
	//当前线程目前x的起始位置，y的起始位置和block的大小
	//这里面放的是三个数值，分别对应的是xy和block
	DimSelectInfo iterInfo = {
		0,//rowBegin
		u16(features->featureSize>>1), //colBegin
		u16(features->featureSize>>1), //blockSize
		{0,0}
	};
	//目前总的迭代次数, 这是用来寻找这两个维度对应的旋转角的
	u16 idIter = 0;
	//遍历n-1个周期，每个周期中每个线程负责两个数字
	//每过一段时间，这个blockSize就会更新一下
	while(iterInfo.blockSize > 0)
	{
		//根据当前的blockSize大小遍历指定的次数
		for(u16 idTime=0;idTime<iterInfo.blockSize;++idTime)
		{
			//初始化本次迭代需要旋转哪两个维度
			initDimForThisIter(&iterInfo,idTime);
			//执行夹角展开
			angularDevelopment(transformer,
				&blockData[iterInfo.iterDim[0]],
				&blockData[iterInfo.iterDim[1]],
				matrix->data[idIter*blockDim.x + threadIdx.x]
			);
			//更新迭代次数
			++idIter;
			__syncthreads();
			//判断是否需要记录中间结果
			if constexpr (RECORD_FLAG==RECORD_ID_ALL)
			{
				//idIter不减1的话，第一个记录位置就是空的
				recordMidRotationResult<2>(blockData,idIter-1,
					intermediateResult
				);
				__syncthreads();
			}
		}
		//迭代完一轮了，更新下一个区域的block
		getNextBlock(&iterInfo);
	}
}

//执行旋转的核函数，注意这个地方需要传值，不能传指针
//RECORD_FLAG表示是否记录旋转的中间结果
//需要在这一层把中间结果转换成Feature3D
template<char RECORD_FLAG,char ID_TENSOR>
__global__ void cudaMatRotation(const RotationMatrix matrix,FeatureLib features,
	const AngleTransformer transformer,
	IntermediateDecoderResult intermediateResult //旋转的中间结果
)
{
	if constexpr (RECORD_FLAG==RECORD_ID_ALL)
	{
		//需要取出的subTensor
		int subDim[] = {(int)ID_TENSOR};
		//取出子tensor
		Feature3D subTensor;
		getSubTensor<4,FeatType,3>(&intermediateResult.qkvRotationResult,
			&subTensor,subDim);
		//调用带中间信息调用的旋转
		cudaMatRotationDevice<RECORD_FLAG>(&matrix,&features,&transformer,&subTensor);
	}
	else
	{
		//直接调用device函数即可 这个单独的测试函数不需要基础附加旋转角
		cudaMatRotationDevice<RECORD_FLAG>(&matrix,&features,&transformer,nullptr);
	}
	
}

//给向量添加bias
template<int FEAT_LENGTH,int THREAD_PER_BLOCK>
__device__ void cudaAddBias(FeatType* dstData,const FeatType* biasData)
{
	//每个线程负责的数据个数 这里目前是默认数据的个数肯定是要大于线程的个数的
	const unsigned threadTaskNum = FEAT_LENGTH / THREAD_PER_BLOCK;
	//当前线程负责的数据头
	FeatType* dataHead = dstData + threadIdx.x*threadTaskNum;
	const FeatType* biasHead = biasData + threadIdx.x*threadTaskNum;
	//把bias的数据直接加在目标数据上，直接允许它越界的那种相加
	for(int id=0;id<threadTaskNum;++id)
	{
		dataHead[id] += biasHead[id];
	}
}

//添加bias的switch函数
//和旋转位置编码的接口不一样，这里直接把token对应的value数据拿过来用的
template<int FEAT_LENGTH,int THREAD_PER_BLOCK,
	char RECORD_FLAG,char ID_QKV
>
__device__ void cudaAddBiasWithSwitch(FeatType* dstData,const FeatType* biasData,
	IntermediateDecoderResult* intermediateResult
)
{
	//正常执行bias的函数
	cudaAddBias<FEAT_LENGTH,THREAD_PER_BLOCK>(
		dstData,biasData
	);
	//记录bias里面的函数
	if constexpr (RECORD_FLAG==RECORD_ID_ALL)
	{
		__syncthreads();
		//只拿一部分线程执行这个逻辑
		if(threadIdx.x < WARP_SIZE)
		{
			//根据当前线程决定的取数据的维度
			int idDims[2] = {(int)ID_QKV,(int)blockIdx.x};
			//取出对应位置的数据头，这是直接精确到feature的
			FeatType* recordHead = getFeatHeadND<3,FeatType,2>(
				&intermediateResult->qkvResult,idDims
			);
			//调用执行数据复制
			dataCopyMultiThread<sizeof(FeatType)*FEAT_LENGTH/WARP_SIZE>(
				(char*)recordHead,(const char*)dstData
			);
		}
		__syncthreads();
	}
}

//对数据做旋转位置编码
//这里传进来的idToken就是当前线程处理的token,每个线程块传入的这个值都不一样
template<unsigned int HEAD_DIM_T,int FEATURE_LEN,int THREAD_PER_BLOCK>
__device__ void rotaryEmbedding(FeatureLib* features,const float baseRotation,
	const unsigned int idToken
)
{
	//计算每个线程需要负责的数据个数
	const unsigned threadTaskNum = FEATURE_LEN / THREAD_PER_BLOCK;
	//当前线程需要处理的初始id
	unsigned idData = threadTaskNum * threadIdx.x;
	//获得线程块的数据头
	FeatType* dataHead = getFeatHead(features,blockIdx.x);
	//遍历需要处理的每个数
	for(int i=0;i<threadTaskNum;++i)
	{
		//计算当前的角度偏移量
		FeatType angleOffset = baseRotation*((idData+i)%HEAD_DIM_T)*idToken;
		//把角度偏移量加到目标数据上
		dataHead[idData+i] += angleOffset;
	}
}

//带record switch的旋转位置编码
template<unsigned int HEAD_DIM_T,
int FEATURE_LEN,int THREAD_PER_BLOCK,
char RECORD_FLAG,char ID_QKV
>
__device__ void rotaryEmbeddingWithRecordSwitch(FeatureLib* features,const float baseRotation,
	const unsigned int idToken,
	IntermediateDecoderResult* intermediateResult //记录中间结果的结构体
)
{
	//走正常的旋转位置编码就可以
	rotaryEmbedding<HEAD_DIM_T,FEATURE_LEN,THREAD_PER_BLOCK>(
		features,baseRotation,idToken);
	//判断是否需要记录中间数据
	if constexpr (RECORD_FLAG==RECORD_ID_ALL)
	{
		//上面的旋转位置编码在结尾是没有同步的
		__syncthreads();
		//只拿一部分线程执行这个逻辑
		if(threadIdx.x < WARP_SIZE)
		{
			//根据当前线程决定的取数据的维度
			int idDims[2] = {(int)ID_QKV,(int)blockIdx.x};
			//取出对应位置的数据头，这是直接精确到feature的
			FeatType* recordHead = getFeatHeadND<3,FeatType,2>(
				&intermediateResult->qkvResult,idDims
			);
			//已经算完的编码结果的数据头
			FeatType* dataHead = getFeatHead(features,blockIdx.x);
			//调用执行数据复制
			dataCopyMultiThread<sizeof(FeatType)*FEATURE_LEN/WARP_SIZE>(
				(char*)recordHead,(const char*)dataHead
			);
		}
		__syncthreads();
	}
}

//从中间结果里面取出qkv用来记录中间结果的数据空间
//然后执行旋转
template<char RECORD_FLAG,int ID_QKV>
__device__ void cudaRotationWithQkvSwitch(const RotationMatrix* matrix,FeatureLib* features,
	const AngleTransformer* const transformer,
	IntermediateDecoderResult* intermediateResult
)
{
	if constexpr (RECORD_FLAG==RECORD_ID_ALL)
	{
		//需要取出的子集
		int idDims[1] = {ID_QKV};
		//从4D特征里面取出3D的子集
		Feature3D subTensor;
		getSubTensor<4,FeatType,3>(&(intermediateResult->qkvRotationResult),
			&subTensor,idDims
		);
		//把subTensor传进去，准备执行旋转的中间过程
		cudaMatRotationDevice<RECORD_FLAG>(matrix,features,transformer,&subTensor);
	}
	else
	{
		//不记录中间结果的情况下直接传入空指针
		cudaMatRotationDevice<RECORD_FLAG>(matrix,features,transformer,nullptr);
	}
}

//执行qkv分别旋转的函数，然后一一对应地去处理
//rotary_cycle指的是旋转位置编码的旋转周期, 它是每个头的维度的一半，其实也就是对每个头的局部旋转
//写成HEAD_DIM_T，加个T是为了跟config.hpp里的那个数区分开
template<unsigned int HEAD_DIM_T,int FEATURE_LEN,int THREAD_PER_BLOCK,
	char RECORD_FLAG //是否记录中间结果
>
__global__ void cudaQkvRotation(const RotationMatrix qMatrix,
	const RotationMatrix kMatrix, 
	const RotationMatrix vMatrix, 
	const FeatureLib valueBias,//value的bias
	FeatureLib qFeature,
	FeatureLib kFeature,
	FeatureLib vFeature,
	const AngleTransformer transformer,
	const float baseRotation, //基础旋转角，这是用来复刻旋转位置编码的
	const unsigned int idToken, //表示目前的token位置，根据token位置给出对应的旋转位置编码
	IntermediateDecoderResult intermediateResult //完整的记录了中间结果的结构体, 没有recorder_flag的话它就是空的
)
{
	//这个旋转位置编码只处理qk
	//block的三个维度分别处理qkv
	if(blockIdx.y==0)
	{
		cudaRotationWithQkvSwitch<RECORD_FLAG,ENUM_ID_QUERY>(&qMatrix,&qFeature,&transformer,&intermediateResult);
		//添加q的旋转位置编码
		rotaryEmbeddingWithRecordSwitch<HEAD_DIM_T,FEATURE_LEN,
			THREAD_PER_BLOCK,RECORD_FLAG,ENUM_ID_QUERY>(
		&qFeature,baseRotation,idToken+blockIdx.x,&intermediateResult);
	}
	else if(blockIdx.y==1)
	{
		cudaRotationWithQkvSwitch<RECORD_FLAG,ENUM_ID_KEY>(&kMatrix,&kFeature,&transformer,&intermediateResult);
		rotaryEmbeddingWithRecordSwitch<HEAD_DIM_T,FEATURE_LEN,
			THREAD_PER_BLOCK,RECORD_FLAG,ENUM_ID_KEY>(
		&kFeature,baseRotation,idToken+blockIdx.x,&intermediateResult);
	}
	else
	{
		cudaRotationWithQkvSwitch<RECORD_FLAG,ENUM_ID_VALUE>(&vMatrix,&vFeature,&transformer,&intermediateResult);
		//添加value的bias
		cudaAddBiasWithSwitch<FEATURE_LEN,THREAD_PER_BLOCK,
			RECORD_FLAG,ENUM_ID_VALUE>(
			getFeatHead(&vFeature,blockIdx.x),valueBias.data,&intermediateResult);
	}
}

//传进来的时候另外还需要角度转换器，这是贯穿全局的角度计算辅助工作
//对矩阵的局部旋转的实现 算完之后结果会直接存在features里面
template<char RECORD_FLAG,char ID_TENSOR>
void localRotation(const RotationMatrix* matrix,FeatureLib* features,const AngleTransformer* const transformer,
	IntermediateDecoderResult* intermediateResult
)
{
	cudaMatRotation<RECORD_FLAG,ID_TENSOR><<<features->featureNum,(features->featureSize>>1)>>>(
			*matrix,*features,*transformer,*intermediateResult);
}

//对qkv同时做旋转的实现，这需要一次性开三份的block,并且还需要把这些东西复制一下
//baseRotation是基础放置角，反正设置成一个比较小的角就行，到时候越大的角度叠加的放置就会越多
template<char RECORD_FLAG>
void qkvRotation(const RotationMatrix* qkvMatrix,FeatureLib* qkvFeatures,const AngleTransformer* const transformer,
	const float baseRotation,const unsigned int idToken, const FeatureLib* valueBias,
	IntermediateDecoderResult* intermediateResult //中间结果
)
{
	//需要确认特征的长度和feature的长度是一样的的
	MY_ASSERT(qkvFeatures->featureSize == FET_LENGTH);
	//这个数据如果最后再打印的话，得到的是一个全零的数据 这个地方打印的还都是正常的数据
	//但经过qkv的旋转后得到的就是全零的数据了
	//printFeatureLib<0,128>(qkvFeatures,3);
	//分别调用对qkv做乘法的核函数
	cudaQkvRotation<HEAD_DIM,FET_LENGTH,FET_LENGTH/2,RECORD_FLAG><<<dim3(qkvFeatures->featureNum,3,1),FET_LENGTH/2>>>(
		qkvMatrix[0],qkvMatrix[1],qkvMatrix[2],valueBias[0],
		qkvFeatures[0],qkvFeatures[1],qkvFeatures[2],
		*transformer,baseRotation,idToken,*intermediateResult
	);
}