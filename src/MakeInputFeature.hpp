#pragma once
#include"Matrix.hpp"
#include<fstream>
#include"Config.hpp"
#include"TinyCompute.hpp"
#include"FeatureLib.hpp"
#include"Optimizer.hpp"

//核函数，选出矩阵里面的对应行
//这里面对feature做完整的复制其实是有多余操作的
//那后面还是应该尽可能对数据做原地运算
//这里用传值的方式传进来，是因为里面的数据指针什么的都是浅拷贝
__global__ void selectRows(int* tokenData,FeatureLib dstFeature,FeatureLib featureLib)
{
	//每个线程负责几个数字的计算
	const unsigned int THREAD_RESPONS_NUM = dstFeature.featureSize >= blockDim.x ? divUp(dstFeature.featureSize,blockDim.x) : 1;
	//每个线程需要复制的数据头
	FeatType* rowHead = getFeatHead(&featureLib,tokenData[blockIdx.x]);
	//当前线程块要处理的数据头
	FeatType* dstRowHead = getFeatHead(&dstFeature,blockIdx.x);
	//要复制的数据的起始地址 如果超过了访问位置后面就不再复制了
	const unsigned int beginCopyId = threadIdx.x * THREAD_RESPONS_NUM >= featureLib.featureSize ? featureLib.featureSize :
		threadIdx.x * THREAD_RESPONS_NUM;
	//计算要复制的数据的长度
	const unsigned int copyLength = beginCopyId + THREAD_RESPONS_NUM >= featureLib.featureSize ? featureLib.featureSize - beginCopyId : THREAD_RESPONS_NUM;
	//执行数据的复制
	memcpy(dstRowHead + beginCopyId, rowHead + beginCopyId, sizeof(FeatType)*copyLength);
}

//用于比较时的中间参考变量
//一个是最大值对应的id,另一个是匹配的分数
typedef PairItem<int,HalfType> CompareItem;

//计算相似度分数 计算这个东西的数据单元是warp,每个warp负责计算一个这个函数
//所以这里面也不需要做同步操作
template<unsigned FEATURE_SIZE>
__device__ HalfType getSimiliarScore(const FeatType* feature1,
	const FeatType* feature2,
	const AngleTransformer* const transformer
)
{
	//每个线程需要负责的数据个数
	const unsigned TASK_NUM_PER_THREAD = FEATURE_SIZE / WARP_SIZE;
	//取自己的数据头
	const FeatType* head1 = feature1 + (threadIdx.x % WARP_SIZE)*TASK_NUM_PER_THREAD;
	const FeatType* head2 = feature2 + (threadIdx.x % WARP_SIZE)*TASK_NUM_PER_THREAD;
	//调用数据头的相加结果
	HalfType localResult = localAngleDis<TASK_NUM_PER_THREAD>(
		head1,head2,transformer
	);
	//用蝶式寻址把每个分段的数据加起来
	for(int idCross=WARP_SIZE/2;idCross>=1;idCross/=2)
	{
		localResult = __hadd(localResult,
			__shfl_xor_sync(unsigned(-1), localResult, idCross, WARP_SIZE));
	}
	//每个线程返回的都是蝶式寻址里面的结果
	return localResult;
}

//选择feature的cuda函数，针对传入的feature,直接从feature库里面选出和这个feature最相似的数据
template<unsigned THREAD_PER_BLOCK,unsigned FEATURE_SIZE>
__global__ void cudaCompareFeature(const FeatureLib queryFeature,
	const FeatureLib featureWeight,
	const AngleTransformer transformer,
	TokenId* dstResult //存储目标结果的地方
)
{
	//warp的数量
	const unsigned WARP_PER_BLOCK = THREAD_PER_BLOCK / WARP_SIZE;
	//开辟共享内存，用于存储每个线程的最终结果, 但最后每个共享内存只需要存储一份数据
	__shared__ CompareItem warpResult[WARP_PER_BLOCK];
	//每个warp处理的数据量 这里打算让每个warp同步计算各个数据的相似度
	unsigned TASK_PER_WARP = divUp(featureWeight.featureNum,WARP_PER_BLOCK);
	//当前线程所属的warp
	unsigned idWarp = threadIdx.x / WARP_SIZE;
	//当前线程在warp内的偏移量
	unsigned offsetInWarp = threadIdx.x % WARP_SIZE;
	//当前线程的目标操作单元
	CompareItem* warpItem = warpResult + idWarp;
	//当前线程访问的数据起始位置
	unsigned idBegin = idWarp * TASK_PER_WARP;
	//当前线程块需要处理的数据头
	const FeatType* queryHead = getFeatHead(&queryFeature,blockIdx.x);
	//检查实际的计算量，判断有没有超过限制
	if(idBegin >= featureWeight.featureNum)
		TASK_PER_WARP = 0;
	else if(featureWeight.featureNum - idBegin < TASK_PER_WARP)
		TASK_PER_WARP = featureWeight.featureNum - idBegin;
	//由第一个线程把相似度分数初始化成0
	if(offsetInWarp == 0)
	{
		warpItem->second = __float2half(-1000.f);
	}
	//在warp层级上遍历需要处理的每个数据
	for(int idData=0;idData<TASK_PER_WARP;++idData)
	{
		//取出当前需要处理的数据头
		const FeatType* dataHead = getFeatHead(&featureWeight,idData + idBegin);
		//调用实际的比较函数，计算当前的相似度分数
		HalfType tempScore = getSimiliarScore<FEATURE_SIZE>(
			dataHead,queryHead,&transformer);
		//判断是不是得到了更好的相似度
		if(offsetInWarp == 0 && __hgt(tempScore,warpItem->second))
		{
			warpItem->second = tempScore;
			warpItem->first = idData + idBegin;
		}
	}
	__syncthreads();
	//把所有的warpResult合并起来，得到最终的输出结果
	for(int addStep = 1;addStep<WARP_PER_BLOCK;addStep<<=1)
	{
		//判断记录当前位置和扩展位置都是有效位
		if(threadIdx.x%(addStep<<1) == 0 && threadIdx.x + addStep < WARP_PER_BLOCK)
		{
			//当前线程的访问位置
			CompareItem* currItem = &warpResult[threadIdx.x];
			//被比较的目标
			CompareItem* cmpItem = &warpResult[threadIdx.x + addStep];
			//判断目前位置是否比自己更高
			if(__hgt(cmpItem->second,currItem->second))
				currItem[0] = cmpItem[0];
		}
		__syncthreads();
	}
	//最后完全由第一个线程记录inputId的结果
	if(threadIdx.x == 0)
	{
		dstResult[blockIdx.x] = warpResult[0].first;
	}
}

//计算输入的第一个数值增大的方向对应的余弦的角度
//正常情况下，这里求的是d(cos(angle1-angle2))/d(angle1)
//但这个前提是计算两个角度的真角度差，而不是直接相减,这所谓的真角度差，也就是一个正1,负1的区别
__device__ HalfType getLossDiffOnAngle(FeatType angle1,
	FeatType angle2,const AngleTransformer* const transformer
)
{
	//计算带符号的角度差 注意这是量化到0~256的角度
	i16 diffWithSymbol = getAngleDiffWithSymbol<i16,FeatType>(angle1,angle2);
	//初步记录符号
	bool negtiveFlag = diffWithSymbol < 0;
	//取绝对值
	if(diffWithSymbol < 0)
		diffWithSymbol = -diffWithSymbol;
	//这个角度的余弦的梯度也就是这个角度的负正弦
	HalfType lossGradient = angle2Num(transformer,diffWithSymbol);
	//上面的angle2Num属于是已经取过负号了，所以它如果是正的则需要再取一次负号
	if(!negtiveFlag)
		lossGradient = -lossGradient;
	return lossGradient;
}

//targetToken是原始的输入token
//而那个检查token属于是上面最后一层decoder算出来的token
//需要注意loss是越小越好的，所以到时候算出来的梯度，它们是用来减少的
//意思是说，这个loss对各个变量的导数是用来表示loss增大的方向的，到最后更新weight的时候要减掉它
//这里打算用另一种思路来算，既然记录weight权重的时候可能有大量的同步操作,那直接就用重复计算强行不让它同步
//强行不让同步的思路其实也就是把它算两次，对一次只记录对feature的导数
//第二次则只记录对weight的导数，第二次传入的时候，主遍历的操作是以weight遍历的角度出发的
//但是这里面仍然要始终记住这是在对谁做计算，不同的位置里面，检查targetToken的方法是不一样的
//用线程块去处理的时候，其实可以每个线程的每个子段去处理所有的数据
//至于中间过程的求导数据，后面会把它弄成那种vector更新的形式，地方不够了就删了重开
template<unsigned THREAD_PER_BLOCK>
__global__ void cudaGetDiffOnFeature(const TokenId* targetToken,
	const FeatureLib features,//由最后一个decoder算出来的特征列表
	const FeatureLib featureWeight,//词表里面对应词的特征
	HalfType* diffLossOnFeature, //对输入的feature的求导 这是用来往回传播的
	const AngleTransformer transformer
)
{
	//当前线程负责的query数据,这会具体到某一个数值，到时候也只会访问这一个数值
	//不需要考虑有剩余数据的情况，它一定会是2的幂
	//这属于是一共有3个层级，feature是一层，每个feature又被分成了若干组，每一组是256个数
	//最后一个遍历层级才是具体的数字，每个线程负责一个具体的数字
	const FeatType* queryData = getFeatHead(&features,blockIdx.x) + 
	 	blockIdx.y*THREAD_PER_BLOCK + threadIdx.x;
	//自己这个数字负责的同位索引
	HalfType* targetLoss = diffLossOnFeature + blockIdx.x*features.featureNum +
	 blockIdx.y*THREAD_PER_BLOCK + threadIdx.x;
	//直接遍历每个待访问的feature就可以了
	for(int idWeight = 0;idWeight<featureWeight.featureNum;++idWeight)
	{
		//当前位置对应的weight值
		const FeatType* targetWeight = getFeatHead(&featureWeight,idWeight) + 
			blockIdx.y*THREAD_PER_BLOCK + threadIdx.x;
		//计算临时的loss
		HalfType tempLoss = getLossDiffOnAngle(targetWeight[0],queryData[0],
			&transformer
		);
		//判断当前的weight是不是目标值，如果是目标值的话，那就应该取反
		if(idWeight == targetToken[blockIdx.x])
			targetLoss[0] = __hsub(targetLoss[0],tempLoss);
		else
			targetLoss[0] = __hadd(targetLoss[0],tempLoss);
	}
}

//计算loss对词表weight的偏导
//需要注意，这里面的输入的feature和训练的期望输出是已经对齐过的，每个feature的对应位置的token
//就是它的期望输出
template<int THREAD_PER_BLOCK>
__global__ void cudaGetLossDiffOnTokenWeight(const TokenId* targetToken,//训练数据，也就是期望的输出
	const FeatureLib features,//由最后一个decoder算出来的特征列表
	FeatureLib featureWeight,//词表里面对应词的特征,注意，它是有可能被动态更新的,所以没有写成const
	WeightGradientInfoLib diffLossOnWeight, //对词库里面weight的求导,它和featureWeight一一对应
	const AngleTransformer transformer,
	const UpdateConfiguation configuration //对于更新过程的配置信息，满了之后更新多少之类的
)
{
	//weight对应的访问数据 这和上面的计算feature到token的loss是类似的
	FeatType* queryData = getFeatHead(&featureWeight,blockIdx.x) +
		blockIdx.y*THREAD_PER_BLOCK + threadIdx.x;
	//weight的更新值的同位索引
	WeightGradientInfo* targetLoss = __getFeatHead<WeightGradientInfo>(
		&diffLossOnWeight,blockIdx.x) +
		blockIdx.y*THREAD_PER_BLOCK + threadIdx.x;
	//遍历每个feature
	for(int idFeature=0;idFeature<features.featureNum;++idFeature)
	{
		//获取当前feature里面应该访问的那个数
		FeatType cmpFeature = *(getFeatHead(&features,idFeature) + 
			blockIdx.y*THREAD_PER_BLOCK + threadIdx.x);
		//如果当前feature不是目标feature，就需要把目标角度取反
		if(targetToken[idFeature] != blockIdx.x)
		{
			cmpFeature += ANGLE_MID_VALUE;
		}
		//临时计算损失
		i16 tempLoss = getAngleDiffWithSymbol<i16,FeatType>(cmpFeature,queryData[0]);
		//根据相应的数据去更新损失
		updateGradient(tempLoss,targetLoss,&configuration,queryData);
	}
	// if(blockIdx.x == 3 && blockIdx.y == 6 && threadIdx.x < 128)
	// {
	// 	printf("%d %d\n",(int)queryData[0],(int)targetLoss[0]);
	// }
}

//把输入的token列表转换成特征列表，每个token对应列表里面的一个feature
class InputFeatureMaker
{
public:

	//用于构造输入feature的权重
	//这个地方需要换成feature lib,它不是简单的矩阵，而是一个特征库
	FeatureLib featureWeight;

	//载入权重数据
	void loadWeight(std::fstream& fileHandle)
	{
		featureWeight = loadFeatureLib(fileHandle);
	}

	//把每个token的标号转换成特征的标号列表
	//最后得到这个矩阵对应的feature input
	FeatureLib makeFeatureInput(const TokenId* tokenData,unsigned tokenNum)
	{
		//把数据转换到cuda内存
		TokenId* cudaToken = (TokenId*)initFromCpuData((const char*)tokenData,tokenNum*sizeof(int));
		//初始化cuda里面转换出来的矩阵 这就是单纯对featureLib的初始化函数
		auto ansMat = initFeatureLib(tokenNum,FET_LENGTH);
		//从特征矩阵里面选出对应的行
		selectRows<<<tokenNum,256>>>(cudaToken,ansMat,featureWeight);
		//释放cuda的token信息
		handleError(cudaFree(cudaToken));
		//返回算出来的矩阵
		return ansMat;
	}

	//计算Loss对词表里面feature的导数
	//features是上层的decoder输出得到的特征列表
	//这里面会算出来两组导数，一组是loss_weight,这会直接存在属性里面
	//另一组导数是loss_feature,它会放在出参里面
	void getDiffLossOnFeature(FeatureLib* features,
		HalfType* diffLossOnInput,
		TokenId* targetToken, //这东西需要是gpu版本的target token
		const AngleTransformer* const transformer,
		Optimizer& optInstance
	)
	{
		const int THREAD_PER_BLOCK = 256;
		//传入weight和features,提取里面的loss信息
		cudaGetDiffOnFeature<THREAD_PER_BLOCK><<<
			dim3(features->featureNum,features->featureSize/THREAD_PER_BLOCK,1),
			THREAD_PER_BLOCK>>>(
			targetToken,*features,
			this->featureWeight,diffLossOnInput,*transformer);
		//计算对weight值的更新
		cudaGetLossDiffOnTokenWeight<THREAD_PER_BLOCK><<<
			dim3(featureWeight.featureNum,featureWeight.featureSize/THREAD_PER_BLOCK,1),
			THREAD_PER_BLOCK>>>(targetToken,features[0],featureWeight,
			optInstance.getWgi((const char*)&featureWeight),
			*transformer,optInstance.updateConfiguration
		);
	}

	//输出特征的反选
	//最后返回出来的是token的列表
	//这个纯属就是用来输出看效果的,到时候如果需要训练，再用那个期望的loss去反推
	template<char CPU_FLAG>
	void compareFeature(FeatureLib* features,TokenId* dstResult,
		const AngleTransformer* const transformer
	)
	{
		//用于存储cuda结果
		TokenId* cudaResult;
		if(CPU_FLAG)
			cudaResult = (TokenId*)dataAllocate(features->featureNum*sizeof(TokenId));
		else
			cudaResult = dstResult;
		//直接启动feature的结果 这里有问题，实际上feature的大小还是那么多，这每个位置的output都属于是下一个位置的
		//所以实际上不存在什么featureNum + 1,一共就featureNum那么多个数据，其实每个位置都属于是对下一个位置的预测
		cudaCompareFeature<512,FET_LENGTH><<<features->featureNum,512>>>(
			features[0],featureWeight,*transformer,cudaResult
		);
		if(CPU_FLAG)
		{
			//把数据复制回cpu 这大概是报错出现的位置
			handleError(cudaMemcpy(dstResult,
				cudaResult,sizeof(TokenId)*(features->featureNum),
				cudaMemcpyDeviceToHost
			));
			//释放cuda内存
			releaseCudaMemory((char*)cudaResult);
		}
	}
};