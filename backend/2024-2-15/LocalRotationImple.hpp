#pragma once
#include "RotationMatrix.hpp"
#include "FeatureLib.hpp"

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
	const bool upperFlag = threadIdx.x < (offsetInBlock>>1);
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

//对两个输入的数据执行对应角度的旋转
__device__ void rotateOfTwoDim(const AngleTransformer* const transformer,
	FeatType* data1,FeatType* data2,const FeatType roteAngle)
{
	//临时的结果
	const FeatType tempAns[2] = {
		sinsinAdd(transformer,
			sincosMul(transformer,*data1,roteAngle),
			128U + sinsinMul(transformer,*data2,roteAngle)
		),
		sinsinAdd(transformer,
			sinsinMul(transformer,*data1,roteAngle),
			sincosMul(transformer,*data2,roteAngle)
		)
	};
	// if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==20)
	// {
	// 	printf("rowDimRotate: %d %d %d %d %d\n",(int)tempAns[0],(int)tempAns[1],(int)data1[0],(int)data2[0],(int)roteAngle);
	// 	//打印第一个角度的两个中间结果
	// 	printf("mul mid result: %d %d\n",(int)sincosMul(transformer,*data1,roteAngle),(int)sinsinMul(transformer,*data2,roteAngle));
	// }
	//把结果保存到输入的角度里
	*data1 = tempAns[0];
	*data2 = tempAns[1];
}

//对特征进行乘法操作的kernel
//每个线程块的大小都是1024,这个计算密集程度只能这样处理
//目前这个算法最大也就能处理2048的长度，如果以后要扩展更大的长度，需要实现另外的算法
//这里是完全不考虑通用性的，线程数就是feature的size的一半
//baseRotation是附加旋转角，其实是模仿的旋转位置编码，为了不过分影响原本的向量，只在最后一次旋转的时候才叠加这个旋转矩阵
__device__ void cudaMatRotationDevice(const RotationMatrix* matrix,FeatureLib* features,
	const AngleTransformer* const transformer, const float baseRotation
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
			//两个数据形成的角度比值对应的正切
			const FeatType angleAtan = sinsinAtan(transformer,blockData[iterInfo.iterDim[0]],
				blockData[iterInfo.iterDim[1]]);
			//实际参与运算的旋转角
			const FeatType roteAngle = iterInfo.blockSize != 1 ? 
				matrix->data[idIter*blockDim.x + threadIdx.x]*angleAtan : 
				matrix->data[idIter*blockDim.x + threadIdx.x]*angleAtan + baseRotation;
			//持续追踪0号数据的旋转情况
			if(blockIdx.y==0 && blockIdx.x==3)
				if(iterInfo.iterDim[0]==0 || iterInfo.iterDim[1]==0)
				{
					//打印参与旋转的两个线程
					printf("two dim rote: %d %d %d %d %d\n",(int)blockData[iterInfo.iterDim[0]],
						(int)blockData[iterInfo.iterDim[1]],(int)roteAngle,
						(int)iterInfo.iterDim[0],(int)iterInfo.iterDim[1]);
				}
			//执行对应维度的旋转
			rotateOfTwoDim(transformer,
				&blockData[iterInfo.iterDim[0]],
				&blockData[iterInfo.iterDim[1]],
				roteAngle
			);
			//更新迭代次数
			++idIter;
		}
		//迭代完一轮了，更新下一个区域的block
		getNextBlock(&iterInfo);
	}
}

//执行旋转的核函数，注意这个地方需要传值，不能传指针
__global__ void cudaMatRotation(const RotationMatrix matrix,FeatureLib features,const AngleTransformer transformer)
{
	//直接调用device函数即可 这个单独的测试函数不需要基础附加旋转角
	cudaMatRotationDevice(&matrix,&features,&transformer,0);
}

//执行qkv分别旋转的函数，然后一一对应地去处理
//rotary_cycle指的是旋转位置编码的旋转周期, 它是每个头的维度的一半，其实也就是对每个头的局部旋转
template<unsigned int ROTARY_CYCLE>
__global__ void cudaQkvRotation(const RotationMatrix qMatrix,
	const RotationMatrix kMatrix, 
	const RotationMatrix vMatrix, 
	FeatureLib qFeature,
	FeatureLib kFeature,
	FeatureLib vFeature,
	const AngleTransformer transformer,
	const float baseRotation, //基础旋转角，这是用来复刻旋转位置编码的
	const unsigned int idToken //表示目前的token位置，根据token位置给出对应的旋转位置编码
)
{
	//这个旋转位置编码只处理qk
	//block的三个维度分别处理qkv
	if(blockIdx.y==0)
		cudaMatRotationDevice(&qMatrix,&qFeature,&transformer,
			(idToken+blockIdx.x)*baseRotation*(threadIdx.x%ROTARY_CYCLE));
	else if(blockIdx.y==1)
		cudaMatRotationDevice(&kMatrix,&kFeature,&transformer,
			(idToken+blockIdx.x)*baseRotation*(threadIdx.x%ROTARY_CYCLE));
	else
		cudaMatRotationDevice(&vMatrix,&vFeature,&transformer,0);
}

//传进来的时候另外还需要角度转换器，这是贯穿全局的角度计算辅助工作
//对矩阵的局部旋转的实现 算完之后结果会直接存在features里面
void localRotation(const RotationMatrix* matrix,FeatureLib* features,const AngleTransformer* const transformer)
{
	//直接调用cuda kernel执行对feature的旋转操作
	cudaMatRotation<<<features->featureNum,(features->featureSize>>1)>>>(*matrix,*features,*transformer);
}

//对qkv同时做旋转的实现，这需要一次性开三份的block,并且还需要把这些东西复制一下
//baseRotation是基础放置角，反正设置成一个比较小的角就行，到时候越大的角度叠加的放置就会越多
void qkvRotation(const RotationMatrix* qkvMatrix,FeatureLib* qkvFeatures,const AngleTransformer* const transformer,
	const float baseRotation,const unsigned int idToken
)
{
	//这个数据如果最后再打印的话，得到的是一个全零的数据 这个地方打印的还都是正常的数据
	//但经过qkv的旋转后得到的就是全零的数据了
	//printFeatureLib<0,2048>(qkvFeatures,3);
	//分别调用对qkv做乘法的核函数
	cudaQkvRotation<(HEAD_DIM>>1)><<<dim3(qkvFeatures->featureNum,3,1),(qkvFeatures->featureSize >> 1)>>>(
		qkvMatrix[0],qkvMatrix[1],qkvMatrix[2],
		qkvFeatures[0],qkvFeatures[1],qkvFeatures[2],*transformer,baseRotation,idToken);
	cudaDeviceSynchronize();
	throw -1;
}