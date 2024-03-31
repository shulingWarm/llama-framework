#pragma once
#include"FeatureND.hpp"

//记录中间的旋转结果，LocalRotationImpl.hpp会依赖这个头文件
template<int TASK_PER_THREAD>
__device__ void recordMidRotationResult(const FeatType* srcData, //当前位置位置的旋转周期
	int idIter, //当前位置的迭代周期
	Feature3D* dstRecord
)
{
	//要访问的id
	int idDims[2] = {
		idIter, //表示当前是第几个迭代周期
		(int)blockIdx.x //表示处理第几个token的数据
	};
	//获取当前位置的数据头
	FeatType* dstHead = getFeatHeadND<3,FeatType,2>(dstRecord,idDims);
	//存储目标位置的数据头
	const int idThreadBegin = TASK_PER_THREAD*threadIdx.x;
	memcpy(dstHead+idThreadBegin,srcData+idThreadBegin,sizeof(FeatType)*TASK_PER_THREAD);
	//查看最后一个旋转周期的效果
	// if(blockIdx.x == 0 && blockIdx.y==0 && threadIdx.x < 64 && idIter == 1024)
	// {
	// 	printf("%d %d: %d %d\n",threadIdx.x,idIter,(int)dstHead[idThreadBegin],
	// 		(int)dstHead[idThreadBegin+1]);
	// }
}