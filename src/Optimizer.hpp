#pragma once
#include<unordered_map>
#include"Config.hpp"
#include"FeatureLib.hpp"
#include "IntermediateGradient.hpp"

//参数更新的配置，导数乘的幅值和累积导数达到时的更新量
struct UpdateConfiguation
{
	//这个参数到时候是需要手动调整的
	float gradientScalar;
	//梯度累积满后的更新量
	FeatType updateSize;
};

//保存weight梯度信息的中间值
//做这个东西是为了如果后面需要转回adam算法的话，可以比较方便地在这个地方调整
typedef FeatType WeightGradientInfo;

//具体的梯度类型，WeightGradientInfo以后可能变成一个结构体啥的
//但这个梯度类型以后估计都是这个类型
typedef FeatType WeightGradient;

//更新权重时会使用的weight信息，它也是每个tensor有一个对应的信息
typedef __FeatureLib<WeightGradientInfo> WeightGradientInfoLib;

//初始化参数更新量
static void initUpdateConfiguration(UpdateConfiguation& updateConfiguration)
{
	//从第一次测试来看，它整体的梯度并不大,先乘个2倍试试吧
	updateConfiguration.gradientScalar = 0.1f;
	updateConfiguration.updateSize = 1;
}

//提供训练过程中相关的中间变量，但并不负责直接更新权重
class Optimizer
{
public:
	//中间层会用到的信息，它属于是梯度反向传播的载体
	IntermediateGradient midGradient;
	//参数更新的配置
	UpdateConfiguation updateConfiguration;
	//各种weight的梯度信息的字典 wgi就是Weight Gradient Info
	std::unordered_map<const char*,WeightGradientInfoLib> wgiMap;

	//初始化优化器，主要是准备中间激活层会用到的feature
	void initOptimizer(int tokenNum,int featureSize)
	{
		midGradient.init(tokenNum,featureSize);
		//对参数更新量的初始化
		initUpdateConfiguration(this->updateConfiguration);
	}

	//输入权重变量对应的指针，输出专门负责维护这块内存的更新梯度管理器
	WeightGradientInfoLib& getWgi(const char* weightPtr)
	{
		return wgiMap.at(weightPtr);
	}

	//向从weight到梯度信息的映射表里添加新的梯度值
	void addWgi(const FeatureLib* features)
	{
		MY_ASSERT(wgiMap.count((const char*)features) == 0);
		auto& newInfoLib = wgiMap[(const char*)features];
		//初始化这个新的权重的信息内容的空间
		initFeatureLibShape<WeightGradientInfo>(newInfoLib,
			features->featureNum,features->featureSize
		);
	}

};

//根据临时算出来的梯度数据去更新损失
//把新算出来的梯度叠加进去， 如果发现它已经超过限制了就对目标数据执行更新
__device__ void updateGradient(const i16 newGradient,//新算出来的loss对这个数的偏导
	WeightGradientInfo* info,//属于这个数的梯度信息
	const UpdateConfiguation* const configuration, //对于更新过程的配置信息，满了之后更新多少之类的
	FeatType* linkingWeight //这是这个所谓的权重信息真正在处理的目标数据
)
{
	//叠加累积之后的loss
	int accumulateLoss = newGradient * configuration->gradientScalar + info[0];
	//判断是否突破了下界
	if(accumulateLoss < ACCUMULATE_LOW_RANGE)
	{
		info[0] = ACCUMULATE_INIT;
		//执行更新
		linkingWeight[0] -= configuration->updateSize;
	}
	else if(accumulateLoss > ACCUMULATE_UP_RANGE)
	{
		info[0] = ACCUMULATE_INIT;
		//执行加法上的更新
		linkingWeight[0] += configuration->updateSize;
	}
	else
	{
		info[0] = accumulateLoss;
	}
}

//从info结构体里面获取梯度信息
__device__ WeightGradient* getGradientFromInfo(WeightGradientInfo* info)
{
	return info;
}