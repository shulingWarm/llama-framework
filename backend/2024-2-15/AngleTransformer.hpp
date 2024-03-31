#pragma once
#include "Config.hpp"
#include<cmath>
#include"CudaAllocate.hpp"
#include<iostream>

//把一个float角度转换成angle的角度，其实都是角度，只不过把它分配到了256的范围内
FeatType floatAngle2Feat(float data, int sampleNum)
{
	return std::round((data/(2*PI) + 1) * sampleNum);
}

//初始化每个三角函数的sin值
//采样数指的是把0~2pi之间分成多少个采样点
void initSinValueChart(HalfType* dstData,int sampleNum)
{
	//计算步长
	float stepLength = 2*PI/sampleNum;
	//目前的数值
	float currAngle = 0;
	//遍历每一步
	for(int idStep=0;idStep<sampleNum;++idStep)
	{
		//计算正弦值
		dstData[idStep] = __float2half(std::sin(currAngle));
		//步长叠加
		currAngle += stepLength;
	}
}

//初始化cpu端的两个sin的相乘结果
void initSinSinAns(FeatType* dstData,int sampleNum)
{
	//计算步长
	float stepLength = 2*PI/sampleNum;
	// int opCount=0;
	// std::cout<<"initSinSinAns print"<<std::endl;
	//双重循环去处理每一步
	for(int iStep=0; iStep < sampleNum; ++iStep)
	{
		//最外层的sin值
		float currSin = std::sin(iStep*stepLength);
		for(int kStep=0;kStep<sampleNum;++kStep)
		{
			//计算反正弦的结果
			float asinAns = std::asin(currSin * std::sin(kStep*stepLength));
			//把它转换成float的数据类型
			dstData[iStep*sampleNum + kStep] = (FeatType)std::round((asinAns/(2*PI) + 1) * sampleNum);
		}
	}
}

//初始化cpu端的两个正弦相加的结果
void initSinSinAddAns(FeatType* dstData,const int sampleNum)
{
	//计算步长
	const float stepLength = 2*PI/sampleNum;
	//用双重循环去处理每一步
	for(int iStep=0; iStep < sampleNum; ++iStep)
	{
		//最外层的sin值
		float currSin = std::sin(iStep*stepLength);
		for(int kStep=0;kStep<sampleNum;++kStep)
		{
			//计算相加的结果
			const float addAns = currSin + std::sin(kStep*stepLength);
			//目标数据
			auto& dstAns = dstData[iStep*sampleNum + kStep];
			//判断相加结果是否满足要求
			if(addAns>1)
				dstAns = 64; //1的情况
			else if(addAns<-1)
				dstAns = 192;//-1的情况
			else
				dstAns = (FeatType)std::round((std::asin(addAns)/(2*PI) + 1) * sampleNum);
		}
	}
}

//初始化从数值反推角度的查表
//这个地方指的是从half到angle分成了sampleNum个量化阶，从angle到half分成了valueSampleNum个量化阶
void initNumToAngleChart(FeatType* dstData,const int sampleNum,const int valueSampleNum)
{
	//计算步长
	const float stepLength = 2.f/sampleNum;
	int idData = 0;
	//遍历从-1到1的每个数
	for(float currNum = -1;currNum<=1,idData<sampleNum;currNum+=stepLength,idData++)
	{
		//计算当前位置的反正弦
		dstData[idData] = (FeatType)std::round((std::asin(currNum)/(2*PI) + 1) * valueSampleNum);
	}
}

//判断一个float函数是否接近零
bool judgeNumNearZero(float data)
{
	if(data==0 || (data>0 && data<1e-6) || (data<0 && data>-1e-6))
		return true;
	return false;
}

//计算两个float的比值的反正切
float getASinFloat(float upNum,float downNum)
{
	//如果分子是0那就是0
	if(judgeNumNearZero(upNum))
		return 0;
	if(judgeNumNearZero(downNum))
	{
		if(upNum>0)
			return PI/2;
		return PI*3/2;
	}
	return std::atan(upNum/downNum);
}

//初始化反正切的查找表，每个位置的正弦的分子比分母然后反推到90度
//这里是计算当比值的两个角度确定之后，它对应的正切值
void initAtanAngleChart(FeatType* dstData,const int sampleNum)
{
	//计算角度的步长
	const float stepLength = 2*PI/sampleNum;
	//遍历第一维的角度
	for(int iStep=0; iStep < sampleNum; ++iStep)
	{
		//最外层的sin值
		float currSin = std::sin(iStep*stepLength);
		//目标数据头
		FeatType* rowHead = dstData + iStep*sampleNum;
		for(int kStep=0;kStep<sampleNum;++kStep)
		{
			//第二个角度的正弦
			float sin2 = std::sin(kStep*stepLength);
			//记录反正切的结果
			rowHead[kStep] = floatAngle2Feat(getASinFloat(currSin,sin2),sampleNum);
		}
	}
}

//角度转换器，把中间存储的特征数据类型做各种情况的转换
struct AngleTransformer
{
public:
	//从角度到数值的float转换
	HalfType* angleToNum;
	//从数值到角度的转换
	FeatType* numToAngle;
	//正弦和正弦相乘结果的等效正弦值
	FeatType* sinsinMulAns;
	//正弦和正弦相加结果的等效正弦值
	FeatType* sinsinAddAns;
	//反正切的查表集 输入的是分子分母的两个sin值，然后获得对应的查表
	FeatType* atanAns;

	//初始化角度的transformer
	void init(int sampleNum)
	{
		//初始化float形式的角度到数值的转换表
		HalfType cpuChart[ANGLE_CHART_SIZE];
		//初始化正弦相乘结果对应的反正弦 这个地方要开65535的大小，如果开在栈上会栈溢出
		FeatType* cpuSinSinAns = (FeatType*)malloc(sizeof(FeatType)*ANGLE_CHART_SIZE*ANGLE_CHART_SIZE);
		//在这里面更新sin的查表数据
		initSinValueChart(cpuChart,ANGLE_CHART_SIZE);
		//把数据转换到gpu上
		angleToNum = (HalfType*)initFromCpuData((char*)cpuChart,sizeof(HalfType)*ANGLE_CHART_SIZE);
		initSinSinAns(cpuSinSinAns,sampleNum);
		//把数据转换到gpu上
		sinsinMulAns = (FeatType*)initFromCpuData((char*)cpuSinSinAns,sizeof(FeatType)*ANGLE_CHART_SIZE*ANGLE_CHART_SIZE);
		//初始化正弦之间相加结果的查表
		initSinSinAddAns(cpuSinSinAns,sampleNum);
		sinsinAddAns = (FeatType*)initFromCpuData((char*)cpuSinSinAns,sizeof(FeatType)*ANGLE_CHART_SIZE*ANGLE_CHART_SIZE);
		//初始化从数值反推角度的查表
		initNumToAngleChart(cpuSinSinAns,NUM_ANGLE_CHART_SIZE,ANGLE_CHART_SIZE);
		numToAngle = (FeatType*)initFromCpuData((char*)cpuSinSinAns,sizeof(FeatType)*NUM_ANGLE_CHART_SIZE);
		//初始化反正切的查找表
		initAtanAngleChart(cpuSinSinAns,ANGLE_CHART_SIZE);
		atanAns = (FeatType*)initFromCpuData((char*)cpuSinSinAns,sizeof(FeatType)*ANGLE_CHART_SIZE*ANGLE_CHART_SIZE);
		//在结束的时候释放内存
		free(cpuSinSinAns);
	}

};

//把输入的数据换成对应的sin值
__device__ HalfType angle2Num(const AngleTransformer* const transformer, const FeatType angle)
{
	return transformer->angleToNum[angle];
}

//把数字反转换成角度
template<class T>
__device__ FeatType num2Angle(const AngleTransformer* const transformer, const T data)
{
	return 0;
}

//float情况下的特化
template<>
__device__ FeatType num2Angle(const AngleTransformer* const transformer, const float data)
{
	//正常的查表操作
	int idChart = (data + 1)/2.f * NUM_ANGLE_CHART_SIZE;
	//判断如果数据大于1就返回90度
	if(idChart >= NUM_ANGLE_CHART_SIZE)
		return 64;
	if(idChart < 0)
		return 192;
	//访问目标位置
	return transformer->numToAngle[idChart];
}

//传入half数据版本的特化
template<>
__device__ FeatType num2Angle(const AngleTransformer* const transformer, const HalfType data)
{
	return num2Angle<float>(transformer,__half2float(data));
}

//把一个正弦的角度转换成余弦的角度
__device__ FeatType cosAngle2SinAngle(FeatType angle)
{
	return (320U - angle);
}

//正弦余弦的角度相乘，然后直接把相乘的结果的反正弦
//先传正弦再传余弦
__device__ FeatType sincosMul(const AngleTransformer* const transformer,FeatType angle1,FeatType angle2)
{
	return transformer->sinsinMulAns[(unsigned)angle1*ANGLE_CHART_SIZE + cosAngle2SinAngle(angle2)];
}

//正弦和正弦相乘的结果，也是通过查表实现的
__device__ FeatType sinsinMul(const AngleTransformer* const transformer,FeatType angle1,FeatType angle2)
{
	return transformer->sinsinMulAns[(unsigned)angle1*ANGLE_CHART_SIZE + angle2];
}

//两个正弦值相加的结果
__device__ FeatType sinsinAdd(const AngleTransformer* const transformer,FeatType angle1,FeatType angle2)
{
	return transformer->sinsinAddAns[(unsigned)angle1*ANGLE_CHART_SIZE + angle2];
}

//输入两个正弦的角度，求反正切
__device__ FeatType sinsinAtan(const AngleTransformer* const transformer,FeatType angle1,FeatType angle2)
{
	return transformer->atanAns[(unsigned)angle1*ANGLE_CHART_SIZE + angle2];
}