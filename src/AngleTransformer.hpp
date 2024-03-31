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

//判断一个角度是不是在左半轴
//首先是默认角度在0~2PI的
bool judgeAngleAtLeft(const float angle)
{
	//判断角度的范围
	return angle > PI/2 && angle < 3*PI/2;
}

//2024-2-22
//初始化角度正弦再反转回角度
//写这个东西是为了服务于激活函数的
void initAngleToSinAsAngle(FeatType* dstData,const unsigned sampleNum)
{
	//每一步的位置对应的角度
	const float stepLength = 2*PI/sampleNum;
	//遍历角度数据的每一位
	for(int i=0;i<sampleNum;++i)
	{
		//当前位置的角度
		const float currentAngle = stepLength * i;
		//当前位置的正弦 但把它映射到0 ~ 1
		const float sinValue = (std::sin(currentAngle) + 1)/2;
		//判断一个角度是不是在左半轴
		bool angleAtLeft = judgeAngleAtLeft(currentAngle);
		//计算这个角度在128条件下的分度
		FeatType angleOffset = std::round(sinValue * ANGLE_MID_VALUE);
		//做角度上的添加
		FeatType finalAngle = angleOffset + ANGLE_QUANT_270;
		//如果是在左半区间的话，还需要把这个角度换到左边去
		if(angleAtLeft)
		{
			finalAngle = ANGLE_MID_VALUE - finalAngle;
		}
		//把计算完的角度放到目标位置上
		dstData[i] = finalAngle;
	}
}

//计算256个位置每个位置对应的导数
//其实就是求这个角度的正弦，这纯属先试着写一下
//这个代码仅仅就是个示范，并不会真的去调用
void initCosDiffGradient(i16* gradientList,//最后算出来的梯度结果
	int sampleNum
)
{
	//每个角度的步长
	const float stepLength = 2*PI/sampleNum;
	//遍历角度的每一位
	for(int i=0;i<sampleNum;++i)
	{
		float currAngle = stepLength * i;
		//如果角度超过了一半，就给它弄到负数去
		if(currAngle > PI)
		{
			currAngle -= 2*PI;
		}
		//计算角度对loss的导数
		float lossDiffOnAngle = -std::sin(currAngle);
		//如果是负角度，就乘个负数，这算是角度对角度差的导数
		if(currAngle < 0)
		{
			lossDiffOnAngle = -lossDiffOnAngle;
		}
		//但真正对应到那个数值，angle = num/128*PI
		//现在先不处理这个num/128*PI，这东西本来就是可以通过学习率来调整的
		gradientList[i] = lossDiffOnAngle;
	}
}

//对于单个角度，处理y对x的导数
HalfType getActivateGradient(FeatType inputAngle)
{
	float finalGradient = 0;
	//判断是不是90~270度
	if(inputAngle >= ANGLE_QUANT_90 && inputAngle<ANGLE_QUANT_270)
	{
		// y = 128 - sin(x/256*2*PI)*64
		finalGradient = - 64*std::cos(
			(float)inputAngle/(float)ANGLE_CHART_SIZE)/ANGLE_CHART_SIZE*2*PI;
	}
	else //-90 ~ 90度
	{
		// y = sin(x/256*2*PI) * 64
		finalGradient = 64 * std::cos(
			(float)inputAngle/(float)ANGLE_CHART_SIZE)/ANGLE_CHART_SIZE*2*PI;
	}
	return __float2half(finalGradient);
}

//初始化正弦激活的求导，就算这个函数最后不用，当作一个数学上的表达式也是可以的
void initActivateGradient(HalfType* dstData)
{
	//遍历可能输入的每个数字
	for(int i=0;i<TYPE_SAMPLE_NUM;++i)
	{

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
	//角度到正弦再反映射回角度的active
	//这是把这个东西当作激活函数在使用
	FeatType* sinActivate;

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
		//初始化从角度到正弦再反映射回角度
		//同时需要注意映射的时候把反转回来的角度再换到原来的象限
		initAngleToSinAsAngle(cpuSinSinAns,ANGLE_CHART_SIZE);
		sinActivate = (FeatType*)initFromCpuData((char*)cpuSinSinAns,sizeof(FeatType)*ANGLE_CHART_SIZE);
		//在结束的时候释放内存
		free(cpuSinSinAns);
	}

};

//把一个正弦的角度转换成余弦的角度
__device__ FeatType cosAngle2SinAngle(FeatType angle)
{
	return (320U - angle);
}

//把输入的数据换成对应的sin值
__device__ HalfType angle2Num(const AngleTransformer* const transformer, const FeatType angle)
{
	return transformer->angleToNum[angle];
}

//取一个角度的余弦值
__device__ HalfType angle2CosNum(const AngleTransformer* const transformer, const FeatType angle)
{
	return transformer->angleToNum[cosAngle2SinAngle(angle)];
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

//执行对一个角度的scalar操作
__device__ FeatType rotateAngleByScalar(const AngleTransformer* const transformer,
	FeatType angle, FeatType rotateAngle
)
{
	//把旋转角转换成正弦值
	float sinValue = __half2float(angle2Num(transformer,rotateAngle));
	//把角度乘上1+sin
	return (1.f + sinValue) * angle;
}

//计算两个角度的差值，返回出来的是一个0~128的数字
//这里返回的是一个绝对值的差
__device__ FeatType getAngleDiff(const FeatType angle1,const FeatType angle2)
{
	const FeatType angleDiff = angle1 - angle2;
	//如果角度的差超过一半的话，就从另一边取
	if(angleDiff > ANGLE_MID_VALUE)
	{
		return angle2 - angle1;
	}
	return angleDiff;
}


//计算两个角度的中间值
__device__ FeatType getMidAngle(const FeatType angle1,const FeatType angle2)
{
	//求两个点的临时中值
	const FeatType tempMid = ((int)angle1 + (int)angle2)>>1;
	//对位角度
	const FeatType oppositeAngle = tempMid + ANGLE_MID_VALUE;
	//查看哪个中间值的角度更小
	if(getAngleDiff(tempMid,angle1) > getAngleDiff(oppositeAngle,angle1))
	{
		return oppositeAngle;
	}
	return tempMid;
}