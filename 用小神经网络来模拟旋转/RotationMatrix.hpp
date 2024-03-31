#pragma once
#include"Config.hpp"

//旋转矩阵，里面只有n(n-1)/2个参数
//而且这里面存的都是角度信息
struct RotationMatrix
{
public:
	//对应的数据
	FeatType* data;

	//矩阵的大小
	unsigned matSize;
};