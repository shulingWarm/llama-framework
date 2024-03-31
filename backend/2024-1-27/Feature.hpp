#pragma once
#include"Config.hpp"

//这是用多个char和一个float表示的向量
struct Feature
{
public:
	FeatType* data;
	float scale;
	unsigned length;
};