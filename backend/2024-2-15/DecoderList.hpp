#pragma once
#include"AttentionDecoder.hpp"
#include"Config.hpp"
#include"FeatureLib.hpp"
#include<fstream>
#include<vector>

class DecoderList
{

	//每一层的Attention Decoder
	std::vector<AttentionDecoder> decoders;

public:

	//对每一层decoder的初始化
	void init(std::fstream& fileHandle)
	{
		//需要先读取attention的layer层数
		int layerNum = StreamReader::read<int>(fileHandle);
		//临时给它改成1，debug用的
		layerNum = 1;
		decoders.resize(1);
		//遍历每一层的decoder
		for(auto& eachLayer : decoders)
		{
			eachLayer.init(fileHandle);
		}
	}
	
	//对每一层的decoder的前向推理
	void forward(FeatureLib* features,const AngleTransformer* const transformer)
	{
		for(auto& eachLayer : decoders)
		{
			eachLayer.forward(features,transformer);
		}
	}
};