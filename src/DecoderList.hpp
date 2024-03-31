#pragma once
#include"AttentionDecoder.hpp"
#include"Config.hpp"
#include"FeatureLib.hpp"
#include<fstream>
#include<vector>
#include"IntermediateDecoderResult.hpp"

class DecoderList
{

	//每一层的Attention Decoder
	std::vector<AttentionDecoder> decoders;

	//准备一下qkv的中间数据 新输入进来一个feature的时候会先把它们复制三份，然后得到三份qkv的结果
	FeatureLib copyQkv[3];

public:

	//对每一层decoder的初始化
	void init(std::fstream& fileHandle)
	{
		//需要先读取attention的layer层数
		int layerNum = StreamReader::read<int>(fileHandle);
		decoders.resize(layerNum);
		//遍历每一层的decoder
		for(auto& eachLayer : decoders)
		{
			eachLayer.init(fileHandle);
		}
	}
	
	//对每一层的decoder的前向推理
	template<char RECORD_FLAG>
	void forward(FeatureLib* features,const AngleTransformer* const transformer,
		IntermediateDecoderResult* intermediateResult = nullptr
	)
	{
		int idLayer = 0;
		for(auto& eachLayer : decoders)
		{
			std::cout<<"layer: "<<idLayer<<std::endl;
			++idLayer;
			eachLayer.forward<RECORD_FLAG>(features,transformer,copyQkv,intermediateResult);
		}
	}

	//对输入操作的反向求导
	void backward(FeatureLib* historyOutput, //这是上次运行时记录下的输出结果
		Optimizer& optInstance, //优化器，里面会存储loss对output里面每个数字的求导
		IntermediateDecoderResult& intermediateResult, //用于存储decoder中间结果的结构体
		const AngleTransformer* const transformer
	)
	{
		//从后向前遍历每一层的decoder
		for(int id=decoders.size()-1;id>=0;--id)
		{
			//当前位置的decoder
			auto& currDecoder = decoders[id];
			//先临时调用对当前的decoder中间结果的记录
			currDecoder.forward<RECORD_ID_ALL>(
				&currDecoder.getHistoryInput(),transformer,
				copyQkv,
				&intermediateResult
			);
			//然后调用子层的backward
			currDecoder.backward(historyOutput,optInstance,
				intermediateResult);
		}
	}
};