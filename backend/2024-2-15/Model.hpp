#pragma once
#include<string>
#include"Tokenizer.hpp"
#include<vector>
#include<fstream>
#include"MakeInputFeature.hpp"
#include"AngleTransformer.hpp"
#include"DecoderList.hpp"
#include<iostream>

//核心的模型
class Model
{
	Tokenizer tokenizer;
	//生成输入特征用到的工具
	InputFeatureMaker featureMaker;

	Tokenizer& getTokenizer()
	{
		return tokenizer;
	}

	//初始化角度转换器，全局都会用得到
	AngleTransformer angleTrnasformer;

	//各层的attention
	DecoderList decoders;

public:

	//载入模型
	void loadModel(std::fstream& fileHandle)
	{
#ifdef PRINT_INTERNAL
		std::cout<<"prepare load vocabulary"<<std::endl;
#endif
		//读取单词表
		tokenizer.loadVocabulary(fileHandle);
#ifdef PRINT_INTERNAL
		std::cout<<"loadWeight"<<std::endl;
#endif
		//载入对token的编码器
		featureMaker.loadWeight(fileHandle);
#ifdef PRINT_INTERNAL
		std::cout<<"decoders.init"<<std::endl;
#endif
		//载入attention里面的decoder的权重
		decoders.init(fileHandle);
		//初始化角度转换器
		angleTrnasformer.init(TYPE_SAMPLE_NUM);
		std::cout<<"angle finish out"<<std::endl;
	}


	std::string inference(std::string promote)
	{
#ifdef PRINT_INTERNAL
		std::cout<<"begin inference"<<std::endl;
#endif
		//把模型转换成token列表
		std::vector<int> tokenList;
		//调用单词表进行分词
		tokenizer.str2Token(promote,tokenList);
#ifdef PRINT_INTERNAL
		cudaDeviceSynchronize();
		std::cout<<"finish str2Token"<<std::endl;
#endif
		//把输入的token序列转换成特征序列
		auto inputFeature = featureMaker.makeFeatureInput(tokenList.data(),tokenList.size());
#ifdef PRINT_INTERNAL
		cudaDeviceSynchronize();
		std::cout<<"finish makeFeatureInput"<<std::endl;
#endif
		//下一步要进入到attention层的计算了
		decoders.forward(&inputFeature,&angleTrnasformer);
		return "";
	}
};