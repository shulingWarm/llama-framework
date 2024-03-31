#pragma once
#include<string>
#include"Tokenizer.hpp"
#include<vector>
#include<fstream>
#include"MakeInputFeature.hpp"
#include"AngleTransformer.hpp"

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

public:

	//载入模型
	void loadModel(std::fstream& fileHandle)
	{
		//读取单词表
		tokenizer.loadVocabulary(fileHandle);
		//载入对token的编码器
		featureMaker.loadWeight(fileHandle);
		//初始化角度转换器
		angleTrnasformer.init();
	}


	std::string inference(std::string promote)
	{
		//把模型转换成token列表
		std::vector<int> tokenList;
		//调用单词表进行分词
		tokenizer.str2Token(promote,tokenList);
		//把输入的token序列转换成特征序列
		auto inputFeature = featureMaker.makeFeatureInput(tokenList.data(),tokenList.size());
		return "";
	}
};