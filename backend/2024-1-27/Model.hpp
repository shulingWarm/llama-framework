#pragma once
#include<string>
#include"Tokenizer.hpp"
#include<vector>
#include<fstream>

//核心的模型
class Model
{
	Tokenizer tokenizer;

	Tokenizer& getTokenizer()
	{
		return tokenizer;
	}

public:

	//载入模型
	void loadModel(std::fstream& fileHandle)
	{
		//读取单词表
		tokenizer.loadVocabulary(fileHandle);
		//载入对token的编码器
	}


	std::string inference(std::string promote)
	{
		//把模型转换成token列表
		std::vector<int> tokenList;
		//调用单词表进行分词
		tokenizer.str2Token(promote,tokenList);
		//把输入的token序列转换成特征序列

	}
};