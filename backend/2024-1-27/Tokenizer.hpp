#pragma once
#include<string>
#include<vector>
#include<unordered_map>
#include<iostream>
#include<fstream>
#include"StreamReader.hpp"

//把字符串转换成token的工具
class Tokenizer
{
	//单词的索引表
	std::unordered_map<std::string,int> tokenMap;
public:

	//分词试探的最大距离
	const static unsigned int MAX_TRY_DIS = 64;

	//构造函数，初始化token的列表
	Tokenizer()
	{
		
	}

	//载入单词表
	void loadVocabulary(std::fstream& fileHandle)
	{
		//读取单词个数
		auto pairNum = StreamReader::read<int>(fileHandle);
		//依次读取每个pair
		for(auto idPair=0;idPair<pairNum;++idPair)
		{
			//读取字符串
			auto str = StreamReader::readStr(fileHandle);
			//读取对应的id
			auto tokenId = StreamReader::read<int>(fileHandle);
			//在map里面记录这个pair
			tokenMap[str] = tokenId;
		}
	}

	//从单词到token的转换
	int word2Token(const std::string& word)
	{
		//判断map里面是否存在
		if(tokenMap.count(word)==0)
			return -1;
		return tokenMap.at(word);
	}

	//这后面有一个优化点，直接用单词树来存储，推理的时候会判断一下后面还有没有可能找到新的点
	//如果后面有可能找到新的匹配token的话再继续向后推理，如果不可能的话就直接不推理了
	//把字符串转换成token列表
	void str2Token(const std::string& str,std::vector<int>& tokenList)
	{
		//目前已经处理到的id位置
		auto currId = 0U;
		//临时的字符串buffer
		std::string buffer = "";
		buffer.reserve(MAX_TRY_DIS);
		//遍历每个id的位置，取尽可能长的单词
		while(currId < str.size())
		{
			buffer.clear();
			//往buffer里面添加第1个单词
			buffer.push_back(str[currId]);
			//目前的试探位置
			auto tryLocal = currId + 1;
			//结束迭代的位置
			const auto tryEnd = currId + MAX_TRY_DIS > str.size() ? str.size() : currId + MAX_TRY_DIS;
			//目前能处理的最佳长度
			auto bestLength = tryLocal;
			//最佳长度对应的token
			auto bestToken = word2Token(buffer);
			//持续向前遍历
			for(;tryLocal<tryEnd;++tryLocal)
			{
				//添加新的字符
				buffer.push_back(str[tryLocal]);
				//转换出新的token
				auto tempToken = word2Token(buffer);
				//判断是不是有效的token
				if(tempToken>=0){
					//更新目前的最佳长度
					bestLength = tryLocal;
					//更新目前的最佳token
					bestToken = tempToken;
				}
			}
			//更新目前的迭代位置
			currId = bestLength;
			//判断是不是有新的token可以添加
			if(bestToken>0)
			{
				tokenList.push_back(bestToken);
			}
		}

	}
};