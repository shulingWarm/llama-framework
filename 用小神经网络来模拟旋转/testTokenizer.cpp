#include"Tokenizer.hpp"
#include<string>
#include<iostream>
#include<fstream>

//从文件流读取分词器
void readTokenizerFromFile()
{
	std::string vocabularyPath = "/media/zzh/data/temp/vocabulary.bin";
	std::fstream fileHandle;
	fileHandle.open(vocabularyPath,std::ios::in|std::ios::binary);
	//生成分词器
	Tokenizer tokenTool;
	tokenTool.loadVocabulary(fileHandle);
	//用于测试的输入命令
	std::string promote = "请介绍一下微软这家企业。";
	//送去做分词
	std::vector<int> tokenList;
	tokenTool.str2Token(promote,tokenList);
	//遍历打印每个token
	for(auto eachToken : tokenList)
	{
		std::cout<<eachToken<<" ";
	}
	std::cout<<std::endl;
}

//这是一个分词器的测试样例
int main()
{
	readTokenizerFromFile();
	//输入的测试命令
	// std::string promote = "三三三星西西电三国杀";
	// //分词器的实例
	// Tokenizer tokenTool;
	// //对字符串做转换
	// std::vector<int> tokenList;
	// tokenTool.str2Token(promote,tokenList);
	// //遍历打印每个token
	// for(auto eachToken : tokenList)
	// {
	// 	std::cout<<eachToken<<" ";
	// }
	// std::cout<<std::endl;
}