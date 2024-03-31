#include<iostream>
#include<string>
#include"Model.hpp"

int main()
{
	//要输入的推理文本
	std::string promote = "杀戮尖塔是";
	//要读取的权重所在的文件
	std::string weightPath = "/media/zzh/data/temp/randWeight.bin";
	//新建模型
	Model model;
	//打开文件输入流
	std::fstream fileHandle;
	fileHandle.open(weightPath,std::ios::in|std::ios::binary);
	//载入模型
	model.loadModel(fileHandle);
	model.inference(promote);
}