#include<iostream>
#include<string>
#include"Model.hpp"

int main()
{
	//要输入的推理文本
	std::string promote = "杀戮尖塔是";
	//新建模型
	Model model;
	model.inference(promote);
}