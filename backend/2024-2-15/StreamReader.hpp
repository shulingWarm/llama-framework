#pragma once
#include<fstream>
#include<string>

//封装的各种从数据流里面读取内容的函数
//模仿java那种接口
class StreamReader
{
public:
	template<class T>
	static T read(std::istream& stream)
	{
		T temp;
		stream.read((char*)&temp,sizeof(temp));
		return temp;
	}

	//读取字符串
	static std::string readStr(std::istream& stream)
	{
		//读取一个两字节的数字，表示字符串的长度
		auto strLength = read<unsigned short int>(stream);
		//读取对应长度的字符串
		char str[512] = {'\0'};
		stream.read(str,strLength);
		return {str};
	}
};