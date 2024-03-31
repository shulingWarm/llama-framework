#pragma once
#include<string>
#include"Tokenizer.hpp"
#include<vector>
#include<fstream>
#include"MakeInputFeature.hpp"
#include"AngleTransformer.hpp"
#include"DecoderList.hpp"
#include<iostream>
#include"Optimizer.hpp"
#include"IntermediateDecoderResult.hpp"

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

	//优化过程中会用到的相关变量，提前开辟相关的空间
	//这东西属于是给训练的时候用的 到后面如果想把代码弄简洁点，可以弄个子类，把这个东西放子类里面
	Optimizer optInstance;

	//用于存储中间结果, 其实是训练的时候会用到，但为了接口的一致性，总归是要传入一下这个东西
	IntermediateDecoderResult intermediateResult;

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

	//初始化优化器，调用之前需要确保已经载入过模型了
	//把已经载入过的weight链接到optimizer里面的一个权重记录器里面
	void linkWeightToOptimizer()
	{
		//现在只链接一个词库编码信息，decoder里面的那些权重等实现它们的反向推理的时候再说
		optInstance.addWgi(&featureMaker.featureWeight);
	}

	//训练模型的主逻辑，和推理基本上是一样的，只不过到时候需要调用一个反向求导去更新功能
	void trainInference(std::string promote)
	{
		//先把模型转换成token列表
		std::vector<TokenId> tokenList;
		//调用单词表做分词
		tokenizer.str2Token(promote,tokenList);
		//打印转换出来的token列表
		for(auto eachToken : tokenList)
		{
			std::cout<<eachToken<<" ";
		}
		std::cout<<std::endl;
		std::cout<<tokenizer.token2Str(tokenList.data(),tokenList.size())<<std::endl;

		//确定分词的token数后，就可以开始准备优化器里面的激活相关的变量了
		//正常训练的时候，这句代码不会被写在这里
		//这里减1是因为没必要输入最后一个token,最后一个token是对下一个位置的预测
		//没人知道正确答案
		optInstance.initOptimizer(tokenList.size()-1,FET_LENGTH);

		//把输入的token序列转换成特征序列 这个后面还需要注意管理内存泄漏的问题
		//少输入一个token,最后一个token是用来让它预测的，没必要告诉他
		auto inputFeature = featureMaker.makeFeatureInput(tokenList.data(),tokenList.size()-1);
		//token的数据虽然在这个makeFeatureInput里面初始化过一次，但这里就不把它取出来了
		//省得再给代码添加冗余操作
		//这里不必输入第一个token,第一个token不用被预测，这个token信息仅仅是用来算loss的
		TokenId* cudaTokenId = (TokenId*)initFromCpuData((const char*)(tokenList.data()+1),
			sizeof(TokenId)*(tokenList.size()-1));

		//临时初始化中间结果的空间
		initDecoderIntermediateContainer<FET_LENGTH>(intermediateResult,tokenList.size()-1);

		//下一步要进入到attention层的计算了
		//这里本来应该不使用中间结果的记录的，这里仅仅是做测试
		decoders.forward<RECORD_ID_TRAIN>(
			&inputFeature,&angleTrnasformer,&intermediateResult);

		//为了测试临时设置的变量，用于对输入的特征做解码
		std::vector<TokenId> tokenResult(inputFeature.featureNum);

		//对输出的特征解码成token列表的形式
		featureMaker.compareFeature<1>(&inputFeature,tokenResult.data(),&angleTrnasformer);
		//把token列表转换成字符串
		std::cout<<tokenizer.token2Str(tokenResult.data(),tokenResult.size())<<std::endl;
		//执行求导过程，更新词表向量里面的权重
		featureMaker.getDiffLossOnFeature(&inputFeature,
			optInstance.midGradient.getMainPtr(),cudaTokenId,&angleTrnasformer,
			optInstance
		);
		//调用decoder列表里面的反向传播
		//decoders.backward(&inputFeature,optInstance);
	}

// 	std::string inference(std::string promote)
// 	{
// #ifdef PRINT_INTERNAL
// 		std::cout<<"begin inference"<<std::endl;
// #endif
// 		//把模型转换成token列表
// 		std::vector<TokenId> tokenList;
// 		//调用单词表进行分词
// 		tokenizer.str2Token(promote,tokenList);
// #ifdef PRINT_INTERNAL
// 		cudaDeviceSynchronize();
// 		std::cout<<"finish str2Token"<<std::endl;
// #endif
// 		//把输入的token序列转换成特征序列 这个后面还需要注意管理内存泄漏的问题
// 		auto inputFeature = featureMaker.makeFeatureInput(tokenList.data(),tokenList.size());
// #ifdef PRINT_INTERNAL
// 		cudaDeviceSynchronize();
// 		std::cout<<"finish makeFeatureInput"<<std::endl;
// #endif
// 		//下一步要进入到attention层的计算了
// 		decoders.forward(&inputFeature,&angleTrnasformer);
// 		//调用信息解码，把最终输出的特征列表重新转换成token
// 		std::vector<TokenId> tokenResult(inputFeature.featureNum);
// 		featureMaker.compareFeature<1>(&inputFeature,tokenResult.data(),&angleTrnasformer);
// 		//把token id解码成字符串
// 		return tokenizer.token2Str(tokenResult.data(),tokenResult.size());
// 	}
};