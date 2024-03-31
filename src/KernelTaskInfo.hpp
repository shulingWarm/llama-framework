#pragma once

//每个线程的任务量
struct ThreadTaskInfo
{
	//每个线程负责的任务数
	int taskNum;
	//每个线程负责的起始id的位置
	int idBegin;
	//总的数据头
	char* dataHead;
};

//初始化任务量
void initKernelTask()
{

}