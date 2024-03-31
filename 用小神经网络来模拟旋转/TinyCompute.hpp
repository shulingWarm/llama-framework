#pragma once

//各种小的计算

//向上取整
__device__ int divUp(int num1,int num2)
{
	return (num1 + num2 - 1)/num2;
}