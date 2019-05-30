//#include "~/cnn.hpp"
#include <iostream>
#include <Eigen/Eigen>

Eigen::Matrixxd
ReLU(Eigen::Matrixxd &input){
	row = rows(input);
	col = cols(input);
	for (int i = 0; i < row; i ++){
		for (int j = 0; j < col; j ++){
			if (input(i,j) < 0)
				input(i,j) = 0;
		}
	}
}

double
conv(Eigen::Matrixd input, Eigen::Matrixxd kernel){
	double result = 0;
	row = rows(kernel);
	col = cols(kernel);
	for (int i = 0; i < row; i ++){
		for (int j = 0; j < col; j ++){
			result += input(i,j) * kernel(i,j);
		}
	}
	return result;
}

Eigen::Matrixxd 
convLayer(Eigen::Matrixxd input, Eigen::Matrixxd kernel,int step, string padding){
	if (padding == "SAME"){
		int extend_size = rows(kernel);
		row = rows(input);
		col = cols(input);
		for (int i = 0; i < row; i ++){
			for (int j = 0; j < col ; j ++){

			}
		}
	}	
}

Eigen::Matrixxd
poolLayer(Eigen::Matrixxd input, k, padding){
	int input_row = rows(input);
	int input_col = cols(input);
	int result_row = ceil(input_row / k);
	int result_col = ceil(input_col / k); 
	Eigen::Matrixxd result(result_row, result_col);

	if (padding == "MAX"){
		for (int i = 0; i < result_row - 1; i ++){
			for (int j = 0; j < result_col - 1; j ++){
				result(i,j) = input.block(i*k, j*k, k, k).maxCoeff();
			}
			int j = result_col - 1;
			result(i,j) = input.block(i*k, j*k, k, input_col - j*k ).maxCoeff(); 
		}
		int i = result_row - 1;
		for (int j = 0; j < result_col - 1; j ++){
			result(i,j) = input.block(i*k, j*k, input_row - i*k, k).maxCoeff();
		}
		int j = result_col - 1;
		result(i,j) = input.block(i*k, j*k, input_row - i*k, input_col - j*k ).maxCoeff();
	}
	if (padding == "AVERAGE"){
		for (int i = 0; i < result_row - 1; i ++){
			for (int j = 0; j < result_col - 1; j ++){
				result(i,j) = input.block(i*k, j*k, k, k).sum() / (k*k);
			}
			int j = result_col - 1;
			result(i,j) = input.block(i*k, j*k, k, input_col - j*k ).sum() / (k*(input_col - j*k)); 
		}
		int i = result_row - 1;
		for (int j = 0; j < result_col - 1; j ++){
			result(i,j) = input.block(i*k, j*k, input_row - i*k, k).sum() / ((input_row - i*k) * k);
		}
		int j = result_col - 1;
		result(i,j) = input.block(i*k, j*k, input_row - i*k, input_col - j*k ).sum() / ((input_row - i*k) * (input_col - j*k));
	}
	return result;
}

int main(int argc, char const *argv[])
{
	Eigen::matrixxd input(2,2);
	input << (1,2
			  2,1);
	Eigen::matrixxd kernel(2,2);
	kernel << (1,2,
			   2,1);
	std::cout << conv(input,kernel)<< "\n";		   
	return 0;
}