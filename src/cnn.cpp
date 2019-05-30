//#include "cnn.hpp"
#include <iostream>
#include <Eigen/Eigen>
#include <string>
#include <vector>

using namespace Eigen;
using namespace std;

MatrixXd ReLU(MatrixXd &input){
	int row = input.rows();
	int col = input.cols();
	for (int i = 0; i < row; i ++){
		for (int j = 0; j < col; j ++){
			if (input(i,j) < 0)
				input(i,j) = 0;
		}
	}
}

double
conv(MatrixXd input, MatrixXd kernel){
	// std::cout << "input = \n" << input << "\n";
	// std::cout << "kernel = \n" << kernel << "\n";

	double result = 0;
	int row = kernel.rows();
	int col = kernel.cols();
	for (int i = 0; i < row; i ++){
		for (int j = 0; j < col; j ++){
			result += input(i,j) * kernel(i,j);
		}
	}
	return result;
}



//default step is 1; default mode is SAME
MatrixXd 
convLayer(MatrixXd x, MatrixXd kernel, double b_conv){
	int extend_size = kernel.rows() - 1;
	int row = x.rows();
	int col = x.cols();
	MatrixXd input( row + extend_size, col + extend_size );
	auto result = x;
	input.setZero();
	for (int i = 0; i < row; i++){
		for (int j =0; j < col; j++){
			input(i,j) = x(i,j);
		}
	}

	for (int i = 0; i < row; i ++ ){
		for (int j = 0; j < col ; j ++ ){
			result(i,j) = conv(input.block(i,j,kernel.rows(),kernel.rows()),kernel) + b_conv;
		}
	}
	return result;		
}

MatrixXd
poolLayer(MatrixXd input, int k, string padding){
	int input_row = input.rows();
	int input_col = input.cols();
	int result_row = (input_row + k - 1) / k;
	int result_col = (input_col + k - 1) / k; 
	MatrixXd result(result_row, result_col);
	// std::cout << "row = " << input_row << "\n";
	// std::cout << "result_row = " << result_row << "\n";

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

// int main(int argc, char const *argv[])
// {
// 	MatrixXd input(2,2);
// 	// input << 1,1,
// 			 // 1,1;
// 	MatrixXd kernel(2,2);
// 	kernel << 1,2,
// 			  2,1;
// 	std::cout << "before = \n" <<input << "\n";
// 	input += kernel;
// 	std::cout << "after = \n" << input << "\n"; 


// 	//std::cout << input.maxCoeff();
// 	// std::cout << "after poolLayer:\n" << poolLayer(result ,2,"MAX") << "\n";

// 	return 0;
// }
void input_parameter_conv1(vector<MatrixXd> &w_conv1, vector<double> &b_conv1){

}
void input_parameter_conv2(vector<vector<MatrixXd>> &w_conv2, vector<vector<double>> &b_conv2){

}
void input_parameter_fc1(MatrixXd & w_fc1, VectorXd & b_fc1){

}

void input_parameter_fc2(MatrixXd & w_fc2, VectorXd & b_fc2){

}



int main(int argc, char const *argv[])
{
	int init_size = 28;
	int size2 = init_size / 2;
	int size3 = size2 / 2;

	int neural_amount1 = 32;
	int neural_amount2 = 64;
	int neural_amount3 = 1024;

	int kernel_size = 5;


	// get the init picture
	MatrixXd init_picture(init_size,init_size);

	//first convulution layer and pool layer
	vector<MatrixXd> w_conv1(neural_amount1);
	for (int i = 0; i < neural_amount1; i++){
		w_conv1[i].resize(kernel_size,kernel_size);
	}	
	vector<double> b_conv1(neural_amount1);

	
	vector<MatrixXd> picture_after_c1(neural_amount1);
	for (int i = 0; i < neural_amount1; i++){
		picture_after_c1[i].resize(init_size,init_size);
	}
	vector<MatrixXd> picture_after_p1(32);
	for (int i = 0; i < neural_amount1; i++){
		picture_after_p1[i].resize(size2,size2);
	}

	input_parameter_conv1(w_conv1,b_conv1);  //load the parameters from trianed model
	
	for (int i = 0; i < neural_amount1; i++){
		picture_after_c1[i] = convLayer(init_picture,w_conv1[i],b_conv1[i]);
		ReLU(picture_after_c1[i]);
		picture_after_p1[i] = poolLayer(picture_after_c1[i],2,"MAX");
	}

	//second convulution layer and pool layer
	
	vector<vector<MatrixXd>> w_conv2(neural_amount1);
	vector<vector<double>> b_conv2(neural_amount1);
	for (int i = 0; i < neural_amount1; i ++){
		w_conv2[i].resize(neural_amount2);
		b_conv2[i].resize(neural_amount2);
		for (int j = 0; j < neural_amount2; j++){
			w_conv2[i][j].resize(kernel_size,kernel_size);
		}
	}
	
	
	vector<MatrixXd> picture_after_c2(neural_amount2);
	for (int i = 0; i < neural_amount2; i++){
		picture_after_c2[i].resize(size2,size2);
	}
	vector<MatrixXd> picture_after_p2(neural_amount2);
	for (int i = 0; i < neural_amount2; i++){
		picture_after_p2[i].resize(size3,size3);
	}

	
	input_parameter_conv2(w_conv2,b_conv2); //load the parameters from trianed model

	for (int j = 0; j < neural_amount2; j ++){
		for (int i = 0; i < neural_amount1; i ++){
			picture_after_c2[j] += convLayer(picture_after_p1[i], w_conv2[i][j] ,b_conv2[i][j]);
		}
		ReLU(picture_after_c2[j]);
		picture_after_p2[j] = poolLayer(picture_after_c2[j], 2, "MAX");
	}

	
	//reshape the 64 7*7 matrix to a (64*7*7) * 1 matrix  
	VectorXd reshaped_picture(neural_amount2 * size3 * size3);
	VectorXd picture_after_link(neural_amount3);
	MatrixXd w_fc1(neural_amount3, neural_amount2 * size3 * size3);
	VectorXd b_fc1(neural_amount3);

	input_parameter_fc1(w_fc1, b_fc1);  //load the parameters from trianed model
	picture_after_link = w_fc1 * reshaped_picture + b_fc1;

	
	VectorXd result(10);
	MatrixXd w_fc2(10, neural_amount3);
	VectorXd b_fc2(10);
	input_parameter_fc2(w_fc2, b_fc2);  //load the parameters from trianed model
	result = w_fc2  * picture_after_link + b_fc2;

	int index;
	double temp = result.maxCoeff(&index);

	std::cout << "The result is \n" << result << "\n";
	std::cout << "Thus the guess number is " << index << "\n";	
	return 0;
}

