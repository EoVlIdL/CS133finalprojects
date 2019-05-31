//#include "cnn.hpp"
#include "mnist_reader.h"
#include <iostream>
#include <random>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <Eigen/Eigen>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
using namespace Eigen;
using namespace std;

mnist_reader::mnist_reader(string pathImage, string pathLabel)
{
    mFullPathImage = pathImage;
    mFullPathLabel = pathLabel;
}


int mnist_reader::reverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}


void mnist_reader::ReadMNIST(vector<Eigen::MatrixXd> &mnist, Eigen::MatrixXi &label)
{
    ifstream file (mFullPathImage,ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        mnist.resize(number_of_images);
        cout<<"number_of_images : "<<number_of_images<<endl;
        for(auto itr(mnist.begin()); itr != mnist.end(); itr++)
        {
            itr->resize(n_rows,n_cols);
        }
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    mnist[i](r,c) = ((double)temp)/255;
                }
            }
        }
    }

    ifstream file2 (mFullPathLabel,ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        file2.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file2.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        label.resize(number_of_images,1);
        for(int i=0;i<number_of_images;++i)
        {
            unsigned char temp=0;
            file2.read((char*)&temp,sizeof(temp));
            label(i,0) = (int)temp;
        }
    }
}


void ReLU(MatrixXd &input){
	int row = input.rows();
	int col = input.cols();
	// std::cout << "row = " << row << "  col = " << col<< "\n";
	for (int i = 0; i < row; i ++){
		// std::cout << "i = " << i << "\n";
		for (int j = 0; j < col; j ++){
			if (input(i,j) < 0)
				input(i,j) = 0;
		}
		// std::cout << "input = \n" << input << "\n";
	}
	// std::cout << "ReLU finished(inside)\n";
	// std::cout << input << "\n";

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
			result += input(i,j) * kernel(row - i -1,col - j-1);
		}
	}
	return result;
}



//default step is 1; default mode is SAME
MatrixXd 
convLayer(MatrixXd x, MatrixXd kernel){
	int extend_size = kernel.rows() - 1;
	int row = x.rows();
	int col = x.cols();
	MatrixXd input( row + extend_size, col + extend_size );
	auto result = x;
	input.setZero();
	for (int i = 0; i < row; i++){
		for (int j =0; j < col; j++){
			input(i + 1,j + 1) = x(i,j);
		}
	}

	for (int i = 0; i < row; i ++ ){
		for (int j = 0; j < col ; j ++ ){
			result(i,j) = conv(input.block(i,j,kernel.rows(),kernel.rows()),kernel);
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


void bias(MatrixXd & x , double b){
	int row = x.rows();
	int col = x.cols();
	for (int i = 0; i < row; i ++){
		for (int j = 0; j < col ; j ++){
			x(i,j) += b;
		}
	}
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
	fstream wcon1("w_con1.txt");
	if(!wcon1.is_open()){
		cout<<"Error opening file w_conv1.txt"<<endl;
		exit(1);
	}
	string line;
    int row_count = 0;
    int col_count = 0;
    int row = w_conv1[0].rows();
    //cout<<row<<"********"<<endl;
    int col = w_conv1[0].cols();
    while (getline(wcon1,line)){
    	stringstream input(line);
    	vector<double> matrix;
    	string tmp;
    	while(input>>tmp){
    		matrix.push_back(stod(tmp));
    	}
    	//for(int i = 0; i<8; i++ ){	
    	//cout<<"innerloop"<<matrix[i]<<endl;
    	//} 	
    	for(int i = 0; i < 8; i++){
    		w_conv1[i](row_count,col_count) = matrix[i];
		}
		col_count ++;
    	if (col_count == col){
	    	col_count = 0;
	    	row_count += 1;
	    }   
	    //cout<<"loop "<<row_count<<endl; 	
    }
    // for(int i = 0; i<w_conv1.size(); i++){
    // 	cout<<"out loop\n "<<w_conv1[i]<<endl; 
    // }
    
    // for(int i = 0; i<8; i++ ){
    // 	cout<<w_conv1[i]<<"/*****//"<<endl;
    // }

	// string buffer;
	// while(!wcon1.eof()){
	// 	getline(wcon1,buffer);
	// }
	wcon1.close();
	fstream bcon1("b_con1.txt");
	if(!bcon1.is_open()){
		cout<<"Error opening file w_conv1.txt"<<endl;
		exit(1);
	}
	string bias;
	while (getline(bcon1,bias)){
    	stringstream input(bias);
    	string tmp;
    	vector<double> pre;
    	while(input>>tmp){
    		pre.push_back(stod(tmp));
    	}
    	for(int i =0;i<b_conv1.size();i++){
    		b_conv1[i] = pre[i];
    	}
	}
	bcon1.close();
	// for(int i =0;i<b_conv1.size();i++){
	// 	cout<<"begin"<<endl;
	// 	cout<<b_conv1[i]<<endl;
	// }
	// std::cout << "w_conv1[0] = \n" << w_conv1[0] << "\n";
	// std::cout << "b_conv1[0] = " << b_conv1[0] << "\n";
}
void input_parameter_conv2(vector<vector<MatrixXd>> &w_conv2, vector<double> &b_conv2){
	fstream bcon2("b_con2.txt");
	if(!bcon2.is_open()){
		cout<<"Error opening file w_conv1.txt"<<endl;
		exit(1);
	}
	string bias;
	while (getline(bcon2,bias)){
    	stringstream input(bias);
    	string tmp;
    	vector<double> pre;
    	while(input>>tmp){
    		pre.push_back(stod(tmp));
    	}
    	for(int i =0;i<b_conv2.size();i++){
    		b_conv2[i] = pre[i];
    	}
	}
	bcon2.close();
	// for(int i =0;i<b_conv2.size();i++){
 //    		cout<<b_conv2[i]<<endl;
 //    	}
	fstream wcon2("w_con2.txt");
	if(!wcon2.is_open()){
		cout<<"Error opening file w_conv1.txt"<<endl;
		exit(1);
	}
	string line;
	int matrix_count = 0;
    int row_count = 0;
    int col_count = 0;
    int row = w_conv2[0][0].rows(); 
    int col = w_conv2[0][0].cols();
    int loop_count = 0;
    while (getline(wcon2,line)){
    	//cout<<"enter loop "<<loop_count<<endl;
    	stringstream input(line);
    	vector<double> matrix;
    	string tmp;
    	while(input>>tmp){
    		matrix.push_back(stod(tmp));
    	}
    	
  //   	for(int i = 0; i < 8; i++){
  //   		for(int j = 0; j < 16; j++){
  //   			w_conv2[i][j](row_count,col_count) = matrix[j];
  //   		}
		// }
		if(matrix_count == 8){
			matrix_count = 0;
			col_count += 1;
			if (col_count == col){
	    		col_count = 0;
	    		row_count += 1;
	    		//if(row_count == row){

	    		//}
	    	}   	
		}
		for(int i = 0; i < 16; i++){
    		// cout<<"enter inner loop"<<endl;
    		// cout<<matrix_count<<"***"<<col_count<<"***"<<row_count<<endl;
    		// cout<<"row"<<row<<endl;
    		w_conv2[matrix_count][i](row_count,col_count) = matrix[i];
    	}
    	
    	matrix_count++;
    	loop_count++;
    	//cout<<"********\n"<<w_conv2[7][15]<<"*********"<<endl;
    }
   // cout<<"exit loop"<<endl;
    // for(int i = 0; i < 8 ; i++){
    // 	for(int j = 0; j < 16; j++){
    // 		cout<<w_conv2[i][j]<<"********"<<endl;
    // 	}
    // }
	wcon2.close();
	//cout<<"crash in conv2"<<endl;
	// std::cout << "w_conv2[0][0] = \n" << w_conv2[0][0] << "\n";
	// std::cout << "b_conv2[0] = " << b_conv2[0] << "\n";
}
void input_parameter_fc1(MatrixXd & w_fc1, VectorXd & b_fc1){
	fstream wfc1("w_fc1.txt");
	if(!wfc1.is_open()){
		cout<<"Error opening file w_fc1.txt"<<endl;
		exit(1);
	}
	// cout<<"goes wfc1"<<endl;
	int row_count = 0;
    int col_count = 0;
    int row = w_fc1.rows(); 
    int col = w_fc1.cols();
    string line;
	while (getline(wfc1,line)){
    	//cout<<"enter loop "<<loop_count<<endl;
    	stringstream input(line);
    	string tmp;
    	vector<double> m;
    	while(input>>tmp){
    		//matrix.push_back(stod(tmp));
    		m.push_back(stod(tmp));
    	}
    	for(int i = 0; i < 4; i++){
    		w_fc1(row_count,col_count) = m[i];
    			row_count ++;
    			if (row_count == row){
	    			row_count = 0;
	    			col_count += 1;
	    		}
    	}	
    }
    //cout<<w_fc1<<endl;
    wfc1.close();
    fstream bfc1("b_fc1.txt");
	if(!bfc1.is_open()){
		cout<<"Error opening file b_fc1.txt"<<endl;
		exit(1);
	}
	string bias;
	while (getline(bfc1,bias)){
    	//cout<<"enter loop "<<loop_count<<endl;
    	stringstream input(bias);
    	string tmp;
    	vector<double> m;
    	while(input>>tmp){
    		//matrix.push_back(stod(tmp));
    		m.push_back(stod(tmp));
    	}
    	for(int i = 0; i < b_fc1.size(); i++){
    		b_fc1[i] = m[i];
    	}
    }
    bfc1.close();
}

void input_parameter_fc2(MatrixXd & w_fc2, VectorXd & b_fc2){
	fstream wfc2("w_fc2.txt");
	if(!wfc2.is_open()){
		cout<<"Error opening file w_fc2.txt"<<endl;
		exit(1);
	}
	// cout<<"goes wfc2"<<endl;
	int row_count = 0;
    int col_count = 0;
    int row = w_fc2.rows(); 
    int col = w_fc2.cols();
    string line;
	while (getline(wfc2,line)){
    	//cout<<"enter loop "<<loop_count<<endl;
    	stringstream input(line);
    	string tmp;
    	vector<double> m;

    	while(input>>tmp){
    		//matrix.push_back(stod(tmp));
    		m.push_back(stod(tmp));
    	}
    	
    	for(int i = 0; i<m.size(); i++){
    		w_fc2(i,col_count) = m[i];		
    	}
    	// cout<<"col_count "<<col_count<<endl;
    	col_count++;
    }
    // cout<<w_fc2<<endl;
    wfc2.close();
    fstream bfc2("b_fc2.txt");
	if(!bfc2.is_open()){
		cout<<"Error opening file b_fc2.txt"<<endl;
		exit(1);
	}
	string bias;
	while (getline(bfc2,bias)){
    	//cout<<"enter loop "<<loop_count<<endl;
    	stringstream input(bias);
    	string tmp;
    	vector<double> m;
    	while(input>>tmp){
    		//matrix.push_back(stod(tmp));
    		m.push_back(stod(tmp));
    	}
    	for(int i = 0; i < b_fc2.size(); i++){
    		b_fc2[i] = m[i];
    	}
    }
    bfc2.close();
}

double test(MatrixXd init_picture, double label)
{
	int init_size = 28;
	int size2 = init_size / 2;
	int size3 = size2 / 2;

	int neural_amount1 = 8;
	int neural_amount2 = 16;
	int neural_amount3 = 256;

	int kernel_size = 3;
	// MatrixXd init_picture(init_size,init_size);

	// get the init picture
	// MatrixXd init_picture(init_size,init_size);

	// MatrixXd picture(14,4); 
	// picture <<  0.6, 0.8 , 0  , 0,
	// 		    0.7, 1.0 , 0  , 0,
	// 		    0.7, 1.0 , 0  , 0,
	// 			0.7, 1.0 , 0  , 0,
	// 			0.7, 1.0 , 0  , 0,
	// 		    0.5, 1.0 , 0.4, 0,
	// 		      0, 1.0 , 0.4, 0,
	// 		      0, 1.0 , 0.4, 0,
	// 			  0, 1.0 , 0.7, 0,
	// 			  0, 1.0 , 0.7, 0,
	// 			  0, 1.0 , 0.7, 0,
	// 			  0, 1.0 , 1.0, 0,
	// 			  0, 0.9 , 0.7, 0.1,
	// 			  0, 0.3 , 0.7, 0.1;
	// for (int i = 0; i < 14; i++){
	// 	for (int j = 0; j < 4; j++){
	// 		init_picture(i + 5, j + 8) = picture(i,j);
	// 	}
	// }

	// std::cout << "init_picture = \n" << init_picture << "\n";



	//first convulution layer and pool layer
	vector<MatrixXd> w_conv1(neural_amount1);
	for (int i = 0; i < neural_amount1; i++){
		w_conv1[i].resize(kernel_size,kernel_size);
	}	
	vector<double> b_conv1(neural_amount1);

	
	vector<MatrixXd> picture_after_c1(neural_amount1);
	vector<MatrixXd> picture_after_p1(neural_amount1);
	for (int i = 0; i < neural_amount1; i++){
		picture_after_c1[i].resize(init_size,init_size);
		picture_after_p1[i].resize(size2,size2);
	}
	cout<<"conv1"<<endl;
	input_parameter_conv1(w_conv1,b_conv1);  //load the parameters from trianed model
	
	for (int i = 0; i < neural_amount1; i++){
		// w_conv1[i].transposeInPlace();
		picture_after_c1[i] = convLayer(init_picture,w_conv1[i]);
		bias(picture_after_c1[i], b_conv1[i]);
		ReLU(picture_after_c1[i]);
		picture_after_p1[i] = poolLayer(picture_after_c1[i],2,"MAX");
	}

std::cout << "picture_after_p1\n";
for (int i = 0; i < neural_amount1; i ++){
	std::cout << picture_after_p1[i] << "\n\n";
}

	//second convulution layer and pool layer
	
	vector<vector<MatrixXd>> w_conv2(neural_amount1);
	vector<double> b_conv2(neural_amount2);
	for (int i = 0; i < neural_amount1; i ++){
		w_conv2[i].resize(neural_amount2);
		for (int j = 0; j < neural_amount2; j++){
			w_conv2[i][j].resize(kernel_size,kernel_size);
		}
	}
	
	vector<MatrixXd> picture_after_c2(neural_amount2);
	vector<MatrixXd> picture_after_p2(neural_amount2);
	for (int i = 0; i < neural_amount2; i++){
		picture_after_c2[i].resize(size2,size2);
		picture_after_p2[i].resize(size3,size3);
	}
	// cout << "here\n";
	cout<<"conv2"<<endl;
	input_parameter_conv2(w_conv2,b_conv2); //load the parameters from trianed model
	// cout << "here\n";
	for (int j = 0; j < neural_amount2; j ++){
		// cout << "j = " << j <<"\n";
		for (int i = 0; i < neural_amount1; i ++){
			// w_conv2[i][j].transposeInPlace();
			picture_after_c2[j] += convLayer(picture_after_p1[i], w_conv2[i][j]);
		}
		// cout << "here\n";
		bias(picture_after_c2[j], b_conv2[j]);
		// cout << "after bias\n";
		// std::cout << "matrix = \n" << picture_after_c2[j] <<"\n";
		// std::cout <<"before Relu\n";
		ReLU(picture_after_c2[j]);
		// cout << "after Relu(outside)\n";
		picture_after_p2[j] = poolLayer(picture_after_c2[j], 2, "MAX");
	}
	// cout << "here\n";
	
	//reshape the 64 7*7 matrix to a (64*7*7) * 1 matrix  
	VectorXd reshaped_picture(neural_amount2 * size3 * size3);
	VectorXd picture_after_link(neural_amount3);
	MatrixXd w_fc1(neural_amount3, neural_amount2 * size3 * size3);
	VectorXd b_fc1(neural_amount3);
	
	int count = 0;
	for (int index = 0; index < neural_amount1; index ++){
		for (int i = 0; i < size3; i++){
			for (int j = 0; j < size3 ; j++){
				reshaped_picture(count) = picture_after_p2[index](i,j);
				count ++;
			}
		}
	}

	cout<<"fc1"<<endl;
	input_parameter_fc1(w_fc1, b_fc1);  
	//load the parameters from trianed model
	picture_after_link = w_fc1 * reshaped_picture + b_fc1;

	
	VectorXd result(10);
	MatrixXd w_fc2(10, neural_amount3);
	VectorXd b_fc2(10);
	cout<<"fc2"<<endl;
	input_parameter_fc2(w_fc2, b_fc2);  //load the parameters from trianed model
	result = w_fc2  * picture_after_link + b_fc2;

	int index;
	double temp = result.maxCoeff(&index);
	double sum = result.sum();
	for (int i = 0; i < result.size(); i++){
		result(i) = result(i) / sum;
	}

	// std::cout << "The result is \n" << result << "\n";
	// std::cout << "Thus the guess number is " << index << "\n";	
	if (index == label){
		return 1;
	}
	return 0;
}
int main(int argc, char const *argv[])
{
	mnist_reader readerTrain("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    std::vector<Eigen::MatrixXd> imageTrain;
    Eigen::MatrixXi labelTrain;
    readerTrain.ReadMNIST(imageTrain, labelTrain);

    mnist_reader readerTest("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
    std::vector<Eigen::MatrixXd> imageTest;
    Eigen::MatrixXi labelTest;
    readerTest.ReadMNIST(imageTest, labelTest);

    double sum = 0;
    for (int i = 0; i < 100; i++){
    	sum += test(imageTest[i], labelTest(i,0));
    }
    std::cout << "sum = " << sum << "\n";
	
    // MatrixXd ma(4,4);
    // ma << 1,2,3,4,
    // 	  2,3,4,5,
    // 	  3,4,5,6,
    // 	  4,5,6,7;
   	// MatrixXd kernel(3,3);
   	// kernel << 0,0,0,
   	// 		  0,1,0,
   	// 		  0,0,0;
   	// std::cout << "c = " << convLayer(ma,kernel) << "\n";

	return 0;
}

