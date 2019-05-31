CS133_Group_Project
===================
Multi-layer networks include the following types of layers:
1. K single-layer networks
2. A softmax regression output layer.

Compile the project
-------------------
    g++ -std=c+=11 cnn.cpp

File Location
-------------------
	Mnist Data:    All data are under "MNIST_data" folder.
	Source Code:   All source codes are under "src" folder.
	Header File:   All include header files are under include.
	Eigen package: All Eigen files.

Usage Example
-------------------

Implementation Order
-----
From Mnist Data get corresponding data "t10k-images-idx3-ubyte" and "train-images-idx3-ubyte" with label data. Through "mnist_reader.h" function ReadMNIST trabsfer data to a struct `vector<Eigen::MatrixXd>`.  
Then compute the input matrix with first conv_layer and pool_layer get two vectors vector<MatrixXd> picture_after_c1 and  vector<MatrixXd> picture_after_p1.  
After get first conv and pool, load parameters from train model and do the same operation once more. 
Return vector<MatrixXd> picture_after_c2 and vector<MatrixXd> picture_after_p2. Then reshape the `64*7*7` matrix to a `(64*7*7) * 1` matrix and delivered to softmax regression output layer.  
A layered activation map; each layer is a used filter. The larger rectangle is the
slice to be downsampled.  
The active map is compressed by downsampling.  
A new set of activation maps obtained by having the filter scan the first map heap
that has been downsampled.  
Compresses the second downsampling of the second set of activation maps.  
A fully connected layer that classifies the output by a node and a tag.  
Final get output:
```
	1. The result is \n" << result 
	2. Thus the guess number is " << index 	
```
 
Train and Test
-------------------

Performance
-------------------

Meta
-------------------
34038083

Contributing
-------------------
1. https://skymind.ai/wiki/neural-network
2. https://deeplearning4j.org/cn/multinetwork
3. http://www.tensorfly.cn/tfdoc/tutorials/mnist_download.html
4. https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/79017786
