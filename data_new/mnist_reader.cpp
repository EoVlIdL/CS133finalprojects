#include "mnist_reader.h"
#include <iostream>
#include <random>
#include <eigen3/Eigen/Dense>
#include <functional>

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
        cout<<"n_rows : "<<n_rows<<endl;
        cout<<"n_cols : "<<n_cols<<endl;
        cout<<mnist[0].cols()<<" and "<<mnist[0].rows()<<endl;
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
                    mnist[i](r,c) = ((float)temp)/255;
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

int main()
{   
    srand(time(0));

    try
    {
        mnist_reader readerTrain("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
        std::vector<Eigen::MatrixXf> imageTrain;
        Eigen::MatrixXi labelTrain;
        readerTrain.ReadMNIST(imageTrain, labelTrain);
        cout<<imageTrain[0].rows()<<" and "<<imageTrain[0].cols()<<endl;
        cout<<imageTrain[2]<<endl;
        cout<<"1111111111111"<<endl;
        cout<<labelTrain(2,0)<<"_label"<<endl;

        mnist_reader readerTest("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
        std::vector<Eigen::MatrixXf> imageTest;
        Eigen::MatrixXi labelTest;
        readerTest.ReadMNIST(imageTest, labelTest);
    }

    catch (const std::exception& ex)
    {
        std::cout << "Exception was thrown: " << ex.what() << std::endl;
    }
    return 0;
}
