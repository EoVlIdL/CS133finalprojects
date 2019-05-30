#include <string>
#include <Eigen/Eigen>

Eigen::MatrixXd
ReLU(Eigen::MatrixXd input);

Eigen::MatrixXd 
conv(Eigen::MatrixXd input, Eigen::MatrixXd kernel);

Eigen::MatrixXd
pool(Eigen::MatrixXd input,int k);

Eigen::MatrixXd
connected(Eigen::MatrixXd input)

