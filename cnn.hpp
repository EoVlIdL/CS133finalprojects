#include <Eigen/Eigen>
#include <string>

Eigen::Matrixxd
ReLU(Eigen::Matrixxd input);

Eigen::Matrixxd 
convLayer(Eigen::Matrixxd input, Eigen::Matrixxd kernel,  int step = 1, string padding = "SAME");

Eigen::Matrixxd
poolLayer(Eigen::Matrixxd input, k ,string padding);

Eigen::Matrixxd
connectedLayer(Eigen::Matrixxd input);

