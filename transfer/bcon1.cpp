#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <Eigen/Eigen>
using namespace std;
using namespace Eigen;

int main(){
	string s ="0.17176697 0.09390566 0.12352279 0.15875247 0.07848966 0.07257443 0.09798833 0.11178268";
	vector<string> v;
	vector<double> res;
	for(int i = 0; i < 88; i = i + 11){
		v.push_back(s.substr(i,10));
	}
	for(int i = 0 ; i< 8; i++){
		res.push_back(stod(v[i]));
	}
	// for(int i = 0 ; i< 8; i++){
	// 	cout<<res[i]<<endl;
	// }
}