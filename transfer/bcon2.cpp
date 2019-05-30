#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <Eigen/Eigen>
using namespace std;
using namespace Eigen;

int main(){
	string s ="-0.12026328 -0.02369919 0.0794071 -0.07300829 0.02144894 0.1731819 0.14079559 0.08050600 0.13548067 -0.04575408 0.04896416 -0.02175735 0.17480594 0.07393119 0.03627094 -0.0374101";
	string::size_type pos1, pos2;
	pos2 = s.find(" ");
	string b = " ";
	vector<string> v;
	vector<double> v1;
	pos1 = 0;
	while(string::npos != pos2){
		//cout<<"enterloop"<<endl;
		v.push_back(s.substr(pos1, pos2-pos1));
         
       		pos1 = pos2 + b.size();
        	pos2 = s.find(b, pos1);
	}
	v.push_back(s.substr(pos1,10));
	for(int i = 0; i<16; i++){
		cout<<v[i]<<endl;		
	}
	//cout<<"outloop"<<endl;
	for(int i = 0; i<16; i++){
		v1.push_back(stod(v[i]));		
	}
	//cout<<"outstod"<<endl;
	//for(int i = 0; i<16; i++){
	//	cout<<v1[i]<<endl;		
	//}
}