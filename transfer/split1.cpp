#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <Eigen/Eigen>
using namespace std;
using namespace Eigen;

 
vector<vector<double>> SplitString1(const string& s, vector<string>& v, const string& c)
{
    int i = 1;
    string::size_type pos1, pos2;
    vector<string> v1,v2,v3,v4,v5,v6,v7,v8;
    vector<double> v1_,v2_,v3_,v4_,v5_,v6_,v7_,v8_;
    vector<vector<double>> vec;
    pos2 = s.find(c);
    pos1 = 0;
    while(string::npos != pos2)
    {
		if (i == 1){
			//cout<<"1"<<" "<<endl;
        	v1.push_back(s.substr(pos1, pos2-pos1));
         
       		pos1 = pos2 + c.size();
        	pos2 = s.find(c, pos1);
	    }
		if (i == 2){
			//cout<<"2"<<" "<<endl;
        	v2.push_back(s.substr(pos1, pos2-pos1));
         
        	pos1 = pos2 + c.size();
        	pos2 = s.find(c, pos1);
    	}
		if (i == 3){
			//cout<<"3"<<" "<<endl;
        	v3.push_back(s.substr(pos1, pos2-pos1));
         
        	pos1 = pos2 + c.size();
        	pos2 = s.find(c, pos1);
    	}
		if (i == 4){
			//cout<<"4"<<" "<<endl;
        	v4.push_back(s.substr(pos1, pos2-pos1));
         
        	pos1 = pos2 + c.size();
        	pos2 = s.find(c, pos1);
    	}
		if (i == 5){
			//cout<<"5"<<" "<<endl;
        	v5.push_back(s.substr(pos1, pos2-pos1));
         
        	pos1 = pos2 + c.size();
        	pos2 = s.find(c, pos1);
    	}
		if (i == 6){
			//cout<<"6"<<" "<<endl;
        	v6.push_back(s.substr(pos1, pos2-pos1));
         
        	pos1 = pos2 + c.size();
        	pos2 = s.find(c, pos1);
    	} 
		if (i == 7){
			//cout<<"7"<<" "<<endl;
        	v7.push_back(s.substr(pos1, pos2-pos1));
         
        	pos1 = pos2 + c.size();
        	pos2 = s.find(c, pos1);
    	}
		if (i == 8){
			//cout<<"8"<<" "<<endl;
        	v8.push_back(s.substr(pos1, pos2-pos1));
         
        	pos1 = pos2 + c.size();
        	pos2 = s.find(c, pos1);
        	i = i-8;
    	}	
    	i++;
    }
    if(pos1 != s.length())
        v8.push_back(s.substr(pos1));
    // v.clear();

    // v.insert(v.end(),v1.begin(),v1.end());
    // v.insert(v.end(),v2.begin(),v2.end());
    // v.insert(v.end(),v3.begin(),v3.end());
    // v.insert(v.end(),v4.begin(),v4.end());
    // v.insert(v.end(),v5.begin(),v5.end());
    // v.insert(v.end(),v6.begin(),v6.end());
    // v.insert(v.end(),v7.begin(),v7.end());
    // v.insert(v.end(),v8.begin(),v8.end());
    for(auto i = 0; i < v1.size() ;i++){
        //cout<<"goes ite here"<<endl;
        v1_.push_back(stod(v1[i]));
    }
     for(auto i = 0; i < v2.size() ;i++){
        //cout<<"goes ite here"<<endl;
        v2_.push_back(stod(v2[i]));
    }
    for(auto i = 0; i < v3.size() ;i++){
        //cout<<"goes ite here"<<endl;
        v3_.push_back(stod(v3[i]));
    }
    for(auto i = 0; i < v4.size() ;i++){
        //cout<<"goes ite here"<<endl;
        v4_.push_back(stod(v4[i]));
    }
     for(auto i = 0; i < v5.size() ;i++){
        //cout<<"goes ite here"<<endl;
        v5_.push_back(stod(v5[i]));
    }   
     for(auto i = 0; i < v6.size() ;i++){
        //cout<<"goes ite here"<<endl;
        v6_.push_back(stod(v6[i]));
    }  
    for(auto i = 0; i < v7.size() ;i++){
        //cout<<"goes ite here"<<endl;
        v7_.push_back(stod(v7[i]));
    }
    for(auto i = 0; i < v8.size() ;i++){
        //cout<<"goes ite here"<<endl;
        v8_.push_back(stod(v8[i]));
    }
     vec.push_back(v1_);
    vec.push_back(v2_);
     vec.push_back(v3_);
     vec.push_back(v4_);
     vec.push_back(v5_);
     vec.push_back(v6_);
     vec.push_back(v7_);
     vec.push_back(v8_);
     return vec;
}


 

int main(){
    string s = "[[[[ 0.07659888  0.12986547 -0.21852978 -0.06858756  0.18854319 0.15537162  0.19695011  0.03963752]] [[-0.01627626  0.04426364  0.0277969  -0.13956425  0.10774631 -0.03528672  0.06751835  0.14146826]] [[-0.17298096  0.02331222 -0.23987716 -0.0099359   0.16202138 0.07302735  0.06242476  0.08487885]]] [[[-0.01069006  0.02880032 -0.04673419 -0.05329831  0.12605935 0.10972764  0.29924333 -0.0479504 ]] [[-0.15540549  0.08806492 -0.08930702 -0.07574815  0.22624114 0.07298056  0.090374    0.04996163]] [[-0.04289445  0.1466314  -0.03071477 -0.1938773   0.09341886 0.25639138  0.07338389  0.03533709]]] [[[-0.25311217  0.00135017  0.1280048   0.02306318 -0.15690267 0.12677439 -0.03038098 -0.22317116]] [[-0.14279452  0.26443243  0.13310254 -0.01555136 -0.03749768 0.15923801 -0.02370924 -0.05709661]] [[-0.03494845  0.17450118  0.07703213  0.14864519  0.03469293 0.11006013  0.10101435 -0.01718534]]]]";
    string s1 = "0.07659888 0.12986547 -0.21852978 -0.06858756 0.18854319 0.15537162 0.19695011 0.03963752 -0.01627626 0.04426364 0.0277969 -0.13956425 0.10774631 -0.03528672 0.06751835 0.14146826 -0.17298096 0.02331222 -0.23987716 -0.0099359 0.16202138 0.07302735 0.06242476 0.08487885 -0.01069006 0.02880032 -0.04673419 -0.05329831 0.12605935 0.10972764 0.29924333 -0.0479504 -0.15540549 0.08806492 -0.08930702 -0.07574815 0.22624114 0.07298056 0.090374 0.04996163 -0.04289445 0.1466314 -0.03071477 -0.1938773 0.09341886 0.25639138 0.07338389 0.03533709 -0.25311217 0.00135017 0.1280048 0.02306318 -0.15690267 0.12677439 -0.03038098 -0.22317116 -0.14279452 0.26443243 0.13310254 -0.01555136 -0.03749768 0.15923801 -0.02370924 -0.05709661 -0.03494845 0.17450118 0.07703213 0.14864519 0.03469293 0.11006013 0.10101435 -0.01718534";
    vector<string> v;
    vector<MatrixXd> res;
    vector<vector<double>> va;
	cout<<"123456"<<" "<<endl;
    va = SplitString1(s1, v," "); 
    MatrixXd m(3,3);
    //for(vector<string>::size_type i = 0; i != v.size(); ++i)
    //    cout << v[i] << " ";
    for (int j = 0; j< va.size(); j++){
         m <<va[j][0],va[j][1],va[j][2],
                va[j][3],va[j][4],va[j][5],
                va[j][6],va[j][7],va[j][8];
            res.push_back(m);
    }
    for(int i = 0; i<8; i++){
        cout<<res[i]<<endl;
    }
    cout << endl;
}
