#include "iostream"
#include "malloc.h"
#include "cstdlib"
#include "cmath"
using namespace std;
double** memalloc(int m,int n){
	double **mem=(double **)malloc(m*sizeof(double*));
	for(int i=0;i<m;i++){
		mem[i]=(double *)malloc(n*sizeof(double));
	}
	return mem;
}
void memfree(double **mem,int m){
	for(int i=0;i<m;i++)
		free(mem[i]);
	free(mem);
}
double actf(double x){
	return 1/(1+exp(-x));
	//return tanh(x);
}
double dactf(double x){
	return x*(1-x);
	//return 1-x*x;
}
class network{
	double **weighthid,**weightout,**actin,**acthid,**actout,**wchhidden,**wchoutput,**outputdelta,**hiddendelta;
	int inputn,hiddenn,outputn;
	double temp,e,change;
	int temp1;
	public:
		network(int input,int hidden,int output){
			inputn=input+1;
			hiddenn=hidden;
			outputn=output;
			weighthid=memalloc(inputn,hiddenn);
			weightout=memalloc(hiddenn,outputn);
			wchhidden=memalloc(inputn,hiddenn);
			wchoutput=memalloc(hiddenn,outputn);
			actin=memalloc(inputn,1);
			acthid=memalloc(hiddenn,1);
			actout=memalloc(outputn,1);
			outputdelta=memalloc(outputn,1);
			hiddendelta=memalloc(hiddenn,1);
			srand(time(0));
			for(int i=0;i<inputn;i++){
				for(int j=0;j<hiddenn;j++){
					weighthid[i][j]=-1+(rand()/(double)RAND_MAX)*(1+1);
				}
			}
			for(int i=0;i<hiddenn;i++){
				for(int j=0;j<outputn;j++){
					weightout[i][j]=-1+(rand()/(double)RAND_MAX)*(1+1);
				}
			}
			for(int i=0;i<inputn;i++){
				for(int j=0;j<hiddenn;j++){
					wchhidden[i][j]=0;
				}
			}
			for(int i=0;i<hiddenn;i++){
				for(int j=0;j<outputn;j++){
					wchoutput[i][j]=0;
				}
			}
		}
		~network(){
			memfree(actin,inputn);
			memfree(acthid,hiddenn);
			memfree(actout,outputn);
			memfree(outputdelta,outputn);
			memfree(hiddendelta,hiddenn);
			memfree(weighthid,inputn);			
			memfree(weightout,hiddenn);
			memfree(wchhidden,inputn);
			memfree(wchoutput,hiddenn);
		}
		void debug_out(){
			/*for (int i=0;i<inputn;i++){
				cout<<"input "<<i<<"= "<<actin[i][0]<<endl;
			}*/
			/*for (int i=0;i<inputn;i++){
				for(int j=0;j<hiddenn;j++){
					cout<<"weight of input "<<i<<" to hidden "<<j<<" = "<<weighthid[i][j]<<endl;
				}
			}
			for (int i=0;i<hiddenn;i++){
				cout<<"hidden "<<i<<"= "<<acthid[i][0]<<endl;
			}
			for (int i=0;i<hiddenn;i++){
				for(int j=0;j<outputn;j++){
					cout<<"weight of hidden "<<i<<" to output "<<j<<" = "<<weightout[i][j]<<endl;
				}
			}*/
			for (int i=0;i<outputn;i++){
				cout<<"output "<<i<<"= "<<actout[i][0]<<endl;
			}
		}
		double** update(double *mat,int matlen){
			if(matlen!=inputn-1){
				
				exit(0);	
			}
			for(int i=0;i<inputn-1;i++)
				actin[i][0]=mat[i];
			actin[inputn-1][0]=1;
			temp=0;
			for(int i=0;i<hiddenn;i++){
				temp=0;
				for(int k=0;k<inputn;k++){
						temp+=actin[k][0]*weighthid[k][i];
				}
				acthid[i][0]=actf(temp);
			}
			for(int i=0;i<outputn;i++){
				temp=0;
				for(int k=0;k<hiddenn;k++){
					temp+=acthid[k][0]*weightout[k][i];
				}
				actout[i][0]=actf(temp);
			}
			return actout;
		}
		void backprop(float lrate,float momentum,double *target,int targetlen){
			if (targetlen!=outputn){
				exit(0);
			}
			for(int i=0;i<outputn;i++){
				outputdelta[i][0]=(target[i]-actout[i][0])*dactf(actout[i][0]);
			}
			for(int i=0;i<hiddenn;i++){
				e=0;
				for(int j=0;j<outputn;j++){
					e+=outputdelta[j][0]*weightout[i][j];
				}
				hiddendelta[i][0]=e*dactf(acthid[i][0]);
			}
			for(int i=0;i<hiddenn;i++){
				for(int j=0;j<outputn;j++){
					change=outputdelta[j][0]*acthid[i][0];
					weightout[i][j]+=lrate*change+momentum*wchoutput[i][j];
					wchoutput[i][j]=change;
				}
			}
			for(int i=0;i<inputn;i++){
				for(int j=0;j<hiddenn;j++){
					change=hiddendelta[j][0]*actin[i][0];
					weighthid[i][j]+=lrate*change+momentum*wchhidden[i][j];
					wchhidden[i][j]=change;
				}
			}
		}
		void training(double *mat,double *target,int matlen,int targetlen,int it,float lrate,float momntm){
			for(int i=0;i<it;i++){
				//cout<<it<<endl;
				for(int j=0;j<1;j++){
					this->update(mat,matlen);
					this->backprop(lrate,momntm,target,targetlen);
					//this->debug_out();
				}
				cout<<endl;
			}
		}
};
