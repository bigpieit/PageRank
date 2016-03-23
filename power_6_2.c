#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "mkl.h"
/*  Matrix Generate  */
void matcolgenrt(double *a,int row,int col, double val){
  int i,j;
  for (i = 0; i < row; i++ ){
    for (j = 0; j < col; j++){
      a[i+j*row]=val;
    }
  }
}
/*   Matrix Print    */
void matcolprint(double *a,int row,int col){
  int i,j;
  for (i = 0; i < row; i++ ){
    for (j = 0; j < col; j++){
      printf("%lf ",a[i+j*row]);
    }
    printf("\n");
  }
}
/*  Matrix Tranpose */
void mattranspose(double *X,int row,int col,double *XT){
  int i,j;
  for (i=0;i<row;i++){
    for (j=0;j<col;j++){
      XT[i*col+j]=X[j*row+i];
    }
  }
}
/*   Matrix Combination   */
void matadd(double *A,double *S,double *M,int row,int col, double cof){
  int i;
  int sum=row*col;
  for (i=0;i<sum;i++){
    M[i]=(1-cof)*A[i]+cof*S[i];
  }
}
/*   Matrix Combination   */
void matshift(double *A,int row, int col,double sigma,double *B){
  int i,j;
  for (i=0;i<row;i++){
    for (j=0;j<col;j++){
      if (i==j){
        B[i*col+j]=A[i*col+j]-sigma;
      }
    }
  }
}


/*   Vector Print   */
void vectprint(double *a,int n){
  int i;
  for (i=0; i < n; i++){
    printf("%lf\n",a[i]);
  }
}
/*   Vector Exchange   */
void vectexchange(double *X,double *XTEMP,int size){
  int i;
  for (i=0;i<size;i++){
    X[i]=XTEMP[i];
  }
}
/*   Vector Check   */
int vectcheck(double *X,double *XTEMP,int size,double criterion){
  int i;
  double diff;
  for (i=0;i<size;i++){
    diff=fabs(X[i]-XTEMP[i]);
    if(diff>criterion){
      return 0;
    }
  }
  return 1;
}

void vectnormalize(double *X,int size,double norm2){
  int i;
  for (i=0;i<size;i++){
    X[i]=X[i]/norm2;
  }
}
void vectnormalize2(double *X,int size){
  int i;
  double max=fabs(X[0]);
  for (i=0;i<size;i++){
    if (fabs(X[i])>max){
      max=fabs(X[i]);
    }
  }
  for (i=0;i<size;i++){
    X[i]=X[i]/max;
  }
}
void vectnormalize1(double *X,int size){
  int i;
  double sum=0.0;
  for (i=0;i<size;i++){
    sum+=X[i];
  }
  for (i=0;i<size;i++){
    X[i]=X[i]/sum;
  }
}
double vectproduct(double *X,double *Y,int size){
  int i;
  double sum=0.0;
  for (i=0;i<size;i++){
    sum+=X[i]*Y[i];
  }
  return sum;
}
/* Vector Addition R=Y+u X*/
void vectadd(double *Y,double *X,double *R,double mu,int size){
  int i;
  for (i=0;i<size;i++){
    R[i]=Y[i]+X[i]*mu;
  } 
}
/* Vector Norm 2 */
double vectnorm2(double *X,int size){
  int i;
  double sum=0.0;
  for (i=0;i<size;i++){
    sum+=X[i]*X[i];
  }
  double val=sqrt(sum);
  return val;
}

double eigenval(double *A,double *X,int size){
  double eigen1,eigen2;
  double *XTEMP;
  size_t size2 =  size*sizeof(double);
  XTEMP = (double *) malloc(size2);
  int incx=1,incy=1;
  double alpha=1.0,beta=0.0;

  eigen1=vectproduct(X,X,size);
  cblas_dgemv (CblasColMajor, CblasNoTrans, size, size, alpha, A, size, X, incx, beta, XTEMP, incy);
  eigen2=vectproduct(X,XTEMP,size);
  return eigen2/eigen1;
}

void rankswap(double (*rank)[2],int size){
  double rtemp0,rtemp1;
  int i,j;
  for (i=1;i<size;i++){
    for (j=i;j>0;j--){
      if (rank[j][0]>rank[j-1][0]){
        rtemp0=rank[j][0];
        rtemp1=rank[j][1];
        rank[j][0]=rank[j-1][0];
        rank[j][1]=rank[j-1][1];
        rank[j-1][0]=rtemp0;
        rank[j-1][1]=rtemp1;
      }
    }     
  }
}
void rankprint(double (*rank)[2],int size){
  int i;
  for (i=0;i<size;i++){
    printf(" %d     %lf    %d\n",(int)(rank[i][1]),rank[i][0],i+1);
  }
}


/***********  Main   Code    ***********/
int main(int argc, char **argv){
  /*     Input     */
  char *filename1 = argv[1];
  char *filename2 = argv[2];
  double sigma = atof(argv[3]);
 
  /*   Read File for A  */ 
  double *A;
  FILE *file1 = fopen(filename1,"r");
  int nnode,i,j;
  int nidx,idx;
  fscanf(file1,"%d", &nnode);
  size_t size = nnode*nnode*sizeof(double);
  A = (double *) malloc(size);
  for (i=0;i<nnode;i++){
    fscanf(file1,"%i",&nidx);
    for (j=0;j<nidx;j++){
      fscanf(file1,"%i",&idx);
      A[i*nnode+idx-1]=1.0/nidx;
    }
  }
  fclose(file1);
  printf("The Web Matrix A\n");
  matcolprint(A,nnode,nnode); 

/*   Read File for Initial X  */ 
  double *X,*XTEMP;
  size_t size2 =  nnode*sizeof(double);
  FILE *file2 = fopen(filename2,"r");
  X = (double *) malloc(size2);
  XTEMP = (double *) malloc(size2);
  double inter; 
  for (i=0;i<nnode;i++){
    fscanf(file2,"%lf",&inter);
    X[i]=inter;
    XTEMP[i]=0.0;
  }  
  fclose(file2);

/***************   Methods  ***************/
  /*  Power Method AX=X  */
  printf("The Initial Scores of X\n");
  vectprint(X,nnode);
  double norm2=vectnorm2(X,nnode);
  vectnormalize(X,nnode,norm2);
  double *R;
  R = (double *) malloc(size2); 
  int incx=1,incy=1;
  double alpha=1.0,beta=0.0;
  cblas_dgemv (CblasColMajor, CblasNoTrans, nnode, nnode, alpha, A, nnode, X, incx, beta, XTEMP, incy);
  double mu=vectproduct(X,XTEMP,nnode);
  vectadd(XTEMP,X,R,-1*mu,nnode);
  double flag=vectnorm2(R,nnode);
  vectadd(XTEMP,X,X,-1*sigma,nnode);

  double crt=0.0001;
  int count=1;
  while (flag>crt){
    norm2=vectnorm2(X,nnode);
    vectnormalize(X,nnode,norm2);
    cblas_dgemv (CblasColMajor, CblasNoTrans, nnode, nnode, alpha, A, nnode, X, incx, beta, XTEMP, incy);
    mu=vectproduct(X,XTEMP,nnode);
    vectadd(XTEMP,X,R,-1*mu,nnode);
    flag=vectnorm2(R,nnode);
    vectadd(XTEMP,X,X,-1*sigma,nnode);
    count++;    
  }
  printf("Final Scores of  X\n");
  vectprint(X,nnode);
  /*  Rank Final X  */
  double rank[nnode][2];
  for (i=0;i<nnode;i++){
    rank[i][0]=X[i];
    rank[i][1]=i+1;
  }
  rankswap(rank,nnode);
  printf("Node     Score    Rank\n");
  rankprint(rank,nnode);
  
  printf("X\n");
  vectprint(X,nnode);
  printf("XTEMP\n");
  vectprint(XTEMP,nnode);
  
  double eigen;
  eigen=eigenval(A,X,nnode);
  printf("Eigen Value of X\n");
  printf("%lf\n",eigen);
  printf("sigma of A - I * sigma\n");
  printf("%lf\n",sigma);
  printf("count\n");
  printf("%d\n",count);
  return 0;
}
