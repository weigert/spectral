#include <functional>
#include <Eigen/Dense>
#include <complex>

namespace spectral {
using namespace Eigen;
using namespace std;

#define PI 3.14159265f

#define DN 1
#define DM 1

typedef float T;                          //Scalar-Type
typedef Matrix<T, DN, 1> avec;            //Vector-Type (In)
typedef Matrix<T, DM, 1> bvec;            //Vector-Type (Out)
typedef Matrix<T, Dynamic, 1> wvec;       //Vector-Type (Weights)
typedef Matrix<T, DM, DN> mat;            //Matrix-Type

typedef Matrix<T, Dynamic, Dynamic> XMAT; //
typedef Matrix<T, Dynamic, 1> XVEC;       //

typedef pair<T,T> D;                      //Domain
typedef function<bvec(avec)> M;           //Mapping Function
typedef pair<avec,bvec> S;                //Sample Point

/*
================================================================================
                  Basis Functions Orthogonal / Non-Orthogonal
================================================================================
*/

class basis {
public:

  bvec f(int k, avec x){ return bvec::Zero(); };  //Basis Function
  wvec w;                                                 //Basis Weights
  int K = 0;                                           //Number of Basis Functions

  //Boundary and Domain Handling
  M inhom = [](avec x){ return bvec::Zero(); };
  D domain = {-1, 1};

  Matrix<T, Dynamic, Dynamic> A;
  Matrix<T, Dynamic, 1> b;
  JacobiSVD<Matrix<T, Dynamic, Dynamic>> svd;

  basis(){}
  basis(int _K){
    K = _K;
    w = VectorXf::Zero(K);
  }
  basis(int _K, D _domain):basis(_K){
    domain = _domain;
  }

  basis(int _K, D _domain, M _inhom):basis(_K, _domain){
    inhom = _inhom;
  }

  bvec sample(avec x){
    bvec val = inhom(x);
    for(int k = 0; k < K; k++)
      val += w(k)*f(k, x);
    return val;
  }

};

class cosine: public basis {
public:

  cosine(int _K):basis(_K){}
  cosine(int _K, D domain):basis(_K, domain){}
  cosine(int _K, D domain, M inhom):basis(_K, domain, inhom){}

  D domain = {-PI, PI};

  bvec f(int k, avec x){
    bvec out;
    out << cos((T)k*2.0f*PI*x(0)/(domain.second - domain.first));
    return out;
  }

  bvec sample(avec x){
    bvec val = inhom(x);
    for(int k = 0; k < K; k++)
      val += w(k)*f(k, x);
    return val;
  }

};

class taylor: public basis {
public:

  taylor(int _K):basis(_K){}
  taylor(int _K, D domain):basis(_K, domain){}
  taylor(int _K, D domain, M inhom):basis(_K, domain, inhom){}

  bvec f(int k, avec x){
    x(0) = 2.0f*(x(0) - domain.first)/(domain.second - domain.first) - 1.0f;
    bvec out;
    out << pow(x(0), k);
    return out;
  }

  bvec sample(avec x){
    bvec val = inhom(x);
    for(int k = 0; k < K; k++)
      val += w(k)*f(k, x);
    return val;
  }

};

class chebyshev: public basis {
public:

  vector<function<bvec(int, avec)>> tfuncs = {
    [](int k, avec x){ bvec out = bvec::Ones(); return out; },
    [](int k, avec x){ bvec out = x(0)*bvec::Ones(); return out; },
  };

  void init(){
    for(int j = 2; j < K; j++)
    tfuncs.push_back([&](int k, avec x){
      return 2.0f*x(0)*tfuncs[k-1](k-1, x) - tfuncs[k-2](k-2, x);
    });
  }

  chebyshev(int _K):basis(_K){ init(); }
  chebyshev(int _K, D domain):basis(_K, domain){ init(); }
  chebyshev(int _K, D domain, M inhom):basis(_K, domain, inhom){ init(); }

  bvec f(int k, avec x){
    x(0) = 2.0f*(x(0) - domain.first)/(domain.second - domain.first) - 1.0f;
    return tfuncs[k](k, x);
  }

  bvec sample(avec x){
    bvec val = inhom(x);
    for(int k = 0; k < K; k++)
      val += w(k)*f(k, x);
    return val;
  }

};

class fourier: public basis {
public:

  VectorXcf w;                       //Basis Weights
  int HK;

  Matrix<complex<T>, Dynamic, Dynamic> A;
  Matrix<complex<T>, Dynamic, 1> b;
  JacobiSVD<Matrix<complex<T>, Dynamic, Dynamic>> svd;

  fourier(int _K){
    HK = _K;
    K = 2*HK+1;
    w = VectorXcf::Zero(K);
  }
  fourier(int _K, D _domain):fourier(_K){
    domain = _domain;
  }
  fourier(int _K, D _domain, M _inhom):fourier(_K, _domain){
    inhom = _inhom;
  }

  Matrix<complex<float>, DM, 1> f(int k, avec x){
    Matrix<complex<float>, DM, 1> out;
    out << exp(1if*2.0f*PI/(domain.second-domain.first)*(float)(k-HK)*x(0));
    return out;
  }

  bvec sample(avec x){
    bvec val = inhom(x);
    for(int k = 0; k < K; k++)
      val += (w(k)*f(k, x)).real();
    return val;
  }

};

/*
================================================================================
                    Weighted Residual Minimizing Methods
================================================================================
*/

template<typename B>
void leastsquares(B* basis, vector<S>& samples){

  const int K = basis->K;                 //Size of System
  const int N = samples.size();

  basis->A = basis->A.Zero(K, K);         //Linear System
  basis->b = basis->b.Zero(K);

  for(int j = 0; j < K; j++)              //Fill Matrix
  for(int k = 0; k < K; k++)
  for(int n = 0; n < N; n++)
    basis->A(j,k) += basis->f(j, samples[n].first)(0)*basis->f(k, samples[n].first)(0);

  for(int j = 0; j < K; j++)              //Fill Vector
  for(int n = 0; n < N; n++)
    basis->b(j) += basis->f(j, samples[n].first)(0)*(samples[n].second - basis->inhom(samples[n].first))(0);

  basis->svd.compute(basis->A, ComputeThinU | ComputeThinV);
  basis->w = basis->svd.solve(basis->b);  //Solve

}

template<typename T>
void galerkin(T* basis, vector<S>& samples){

  const int K = basis->K;                 //Size of System
  const int N = samples.size();

  basis->A = basis->A.Zero(K, K);         //Linear System
  basis->b = basis->b.Zero(K);

  for(int j = 0; j < K; j++)              //Fill Matrix
  for(int k = 0; k < K; k++)
  for(int n = 0; n < N; n++)
    basis->A(j,k) += basis->f(j, samples[n].first)(0)*basis->f(k, samples[n].first)(0);

  for(int j = 0; j < K; j++)              //Fill Vector
  for(int n = 0; n < N; n++)
    basis->b(j) += basis->f(j, samples[n].first)(0)*(samples[n].second - basis->inhom(samples[n].first))(0);

  basis->svd.compute(basis->A, ComputeThinU | ComputeThinV);
  basis->w = basis->svd.solve(basis->b);  //Solve

}

template<typename T>
void collocation(T* basis, vector<S>& samples){

  const int K = basis->K;                 //Size of System
  const int N = samples.size();

  basis->A = basis->A.Zero(N, K);         //Linear System
  basis->b = basis->b.Zero(N);

  for(int n = 0; n < N; n++)              //Fill Matrix
  for(int k = 0; k < K; k++)
    basis->A(n,k) += basis->f(k, samples[n].first)(0);

  for(int n = 0; n < N; n++)              //Fill Vector
    basis->b(n) += samples[n].second(0) - basis->inhom(samples[n].first)(0);

  basis->svd.compute(basis->A, ComputeThinU | ComputeThinV);
  basis->w = basis->svd.solve(basis->b);  //Solve

}

/*
================================================================================
                            Utility Functions
================================================================================
*/

vector<S> sample(int N, D domain, M mapping){
  vector<S> samples;
  for(int n = 0; n < N; n++){
    S newsample;
    newsample.first(0) = domain.first + (T)n/(T)(N-1)*(domain.second - domain.first);
    newsample.second = mapping(newsample.first);
    samples.push_back(newsample);
  }
  return samples;
}

template<typename B>
T err(B* basis, vector<S>& samples){
  T msqerr = 0.0f;
  for(auto& s: samples){
    bvec ny = basis->sample(s.first);
    msqerr += (ny-s.second).dot(ny-s.second)/(float)samples.size();
  }
  return msqerr;
}


}
