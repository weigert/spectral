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
  size_t K = 0;                                           //Number of Basis Functions

  //Boundary and Domain Handling
  M inhom = [](avec x){ return bvec::Zero(); };
  D domain = {-1, 1};

  basis(){}
  basis(size_t _K){
    K = _K;
    w = VectorXf::Zero(K);
  }
  basis(size_t _K, D _domain):basis(_K){
    domain = _domain;
  }

  basis(size_t _K, D _domain, M _inhom):basis(_K, _domain){
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

  cosine(size_t _K):basis(_K){}
  cosine(size_t _K, D domain):basis(_K, domain){}
  cosine(size_t _K, D domain, M inhom):basis(_K, domain, inhom){}

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

  taylor(size_t _K):basis(_K){}
  taylor(size_t _K, D domain):basis(_K, domain){}
  taylor(size_t _K, D domain, M inhom):basis(_K, domain, inhom){}

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
    for(size_t j = 2; j < K; j++)
    tfuncs.push_back([&](int k, avec x){
      return 2.0f*x(0)*tfuncs[k-1](k-1, x) - tfuncs[k-2](k-2, x);
    });
  }

  chebyshev(size_t _K):basis(_K){ init(); }
  chebyshev(size_t _K, D domain):basis(_K, domain){ init(); }
  chebyshev(size_t _K, D domain, M inhom):basis(_K, domain, inhom){ init(); }

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

/*
================================================================================
                    Weighted Residual Minimizing Methods
================================================================================
*/

template<typename B>
void leastsquares(B* basis, vector<S>& samples){

  const size_t K = basis->K;          //Size of System
  const size_t N = samples.size();

  XMAT A = XMAT::Zero(K, K);          //Linear System
  XVEC b = XVEC::Zero(K);

  for(size_t j = 0; j < K; j++)       //Fill Matrix
  for(size_t k = 0; k < K; k++)
  for(size_t n = 0; n < N; n++)
    A(j,k) += basis->f(j, samples[n].first)*basis->f(k, samples[n].first);

  for(size_t j = 0; j < K; j++)       //Fill Vector
  for(size_t n = 0; n < N; n++)
    b(j) += basis->f(j, samples[n].first)*(samples[n].second - basis->inhom(samples[n].first));

  JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
  basis->w = svd.solve(b);            //Solve

}

template<typename T>
void galerkin(T* basis, vector<S>& samples){

  const size_t K = basis->K;          //Size of System
  const size_t N = samples.size();

  XMAT A = XMAT::Zero(K, K);          //Linear System
  XVEC b = XVEC::Zero(K);

  for(size_t j = 0; j < K; j++)       //Fill Matrix
  for(size_t k = 0; k < K; k++)
  for(size_t n = 0; n < N; n++)
    A(j,k) += basis->f(j, samples[n].first)*basis->f(k, samples[n].first);

  for(size_t j = 0; j < K; j++)       //Fill Vector
  for(size_t n = 0; n < N; n++)
    b(j) += basis->f(j, samples[n].first)*(samples[n].second - basis->inhom(samples[n].first));

  JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
  basis->w = svd.solve(b);            //Solve

}

template<typename T>
void collocation(T* basis, vector<S>& samples){

  const size_t K = basis->K;          //Size of System
  const size_t N = samples.size();

  XMAT A = XMAT::Zero(N, K);          //Linear System
  XVEC b = XVEC::Zero(N);

  for(size_t n = 0; n < N; n++)       //Fill Matrix
  for(size_t k = 0; k < K; k++)
    A(n,k) += basis->f(k, samples[n].first)(0);

  for(size_t n = 0; n < N; n++)       //Fill Vector
    b(n) += samples[n].second(0) - basis->inhom(samples[n].first)(0);

  JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
  basis->w = svd.solve(b);            //Solve

}

/*
================================================================================
                    Specializations for Complex Functions
================================================================================
*/

class fourier: public basis {
public:

  VectorXcf w;                       //Basis Weights

  fourier(size_t _K){
    K = _K;
    w = VectorXcf::Zero(2*K+1);
  }
  fourier(size_t _K, D _domain):fourier(_K){
    domain = _domain;
  }
  fourier(size_t _K, D _domain, M _inhom):fourier(_K, _domain){
    inhom = _inhom;
  }

  complex<float> f(int k, avec x){
    return exp(1if*2.0f*PI/(domain.second-domain.first)*(float)k*x(0));
  }

  bvec sample(avec x){
    bvec val = inhom(x);
    for(int k = 0; k < 2*K+1; k++)
      val(0) += (w(k)*f(k-K, x)).real();
    return val;
  }

};

template<>
void leastsquares<spectral::fourier>(spectral::fourier* basis, vector<S>& samples){

  //Size of System
  const size_t K = basis->K;
  const size_t N = samples.size();

  //Fill Dense Linear System
  MatrixXcf A = MatrixXcf::Zero(2*K+1, 2*K+1);
  VectorXcf b = VectorXcf::Zero(2*K+1);

  for(size_t j = 0; j < 2*K+1; j++)
  for(size_t k = 0; k < 2*K+1; k++)
  for(size_t n = 0; n < N; n++)
    A(j,k) += basis->f(j-K, samples[n].first)*basis->f(k-K, samples[n].first);

  for(size_t j = 0; j < 2*K+1; j++)
  for(size_t n = 0; n < N; n++)
    b(j) += basis->f(j-K, samples[n].first)*(samples[n].second(0) - basis->inhom(samples[n].first)(0));

  //Solve Linear System
  JacobiSVD<MatrixXcf> svd(A, ComputeThinU | ComputeThinV);
  basis->w = svd.solve(b);

}

template<>
void collocation(spectral::fourier* basis, vector<S>& samples){

  const size_t K = basis->K;          //Size of System
  const size_t N = samples.size();

  MatrixXcf A = MatrixXcf::Zero(N, 2*K+1);          //Linear System
  VectorXcf b = VectorXcf::Zero(N);

  for(size_t n = 0; n < N; n++)       //Fill Matrix
  for(size_t k = 0; k < 2*K+1; k++)
    A(n,k) += basis->f(k-K, samples[n].first);

  for(size_t n = 0; n < N; n++)       //Fill Vector
    b(n) += samples[n].second(0) - basis->inhom(samples[n].first)(0);

  JacobiSVD<MatrixXcf> svd(A, ComputeThinU | ComputeThinV);
  basis->w = svd.solve(b);            //Solve

}

/*
================================================================================
                            Utility Functions
================================================================================
*/

vector<S> sample(size_t N, D domain, M mapping){
  vector<S> samples;
  for(size_t n = 0; n < N; n++){
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
