#include <Eigen/Dense>
#include <functional>
#include <complex>

namespace spectral {
using namespace Eigen;
using namespace std;

#define PI 3.14159265f

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
                    Domain Handling and Array Flattening
================================================================================
*/

int flatsize(avec DK){
  return DK.prod();
}

int ptoi(avec& pos, avec& dim){ //Convert Vector to Index
  int i = 0;
  for(unsigned int j = 0; j < DN; j++){
    int F = 1;
    for(unsigned int k = DN-1; k > j; k--)
      F *= dim(k);
    i += F*pos(j);
  }
  return i;
}

avec itop(int i, avec& dim){     //Convert Index to Vector
  avec n;
  for(unsigned int j = 0; j < DN; j++){
    int F = 1;
    for(unsigned int k = DN-1; k > j; k--)
      F *= dim(k);
    n(j) = (int)(i/F);
    i -= F*n(j);
  }
  return n;
}

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


/*
================================================================================
                  Basis Functions Orthogonal / Non-Orthogonal
================================================================================
*/

class basis {
public:

  virtual bvec f(int k, avec x){ return bvec::Zero(); };  //Basis Function

  avec DK;                                                //Vector of Basis Function Resolution
  int K;                                                  //Number of Basis Functions (Total)
  wvec w;                                                 //Basis Weights

  //Boundary and Domain Handling
  M inhom = [](avec x){ return bvec::Zero(); };
  D domain = {-1, 1};

  Matrix<T, Dynamic, Dynamic> A;
  Matrix<T, Dynamic, 1> b;
  JacobiSVD<Matrix<T, Dynamic, Dynamic>> svd;

  basis(){}
  basis(avec _DK):DK{_DK},K{flatsize(_DK)}{}
  basis(avec _DK, D _domain):basis(_DK){
    domain = _domain;
  }
  basis(avec _DK, D _domain, M _inhom):basis(_DK, _domain){
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

  cosine(avec _DK):basis(_DK){}
  cosine(avec _DK, D domain):basis(_DK, domain){}
  cosine(avec _DK, D domain, M inhom):basis(_DK, domain, inhom){}

  D domain = {-PI, PI};

  bvec f(int k, avec x){

    bvec out = bvec::Ones();  //Output Vector
    avec _K = itop(k, DK);    //Weight Indexing

    for(unsigned int i = 0; i < DN; i++)
      out(0) *= cos(_K(i)*x(i)*2.0f*PI/(domain.second - domain.first));

    return out;

  }

};


class taylor: public basis {
public:

  taylor(avec _K):basis(_K){}
  taylor(avec _K, D domain):basis(_K, domain){}
  taylor(avec _K, D domain, M inhom):basis(_K, domain, inhom){}

  bvec f(int k, avec x){

    bvec out = bvec::Ones();  //Output Vector
    avec _K = itop(k, DK);    //Weight Indexing

    for(unsigned int i = 0; i < DN; i++){
      T z = 2.0f*(x(i) - domain.first)/(domain.second - domain.first) - 1.0f;
      out(0) *= pow(z, _K(i));
    }

    return out;

  }

};

class chebyshev: public basis {
private:

  //Pre-Compute Chebyshev Polynomials Recursively

  vector<function<T(const int, const T)>> tfuncs = {
    [](const int k, const T x){ T out = 1.0f; return out; },
    [](const int k, const T x){ T out = x; return out; },
  };

  void init(const avec _K){
    for(int j = 2; j < _K.maxCoeff(); j++)
    tfuncs.push_back([&](const int k, const T x){
      return 2.0f*x*tfuncs[k-1](k-1, x) - tfuncs[k-2](k-2, x);
    });
  }

public:

  chebyshev(avec _K):basis(_K){ init(_K); }
  chebyshev(avec _K, D domain):basis(_K, domain){ init(_K); }
  chebyshev(avec _K, D domain, M inhom):basis(_K, domain, inhom){ init(_K); }

  bvec f(int k, avec x){

    bvec out = bvec::Ones();  //Output Vector
    avec _K = itop(k, DK);    //Weight Indexing

    for(unsigned int i = 0; i < DN; i++){
      T z = 2.0f*(x(i) - domain.first)/(domain.second - domain.first) - 1.0f;
      out(0) *= tfuncs[_K(i)](_K(i), z);
    }

    return out;

  }

};

//Fourier is Seperate, Because Complex!

class fourier {
public:

  VectorXcf w;                       //Basis Weights
  avec HK;
  avec DK;
  int K;
  D domain = {-PI, PI};
  M inhom;

  Matrix<complex<T>, Dynamic, Dynamic> A;
  Matrix<complex<T>, Dynamic, 1> b;
  JacobiSVD<Matrix<complex<T>, Dynamic, Dynamic>> svd;

  fourier(avec _DK){
    HK = _DK;
    DK = 2.0f*HK+avec::Ones();
    K = flatsize(DK);      //Shift to 2D
  }
  fourier(avec _DK, D _domain):fourier(_DK){
    domain = _domain;
  }
  fourier(avec _DK, D _domain, M _inhom):fourier(_DK, _domain){
    inhom = _inhom;
  }

  Matrix<complex<T>, DM, 1> f(int k, avec x){

    Matrix<complex<T>, DM, 1> out = Matrix<complex<T>, DM, 1>::Ones();
    avec _K = itop(k, DK);

    for(unsigned int i = 0; i < DN; i++)
      out(0) *= exp(1if*2.0f*PI/(domain.second-domain.first)*(_K(i)-HK(i))*x(i));

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
void leastsquares(B& basis, const vector<S>& samples){

  const int K = basis.K;                 //Size of System
  const int N = samples.size();

  basis.A = basis.A.Zero(K, K);         //Linear System
  basis.b = basis.b.Zero(K);

  for(int j = 0; j < K; j++)              //Fill Matrix
  for(int k = 0; k < K; k++)
  for(int n = 0; n < N; n++)
    basis.A(j,k) += basis.f(j, samples[n].first)(0)*basis.f(k, samples[n].first)(0);

  for(int j = 0; j < K; j++)              //Fill Vector
  for(int n = 0; n < N; n++)
    basis.b(j) += basis.f(j, samples[n].first)(0)*(samples[n].second - basis.inhom(samples[n].first))(0);

  basis.svd.compute(basis.A, ComputeThinU | ComputeThinV);
  basis.w = basis.svd.solve(basis.b);  //Solve

}

template<typename B>
void galerkin(B& basis, const vector<S>& samples){

  const int K = basis.K;                 //Size of System
  const int N = samples.size();

  basis.A = basis.A.Zero(K, K);         //Linear System
  basis.b = basis.b.Zero(K);

  for(int j = 0; j < K; j++)              //Fill Matrix
  for(int k = 0; k < K; k++)
  for(int n = 0; n < N; n++)
    basis.A(j,k) += basis.f(j, samples[n].first)(0)*basis.f(k, samples[n].first)(0);

  for(int j = 0; j < K; j++)              //Fill Vector
  for(int n = 0; n < N; n++)
    basis.b(j) += basis.f(j, samples[n].first)(0)*(samples[n].second - basis.inhom(samples[n].first))(0);

  basis.svd.compute(basis.A, ComputeThinU | ComputeThinV);
  basis.w = basis.svd.solve(basis.b);  //Solve

}

template<typename B>
void collocation(B& basis, const vector<S>& samples){

  const int K = basis.K;                 //Size of System
  const int N = samples.size();

  basis.A = basis.A.Zero(N, K);         //Linear System
  basis.b = basis.b.Zero(N);

  for(int n = 0; n < N; n++)              //Fill Matrix
  for(int k = 0; k < K; k++)
    basis.A(n,k) += basis.f(k, samples[n].first)(0);

  for(int n = 0; n < N; n++)              //Fill Vector
    basis.b(n) += samples[n].second(0) - basis.inhom(samples[n].first)(0);

  basis.svd.compute(basis.A, ComputeThinU | ComputeThinV);
  basis.w = basis.svd.solve(basis.b);  //Solve

}

template<typename B>
T err(B& basis, const vector<S>& samples){
  T msqerr = 0;
  for(auto& s: samples){
    bvec ny = basis.sample(s.first);
    msqerr += (ny-s.second).dot(ny-s.second)/(T)samples.size();
  }
  return msqerr;
}

}
