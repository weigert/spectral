#include <functional>
#include <Eigen/Dense>
#include <complex>

namespace spectral {

using namespace std;
using namespace glm;
using namespace Eigen;

#define PI 3.14159265f

/*
================================================================================
                  Basis Functions Orthogonal / Non-Orthogonal
================================================================================
*/

class basis {
public:

  float f(int k, float x);   //Basis Function
  float df(int k, float x);  //Gradient of Basis Function
  VectorXf w;                       //Basis Weights
  size_t K = 0;                     //Number of Basis Functions

  //Boundary and Domain Handling
  function<float(float)> inhom = [](float x){ return 0.0f; };
  pair<float, float> domain;

  basis(){}
  basis(size_t _K){
    K = _K;
    w = VectorXf::Zero(K);
  }

  basis(size_t _K, pair<float, float> _domain):basis(_K){
    domain = _domain;
  }

  basis(size_t _K, pair<float, float> _domain, function<float(float)> _inhom):basis(_K, _domain){
    inhom = _inhom;
  }

  float sample(float x);

};

class cosine: public basis {
public:

  cosine(size_t K):basis(K){}
  cosine(size_t K, pair<float, float> _domain):cosine(K){
    domain = _domain;
  }
  cosine(size_t K, pair<float, float> _domain, function<float(float)> _inhom):cosine(K, _domain){
    inhom = _inhom;
  }

  pair<float, float> domain = {-PI, PI};

  //Define Base Functions
  float f(int k, float x){
    return cos((float)k*2.0f*PI*x/(domain.second - domain.first));
  }

  float df(int k, float x){
    return -(float)k*sin((float)k*x);
  }

  float sample(float x){
    float val = inhom(x);
    for(int k = 0; k < K; k++)
      val += w(k)*f(k, x);
    return val;
  }

};

class polynomial: public basis {
public:

  polynomial(size_t K):basis(K){}
  polynomial(size_t K, pair<float, float> _domain):polynomial(K){
    domain = _domain;
  }
  polynomial(size_t K, pair<float, float> _domain, function<float(float)> _inhom):polynomial(K, _domain){
    inhom = _inhom;
  }
  float f(int k, float x){
    x = 2.0f*(x - domain.first)/(domain.second - domain.first) - 1.0f;
    return pow(x, k);
  }
  float df(int k, float x){
    return k*pow(x, k-1);
  }

  float sample(float x){
    float val = inhom(x);
    for(int k = 0; k < K; k++)
      val += w(k)*f(k, x);
    return val;
  }

};


class fourier: public basis {
public:

  VectorXcf w;                       //Basis Weights

  fourier(size_t _K){
    K = _K;
    w = VectorXcf::Zero(2*K+1);
  }
  fourier(size_t _K, pair<float, float> _domain):basis(_K){
    domain = _domain;
  }
  fourier(size_t _K, pair<float, float> _domain, function<float(float)> _inhom):basis(_K, _domain){
    inhom = _inhom;
  }

  complex<float> f(int k, float x){
    return exp(1if*2.0f*PI/(domain.second-domain.first)*(float)k*x);
  }
  complex<float> df(int k, float x){
    return exp(1if*2.0f*PI/(domain.second-domain.first)*(float)k*x);
  }

  float sample(float x){
    float val = inhom(x);
    for(int k = 0; k < 2*K+1; k++){
      val += real(w(k)*f(k-K, x));
    }
    return val;
  }

};


class chebyshev: public basis {
public:

  vector<function<float(int, float)>> tfuncs = {
    [](int k, float x){ return 1.0f; },
    [](int k, float x){ return x; },
  };

  chebyshev(size_t K):basis(K){
    for(size_t j = 2; j < K; j++){
      tfuncs.push_back([&](int k, float x){
        return 2.0f*x*tfuncs[k-1](k-1, x) - tfuncs[k-2](k-2, x);
      });
    }
  }

  chebyshev(size_t K, pair<float, float> _domain):chebyshev(K){
    domain = _domain;
  }
  chebyshev(size_t K, pair<float, float> _domain, function<float(float)> _inhom):chebyshev(K, _domain){
    inhom = _inhom;
  }

  float f(int k, float x){
    x = 2.0f*(x - domain.first)/(domain.second - domain.first) - 1.0f;
    return tfuncs[k](k, x);
  }
  float df(int k, float x){
    return tfuncs[k](k, x);
  }

  float sample(float x){
    float val = inhom(x);
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

template<typename T>
void leastsquares(T* basis, vector<vec2>& samples){

  //Size of System
  const size_t K = basis->K;
  const size_t N = samples.size();

  //Fill Dense Linear System
  MatrixXf A = MatrixXf::Zero(K, K);
  VectorXf b = VectorXf::Zero(K);

  for(size_t j = 0; j < K; j++)
  for(size_t k = 0; k < K; k++)
  for(size_t n = 0; n < N; n++)
    A(j,k) += basis->f(j, samples[n].x)*basis->f(k, samples[n].x);

  for(size_t j = 0; j < K; j++)
  for(size_t n = 0; n < N; n++)
    b(j) += basis->f(j, samples[n].x)*(samples[n].y - basis->inhom(samples[n].x));

  //Solve Linear System
  JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
  basis->w = svd.solve(b);

}

template<>
void leastsquares<spectral::fourier>(spectral::fourier* basis, vector<vec2>& samples){

  //Size of System
  const size_t K = basis->K;
  const size_t N = samples.size();

  //Fill Dense Linear System
  MatrixXcf A = MatrixXcf::Zero(2*K+1, 2*K+1);
  VectorXcf b = VectorXcf::Zero(2*K+1);

  for(size_t j = 0; j < 2*K+1; j++)
  for(size_t k = 0; k < 2*K+1; k++)
  for(size_t n = 0; n < N; n++)
    A(j,k) += basis->f(j-K, samples[n].x)*basis->f(k-K, samples[n].x);

  for(size_t j = 0; j < 2*K+1; j++)
  for(size_t n = 0; n < N; n++)
    b(j) += basis->f(j-K, samples[n].x)*(samples[n].y - basis->inhom(samples[n].x));

  //Solve Linear System
  JacobiSVD<MatrixXcf> svd(A, ComputeThinU | ComputeThinV);
  basis->w = svd.solve(b);

}

template<typename T>
void galerkin(T* basis, vector<vec2>& samples){

  //Size of System
  const size_t K = basis->K;
  const size_t N = samples.size();

  //Fill Dense Linear System
  MatrixXf A = MatrixXf::Zero(K, K);
  VectorXf b = VectorXf::Zero(K);

  for(size_t j = 0; j < K; j++)
  for(size_t k = 0; k < K; k++)
  for(size_t n = 0; n < N; n++)
    A(j,k) += basis->f(j, samples[n].x)*basis->f(k, samples[n].x);

  for(size_t j = 0; j < K; j++)
  for(size_t n = 0; n < N; n++)
    b(j) += basis->f(j, samples[n].x)*(samples[n].y - basis->inhom(samples[n].x));

  //Solve Linear System
  JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
  basis->w = svd.solve(b);

}

template<typename T>
void collocation(T* basis, vector<vec2>& samples){

  //Size of System
  const size_t K = basis->K;
  const size_t N = samples.size();

  //Fill Dense Linear System
  MatrixXf A = MatrixXf::Zero(N, K);
  VectorXf b = VectorXf::Zero(N);

  for(size_t n = 0; n < N; n++)
  for(size_t k = 0; k < K; k++)
    A(n,k) += basis->f(k, samples[n].x);

  for(size_t n = 0; n < N; n++)
    b(n) += samples[n].y - basis->inhom(samples[n].x);

  JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
  basis->w = svd.solve(b);

}

/*
================================================================================
                              Utility Methods
================================================================================
*/

vector<vec2> sample(size_t N, pair<float, float> domain, function<float(float)> mapping){
  vector<vec2> samples;
  for(size_t n = 0; n < N; n++){
    vec2 newsample;
    newsample.x = domain.first + (float)n/(float)(N-1)*(domain.second - domain.first);
    newsample.y = mapping(newsample.x);
    samples.push_back(newsample);
  }
  return samples;
}

}
