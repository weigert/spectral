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

typedef function<bvec(avec)> M;           //Mapping Function
typedef pair<avec,bvec> S;                //Sample Point

/*
================================================================================
                  Array Flattening and Index Management
================================================================================
*/

int flatsize(avec D){
  return D.prod();
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

/*
================================================================================
                    Domain Handling and Transformations
================================================================================
*/

class domain {
public:

  avec a;
  avec b;

  domain(){}
  domain(initializer_list<T> il){
    vector<T> iv(il);
    if(iv.size() < 2) cout<<"CAN'T CONSTRUCT DOMAIN"<<endl;
    a = iv[0]*a.Ones();
    b = iv[1]*b.Ones();
  }

  avec to(avec x, domain d){
    x = (x - a).cwiseQuotient(b - a);     //Normalize between [0, 1]
    return x.cwiseProduct(d.b - d.a)+d.a; //Tranform between [a, b]
  }

  avec from(avec x, domain d){
    x = (x - d.a).cwiseQuotient(d.b - d.a); //Normalize between [0, 1]
    return x.cwiseProduct(b - a)+a;         //Tranform between [a, b]
  }

  vector<S> sample(int N, M mapping){

    vector<S> samples;

    for(int n = 0; n < N; n++){

      S newsample;
      newsample.first << (float)n;
      newsample.first = from(newsample.first, {0, (T)(N-1)});
      newsample.second = mapping(newsample.first);

      samples.push_back(newsample);

    }
    return samples;

  }

  vector<S> sample(int N1, int N2, M mapping){

    vector<S> samples;

    domain d;
    d.a << 0, 0;
    d.b << (T)(N1-1), (T)(N2-1);

    for(int n1 = 0; n1 < N1; n1++)
    for(int n2 = 0; n2 < N2; n2++){

      S newsample;
      newsample.first << (float)n1, (float)n2;
      newsample.first = from(newsample.first, d);
      newsample.second = mapping(newsample.first);

      samples.push_back(newsample);

    }
    return samples;

  }

};

/*
================================================================================
                  Basis Functions Orthogonal / Non-Orthogonal
================================================================================
*/

class basis {
public:

  virtual bvec f(int k, avec x){ return bvec::Zero(); };  //Basis Function

  avec D;                                                //Vector of Basis Function Resolution
  wvec w;                                                 //Basis Weights

  //Boundary and dom Handling
  M inhom = [](avec x){ return bvec::Zero(); };
  domain dom = {-1, 1};

  Matrix<T, Dynamic, Dynamic> A;
  Matrix<T, Dynamic, 1> b;
  JacobiSVD<Matrix<T, Dynamic, Dynamic>> svd;

  basis(){}
  basis(avec _D):D{_D}{}
  basis(avec _D, domain _dom):basis(_D){ dom = _dom; }
  basis(avec _D, domain _dom, M _inhom):basis(_D, _dom){ inhom = _inhom; }

  bvec sample(avec x){
    bvec val = inhom(x);
    for(int k = 0; k < flatsize(D); k++)
      val += w(k)*f(k, x);
    return val;
  }

};

class cosine: public basis {
public:

  cosine(avec _D):basis(_D){}
  cosine(avec _D, domain _dom):basis(_D, _dom){}
  cosine(avec _D, domain _dom, M inhom):basis(_D, _dom, inhom){}

  domain dom = {-PI, PI};

  bvec f(int k, avec x){

    bvec out = bvec::Ones();  //Output Vector
    avec K = itop(k, D);    //Weight Indexing

    x = dom.to(x, {-PI, PI});
    for(unsigned int i = 0; i < DN; i++)
      out(0) *= cos(K(i)*x(i));

    return out;

  }

};


class taylor: public basis {
public:

  taylor(avec _D):basis(_D){}
  taylor(avec _D, domain _dom):basis(_D, _dom){}
  taylor(avec _D, domain _dom, M inhom):basis(_D, _dom, inhom){}

  domain dom = {-1, 1};

  bvec f(int k, avec x){

    bvec out = bvec::Ones();  //Output Vector
    avec K = itop(k, D);    //Weight Indexing

    x = dom.to(x, {-1, 1});
    for(unsigned int i = 0; i < DN; i++)
      out(0) *= pow(x(i), K(i));

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

  void init(const avec _D){
    for(int j = 2; j < _D.maxCoeff(); j++)
    tfuncs.push_back([&](const int k, const T x){
      return 2.0f*x*tfuncs[k-1](k-1, x) - tfuncs[k-2](k-2, x);
    });
  }

public:

  chebyshev(avec _D):basis(_D){ init(_D); }
  chebyshev(avec _D, domain _dom):basis(_D, _dom){ init(_D); }
  chebyshev(avec _D, domain _dom, M inhom):basis(_D, _dom, inhom){ init(_D); }

  domain dom = {-1, 1};

  bvec f(int k, avec x){

    bvec out = bvec::Ones();  //Output Vector
    avec K = itop(k, D);    //Weight Indexing

    x = dom.to(x, {-1, 1});
    for(unsigned int i = 0; i < DN; i++)
      out(0) *= tfuncs[K(i)](K(i), x(i));

    return out;

  }

};

//Fourier is Seperate, Because Complex!

class fourier {
public:

  VectorXcf w;                       //Basis Weights
  avec H;
  avec D;

  domain dom = {-PI, PI};
  M inhom = [](avec x){ return bvec::Zero(); };

  Matrix<complex<T>, Dynamic, Dynamic> A;
  Matrix<complex<T>, Dynamic, 1> b;
  JacobiSVD<Matrix<complex<T>, Dynamic, Dynamic>> svd;

  fourier(avec _D){
    H = _D;
    D = 2.0f*H+avec::Ones();
  }
  fourier(avec _D, domain _dom):fourier(_D){
    dom = _dom;
  }
  fourier(avec _D, domain _dom, M _inhom):fourier(_D, _dom){
    inhom = _inhom;
  }

  Matrix<complex<T>, DM, 1> f(int k, avec x){

    Matrix<complex<T>, DM, 1> out = Matrix<complex<T>, DM, 1>::Ones();
    avec K = itop(k, D);

    x = dom.to(x, {-PI, PI});
    for(unsigned int i = 0; i < DN; i++)
      out(0) *= exp(1if*(K(i)-H(i))*x(i));

    return out;
  }

  bvec sample(avec x){
    bvec val = inhom(x);
    for(int k = 0; k < flatsize(D); k++)
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

  const int K = flatsize(basis.D);       //Size of System
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

  const int K = flatsize(basis.D);       //Size of System
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

  const int K = flatsize(basis.D);       //Size of System
  const int N = samples.size();

  basis.A = basis.A.Zero(N, K);           //Linear System
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
