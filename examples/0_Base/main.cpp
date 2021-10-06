#include <iostream>
#include "../../spectral.h"

int main( int argc, char* args[] ) {
using namespace std;

  /*

    First, we define a set of samples.
    We generate 50 equidistant points in a domain  {-2, 2} using a lambda expression.
    Here, we are approximating a gaussian.

  */

  vector<spectral::S> samples;
  samples = spectral::sample(50, {-2, 2}, [](spectral::avec x){
    x(0) = exp(-2.0f*x.dot(x));
    return x;
  });

  /*

    Next, we construct a basis set representation "solution" of our data.
    We can pick one of many base classes to represent the solution.
    We define the number of basis functions, the domain in which we solve and
    additionally, we define our imposed inhomogeneity function.

  */

  spectral::cosine solution(16, {-2, 2}, [](spectral::avec x){
    return spectral::bvec::Zero();
  });

  /*

    Finally, we choose our specific method of weighted residuals and solve
    for the weights of our basis set representation using our samples.
    We then output the mean squared error of our fit.

  */

  spectral::leastsquares(&solution, samples);
  cout<<"MSQErr: "<<spectral::err(&solution, samples)<<endl;

	return 0;

}
