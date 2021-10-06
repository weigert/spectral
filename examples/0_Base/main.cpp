#include "../../spectral.h"
#include <iostream>

int main( int argc, char* args[] ) {
using namespace std;

  /*

    First, we define a set of samples.
    We generate 50 equidistant points in a domain  {-2, 2} using a lambda expression.
    Here, we are approximating a gaussian.

  */

  vector<glm::vec2> samples;
  samples = spectral::sample(50, {-2, 2}, [](float x){
    return exp(-2.0f*x*x);
  //  if(x > 0) return 1.0f;
  //  else return 0.0f;
  });

  /*

    Next, we construct a basis set representation "solution" of our data.
    We can pick one of many base classes to represent the solution.
    We define the number of basis functions, the domain in which we solve and
    additionally, we define our imposed inhomogeneity function.

  */

  spectral::fourier solution(10, {-2, 2}, [](float x){
    return 0.5f;
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
