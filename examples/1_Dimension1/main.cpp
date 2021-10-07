#include <TinyEngine/TinyEngine>

#define DN 1
#define DM 1

#include "../../spectral.h"

int main( int argc, char* args[] ) {
using namespace std;

  //Define our Input Positions
  vector<spectral::S> samples;
  samples = spectral::sample(50, {-2, 2}, [](spectral::avec x){
    //x(0) = exp(-2.0f*x.dot(x));
    x(0) = (x(0) > 0)?1.0f:0.0f;//exp(-2.0f*x.dot(x));
    return x;
  });

  //Construct a Cosine Representation of Samples with Inhomogeneity
  spectral::fourier solution(25.0f*spectral::avec::Ones(), {-2, 2}, [](spectral::avec x){
    return spectral::bvec::Zero();
  });

  //Perform a Fit with a Weighted Resiual Method
  spectral::collocation(solution, samples);
  cout<<"MSQErr: "<<spectral::err(solution, samples)<<endl;

  //Sample the Solution
  vector<spectral::S> approximations;
  approximations = spectral::sample(500, {-2, 2}, [&](spectral::avec x){
    return solution.sample(x);
  });




	//Initialize a Window
  Tiny::view.pointSize = 4.0f;
	Tiny::window("Example Window", 600, 400);

  Shader lineshader({"shader/line.vs", "shader/line.fs"}, {"in_Quad"});

  Buffer samplebuf;
  samplebuf.fill(samples);

  Buffer approximationbuf;
  approximationbuf.fill(approximations);

  Model samplemesh({"in_Quad"});
  samplemesh.bind<pair<float,float>>("in_Quad", &samplebuf);  //Update particle system
  samplemesh.SIZE = samples.size();

  Model approximationmesh({"in_Quad"});
  approximationmesh.bind<pair<float,float>>("in_Quad", &approximationbuf);  //Update particle system
  approximationmesh.SIZE = approximations.size();





	//Add the Event Handler
	Tiny::event.handler = [&](){
	};

	//Set up an ImGUI Interface here
	Tiny::view.interface = [&](){
	};

	//Define the rendering pipeline
	Tiny::view.pipeline = [&](){

		Tiny::view.target(glm::vec3(1));	//Clear Screen to white

    lineshader.use();

    lineshader.uniform("color", glm::vec3(0,0,0));
    approximationmesh.render(GL_LINE_STRIP);

    lineshader.uniform("color", glm::vec3(1,0,0));
    samplemesh.render(GL_POINTS);

	};

	//Execute the render loop
	Tiny::loop([&](){
	});

	Tiny::quit();

	return 0;

}
