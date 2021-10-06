#include <TinyEngine/TinyEngine>

#include "../../spectral.h"

int main( int argc, char* args[] ) {
using namespace std;

  //Define our Input Positions
  vector<pair<float,float>> samples;
  samples = spectral::sample(50, {-2, 2}, [](float x){
    if(x > 0) return 1.0f;
    else return 0.0f;
  });

  //Construct a Cosine Representation of Samples with Inhomogeneity
  spectral::fourier solution(11, {-2, 2}, [](float x){
    return 0.5f;
  });

  //Perform a Fit with a Weighted Resiual Method
  spectral::leastsquares(&solution, samples);

  //Sample the Solution
  vector<pair<float,float>> approximations;
  approximations = spectral::sample(500, {-2, 2}, [&](float x){
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
