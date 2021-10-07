#include <TinyEngine/TinyEngine>
#include <TinyEngine/camera>

#define DN 2
#define DM 1

#include "../../spectral.h"

#include "vertexpool.h"

#define SIZEX 20
#define SIZEY 20

int main( int argc, char* args[] ) {
using namespace std;
using namespace Eigen;




  //Generate Samples
  vector<spectral::S> samples;

  spectral::M target = [](spectral::avec in){
    spectral::bvec out; //Input Vector

    spectral::avec shift;
    shift << 0.3, -0.3;

    float a = exp(-4.0*in.dot(in));
//    a += 0.5f * exp(-4.0*(in+0.5f*spectral::avec::Ones()).dot(in+0.5f*spectral::avec::Ones()));
//    a -= 0.5f * exp(-4.0*(in+shift).dot(in+shift));
    out << a;
  //  out << ((in.dot(in) < 0.25f)?1.0f:0.0f);
    return out;
  };

  for(int i = 0; i < SIZEX; i++)
  for(int j = 0; j < SIZEY; j++){

    spectral::avec in; //Input Vector
    in << i, j;
    in = in/10.0f;
    in = in - 0.95f*spectral::avec::Ones();//1.0f;   //Normalize Position

    spectral::S sample(in, target(in));
    samples.push_back(sample);

  }

  //Asymmetric Fourier
  spectral::avec K;
  K << 5, 5;

  spectral::cosine solution(K, {-2, 2}, [](spectral::avec x){
    return spectral::bvec::Zero();
  });


  //cout<<solution.w.size()<<endl;

  spectral::collocation(solution, samples);
  cout<<"Err: "<<spectral::err(solution, samples)<<endl;
//  cout<<solution.w<<endl;
  //cout<<solution.w.size()<<endl;

  vector<spectral::S> approximations;

  for(int i = 0; i < 50; i++)
  for(int j = 0; j < 50; j++){

    spectral::avec in; //Input Vector
    in << i, j;
    in = in/25.0f;
    in = in - spectral::avec::Ones();//1.0f;   //Normalize Position

    spectral::S sample(in, solution.sample(in));
    approximations.push_back(sample);

  }












	//Initialize a Window
  Tiny::view.pointSize = 4.0f;
	Tiny::window("Example Window", 1200, 800);

  cam::near = -200.0f;
  cam::far = 200.0f;
  cam::moverate = 0.1f;
  cam::zoomrate = 15.0f;
  cam::look = glm::vec3(0, 0, 0);
  cam::init(500.0, cam::ORTHO);

  Shader surfshader({"shader/surf.vs", "shader/surf.fs"}, {"in_Position", "in_Normal", "in_Color"});

  Vertexpool<Vertex> vertexpool(SIZEX*SIZEY, 1);
  uint* section = vertexpool.section(SIZEX*SIZEY, 0, glm::vec3(0));
  for(size_t i = 0; i < samples.size(); i++)
  vertexpool.fill(section, i,
    vec3(samples[i].first(0), samples[i].second(0), samples[i].first(1))
  );

  for(int i = 0; i < SIZEX-1; i++){
  for(int j = 0; j < SIZEY-1; j++){

    vertexpool.indices.push_back(i*SIZEY+j);
    vertexpool.indices.push_back(i*SIZEY+(j+1));
    vertexpool.indices.push_back((i+1)*SIZEY+j);

    vertexpool.indices.push_back((i+1)*SIZEY+j);
    vertexpool.indices.push_back(i*SIZEY+(j+1));
    vertexpool.indices.push_back((i+1)*SIZEY+(j+1));

  }}

  vertexpool.resize(section, vertexpool.indices.size());
  vertexpool.index();
  vertexpool.update();



  Vertexpool<Vertex> solutionsurf(50*50, 1);
  uint* section2 = solutionsurf.section(50*50, 0, glm::vec3(0));
  for(size_t i = 0; i < 50*50; i++)
  solutionsurf.fill(section2, i,
    vec3(approximations[i].first(0), approximations[i].second(0), approximations[i].first(1))
  );

  for(int i = 0; i < 50-1; i++){
  for(int j = 0; j < 50-1; j++){

    solutionsurf.indices.push_back(i*50+j);
    solutionsurf.indices.push_back(i*50+(j+1));
    solutionsurf.indices.push_back((i+1)*50+j);

    solutionsurf.indices.push_back((i+1)*50+j);
    solutionsurf.indices.push_back(i*50+(j+1));
    solutionsurf.indices.push_back((i+1)*50+(j+1));

  }}

  solutionsurf.resize(section2, solutionsurf.indices.size());
  solutionsurf.index();
  solutionsurf.update();



	//Add the Event Handler
	Tiny::event.handler = cam::handler;

	//Set up an ImGUI Interface here
	Tiny::view.interface = [&](){
	};

	//Define the rendering pipeline
	Tiny::view.pipeline = [&](){

		Tiny::view.target(glm::vec3(1));	//Clear Screen to white

    surfshader.use();
    surfshader.uniform("vp", cam::vp);

    surfshader.uniform("color", glm::vec4(0,0,0,1));
    vertexpool.render(GL_POINTS);

    surfshader.uniform("color", glm::vec4(1,0,0,1));
    solutionsurf.render(GL_LINES);

	};

	//Execute the render loop
	Tiny::loop([&](){
	});

	Tiny::quit();

	return 0;

}
