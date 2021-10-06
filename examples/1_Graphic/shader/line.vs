#version 330 core

in vec2 in_Quad;

void main(){
  vec2 ex_Quad = in_Quad / 2.5f;
  gl_Position = vec4(ex_Quad, -1.0, 1.0);
}
