#extension GL_EXT_draw_buffers : require
precision mediump float;

#define T0(p) texture2D(iChannel0, mod(p,R)/R)
#define T1(p) texture2D(iChannel1, mod(p,R)/R)
#define T2(p) texture2D(iChannel2, mod(p,R)/R)
#define T3(p) texture2D(iChannel3, mod(p,R)/R)
#define T4(p) texture2D(iChannel4, mod(p,R)/R)
//useful defines
#define loop(i,x) for(int i = 0; i < x; i++)
#define range(i,a,b) for(int i = a; i <= b; i++)

varying vec2 uv;
uniform int iFrame;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
uniform sampler2D iChannel4;
uniform vec2 R;
uniform float dt;
uniform float diffusion;
uniform float rho;
void main () {
  vec2 p = floor(vec2(R)*uv);
  vec4 pv = vec4(0); float m = 0.; 
  mat2 stress = mat2(0);
  //deformation gradient 
  mat2 dgrad = mat2(0.);
  //velocity gradient
  mat2 vgrad = mat2(0.);

  range(i, -1, 1) range(j, -1, 1)
  {
    vec2 dp = vec2(i,j);
    vec4 pv0 = T0(p + dp);
    vec4 m0 = T1(p + dp); 
    mat2 dgrad0 = mat2(T2(p + dp)); 
    mat2 vgrad0 = mat2(T3(p + dp)); 
    //dynamic distribution size
    vec2 K = clamp(1.0 - 2.0*abs(pv0.xy), 0.001, 1.0) + clamp(diffusion*abs(m0.y), 0., 0.5)*dt;
    //update particle position in time and relative to cur cell
    pv0.xy += dp + dt*pv0.zw;

    //box overlaps
    vec4 aabbX0 = clamp(vec4(pv0.xy - K*0.5, pv0.xy + K*0.5), vec4(-0.5), vec4(0.5)); //overlap aab
    vec2 size = aabbX0.zw - aabbX0.xy; 
    vec2 center = 0.5*(aabbX0.xy + aabbX0.zw);
    //the deposited mass into this cell
    float dm = m0.x*size.x*size.y/(K.x*K.y);
    vec2 dx = center - pv0.xy;

    dgrad += dgrad0*dm;
    pv += vec4(center, pv0.zw + vgrad*dx)*dm;
    m += dm;
  }

  if(m > 0.0) 
  {
    dgrad/=m;
    vgrad/=m;
    pv/=m;
  }
  else 
  {
    dgrad = mat2(1.);
  }

  if(iFrame < 5)
  {
    m = smoothstep(R.x*0.6, R.x*0.5, p.x)*(0.5 + 0.5*sin(p.x*0.1)*sin(p.y*0.1));
    pv = vec4(0); 
    dgrad = mat2(1.);
  }

  vec4 pstress = T4(p);
  //clamp the gradient to not go insane
  dgrad = mat2(clamp(vec4(dgrad - mat2(1.)), vec4(-4.0), vec4(4.0))) + mat2(1.);


  gl_FragData[0] = pv;
  gl_FragData[1] = vec4(m, abs(pstress.x)  + abs(pstress.w), 0.0, 0.0);
  gl_FragData[2] = vec4(dgrad);
}