#extension GL_EXT_draw_buffers : require
precision mediump float;

#define T0(p) texture2D(iChannel0, mod(p,R)/R)
#define T1(p) texture2D(iChannel1, mod(p,R)/R)

//useful functions
#define GS(x) exp(-dot(x,x))
#define GS0(x) exp(-length(x))
#define CI(x) smoothstep(1.0, 0.9, length(x))
#define Dir(ang) vec2(cos(ang), sin(ang))
#define Rot(ang) mat2(cos(ang), sin(ang), -sin(ang), cos(ang))
#define loop(i,x) for(int i = 0; i < x; i++)
#define range(i,a,b) for(int i = a; i <= b; i++)

varying vec2 uv;
uniform int iFrame;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform vec2 R;
uniform float dt;

vec3 minc(vec3 a, vec3 b)
{
  return (a.z > b.z)?b:a;
}

vec3 border(vec2 p)
{
  return minc(minc(vec3(0.0, -1.0, R.y - p.y), vec3(0.0, 1.0, p.y)),
              minc(vec3(-1.0, 0.0, R.x - p.x), vec3(1.0, 0.0, p.x)));
}

void main () {
  vec2 p = floor(vec2(R)*uv);
  vec4 pv = T0(p); 
  vec4 m = T1(p);
  if(m.x > 0.0)
  {
    vec2 F = vec2(0., 0.); 

    range(i, -2, 2) range(j, -2, 2)
    {
      vec2 dp = vec2(i,j);
      vec4 pv0 = T0(p + dp);
      vec4 m0 = T1(p + dp); 

      vec2 dx = pv0.xy + dp - pv.xy;
      vec2 dv = pv0.zw - pv.zw;
      F -= m.x*m0.x*(0.1*dx*(0.5*(m.x+m0.x) - 0.76) - 0.25*dx*dot(dx,dv))*GS(dx);
    }

    //gravity
    F += m.x*vec2(0., -0.004);
    pv.zw += dt*F/m.x;
    
    //border
    vec3 b = border(pv.xy+p);
    pv.zw += 3.*b.xy*smoothstep(10., 2., b.z);

    //velocity limit
    float v = length(pv.zw);
    pv.zw /= (v > 1.)?1.*v:1.;
  }
  // just output geometry data.
  gl_FragData[0] = pv;
  gl_FragData[1] = m;
}