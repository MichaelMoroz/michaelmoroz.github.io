#extension GL_EXT_draw_buffers : require
precision mediump float;

#define T0(p) texture2D(iChannel0, mod(p,R)/R)
#define T1(p) texture2D(iChannel1, mod(p,R)/R)

//useful defines
#define loop(i,x) for(int i = 0; i < x; i++)
#define range(i,a,b) for(int i = a; i <= b; i++)

varying vec2 uv;
uniform int iFrame;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform vec2 R;
uniform float dt;
void main () {
  vec2 p = floor(vec2(R)*uv);
  vec4 pv; float m;

  range(i, -1, 1) range(j, -1, 1)
  {
    vec2 dp = vec2(i,j);
    vec4 pv0 = T0(p + dp);
    vec4 m0 = T1(p + dp); 
    
    //dynamic distribution size
    vec2 K = clamp(1.0 - 2.0*abs(pv0.xy), 0.01, 1.0) + 0.*dt;
    
    //update particle position in time and relative to cur cell
    pv0.xy += dp + dt*pv0.zw;

    //box overlaps
    vec4 aabbX = vec4(max(-vec2(0.5), pv0.xy - K*0.5), min(vec2(0.5), pv0.xy + K*0.5)); //overlap aabb
    vec2 center = 0.5*(aabbX.xy + aabbX.zw); //center of mass
    vec2 size = max(aabbX.zw - aabbX.xy, 0.); //only positive

    //the deposited mass into this cell
    float dm = m0.x*size.x*size.y/(K.x*K.y); 

    pv += vec4(center, pv0.zw)*dm;
    m += dm;
  }

  if(m > 0.0) 
  {
    pv/=m;
  }

  if(iFrame < 5)
  {
    m = 0.5 + 0.5*sin(p.x*0.1)*sin(p.y*0.1);
  }

  gl_FragData[0] = pv;
  gl_FragData[1] = vec4(m, 0.0, 0.0, 0.0);
}