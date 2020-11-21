#extension GL_EXT_draw_buffers : require
precision mediump float;
varying vec2 uv;
uniform int iFrame;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
uniform sampler2D iChannel4;
uniform sampler2D iChannel5;
uniform vec2 R;
uniform float dt;
uniform float diffusion;
uniform float rho;

uniform vec3 R3D;
uniform vec2 T;
//coordinate mappings
vec2 D2(vec3 p3d){return p3d.xy + vec2(mod(p3d.z, T.x), floor(p3d.z/T.x))*R3D.xy;}
vec3 D3(vec2 p2d){return vec3(mod(p2d, R3D.xy), floor(p2d.x/R3D.x) + T.x*floor(p2d.y/R3D.y));}

#define V0(p3d) texture2D(iChannel0, D2(mod(p3d, R3D))/R).xyz
#define V1(p3d) texture2D(iChannel1, D2(mod(p3d, R3D))/R).xyz
#define V2(p3d) texture2D(iChannel2, D2(mod(p3d, R3D))/R).xyz
#define V3(p3d) texture2D(iChannel3, D2(mod(p3d, R3D))/R).xyz
#define V4(p3d) texture2D(iChannel4, D2(mod(p3d, R3D))/R).xyz
#define V5(p3d) texture2D(iChannel5, D2(mod(p3d, R3D))/R).xyz


//useful defines
#define GS(x) exp(-dot(x,x))
#define loop(i,x) for(int i = 0; i < x; i++)
#define range(i,a,b) for(int i = a; i <= b; i++)


void main () {
  vec3 p = D3(floor(vec2(R)*uv));
  vec3 p1 = vec3(0); 
  vec3 v1 = vec3(0);
  float m1 = 0.;
  mat3 dgrad1 = mat3(0);

  
  float mi = 0.;
  float a = 0.001;
 
  range(i, -1, 1) range(j, -1, 1) range(k, -1, 1)
  {
    vec3 dp = vec3(i,j,k);
    vec3 p0 = V0(p + dp);
    vec3 v0 = V1(p + dp);
    vec3 m0 = V2(p + dp);
    mat3 dgrad0 = mat3(V3(p + dp), V4(p + dp), V5(p + dp));
    
    //dynamic distribution size
    vec3 K = clamp(1.0 - 2.0*abs(p0), 0.00001, 1.0);
    K += K*clamp(diffusion, 0., 0.1)*dt;
    
    //update particle position in time and relative to cur cell
    p0 += dp + dt*v0;

    //box overlaps
    vec3 aabb0 = clamp(p0 - K*0.5, vec3(-0.5), vec3(0.5)); 
    vec3 aabb1 = clamp(p0 + K*0.5, vec3(-0.5), vec3(0.5)); 
    vec3 size = aabb1 - aabb0; 
    vec3 center = 0.5*(aabb0 + aabb1);
    
    //the deposited mass into this cell
    float dm = m0.x*size.x*size.y*size.z/(K.x*K.y*K.z);

    dgrad1 += dgrad0*dm;
    p1 += center*dm;
    v1 += v0*dm;
    m1 += dm;
    float k0 = GS(0.75*dp);
    a += k0;
    mi += m0.x*k0/(K.x*K.y*K.z);
  }
  mi /= a;

  if(m1 > 0.0) 
  {
    dgrad1/=m1;
    p1/=m1;
    v1/=m1;
  }
  else 
  {
    dgrad1 = mat3(1.);
  }

  if(iFrame < 5)
  {
    m1 = rho*smoothstep(R3D.x*0.6, R3D.x*0.5, p.x)*(0.5 + 0.5*sin(p.x*0.3)*sin(p.y*0.3)*sin(p.z*0.3));
    p1 = vec3(0);
    v1 = vec3(0); 
    dgrad1 = mat3(1.);
  }

  gl_FragData[0].xyz = p1;
  gl_FragData[1].xyz = v1;
  gl_FragData[2].xyz = vec3(m1,0.,mi);
  gl_FragData[3].xyz = dgrad1[0];
  gl_FragData[4].xyz = dgrad1[1];
  gl_FragData[5].xyz = dgrad1[2];
}