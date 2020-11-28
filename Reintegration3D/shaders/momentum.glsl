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
uniform sampler2D iChannel6;
uniform vec2 R;
uniform vec3 R3D;
uniform vec2 T;
uniform float dt;
uniform float dynamic_friction;
uniform float gravityZ;
uniform float gravityY;
//useful functions
#define GS(x) exp(-dot(x,x))
#define range(i,a,b) for(int i = a; i <= b; i++)

//coordinate mappings
vec2 D2(vec3 p3d){return p3d.xy + vec2(mod(p3d.z, T.x), floor(p3d.z/T.x))*R3D.xy;}
vec3 D3(vec2 p2d){return vec3(mod(p2d, R3D.xy), floor(p2d.x/R3D.x) + T.x*floor(p2d.y/R3D.y));}

#define V0(p3d) texture2D(iChannel0, D2(mod(p3d, R3D))/R).xyz
#define V1(p3d) texture2D(iChannel1, D2(mod(p3d, R3D))/R).xyz
#define V2(p3d) texture2D(iChannel2, D2(mod(p3d, R3D))/R).xyz
#define V3(p3d) texture2D(iChannel3, D2(mod(p3d, R3D))/R).xyz
#define V4(p3d) texture2D(iChannel4, D2(mod(p3d, R3D))/R).xyz
#define V5(p3d) texture2D(iChannel5, D2(mod(p3d, R3D))/R).xyz
#define V6(p3d) texture2D(iChannel6, D2(mod(p3d, R3D))/R).xyz

vec4 min3(vec4 a, vec4 b)
{
  return (a.w > b.w)?b:a;
}

vec4 border3d(vec3 p)
{
  return min3(min3(vec4(0.0, 0.0, -1.0, R3D.z - p.z), vec4(0.0, 0.0, 1.0, p.z)),
              min3(min3(vec4(0.0, -1.0, 0.0, R3D.y - p.y), vec4(0.0, 1.0, 0.0, p.y)),
                   min3(vec4(-1.0, 0.0, 0.0, R3D.x - p.x), vec4(1.0, 0.0, 0.0, p.x))));
}

vec2 BI(vec3 rd)
{
    vec3 m = 1.0/rd;
    vec3 k = abs(m)*vec3(0.5);
    vec3 t1 = - k;
    vec3 t2 = + k;

    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
	
    return vec2( tN, tF );
}

void main () {
  vec3 p = D3(floor(vec2(R)*uv));
  vec3 p1 = V0(p);
  vec3 v1 = V1(p);
  vec3 m1 = V2(p);
  mat3 stress1 = mat3(V3(p), V4(p), V5(p));
  vec3 grad = vec3(0.);

  vec3 F = vec3(0.);
  float a = 0.001;
  float a1 = 0.001;
  range(i, -1, 1) range(j, -1, 1) range(k, -1, 1)
  {
    vec3 dp = vec3(i,j,k);
    vec3 m0 = V2(p + dp);
    vec3 p0 = V0(p + dp) + dp;
    vec3 v0 = V1(p + dp);
    
    mat3 stress0 = mat3(V3(p + dp), V4(p + dp), V5(p + dp));
    vec3 dx = p0 - p1;
    float d = length(dx); 
    dx = max(d, 1.0)*dx/(d + 0.01);
    vec3 dx1 = dp; vec3 dv = v0 - v1; float K = GS(1.2*dx);
    F += m0.x*(0.5*(stress0 + stress1)*dx + dynamic_friction*dot(dx,dv)*dx)*K; 
    a += K;
    
    K = GS(1.0*dx);
    grad += m0.x*dx*K;
    a1 += K;
  }
  grad /= a1;
  F /= a;

  vec3 K0 = clamp(1.0 - 2.0*abs(p1), 0.01, 1.0);
  //gravity
  F += -0.01*vec3(0.,  gravityY, gravityZ);
  F = clamp(length(F), 0.0, 0.03)*normalize(F);
  v1 += dt*F;
  
  //border
  vec4 b = border3d(p1+p);
  v1 += clamp(-dot(b.xyz,v1) + 0.03, 0., 1.)*b.xyz*smoothstep(5.0, 2., b.w);

  //velocity limit
  float V = length(v1);
  v1 /= (V > 1.)?1.*V:1.;


  gl_FragData[0].xyz = p1; //* clamp(2.0 - m1.y, 0., 1.0);
  gl_FragData[1].xyz = v1;
  gl_FragData[2].xyz = m1;
  gl_FragData[3].xyz = grad;
  gl_FragData[4].xyz = m1;
}