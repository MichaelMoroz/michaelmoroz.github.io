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
uniform vec3 R3D;
uniform vec2 T;

uniform bool solid;
uniform float rho;
uniform float elastic_lambda;
uniform float elastic_mu;
uniform float fluid_p;

//coordinate mappings
vec2 D2(vec3 p3d){return p3d.xy + vec2(mod(p3d.z, T.x), floor(p3d.z/T.x))*R3D.xy;}
vec3 D3(vec2 p2d){return vec3(mod(p2d, R3D.xy), floor(p2d.x/R3D.x) + T.x*floor(p2d.y/R3D.y));}

#define V0(p3d) texture2D(iChannel0, D2(mod(p3d, R3D))/R).xyz
#define V1(p3d) texture2D(iChannel1, D2(mod(p3d, R3D))/R).xyz
#define V2(p3d) texture2D(iChannel2, D2(mod(p3d, R3D))/R).xyz
#define V3(p3d) texture2D(iChannel3, D2(mod(p3d, R3D))/R).xyz
#define V4(p3d) texture2D(iChannel4, D2(mod(p3d, R3D))/R).xyz
#define V5(p3d) texture2D(iChannel5, D2(mod(p3d, R3D))/R).xyz

//useful functions
#define GS(x) exp(-dot(x,x))
#define range(i,a,b) for(int i = a; i <= b; i++)

float determinant( in mat3 m ) 
{
  float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
  float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
  float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

  float b01 = a22 * a11 - a12 * a21;
  float b11 = -a22 * a10 + a12 * a20;
  float b21 = a21 * a10 - a11 * a20;

  return a00 * b01 + a01 * b11 + a02 * b21;
}

mat3 transpose( in mat3 m ) 
{
   return mat3(m[0].x, m[1].x, m[2].x, 
               m[0].y, m[1].y, m[2].y,
               m[0].z, m[1].z, m[2].z); 
}

mat3 inverse(mat3 m) {
  float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
  float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
  float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

  float b01 = a22 * a11 - a12 * a21;
  float b11 = -a22 * a10 + a12 * a20;
  float b21 = a21 * a10 - a11 * a20;

  float det = a00 * b01 + a01 * b11 + a02 * b21;

  return mat3(b01, (-a22 * a01 + a02 * a21), (a12 * a01 - a02 * a11),
              b11, (a22 * a00 - a02 * a20), (-a12 * a00 + a02 * a10),
              b21, (-a21 * a00 + a01 * a20), (a11 * a00 - a01 * a10)) / det;
}

mat3 strain(mat3 D)
{
    float J = abs(determinant(D)) + 0.001;

    // MPM course, page 46
    float volume = J;

    // useful matrices for Neo-Hookean model
    mat3 F_T = transpose(D);
    mat3 F_inv_T = inverse(F_T);
    mat3 F_minus_F_inv_T = D - F_inv_T;

    // MPM course equation 48
    mat3 P_term_0 = elastic_mu * (F_minus_F_inv_T);
    mat3 P_term_1 = elastic_lambda * log(J) * F_inv_T;
    mat3 P = P_term_0 + P_term_1;

    // equation 38, MPM course
    mat3 stress = (1./J) * P * F_T;

    return volume * stress;
}


void main () {
  vec3 p = D3(floor(vec2(R)*uv));
  vec3 p1 = V0(p);
  vec3 v1 = V1(p);
  vec3 m1 = V2(p);
  mat3 dgrad1 = mat3(V3(p), V4(p), V5(p));

  //velocity gradient
  mat3 vgrad = mat3(0.);
  mat3 stress = mat3(0.);

  if(m1.x > 0.0)
  {  
    float M = 0.;
    float k1 = 0.01;
    range(i, -1, 1) range(j, -1, 1) range(k, -1, 1)
    {
      vec3 dp = vec3(i,j,k);
      vec3 p0 = V0(p + dp) + dp;
      vec3 v0 = V1(p + dp);
      vec3 m0 = V2(p + dp);
      vec3 dx = p0 - p1; vec3 dv = v0 - v1; float w = m0.x*GS(0.9*dx);
      vgrad += mat3(dv*dx.x,dv*dx.y,dv*dx.z)*w;
      M += m0.x*w;
      k1 += w;
    }
    M /= k1;
    vgrad /= k1;

    vec3 K0 = clamp(1.0 - 2.0*abs(p1), 0.001, 1.0);
    float drho = m1.x/(K0.x*K0.y*K0.z) - rho;
    vgrad -= 0.01*mat3(drho)*abs(drho);

    //integrate deformation gradient
    dgrad1 += dt*vgrad*dgrad1;

    float r = 0.002;
    //dgrad = dgrad*(1. - r) + mat2(1.)*r;
    
    if(solid)
    {
      //solid
      stress = strain(dgrad1); 
    }
    else
    {
      //fluid
      stress = mat3(-fluid_p*(M - rho));
      dgrad1 = mat3(1.);
    }
  }

  if(iFrame < 5)
  {
    dgrad1 = mat3(1.); 
  }

  gl_FragData[0].xyz = clamp(dgrad1[0], -5.0, 5.0);
  gl_FragData[1].xyz = clamp(dgrad1[1], -5.0, 5.0);
  gl_FragData[2].xyz = clamp(dgrad1[2], -5.0, 5.0);
  gl_FragData[3].xyz = clamp(stress[0], -3.0, 3.0);
  gl_FragData[4].xyz = clamp(stress[1], -3.0, 3.0);
  gl_FragData[5].xyz = clamp(stress[2], -3.0, 3.0);
}