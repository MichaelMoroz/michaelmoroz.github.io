#extension GL_EXT_draw_buffers : require
precision mediump float;


varying vec2 uv;
uniform int iFrame;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
uniform sampler2D iChannel4;
uniform vec2 R;
uniform float dt;

uniform bool solid;
uniform float rho;
uniform float elastic_lambda;
uniform float elastic_mu;
uniform float fluid_p;


//new pv
#define T0(p) texture2D(iChannel0, mod(p,R)/R)
//new m
#define T1(p) texture2D(iChannel1, mod(p,R)/R)
//new dgrad
#define T2(p) texture2D(iChannel2, mod(p,R)/R)
//old pv
#define T3(p) texture2D(iChannel3, mod(p,R)/R)
//old m
#define T4(p) texture2D(iChannel4, mod(p,R)/R)


//useful functions
#define GS(x) exp(-dot(x,x))
#define GS0(x) exp(-length(x))
#define CI(x) smoothstep(1.0, 0.9, length(x))
#define Dir(ang) vec2(cos(ang), sin(ang))
#define Rot(ang) mat2(cos(ang), sin(ang), -sin(ang), cos(ang))
#define loop(i,x) for(int i = 0; i < x; i++)
#define range(i,a,b) for(int i = a; i <= b; i++)

float determinant( in mat2 m ) { return m[0][0]*m[1][1] - m[0][1]*m[1][0]; }
mat2 transpose( in mat2 m ) { return mat2(m[0].x, m[1].x, m[0].y, m[1].y); }
mat2 inverse(in mat2 m) { return mat2(m[1][1],-m[0][1], -m[1][0], m[0][0]) / (m[0][0]*m[1][1] - m[0][1]*m[1][0]); }

mat2 strain(mat2 D)
{
    float J = abs(determinant(D)) + 0.001;

    // MPM course, page 46
    float volume = J;

    // useful matrices for Neo-Hookean model
    mat2 F_T = transpose(D);
    mat2 F_inv_T = inverse(F_T);
    mat2 F_minus_F_inv_T = D - F_inv_T;

    // MPM course equation 48
    mat2 P_term_0 = elastic_mu * (F_minus_F_inv_T);
    mat2 P_term_1 = elastic_lambda * log(J) * F_inv_T;
    mat2 P = P_term_0 + P_term_1;

    // equation 38, MPM course
    mat2 stress = (1./J)* P * F_T;

    return volume * stress;
}


void main () {
  vec2 p = floor(vec2(R)*uv);
  vec4 pv = T0(p); 
  vec4 m = T1(p);
  mat2 dgrad = mat2(T2(p));
  //velocity gradient
  mat2 vgrad = mat2(0.);
  mat2 stress = mat2(0.);

  if(m.x > 0.0)
  {  
    float m1 = 0.;
    range(i, -1, 1) range(j, -1, 1)
    {
      vec2 dp = vec2(i,j);
      vec4 pv0 = T3(p + dp);
      vec4 m0 = T4(p + dp);
      
      //dynamic distribution size
      vec2 K = vec2(1.5);
      //update particle position in time and relative to cur cell
      pv0.xy += dp + dt*pv0.zw;

      //box overlaps
      vec4 aabbX0 = clamp(vec4(pv0.xy - K*0.5, pv0.xy + K*0.5), vec4(-0.5), vec4(0.5)); //overlap aab
      vec2 size = aabbX0.zw - aabbX0.xy; 

      //the deposited mass into this cell
      float dm = m0.x*size.x*size.y/(K.x*K.y);
      
      //delta new old
      vec2 dx = pv0.xy - pv.xy;
      vec2 dv = pv0.zw - pv.zw; 

      vgrad += mat2(dv*dx.x,dv*dx.y)*dm;
      m1 += dm;
    }
    
    vgrad /= m1;

    vec2 K0 = clamp(1.0 - 2.0*abs(pv.xy), 0.001, 1.0);
    float drho = m.x/(K0.x*K0.y) - rho;
    vgrad -= 0.004*mat2(drho)*abs(drho);

    //integrate deformation gradient
    dgrad += dt*vgrad*dgrad;

    float r = 0.002;
    //dgrad = dgrad*(1. - r) + mat2(1.)*r;
    
    if(solid)
    {
      //solid
      stress = strain(dgrad); 
    }
    else
    {
      //fluid
      stress = mat2(-fluid_p*(m1 - rho));
      dgrad = mat2(1.);
    }
  }

  gl_FragData[0] = vec4(dgrad);
  gl_FragData[1] = vec4(vgrad);
  gl_FragData[2] = clamp(vec4(stress), -4.0, 4.0);
}