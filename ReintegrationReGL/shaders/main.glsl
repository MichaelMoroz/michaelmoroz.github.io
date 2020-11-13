precision mediump float;

#define T0(p) texture2D(iChannel0, mod(p,R)/R)
#define T1(p) texture2D(iChannel1, mod(p,R)/R)

//useful defines
#define loop(i,x) for(int i = 0; i < x; i++)
#define range(i,a,b) for(int i = a; i <= b; i++)
#define GS(x) exp(-dot(x,x))

varying vec2 uv;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform int iFrame;
uniform float scale;
uniform vec2 iPos;
uniform float aspect;
uniform vec2 R;
void main () {
    vec2 p = (uv - 0.5)*scale*vec2(aspect, 1.0) + 0.5 - iPos;
    float border = step(0., p.x)*step(p.x, 1.)*
                   step(0., p.y)*step(p.y, 1.);
    float rho = 0.;
    p*=R;
    range(i, -2, 2) range(j, -2, 2)
    {
      vec2 dp = vec2(i,j);
      vec4 pv0 = T0(p + dp);
      vec4 m0 = T1(p + dp); 

      vec2 dx = pv0.xy + dp - fract(p) + 0.5;
      rho += m0.x*GS(dx);
    }
    gl_FragColor = border*vec4(0.1*rho);
}