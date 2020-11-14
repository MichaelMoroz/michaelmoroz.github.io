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


float sdBox( in vec2 p, in vec2 b )
{
    vec2 d = abs(p)-b;
    return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

void main () {
    vec2 p = ((uv - 0.5)*scale*vec2(aspect, 1.0) + 0.5 - iPos)*vec2(R.y/R.x, 1.0);
    float border = step(0., p.x)*step(p.x, 1.)*
                   step(0., p.y)*step(p.y, 1.);
    p*=R;
    vec2 pi = floor(p);
    vec4 pv = T0(pi);
    vec4 m = T1(pi); 
    p -= 0.5;
    vec2 dsize = clamp(1.0 - 2.0*abs(pv.xy), 0.001, 1.0);
    pv.xy += pi;
    float bsdf = sdBox(p - pv.xy,0.5*dsize);
    float rho = m.x*smoothstep(0.2, -0.2, bsdf)/(dsize.x*dsize.y);
     gl_FragColor = border*vec4(rho);
}