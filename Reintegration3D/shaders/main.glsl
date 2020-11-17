precision mediump float;

varying vec2 uv;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform int iFrame;
uniform float scale;
uniform vec2 iPos;
uniform float aspect;
uniform vec2 R;
uniform vec3 R3D;
uniform vec2 T;
#define MAX_RAY_STEPS 256

//coordinate mappings
vec2 D2(vec3 p3d){return p3d.xy + vec2(mod(p3d.z, T.x), floor(p3d.z/T.x))*R3D.xy;}
vec3 D3(vec2 p2d){return vec3(mod(p2d, R3D.xy), floor(p2d.x/R3D.x) + T.x*floor(p2d.y/R3D.y));}

#define V0(p3d) texture2D(iChannel0, D2(mod(p3d, R3D))/R).xyz
#define V1(p3d) texture2D(iChannel1, D2(mod(p3d, R3D))/R).xyz
#define V2(p3d) texture2D(iChannel2, D2(mod(p3d, R3D))/R).xyz

vec3 V2i(vec3 p3d)
{
    float zi = floor(p3d.z);
    float zf = fract(p3d.z);
    return mix(V2(vec3(p3d.xy,zi-1.0)),V2(vec3(p3d.xy,zi)),zf);
}

//useful defines
#define loop(i,x) for(int i = 0; i < x; i++)
#define range(i,a,b) for(int i = a; i <= b; i++)
#define GS(x) exp(-dot(x,x))


vec2 boxIntersection( in vec3 ro, in vec3 rd, in vec3 rad) 
{
    vec3 m = 1.0/rd;
    vec3 n = m*ro;
    vec3 k = abs(m)*rad;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
	
    if( tN>tF || tF<0.0) return vec2(-1.0); // no intersection
    return vec2( tN, tF );
}

bool getVoxel(ivec3 pi, vec3 ro, vec3 rd, out float td, out vec3 N, out float rho)
{
    vec3 p = vec3(pi);
    vec3 vox = V1(p);
    if(vox.y > 0.18)
    {
        vec3 p0 = V0(p);
        vec3 K = clamp(0.5 - 1.0*abs(p0), 0.001, 0.5);
        vec2 bi = boxIntersection(ro - (p + p0 + 0.5), rd, K);
        if(bi.x != -1.0)
        {
            rho = vox.y;
            N = - normalize(V2i(ro + bi.x*rd));
            td = bi.x;
            return true;
        }
    }
    return false;
}
#define PI 3.14159265
#define FOV PI*0.25

void main () {
    vec2 UV = 2.*uv - 1.0;
    
    vec2 angle = PI*iPos*vec2(1., 0.5);
    vec3 camx = vec3(cos(angle.x)*cos(angle.y), sin(angle.x)*cos(angle.y), sin(angle.y));
    vec3 camy = normalize(cross(camx, vec3(0,0,1)));
    vec3 camz = normalize(cross(camy, camx));

    //camera ray
    vec3 rd = normalize(camx + (UV.x*aspect*camy + UV.y*camz)*FOV);
    //camera position
    vec3 ro = R3D*0.5 - 0.5*scale*length(R3D)*camx;

    vec2 td = boxIntersection(ro-R3D*0.5, rd, R3D*0.5);
    vec3 ro1 = ro + td.x*rd; 
    
    if(td.x != -1.0)
    {
        float depth = 0.; vec3 N; float rho = 0.;
        ivec3 mapPos = ivec3(floor(ro1));
        vec3 deltaDist = abs(vec3(length(rd)) / rd);
        ivec3 rayStep = ivec3(sign(rd));
        vec3 sideDist = (sign(rd) * (vec3(mapPos) - ro1) + (sign(rd) * 0.5) + 0.5) * deltaDist; 
        bvec3 mask;
        for (int i = 0; i < MAX_RAY_STEPS; i++) 
        {
            if (getVoxel(mapPos, ro, rd, depth, N, rho) || distance(vec3(mapPos), ro) > td.y + 1.) break;
            mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
            sideDist += vec3(mask) * deltaDist;
            mapPos += ivec3(vec3(mask)) * rayStep;
        }
        gl_FragColor = vec4(0.5 + 0.5*dot(N, normalize(vec3(1))));
    }
    else
    {
        gl_FragColor = vec4(0.);
    }
}