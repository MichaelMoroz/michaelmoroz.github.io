precision mediump float;

varying vec2 uv;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
uniform int iFrame;
uniform float scale;
uniform vec2 iPos;
uniform float aspect;
uniform vec2 R;
uniform vec3 R3D;
uniform vec2 T;
uniform vec3 CamPos;
uniform float isosurface;
uniform float distortion;
#define MAX_RAY_STEPS 256

//coordinate mappings
vec2 D2(vec3 p3d){return p3d.xy + vec2(mod(p3d.z, T.x), floor(p3d.z/T.x))*R3D.xy;}
vec3 D3(vec2 p2d){return vec3(mod(p2d, R3D.xy), floor(p2d.x/R3D.x) + T.x*floor(p2d.y/R3D.y));}

#define V0(p3d) texture2D(iChannel0, D2(mod(p3d, R3D))/R).xyz
#define V1(p3d) texture2D(iChannel1, D2(mod(p3d, R3D))/R).xyz
#define V2(p3d) texture2D(iChannel2, D2(mod(p3d, R3D))/R).xyz
#define V3(p3d) texture2D(iChannel3, D2(mod(p3d, R3D))/R).xyz

vec3 V2i(vec3 p3d)
{
    float zi = floor(p3d.z);
    float zf = p3d.z - zi;
    return mix(V2(vec3(p3d.xy,zi)),V2(vec3(p3d.xy,zi+1.0)),zf);
}

vec3 V3i(vec3 p3d)
{
    float zi = floor(p3d.z);
    float zf = p3d.z - zi;
    return mix(V3(vec3(p3d.xy,zi)),V3(vec3(p3d.xy,zi+1.0)),zf);
}

//useful defines
#define loop(i,x) for(int i = 0; i < x; i++)
#define range(i,a,b) for(int i = a; i <= b; i++)
#define GS(x) exp(-dot(x,x))

float CellIntersection(vec3 rd)
{
    vec3 t1 = - abs(1.0/rd)*vec3(0.5);
    return max( max( t1.x, t1.y ), t1.z );
}

vec3 getPlane(vec3 p0, out vec3 N)
{
    N = -p0/(length(p0)+1e-4);
    return 2.*p0 - N*CellIntersection(N);
}

float planeIntersection(vec3 p0, vec3 N, vec3 ro, vec3 rd)
{
    float d = dot(N, rd);
    float t =dot(p0 - ro, N)/d;
    return (abs(d)>1e-5 && t > 0.0) ? t : -1.0;
}

vec2 boxIntersection( in vec3 ro, in vec3 rd, in vec3 rad) 
{
    vec3 m = 1.0/rd;
    vec3 n = m*ro;
    vec3 k = abs(m)*rad;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
	return (tN>tF || tF<0.0) ? vec2(-1.0) : vec2(tN, tF);
}

// Triangle intersection. Returns { t, u, v }
vec3 triIntersect( in vec3 ro, in vec3 rd, in vec3 v0, in vec3 v1, in vec3 v2 )
{
    vec3 v1v0 = v1 - v0;
    vec3 v2v0 = v2 - v0;
    vec3 rov0 = ro - v0;

    vec3  n = cross( v1v0, v2v0 );
    vec3  q = cross( rov0, rd );
    float d = 1.0/dot( rd, n );
    float u = d*dot( -q, v2v0 );
    float v = d*dot(  q, v1v0 );
    float t = d*dot( -n, rov0 );   
    t=min(u, min(v, min(1.0-(u+v), t)));

    return vec3( t, u, v );
}

float CappedBox(in vec3 ro, in vec3 rd, in vec3 rad, vec3 p0, vec3 N)
{
    vec2 boxInt = boxIntersection(ro, rd, rad);
    if(boxInt.x < 0.0)
     return -1.0;
    else
    {
        vec3 p = ro + boxInt.x*rd;
        float planeInt = planeIntersection(p0, N, ro, rd);
        if(dot(p0 - p, N) < 0.0)
            return (planeInt < boxInt.y && planeInt > boxInt.x)?planeInt:-1.0;

        return boxInt.x;
    }
}

#define DDist 1

bool getVoxel(ivec3 pi, vec3 ro, vec3 rd, out float td, out vec3 N, out float rho)
{
    vec3 p = vec3(pi);
    vec3 vox = V1(p);
    rho = vox.y; //local density
    if(rho > isosurface)
    {
        vec3 p0 = V0(p);
        #if DDist
            vec3 K = clamp(0.5 - 1.0*abs(p0), 0.0, 0.5);
            float bi = boxIntersection(ro - (p + p0 + 0.5), rd, K).x;
        #else
            vec3 pN;
            vec3 iN = -normalize(V2(p));
            vec3 P0 = getPlane(p0, pN); 
            float bi = CappedBox(ro - (p + 0.5), rd, vec3(0.5), P0, mix(pN, iN, 0.)); 
        #endif
        if(bi >= 0.0)
        {
            N = -normalize(V2i(ro + rd*bi));
            td = bi;
            return true;
        }
    }
    td = distance(ro, p);
    return false;
}
#define PI 3.14159265
#define FOV PI*0.25

vec4 exposure(vec4 I)
{
    return 1. - exp(-I);
}

//distance to isosurface
float d_estm(float rho)
{
    return clamp(10.*(isosurface - rho)*abs(isosurface - rho), -0.5, 0.6);
}

float rho2(vec3 pos, inout vec3 grad)
{
    grad = V2i(pos);
    return V3i(pos - distortion*grad).x;
}

vec4 calcGrad( in vec3 pos )
{
    vec4 e = vec4(0.002,-0.002, 0.25, -0.25);
    vec3 g;
    return   e.zwwz*rho2( pos + e.xyy, g) + 
  			 e.wwzz*rho2( pos + e.yyx, g ) + 
			 e.wzwz*rho2( pos + e.yxy, g ) + 
             e.zzzz*rho2( pos + e.xxx, g ) ;
}

vec3 castRay(vec3 ro, vec3 rd, out vec3 grad, out float sh)
{
    vec2 td = boxIntersection(ro-R3D*0.5, rd, R3D*0.5 - 1.0);
    if(all(greaterThan(ro, vec3(0.))) && all(lessThan(ro, R3D)))
    {
        td.x = 0.;        
    }
    vec3 ro1 = ro + td.x*rd; 
    sh = 1.;
    if(td.x >= 0.0)
    {
        /*float depth = 0.; vec3 N; float rho = 0., density = 0.;
        ivec3 mapPos = ivec3(floor(ro1));
        vec3 deltaDist = abs(vec3(length(rd)) / rd);
        ivec3 rayStep = ivec3(sign(rd));
        vec3 sideDist = (sign(rd) * (vec3(mapPos) - ro1) + (sign(rd) * 0.5) + 0.5) * deltaDist; 
        bvec3 mask;
        for (int i = 0; i < MAX_RAY_STEPS; i++) 
        {
            if (getVoxel(mapPos, ro, rd, depth, N, rho) || distance(vec3(mapPos), ro) > td.y) break;
            
            density += 0.*((rho>isosurface)?0.:rho);
            
            //voxel tracing
            mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
            sideDist += vec3(mask) * deltaDist;
            mapPos += ivec3(vec3(mask)) * rayStep;
        }
        vec3 p = ro + depth*rd;
        vec3 r = reflect(rd, N);
        vec3 l =  normalize(vec3(1,1,1));
        float specular = pow(0.5*dot(r,l) + 0.5,40.);
        float diffuse = 0.5 + 0.5*dot(N, l);
         gl_FragColor = exposure(vec4(diffuse*vec4(0.8,0.9,1.0,1.0) + 4.0*specular));
        */

        //current marching position
        vec3 p1 = ro1; float td1 = td.x;

        for(int i = 0; i < MAX_RAY_STEPS; i++)
        {
            float rho1 = rho2(p1, grad);
            float de = d_estm(rho1);
            td1 += de;
            p1 += de*rd;
            if(td1 > 1.0)
            sh *= exp(-0.1*de*rho1);
            if(td1 > td.y - 0.5) return vec3(-1.);
            if(abs(de) < 0.005) return p1;
        } 
        return p1;
    }
    return vec3(-1.);
}

void main () {
    vec2 UV = 2.*uv - 1.0;
    
    vec2 angle = PI*iPos*vec2(1., 0.5);
    vec3 camx = vec3(cos(angle.x)*cos(angle.y), sin(angle.x)*cos(angle.y), sin(angle.y));
    vec3 camy = normalize(cross(camx, vec3(0,0,1)));
    vec3 camz = normalize(cross(camy, camx));

    //camera ray
    vec3 rd = normalize(camx + (UV.x*aspect*camy + UV.y*camz)*FOV);
    //camera position
    vec3 ro = CamPos;

    vec3 grad; float sh;

    vec3 ip = castRay(ro, rd, grad, sh);
    if(ip.x >= 0.)
    {
        //vec4 G = calcGrad(p1);
        vec3 l =  normalize(vec3(1,1,1)); //light
        vec3 N = -normalize(grad);
       // vec3 shadow = castRay(ip + 0.05*N, l, grad, sh);
        float A = 1.0;//sh;
        vec3 r = reflect(rd, N);
       

        float specular = pow(0.5*dot(r,l) + 0.5,40.);
        float diffuse = 0.5 + 0.5*dot(N, l);
        gl_FragColor = exposure(vec4((A + 0.35)*diffuse*vec4(0.8,0.9,1.0,1.0) + A*4.0*specular));
    }
    else gl_FragColor = vec4(0.);

}