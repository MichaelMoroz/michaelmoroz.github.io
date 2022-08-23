---
layout: post
title: Visualizing General Relativity
image: SpaceEngineBH.jpg
---

When dealing with renders of things like warp drives and black holes we usually just expect to see a simple approximation or an artist rendition, usually just assuming that the math required to pull off something accurate would require someone with at least a PhD in Mathematical Physics, which in most cases is somewhat true, but not necessarily. In this blog post I'll try to explain a way to do actually accurate visualizations within a 100 or so lines of code, for basically any kind of space time for which you can write its metric as code. The detailed mathematical derivation of this approach might be somewhat math heavy though.

The main ingredient of any GR render is figuring out how the rays of light move around. Knowing how light moves we can trace rays from the camera into the scene as see where the light came from. So to render a basic scene without objects we simply trace a ray for each pixel and assignt the color of the pixel to the color of the skybox in the direction in which the ray ends up pointing to. 

- [What are geodesics?](#what-are-geodesics)
- [Mathematical description of shortest path](#mathematical-description-of-shortest-path)
- [Lagrangian description of a geodesic](#lagrangian-description-of-a-geodesic)
- [Hamiltonian description of a geodesic](#hamiltonian-description-of-a-geodesic)
- [Writing a Hamiltonian geodesic tracer in GLSL](#writing-a-hamiltonian-geodesic-tracer-in-glsl)
- [Conclusions](#conclusions)
- [References](#references)
  
---

### What are geodesics?
So how exactly do we trace rays in curves space? Any object inside a curved space follows something called a geodesic.

A geodesic is essentially just a fancy word for path of shortest length between 2 points inside a space, and actually there could be multiple of such paths, which are locally minimal(in the sense that you cant nudge the path to make it shorter, globally there might be a shorter path). It should be noted, however, that in Minkowski space-time the definition is actually a bit more complicated, because of negative distances. But instead of paths between 2 points we're only interested in finding how a ray moves. So essentially we have a position in space and a direction of movement, and we would like to know how the direction of movement changes to minimize the lengthof the path the ray takes.

---

### Mathematical description of shortest path
*Here I'll try to very roughly explain the derivation, a more in-depth explanation would at least require a multi-part series of blog posts. And if you wish to skip over the math part, jump to [Writing a Hamiltonian geodesic tracer in GLSL](#writing-a-hamiltonian-geodesic-tracer-in-glsl).*

Mathematically speaking we have some coordinate system, a path, and a way to compute distances between 2 points. 

A coordinate system being a set of several numbers labeling each point in the space. A path is a function that takes in the path parameter and outputs a coordinate, in GR the path parameter is usually proper time(like a clock moving with the object), but it can be anything really. And the way to compute distances is called a metric, and it's the main source of scary math here.

In physics, or more generally differential geometry, a metric is defined as an integral("sum") of something called the metric tensor. A metric tensor is a bilinear form \\( g(a, b) \\), it essentially maps pairs of vectors to real numbers, and is a generalization of dot product for curved spaces. So using a metric tensor we can find the length of a vector in space, and distances \\( ds \\) between infinitely close points in space.
 
\begin{equation}
   ds^2 = g(dx, dx)
\end{equation}

In our case, where we describe vectors as a set of numbers, a metric is simply a matrix product of some matrix \\( g_{\mu \nu} \\) times the vectors. For our infinitesimal distance \\( ds \\) we get this expression:

\begin{equation}
  ds^2 = \sum_{\mu \nu}^N g_{\mu \nu} dx_\mu dx_\nu
\end{equation}

Usually the sum is just implicitly assumed by [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) [1].

\begin{equation}
 ds^2 = g_{\mu \nu} dx^\mu dx^\nu 
\end{equation} 

Here we can actually see that for some simple choices of \\( g_{\mu \nu} \\) we can get the distances by Pythagoras' theorem. Specifically for the case when the metric tensor matrix is a unit matrix.

\begin{equation}
 ds^2 = dx_1^2 + dx_2^2 + dx_3^2 
\end{equation}

For a flat space-time like space we actually get something similar but with the exception that the time coordinate component is with a negative sign.

\begin{equation}
 ds^2 = - dx_0^2 + dx_1^2 + dx_2^2 + dx_3^2 
 \label{flat}
\end{equation}

Here I used the \\( (- + + +) \\) signature, but signs can actually be flipped without changing the geodesics, and in some cases, like for particle physics, it makes more sense to use the opposite \\( (+ - - -) \\) signature.

Going back to the main question of computing distances, to compute the length between 2 points along some path we simply need to sum the infinitesimal distances together using an integral:

\begin{equation}
 l = \int_A^B \sqrt{g_{\mu \nu} dx^\mu dx^\nu} = \int_A^B \sqrt{g_{\mu \nu} dx^\mu dx^\nu} \frac{dt}{dt} = \int_A^B \sqrt{g_{\mu \nu} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt}} dt
\end{equation} 

Where \\( \frac{dx^i}{dt} \\) is simply how fast the coordinate x changes with respect to the path parameter ("clock"), in some sense can be interpreted as the velocity. 

Now our main question is how do we minimize the path length? Here is where we introduce a thing called calculus of variations, which is rouhtly speaking a way to find how a functional(distance) changes by varying its input function(path). Such derivatives has similar properties to normal function derivatives. And in fact, similarly to calculus, to find the extremum of a function(min, max or stationary point), we simply need to equate the variation to 0.

---

### Lagrangian description of a geodesic
There is an entire branch of physics related to variational principles, and basically any kind of physical system has some kind of value it likes to minimize(or more generally make unchanging under small variations of path). That value is called action, and the function under the integral is called the Lagrangian function of the system. The branch of physics studying Lagrangians of systems is called Lagrangian mechanics. 

In our case the Lagrangian can be written like this:

\begin{equation}
 L = \sqrt{g_{\mu \nu} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt}}
\end{equation}

Turns out we don't actually need the square root for the minimum of the functional to be a geodesic, we can simply use this as our geodesic Lagrangian:

\begin{equation}
 L = g_{\mu \nu} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt} 
\end{equation}

The proof of this you can find [here](https://physics.stackexchange.com/questions/149082/geodesic-equation-from-variation-is-the-squared-lagrangian-equivalent) [2]. The only difference such simplification makes is that the parametrization of the path might be different, but the path itself will be the same.

Also we want this with a 1/2 factor, to simplify the equations down the line.

\begin{equation}
 L = \frac{1}{2} g_{\mu \nu} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt} 
\end{equation}

So our goal right now is to minimize this functional:

\begin{equation}
 S = \int_A^B \frac{1}{2} g_{\mu \nu} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt} dt 
\end{equation} 

In general the minimum of a functional like this can be found by applying the [Euler-Lagrange equations](https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation) [3]:

\begin{equation}
  \frac{\partial L}{\partial x^i} - \frac{d}{dt} \frac{\partial L}{\partial \frac{dx^i}{dt}} = 0 
  \label{el}
\end{equation}

---

<details>
<summary>Euler-Lagrange equation derivation</summary>

In general the Action can be written like

\begin{equation*}
  S = \int_A^B  L(t, x(t), \frac{dx(t)}{dt}) dt 
\end{equation*}

Where the Lagrangian is a function of the path parameter("clock"), the path itself, and the derivative of the path with respect to the path parameter.

To find the minimizing path(or more generally, stationary path) of a functional we need to equate the variation of the action to 0

\begin{equation*}
  \delta S = 0
\end{equation*}

Where the variation of the action is found by adding a small variation \( \delta x \) to the path: \( L(t, x + \delta x, \frac{d(x + \delta x)}{dt}) \)

\begin{equation*}
  \delta S = \int_A^B \left( \frac{\partial L}{\partial x} \delta x + \frac{\partial L}{\partial \frac{dx}{dt}} \frac{d(\delta x)}{dt} \right) dt
\end{equation*}

We use integration by parts to get rid of the derivative \( \frac{d}{dt} \) off the path variation 

\begin{equation*}
  \delta S = \int_A^B \left( \frac{\partial L}{\partial x} \delta x - \frac{d}{dt} \frac{\partial L}{\partial \frac{dx}{dt}} \delta x \right) dt + \left( \frac{\partial L}{\partial \frac{dx}{dt}} \delta x \right) \biggr \rvert_A^B
\end{equation*}

Since we keep the endpoints of the path stationary the last term is equal to zero:

\begin{equation*}
  \delta S = \int_A^B \left( \frac{\partial L}{\partial x} \delta x - \frac{d}{dt} \frac{\partial L}{\partial \frac{dx}{dt}} \delta x \right) dt =
  \int_A^B \left( \frac{\partial L}{\partial x} - \frac{d}{dt} \frac{\partial L}{\partial \frac{dx}{dt}}  \right) \delta x  dt
\end{equation*}

Equating this to zero we get

\begin{equation*}
 \int_A^B \left( \frac{\partial L}{\partial x} - \frac{d}{dt} \frac{\partial L}{\partial \frac{dx}{dt}}  \right) \delta x dt = 0
\end{equation*}

Which holds true when the path satisfies the expression under the integral

\begin{equation*}
  \frac{\partial L}{\partial x} - \frac{d}{dt} \frac{\partial L}{\partial \frac{dx}{dt}} = 0
\end{equation*}

Which is the Euler-Lagrange equation!

</details>

You can find a more detailed derivation [here](https://mathworld.wolfram.com/Euler-LagrangeDifferentialEquation.html) [4]

---

Lets derive the Euler-Lagrange equations for our geodesic Lagrangian (there is an equation for each coordinate \\( x^i \\) ):

\begin{equation}
 \frac{\partial L}{\partial \frac{dx^i}{dt}} = 
    \frac{1}{2} \frac{\partial  }{\partial \frac{dx^i}{dt}} g_{\mu \nu} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt} = 
    \frac{1}{2} g_{i \nu} \frac{dx^\nu}{dt} + \frac{1}{2} g_{\mu i} \frac{dx^\mu}{dt} = 
    g_{i \nu} \frac{dx^\nu}{dt} 
\end{equation}

Then we take the derivative with respect to to the path parameter:

\begin{equation}
 \frac{d}{dt} \left( g_{i \nu} \frac{dx^\nu}{dt} \right) =   \frac{d g_{i \nu} }{dt}  \frac{dx^\nu}{dt} + g_{i \nu} \frac{d^2x^\nu}{dt^2} = 
 \frac{d g_{i \nu} }{dx^\mu} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt} +  g_{i \nu} \frac{d^2x^\nu}{dt^2} 
\label{el1}
\end{equation}

And lastly:

\begin{equation}
 \frac{\partial L}{\partial x^i} = \frac{1}{2} \frac{d g_{\mu \nu} }{dx^i} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt} 
 \label{el2}
\end{equation}

Substituting \eqref{el1} and \eqref{el2} into Euler-Lagrange equations \eqref{el} leads us to the equation of a geodesic:

\begin{equation}
 \frac{1}{2} \frac{d g_{\mu \nu} }{dx^i} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt} - \frac{d g_{i \nu} }{dx^\mu} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt} - g_{i \nu} \frac{d^2x^\nu}{dt^2} = 0 
\end{equation}

Multiplying by the metric tensor inverse \\( - g^{i \nu} \\) we get:

\begin{equation}
 \frac{d^2x^i}{dt^2} +  g^{i \nu} \left( \frac{d g_{i \nu} }{dx^\mu} - \frac{1}{2} \frac{d g_{\mu \nu} }{dx^i}  \right) \frac{dx^\mu}{dt} \frac{dx^\nu}{dt} = 0 
\end{equation}

And that's our final system of equations for a geodesic, we could of course also substitute the Christoffel symbols here, but for our application there is no difference. Of course we could just use these equations for tracing geodesic rays and call it a day, but unfortunately this would require computing a whole lot of derivatives (in 4d space time it's 64 of them to be specific), either manually, or by using numerical differentiation. Thankfully there is a way to avoid this, and in fact simplify the entire algorithm! (at a slight performance cost)

---

### Hamiltonian description of a geodesic
So here comes the star of the show - Hamiltonian mechanics. Hamiltonian equations of motion have a really nice form which allows to easily write a computer program that integrates them by using Euler integration.

\begin{equation}
 \frac{dp^i}{dt} = - \frac{\partial H}{\partial x^i} 
\end{equation}
\begin{equation}
 \frac{dx^i}{dt} =   \frac{\partial H}{\partial p^i} 
\end{equation}

The derivation of Hamilton's equations of motion can be found [here](https://en.wikipedia.org/wiki/Hamiltonian_mechanics#Deriving_Hamilton's_equations) [5].
\\( p \\) is the so called generalized momentum, it's the derivative of the Lagrangian with respect to the path parameter.

\begin{equation}
 p_i = \frac{\partial L}{\partial \frac{dx^i}{dt} } 
\end{equation}

And to get the Hamiltonian itself you need to apply the [Legendre Transform](https://blog.jessriedel.com/2017/06/28/legendre-transform/) [6] on the Lagrangian:

\begin{equation}
 H = \sum_{i}^N p^i \frac{dx^i}{dt} - L 
\end{equation}

And for our case the momentum would be the following, which we already computed when writing down the Euler-Lagrange equations:

\begin{equation}
 p_i = g_{i j} \frac{dx^j}{dt} 
 \label{momentum}
\end{equation}

To get the "time" derivatives you simply need to multiply both sides by the metric tensor inverse:

\begin{equation}
 \frac{dx^i}{dt} = g^{i j} p_j 
 \label{dxdt}
\end{equation}

And the Hamiltonian itself:

\begin{equation}
 H = \sum_{i}^N \frac{dx^i}{dt} p_i - L =  g_{i j} \frac{dx^i}{dt} \frac{dx^j}{dt} - \frac{1}{2} g_{i j} \frac{dx^i}{dt} \frac{dx^j}{dt} =  \frac{1}{2} g_{i j} \frac{dx^i}{dt} \frac{dx^j}{dt} = L
\end{equation}

Turns out that for this simple choice of a geodesic Lagrangian, the Hamiltonian is equal to the Lagrangian!

Also we want to know the Hamiltonian as a function of the generalized momentum by substituting \eqref{dxdt} into the Hamiltonian equation:

\begin{equation}
 H = \frac{1}{2} g_{i j} \frac{dx^i}{dt} \frac{dx^j}{dt} = \frac{1}{2} g^{i j} p_i p_j 
 \label{hamiltonian}
\end{equation}

While the equations of motion will simply be:

\begin{equation}
 \frac{dp_i}{dt} = - \frac{\partial H}{\partial x^i} 
 \label{eqmotion1}
\end{equation}

\begin{equation}
 \frac{dx^i}{dt} = g^{i j} p_j 
 \label{eqmotion2}
\end{equation}

And this is all we need to write a numerical geodesic integrator! 

---

### Writing a Hamiltonian geodesic tracer in GLSL

You might have noticed that in the final Hamilton's equations of motion I didn't write out \\( \frac{\partial H}{\partial x^i} \\), this is actually important! We want to keep the derivative of the Hamiltonian as is, because then instead of computing the 64 derivatives of the metric tensor, we only need 4 to find the Hamiltonian gradient. This is the main simplification of the geodesic tracing algorithm.

Here we will use the GLSL shading language, since it has variables and functions which map quite well to the mathematical operations we will perform here. On top of that we can easily then make a real time GR visualization shader. 

First of all we need a function that evaluates the metric tensor at a 4d point in space and time. Let's use the [Alcubierre warp drive](https://en.wikipedia.org/wiki/Alcubierre_drive) [7] metric as an example, since it is quite simple.

```glsl

mat4 Metric(vec4 x)
{
  //Alcubierre metric  
  const float R = 1.0;
  const float sigma = 35.0; 
  const float v = 1.1;

  float x = v*x.x;
  float r = sqrt(sqr(x.y - x) + x.z*x.z + x.w*x.w);
  float f = 0.5*(tanh(sigma*(r + R)) - tanh(sigma*(r - R)))/tanh(sigma*R);
  float gtt = v*v*f*f - 1.0;
  float gxt = -v*f;
  
  return mat4(gtt, gxt,  0,  0,
              gxt,   1,  0,  0,
                0,   0,  1,  0,
                0,   0,  0,  1);
}

```

In our case x is a 4D vector representing position. The first component `x.x` or `x[0]` being time. As an output we get a 4 by 4 matrix represented by `mat4` in GLSL. 

Then we need to write down the Hamiltonian \eqref{hamiltonian}. The Hamiltonian is a function that takes 2 things, the position in space time, and the 4d momentum vector, and outputs a scalar.

```glsl

float Hamiltonian(vec4 x, vec4 p)
{
  mat4 g_inv = inverse(Metric(x));
  return 0.5*dot(g_inv*p,p);
}

```

As a bonus here is the Lagrangian

```glsl

float Lagrangian(vec4 x, vec4 dxdt)
{
  return 0.5*dot(Metric(x)*dxdt,dxdt);
}

```

Surprisingly enough thats it, GLSL already has a matrix inverse function `inverse()`, on top of it the Hamiltonian is just the dot product(in GLSL sense) of `g_inv*p` and `p`, which are the contravariant and covariant momentum vectors respectively. The contravariant momentum actually just being the time derivative of the coordinate `dxdt`, i.e. `dot(dxdt,p)`.

After this we need to compute the 4D gradient of the Hamiltonian. We can do this by using a forward numerical difference in all 4 spacial directions, using some small value `eps`:

```glsl

vec4 HamiltonianGradient(vec4 x, vec4 p)
{
  const float eps = 0.001;
  return (vec4(Hamiltonian(x + vec4(eps,0,0,0), p),
               Hamiltonian(x + vec4(0,eps,0,0), p),
               Hamiltonian(x + vec4(0,0,eps,0), p),
               Hamiltonian(x + vec4(0,0,0,eps), p)) - Hamiltonian(x,p))/eps;
}

```

Now that we have the Hamiltonian gradient we can finally write down the equation of motion \eqref{eqmotion1} \eqref{eqmotion2} integration code

```glsl

vec4 IntegrationStep(inout vec4 x, inout vec4 p)
{
  const float TimeStep = 0.1;
  p = p - TimeStep * HamiltonianGradient(x, p);
  x = x + TimeStep * inverse(Metric(x)) * p;
}

```

You might ask, "wait, that's it?", and ideed that is all you need to integrate the geodesic. Of course it is quite slow since we do a whopping 6 matrix inverse evaluations, which can be optimized down to 1 actually, by replacing most Hamiltonians with Lagrangians which dont have inverses, since they are equal. Even better is to have the metric inverse already computed analytically, but its not possible for every metric, especially for an implicitly defined one.

There is of course the last problem, while initializing the space-time position is easy, how do we initialize the value of the momentum vector `p` when starting to trace?

Before tracing the geodesic you can use the equation \eqref{momentum}

```glsl

p = Metric(x) * dxdt;

```
But what is dxdt? It's nothing more than the 4D direction the ray moves inside space time. There are actually 3 categories the directions can fall into:
* Time-like, when \\( A < 0 \\)
* Null, when \\( A = 0 \\)
* Space-like, when \\( A > 0 \\)

Where \\(A\\) is
\begin{equation}
  A = g_{\mu \nu} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt} 
\end{equation}

or in GLSL `A = dot(Metric(x) * dxdt, dxdt)`

When simulating how light travels we just want null directions, which lead to null geodesic solutions. On the other hand if you want to model an object moving slower than light you need a time-like geodesic. And space-like geodesics for tachionic stuff, which doesn't happen in real life, so we ignore it.

So assuming a flat space metric \eqref{flat} and some 3D direction in space our p for a light ray would be

```glsl

vec4 GetNullMomentum(vec3 dir)
{
  return Metric(x) * vec4(1.0, normalize(dir));
}

```

And the inverse of this operation

```glsl

vec3 GetDirection(vec4 p)
{
  vec4 dxdt = inverse(Metric(x)) * p;
  return normalize(dxdt.yzw);
}

```

So in the end the final simple algorithm will look like this:

```glsl

void TraceGeodesic(inout vec3 pos, inout vec3 dir, inout float time)
{
  vec4 x = vec4(time, pos);
  vec4 p = GetNullMomentum(dir);

  const int steps = 256;
  for(int i = 0; i < steps; i++)
  {
    IntegrationStep(x, p);
    //you can add a stop condition here when x is below the event horizon for example
  }

  pos = x.yzw;
  time = x.x;
  dir = GetDirection(p);
}

```

Also here is a shadertoy example of this algorithm in action together with some optimizations and Kerr-Newman metric. (Restart if black)

<center><iframe width="640" height="360" frameborder="0" src="https://www.shadertoy.com/embed/NtSGWG?gui=true&t=10&paused=false&muted=false" allowfullscreen></iframe></center>

---

### Conclusions

---

### References 
* [1] [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation)
* [2] [Equivalence of squared Lagrangian to Lagrangian](https://physics.stackexchange.com/questions/149082/geodesic-equation-from-variation-is-the-squared-lagrangian-equivalent)
* [3] [Euler-Lagrange equations](https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation) 
* [4] [Euler-Lagrange equations derivation](https://mathworld.wolfram.com/Euler-LagrangeDifferentialEquation.html)
* [5] [Hamiltonian equations derivation](https://en.wikipedia.org/wiki/Hamiltonian_mechanics#Deriving_Hamilton's_equations)
* [6] [Legendre Transform](https://blog.jessriedel.com/2017/06/28/legendre-transform/)
* [7] [Alcubierre metric](https://en.wikipedia.org/wiki/Alcubierre_drive)

