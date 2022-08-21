---
layout: post
title: Overview of Shadertoy particle algorithms
image: ShadertoyParticles.png
---

Initially Shadertoy was created mostly as a shader sharing website made by demoscener's for demoscener's. At first you could just add a single fragment shader, which was enough to make a lot of impressive looking shader visual demos and rendering tech demos, which quite often utilized ray marching techniques. Since there was no framebuffer feedback back then the only form of interactivity shaders had was by using the mouse/keyboard inputs. But starting from the release version 0.8 in 2016 Shadertoy finally got multipass rendering which expanded the scope of possible shaders quite considerably. You might have expected that having 4 float32 buffers and a single 1024^2 float16 cubemap without any vertex shaders would still be quite limiting, only allowing some multipass effects like gaussian blurs, screen space ambient occlusion, progressive path tracing and maybe storing a state inside a buffer to implement simple camera controls. But surprisingly(or probably not) over the years people figured out ways to do things that look pretty much like black magic, considering the given costraints, some of the crazy examples being [a Windows 98 like GUI](https://www.shadertoy.com/view/Xs2cR1), [an Apple II emulator](https://www.shadertoy.com/view/tlX3W7)(?!) [and a playable DOOM port](https://www.shadertoy.com/view/lldGDr)(how?!). Naturally there are a whole lot more of those, you can check out a few lists on [Fabrice's Shadertoy Unofficial blog](https://shadertoyunofficial.wordpress.com/).

Considering all that you probably won't be surprised if I told that Shadertoy users figured out ways to render and simulate up to millions of particles using a few fragment shaders. In this blog post I will try to classify the algorithms used for this, with a general description of the methods and some examples.

---
# A simple loop over all particles
This one is pretty obvious and doesn't need any explanations. It's pretty expensive, especially for rendering, but surprisingly you can get up to a few hundred particles working with a reasonable framerate this way.

Pros:
* Simple
* No constraints on particles
* Stable simulation 

Cons:
* Extremely slow, scales as \\( O(n^2) \\)

Examples:
* [Cocktail Shaker by xjorma](https://www.shadertoy.com/view/3dVyzt)
* [Freezing and Melting by dr2](https://www.shadertoy.com/view/3dj3Wy)
* [SPH fluid 2d by kuma720](https://www.shadertoy.com/view/tslSWM)
* Simulation only: [Basic : SPH Fluid by Gijs](https://www.shadertoy.com/view/WlfyDM)


---
# Using mipmaps to sum forces
Basically the idea is to compute the particle pair forces inside a, sort of, matrix, where each pixel represents the force between particle i and particle j. If all forces acting on a particle are inside of a \\( 2^n*2^n \\) block of pixels then we can take the n-th level mipmap and effectively compute the average of all the forces, to recover the sum we can then just multiply by \\( 2^{2n} \\)

This method can only be used for the interaction, so to render the particles you need to use another method from this list.

Pros:
* Relatively simple
* Quite a bit faster than a simple loop
* No constraints on particles
* Stable simulation 

Cons:
* Still scales as \\( O(n^2) \\)
* The maximal number of particles is limited by the square root of the total number of pixels, using the cubemap buffer one can get up to 1024 interacting particles

Example:
* [MIPmap interaction by rory618](https://www.shadertoy.com/view/wdfcRN)

---
# Pipelined particle sort

Pros:
* Pretty fast, scales as \\( O(nlog(n)) \\) 

Cons:
* A \\( log(n) \\) frame delay, makes interactivity practically impossible
* Pretty complicated



### BVH based rendering

Examples:
* [Fast 2D BVH by rory618](https://www.shadertoy.com/view/WssfDn)
* [Pixie dust by rory618](https://www.shadertoy.com/view/wllcR7)

### Morton curve search based rendering

Examples:
* [Z Particle Sort Pipeline by emh](https://www.shadertoy.com/view/Mtdyzs)
* [Torus, Grid and Sentinel by emh](https://www.shadertoy.com/view/ltcyW2)

---
# Voronoi particle tracking
This one is probably one of the most interesting of the bunch, the main idea is to try to store the closest particle to this pixel, this naturally leads to the formation of a [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram), thus the name - Voronoi particle tracking. There are different ways to construct such a diagram, but the main way is to sort the closest particles by performing a random seach over the particle array and by checking the current closest particles to neighboring pixels in a similar way to a [jump flood algorithm(JFA)](https://www.comp.nus.edu.sg/~tants/jfa.html).

Another important point that makes this algorithm work is the assumption that after 1 frame the diagram didn't change too much, so we can reuse the previous frame diagram as the initial guess to construct a new one, this is important considering we can only do 4 passes per frame in Shadertoy, which is not enough to do a full JFA voronoi diagram construction. The assumption leads to a limit on the particle velocity, and if a particle went too far from its original position, the inital voronoi diagram guess might not work, and as a result we will lose a particle. 

Pros:
* Extremely fast, scales linearly with the number of pixels \\( O(nm) \\) (m is the number of pixel neighbors + random searches)
* Can be used both for rendering and for simulation

Cons:
* Simulation domain is limited by the voronoi diagram
* Particles have a velocity limit
* Particles can be lost
* Maximum particle density limited to the number of closest particles stored per pixel

## The implicit way
There are mainly 2 ways to store the particles on the diagram, the first one is to simulate the particle in a separate array and track only their ID numbers in each pixel of the voronoi diagram, this is the more natual way to do this. 

You can also store multiple closest particles and do a simple insertion sort within each pixel, this has the additional perk of making it easy to find particle neighbors, and as for rendering, this removes the visible lines between the voronoi cells if the particles are overlapping.

Examples:
* [Super SPH](https://www.shadertoy.com/view/tdXBRf)
* [Storing 4 particle ID's per pixel](https://www.shadertoy.com/view/tdscRr)
* [Storing 8 particle ID's per pixel](https://www.shadertoy.com/view/tslyzH)
* [Bokeh particles - 8x voronoi](https://www.shadertoy.com/view/3dsczN)

## The explicit way
The other way is to store the particle state(velocity, position, etc...) directly in each pixel of the diagram, so we don't need to store ID's. This has both a serious drawback and a nice advantage. Since we are storing the same state in each of the voronoi cell pixels we are doing lots of wasted compute, essentially doing the same particle computation in every pixel, on top of that we need to make sure it does not diverge between pixels, otherwise we are no longer tracking the same particle, and in this lies the advantage of this method, if the particle state starts diverging between pixels of the diagram cell we are essentially creating new particles out of thin air, we can delete particle in practically the same way too.

Examples:
* [zoomable, stored voronoi cells by stb](https://www.shadertoy.com/view/ldGGWK)
* [chaotic particle swarm 2 by FabriceNeyret2](https://www.shadertoy.com/view/WtK3zt)


## Extension to 3D screen space ray tracing
On top of tracking particles on a plane nothing stops us from tracking other 2d shapes with a given signed distance, or even tracking 3D objects by their intersection distance from the camera. This way we can track ray traced objects in screen space.  

Examples:
* [SPH 3D](https://www.shadertoy.com/view/ws2fzz)
* [Underwater boids](https://www.shadertoy.com/view/WdSfzD)

### [A playlist with tons of voronoi tracking examples](https://www.shadertoy.com/playlist/4XdXR8)


---
# Graph particle methods

### Using a neighbor graph for the simulation

Examples:
* This one also uses a stochastic sphere rasterizer for the particles, really cool: [Particle fluid by rory618](https://www.shadertoy.com/view/MsdfW4)
* Seems to be some kind of voronoi/graph hybrid, not sure: [Per-pixel particle structure by emh](https://www.shadertoy.com/view/XttcWM)

### Using a graph for particle rendering


### [A playlist with tons of graph examples](https://www.shadertoy.com/playlist/X3tSz8)


---
# Cellular automaton particle/density tracking

A more extensive description of these methods is given here: [Post about Reintegration tracking](https://michaelmoroz.github.io/Reintegration-Tracking/)


## Cellular automaton particle tracking

### Simple CA particle tracking 
There are a lot of custom implementations of this kind of particle tracking. They all seem to be quite different but the main idea is pretty much the same - track how particles move on a grid by exchanging particle information to cell neighbors in a ceirtain radius.

Pros:
* Extremely fast, scales as \\( O(mn) \\), where m is the number of neighbor cells
* Allows for an insane number of particles

Cons:
* Density limited to 1 particle per pixel
* You can lose particles
* Simulation domain is limited by the grid
* Velocity and interaction radius is limited by the neighbor search radius 

Examples:
* [lots o' particles by stb](https://www.shadertoy.com/view/MdtGDX)
* [Liquid Experiment by P_Malin](https://www.shadertoy.com/view/XdGGWD)
* [Liquid stuff by Nrx](https://www.shadertoy.com/view/4sK3zm)
* [Hot liquid metal by Nrx](https://www.shadertoy.com/view/4sKGDK)
* [Fluid physics by lomateron](https://www.shadertoy.com/view/wdsGWS)

### Conservative CA particle tracking
Mostly just an extension I made to solve the density and particle loss issue, while preserving the total particle number. Simply just dividing particles into integer chunks.

Examples:
* [CA Molecular dynamics](https://www.shadertoy.com/view/3s3cWr)
* [CA Neo-Hookean](https://www.shadertoy.com/view/WdGyWR)
* [CA Paste](https://www.shadertoy.com/view/tsGczh)

### [A playlist with tons of CA tracking examples](https://www.shadertoy.com/playlist/4XtXR8)

## Reintegration tracking
Examples:
* Arguably one of the coolest fluid sims in Shadertoy: [Condensed Cloud Dispenser by UnstableLobster](https://www.shadertoy.com/view/WlcfDs)


### [A playlist with tons of reintegration tracking examples](https://www.shadertoy.com/playlist/XXtXR8)

P.S. The main point of this blog post is to increase the number of people that know about those algorithms, so that maybe someone comes up with something even cooler in the future, as well as maybe figure out useful ways to use some of these outside of Shadertoy.