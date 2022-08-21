---
layout: post
title: Visualizing General Relativity
image: ReintegrationTracking.png
---

### Contents
* Introduction
* What are geodesics?
* Mathematical description of shortest path
* Lagrangian mechanics description of shortest path
* Hamiltonian mechanics and Legandre transform
* Hamiltonian equations of motion for a geodesic
* Writing this as code

When dealing with renders of things like warp drives and black holes we usually just expect to see a simple approximation or an artist rendition, usually just assuming that the math required to pull off something accurate would require someone with at least a PhD in Mathematical Physics, which in most cases is somewhat true, but not necessarily. In this blog post I'll try to explain a way to do actually accurate visualizations within a 100 or so lines of code, for basically any kind of space time for which you can write its metric as code. The detailed mathematical derivation of this approach might be somewhat math heavy though.

The main ingredient of any GR render is figuring out how the rays of light move around. Knowing how light moves we can trace rays from the camera into the scene as see where the light came from. So to render a basic scene without objects we simply trace a ray for each pixel and assignt the color of the pixel to the color of the skybox in the direction in which the ray ends up pointing to. 

So how exactly do we trace rays in curves space? Any object inside a curved space follows something called a geodesic.

A geodesic is essentially just a fancy word for path of shortest lenght between 2 points inside a space, and actually there could be multiple of such paths, which are locally minimal(in the sense that you cant nudge the path to make it shorter, globally there might be a shorter path). It should be noted, however, that in Minkovski space-time the definition is actually a bit more complicated, because of negative distances. But instead of paths between 2 points we're only interested in finding how a ray moves. So essentially we have a position in space and a direction of movement, and we would like to know how the direction of movement changes to minimize the lenght of the path the ray takes.

*Here I'll try to very roughly explain the derivation, a more in-depth explanation would at least require a multi-part series of blog posts. And if you wish to skip over the math part, jump to the last part.*

Mathematically speaking we have some coordinate system, a path, and a way to compute distances between 2 points. 

A coordinate system being a set of several numbers labeling each point in the space. A path is a function that takes in the path parameter and outputs a coordinate, in GR the path parameter is usually proper time(like a clock moving with the object), but it can be anything really. And the way to compute distances is called a metric, and it's the main source of scary math here.

In physics, or more generally differential geometry, a metric is defined as an integral("sum") of something called the metric tensor. A metric tensor is a bilinear form \\( g(a, b) \\), it essentially maps pairs of vectors to real numbers, and is a generalization of dot product for curved spaces. So using a metric tensor we can find the length of a vector in space, and distances \\( ds \\) between infinitely close points in space.

\\[ ds^2 = g(dx, dx) \\] 

In our case, where we describe vectors as a set of numbers, a metric is simply a matrix product of some matrix \\( g_{ij} \\) times the vectors. For our infinitessimal distance we get this expression:

\\[ ds^2 = \sum_{ij}^N g_{ij} dx_i dx_j \\] 

Usually the sum is just implicitly assumed by [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) [1].

\\[ ds^2 = g_{ij} dx^i dx^j \\] 

Here we can actually see that for some simple choices of \\( g_{ij} \\) we can get the distances by Pythagoras' theorem. Specifically for the case when the metric tensor matrix is a diagonal.

\\[ ds^2 = dx_1^2 + dx_2^2 + dx_3^2 \\]

For a flat space-time like space we actually get something similar but with the exception that the time coordinate component is with a negative sign.

\\[ ds^2 = - dx_0^2 + dx_1^2 + dx_2^2 + dx_3^2 \\]

Here I used the (-+++) signature, but signs can actually be flipped without changing the geodesics, and in some cases, like for particle physics, it makes more sense to use the opposite (+---) signature.

Lets go back to the main question of computing distances, to get the lenght between 2 points along some path we simply need to sum the infinitessimal distances together using an integral:

\\[ l = \int_A^B \sqrt{g_{ij} dx^i dx^j} = \int_A^B \sqrt{g_{ij} dx^i dx^j} \frac{dt}{dt} = \int_A^B \sqrt{g_{ij} \frac{dx^i}{dt} \frac{dx^j}{dt}} dt\\] 

Where \\( \frac{dx^i}{dt} \\) is simply how fast the coordinate x changes with respect to the path parameter ("clock"), in some sense can be interpreted as the velocity. 

Now our main question is how do we minimize the path length? Here is where we introduce a thing called calculus of variations, which is rouhtly speaking a way to find how a functional(distance) changes by varying its input function(path). Such derivatives has similar properties to normal function derivatives. And in fact, similarly to calculus, to find the extremum of a function(min, max or stationary point), we simply need to equate the variation to 0.

There is an entire branch of physics related to variational principles, and basically any kind of physical system has some kind of value it likes to minimize(or more generally make unchanging under small variations of path). That value is called action, and the function under the integral is called the Lagrangian function of the system. The branch of physics studying Lagrangians of systems is called Lagrangian mechanics. 





...
TODO





### References 
[1] [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation)
[1] [Legandre Transform](https://blog.jessriedel.com/2017/06/28/legendre-transform/)
