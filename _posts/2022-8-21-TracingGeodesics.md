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

When dealing with renders of things like warp drives and black holes we usually just expect to see a simple approximation or an artist rendition, usually just assuming that the math required to pull off something accurate would require someone with at least a PhD in Mathematical Physics, which in most cases is somewhat true, but not necessarily. In this blog post I'll try to explain a relatively straight forward way to do actually accurate visualizations within a 100 or so lines of code, for basically any kind of space time for which you can write its metric as code.

The main ingredient of any GR render is figuring out how the rays of light move around. Knowing how light moves we can trace rays from the camera into the scene as see where the light came from. So to render a basic scene without objects we simply trace a ray for each pixel and assignt the color of the pixel to the color of the skybox in the direction in which the ray ends up pointing to. 

So how exactly do we trace rays in curves space? Any object inside a curved space follows something called a geodesic.

A geodesic is essentially just a fancy word for path of shortest lenght between 2 points inside a space. In general there might be any number of geodesics between 2 points in a curved space, but we're interested in finding the how a ray moves, not the a path between 2 points. So essentially we have a position in space and a direction of movement, and we would like to know how the direction of movement changes to minimize the lenght of the path the ray takes.

*If you wish to skip over the math part deriving how this works, jump to the last part.*

Mathematically speaking we have some coordinate system, a path, and a way to compute distances between 2 points. 

A coordinate system being a set of several numbers labeling each point in the space. A path is a function that takes in the path parameter and outputs a coordinate, in GR the path parameter is usually proper time(like a clock moving with the object), but it can be anything really. And the way to compute distances is called a metric, and it's the main source of scary math here.

In physics, or more generally differential geometry, a metric is defined as an integral("sum") of something called the metric tensor. A metric tensor is a bilinear form \\( g(a, b) \\), it essentially maps pairs of vectors to real numbers, and is a generalization of dot product for curved spaces. So using a metric tensor we can find the length of a vector in space, and distances \\( ds \\) between infinitely close points in space.

\\[ ds^2 = g(dx, dx) \\] 

In our case, where we describe vectors as a set of numbers, a metric is simply a matrix product of some matrix times the vectors

\\[ ds^2 = \sum_{ij}^N g_{i,j} dx_i dx_j \\] 


Length:

\\[ l = \int_A^B \sqrt{\sum_{ij}^N g_{i,j} dx_i dx_j} dt \\] 

...
TODO





### References 
[1] [Legandre Transform](https://blog.jessriedel.com/2017/06/28/legendre-transform/)
