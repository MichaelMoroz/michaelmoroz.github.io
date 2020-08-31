---
layout: post
title: Reintegration tracking
---

In this blog post I'll explain the advection algorithm I used in my latest simulations, including [Paint streams](https://www.shadertoy.com/view/WtfyDj) and [Everflow](https://www.shadertoy.com/view/ttBcWm).
Before starting I should give a big thanks to my friend [Wyatt](https://www.shadertoy.com/user/wyatt) for giving useful suggestions on building this algorithm.

<center><iframe style="width:640px;height:360px;" frameborder="0" src="https://www.shadertoy.com/embed/WtfyDj?gui=true&amp;t=10&amp;paused=false" allowfullscreen=""></iframe></center>

Looks really smooth doesn't it? It even captures droplets with almost close to pixel level precision. And while it does model a fluid with a boundary it does not use particles directly like in SPH, but is actually a semi-Lagrangian grid based algorithm. 

Let's start with a brief history, the initial idea for this algorithm came from trying to extend screen space [voronoi particle tracking](https://www.facebook.com/groups/shadertoy/?post_id=567902837124080) to avoid particle loss, indeed when trying to store the particle state in screen space as close to the particle as we can, in the case of overlap one of the particles will be lost, and there is no way to avoid it. But actually we can reformulate the problem, what if we don't try to avoid particle overlap but tried to conserve the total mass? Like by adding the overlapping particle masses together and weighting their velocities and positions by mass. 
Actually that was already tried by [stb](https://www.shadertoy.com/view/MdtGDX), but it immidiatly becomes apparent that the total number of particles drops proportionally to the particle/pixel density because of them combining together, you might ask if there a way to separate them back, or if there is a way to achieve approximate particle number conservation? In fact, it is possible, but let's overview how he does the particle tracking first.

## Cellular automaton particle tracking
This algorithm is in fact similar to a [lattice gas automaton](https://en.wikipedia.org/wiki/Lattice_gas_automaton) where the state of each cell is defined by a few discrete levels, is the particle in the cell and in which discrete direction it is moving. In our case instead of using a few discrete states we store the position, velocity and mass of the particle within the cell as floats *(or any other number type, depends on the precision you want to achieve, I actually used 2 int16's per channel since Shadertoy only has a 4 channel output per pixel and I needed to store at least 5 numbers)* 

<center>
<table>
  <tr>
    <th><img src="{{ site.baseurl }}/images/ParticleCAframe1.JPG" style="width:250px;height:250px;"></th>
    <th><img src="{{ site.baseurl }}/images/ParticleCAframe2.JPG" style="width:250px;height:250px;"></th>
  </tr>
  <tr>
    <th><b>Initial Frame</b></th>
    <th><b>Next frame</b></th>
  </tr>
</table>
</center>


```glsl
//this cell position
ivec2 p;
//values stored in the cell
vec2 vel, pos;
float mass;
//find and average the particles 
//that land in this cell after a time step dt

//only check the neighbors at radius R
for(int x = -R; x <= R; x++)
{
    for(int y = -R; y <= R; y++)
    {
        particle P = getParticle(p + ivec2(x,y));
    } 
}
```
