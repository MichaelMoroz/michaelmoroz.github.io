---
layout: post
title: Reintegration tracking
image: ReintegrationTracking.png
---

In this blog post I'll explain this advection algorithm and how to use it to make advanced fluid simulations like the ones I made, including [Paint streams](https://www.shadertoy.com/view/WtfyDj) and [Everflow](https://www.shadertoy.com/view/ttBcWm).
Before starting I should give a big thanks to my friend [Wyatt](https://www.shadertoy.com/user/wyatt) for giving useful suggestions on building this algorithm.

<center><iframe style="width:640px;height:360px;" frameborder="0" src="https://www.shadertoy.com/embed/WtfyDj?gui=true&amp;t=10&amp;paused=false" allowfullscreen=""></iframe></center>

Looks really smooth doesn't it? It even captures droplets with almost close to pixel level precision. And while it does model a fluid with a boundary it does not use particles directly like in SPH, but is actually a semi-Lagrangian grid based algorithm. 

Let's start with a brief history, the initial idea for this algorithm came from trying to extend screen space [voronoi particle tracking](https://www.facebook.com/groups/shadertoy/?post_id=567902837124080) to avoid particle loss, indeed when trying to store the particle state in screen space as close to the particle location as we can, in the case of overlap one of the particles will be lost, and there is no way to avoid it. But actually we can reformulate the problem, what if we don't try to avoid particle overlap but tried to conserve the total mass? Like by adding the overlapping particle masses together and weighting their velocities and positions by mass. 
Actually that was already tried by [stb](https://www.shadertoy.com/view/MdtGDX), but it immidiatly becomes apparent that the total number of particles drops proportionally to the particle/pixel density because of them combining together, you might ask if there is a way to separate them back, or if there is a way to achieve approximate particle number conservation? In fact, it is possible, but let's overview how he does the particle tracking first.

### Cellular automaton particle tracking
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

The main idea of the algorithm is to loop over all neighbors of the current cell(including itself), integrate the position of each neighbor particle and add it if it ends up in this cell. Above is a visualization of how it looks like. Each cell has a particle with a mass, the mass is shown by opacity, so 0 mass is invisible and 1 is completely dark. The red cell is our current cell for which we want to find its future state, the future position of the particles is shown by the arrow. We can see that there is more than one particle moving into the red cell. So in the end all of the particles that end up in the same cell are summed and their positions and velocities are averaged. It's quite a simple algorithm, but it has one limitation, to ensure that we counted every possible particle that might end up in this cell we would either need to loop over all the cells, which is highly expensive, or limit the maximum velocity of the particles to make the search radius finite, and hopefully only 1 pixel wide. 

Obviously the second option is much better in our context, and in fact if we want to track particles with velocities so fast that they traverse the grid in 1 frame its actually cheaper to do lots of smaller steps with a 1 pixel neighborhood instead of counting all cells in one step, since \\( 9 \sqrt{N} < N \\) , where N is the number of cells, and the square root is because we need to apply the operation only a linear amount of times, instead of applying it for every cell. And the 9 is the number of neighbors. In 3D it would equivalently be  \\( 27 N^{1/3} < N \\)  which is even more efficient. I should note that more steps does not mean better quality, since the particles may combine together on the way.

Here is a pseudo-code glsl implementation of this algorithm:
```glsl
//this cell position
ivec2 pos;
//values stored in the cell
vec2 velocity = vec2(0.), position = vec2(0.);
float mass = 0.;
//find and average the particles 
//that land in this cell after a time step dt
for(int x = -R; x <= R; x++) //only check the neighbors at radius R
  for(int y = -R; y <= R; y++)
  {
      //get the particle in this neighbor cell from the previous frame
      particle P = getParticle(pos + ivec2(x,y));
      //integrate the particle position
      P.X += P.V*dt;
      //check if the particle is inside of this cell
      if(inCell(P, pos))
      {
        mass += P.M; //add the particle mass to this cell
        position += P.X*P.M; //add the particle position weighted by mass
        velocity += P.V*P.M; //add the particle velocity weighted by mass(momentum)
      }
  } 

//normalize
if(mass > 0.0) //if not vacuum
{
  position /= mass; //center of mass
  velocity /= mass; //average velocity
}
```

### Dividing particles

To combat the problem of a decreasing particle number we can try and divide each particle into M virtual particles with distributed positions that might end up in different cells and thus increase the total number of particles. 
<center>
<img src="{{ site.baseurl }}/images/ParticleCA_div_frame1.JPG" style="width:250px;height:250px;">
</center>

The radius of the distribution defines how likely is the particle is to multiply. If the all the virtual particles end up inside of a single cell the particle would not divide and stay essentially the same(if the average particle of the virtual particles is equal to the origial particle). To make it properly conservative we just need to need to divide the mass of the particle into a number of equal chunks, and make sure the distribution average is zero.

On the picture above and in the code below we see an example for a 5 virtual particle distribution, the average of the distribution directions is zero and its radius is 0.1 pixes.

Here is the code, the only addition are the diffusion directions and a loop for each of the virtual particles
```glsl
//this cell position
ivec2 pos;
//values stored in the cell
vec2 velocity = vec2(0.), position = vec2(0.);
float mass = 0.;

//diffusion radius
float difR = 0.1;
//diffusion directions
vec2 difDir[5] = {vec2(0,0),vec2(1,0),vec2(-1,0),vec2(0,1),vec2(0,-1)};

//find and average the particles 
//that land in this cell after a time step dt
for(int x = -R; x <= R; x++) //only check the neighbors at radius R
  for(int y = -R; y <= R; y++)
  {
      //get the particle in this neighbor cell from the previous frame
      particle P = getParticle(pos + ivec2(x,y));
      //integrate the particle position
      P.X += P.V*dt;
      for(int i = 0; i < 5; i++) //divide particle into 5
      {
        particle difP = P;
        difP.X += difR*difDir[i]; //move particle in one of the diffusion directions
        difP.M /= 5.0; //divide mass into 5 particles

        //check if the divided particle is inside of this cell
        if(inCell(difP, pos))
        {
          mass += difP.M; //add the particle mass to this cell
          position += difP.X*difP.M; //add the particle position weighted by mass
          velocity += difP.V*difP.M; //add the particle velocity weighted by mass(momentum)
        }
      }
  } 

//normalize
if(mass > 0.0) //if not vacuum
{
  position /= mass; //center of mass
  velocity /= mass; //average velocity
}
```

You might ask if its possible to achieve a perfect particle number conservation, and not just total mass conservation. Actually to do that we can just need to divide the mass into integer chunks that sum into the original mass, the virtual particles need not to be the same, so for example one particle with mass 3 can divide into 2 particles with mass 2 and 1. Dividing into more than 2 virtual particles is pretty much the same, even though a bit more complicated. 

### Using particle distributions

The previous algorithm did solve the problem sufficiently well, but did have some shorcomings, like the additional loop which makes that algorthm far slower to execute and discontinuities in density that made it harder to implement fluid simulations. We can actually do better. Lets look at the limiting case of an infinite number of virtual particles, in such a case we end up with a continuous distribution of mass that we can call \\( \rho(\vec{X}) \\) (the distribution should also be centered on the particle position). So to find the amount of mass (and its center) each particle deposits into this cell we need to integrate the distribution within the current cell bounds. Our equations are:
\\[ M = \int \int_{\Omega} \rho(\vec{X})d\vec{X} -  \textrm{mass} \\] 
\\[ \vec{C} = \frac{1}{M} \int \int_{\Omega} \vec{X}\rho(\vec{X})d\vec{X} - \textrm{center of mass} \\] 
Where \\( \Omega \\) is the cell region. But which distribution should we use? If we try to use a normal distribution we will end up with the problem of how to compute it, since there is no analytical solution for such an integral, and we would need to numerically integrate the distribution, which is not much better performance-wise than the previous algorithm. The simplest distribution that gives an analytical solution we can use is actually a uniform axis alligned box, for which the mass and the center of mass are trivial to compute analytically. We can also try a uniform circular distribution or a nonuniform circular distribution equal to \\( 1 - |\vec{X_0} - \vec{X}|^2 \\) for \\(  |\vec{X_0} - \vec{X}| \leqslant 1\\) and equal to \\( 0 \\) for \\(  |\vec{X_0} - \vec{X}| > 1 \\) where \\( \vec{X_0} \\) is the particle position, but lets try out the simplest approach.
<center>
<table>
  <tr>
    <th><img src="{{ site.baseurl }}/images/Reintegration_0_2.JPG" style="width:250px;height:250px;"></th>
    <th><img src="{{ site.baseurl }}/images/Reintegration_0_55.JPG" style="width:250px;height:250px;"></th>
  </tr>
  <tr>
    <th><b>Diffusion radius 0.2</b></th>
    <th><b>Diffusion radius 0.55</b></th>
  </tr>
</table>
</center>
To find the mass and center of mass of the distribution within the bounds of the cell we only need to figure out the overlap box of the cell and the particle distribution. It's relative area will be the relative mass and its center is just the center of mass.
It can be implemented like this: 
```glsl
//this cell position
ivec2 pos;
//values stored in the cell
vec2 velocity = vec2(0.), position = vec2(0.);
float mass = 0.;
//find and average the particles 
//that land in this cell after a time step dt
for(int x = -R; x <= R; x++) //only check the neighbors at radius R
  for(int y = -R; y <= R; y++)
  {
      //get the particle in this neighbor cell from the previous frame
      particle P = getParticle(pos + ivec2(x,y));
      //integrate the particle position
      P.X += P.V*dt;
      //find the overlap of the diffused particle distribution with this cell
      
      vec3 ovrlp = overlap(P.X, pos, diffusion_radius);
      float overlapRelativeArea = ovrlp.z;
      vec2 overlapCenterOfMass = ovrlp.xy;
      float overlapMass = overlapRelativeArea*P.M;

      mass += overlapMass; //add the overlap mass to this cell
      position += overlapCenterOfMass*overlapMass; //add the overlap center weighted by mass
      velocity += P.V*overlapMass; //add the particle velocity weighted by overlap mass(momentum)
  } 

//normalize
if(mass > 0.0) //if not vacuum
{
  position /= mass; //center of mass
  velocity /= mass; //average velocity
}
```

The axis alligned box overlap calculation is rather straightforward 
```glsl
vec3 overlap(vec2 x, vec2 p, float diffusion_radius)
{
    vec4 aabb0 = vec4(p - 0.5, p + 0.5); //cell box
    vec4 aabb1 = vec4(x - diffusion_radius, x + diffusion_radius); //particle box
    vec4 aabbX = vec4(max(aabb0.xy, aabb1.xy), min(aabb0.zw, aabb1.zw)); //overlap box
    vec2 center = 0.5*(aabbX.xy + aabbX.zw); //center of mass 
    vec2 size = max(aabbX.zw - aabbX.xy, 0.); //only positive
    float m = size.x*size.y/(4.0*diffusion_radius*diffusion_radius); //relative area
    //if any of the dimensions are 0 then the mass ratio is 0
    return vec3(center, m);
}
```
As you can see we don't need to loop over virtual particles anymore since the solution is analytical, so the performance of this approach is the same as in the original cellular automaton particle tracker with the added benefit of giving much smoother results.

And here is a real time visualization of the reintegration process with diffusion radius 0.35
<center><iframe style="width:640px;height:360px;" frameborder="0" src="https://www.shadertoy.com/embed/WlSfWD?gui=true&amp;t=10&amp;paused=false" allowfullscreen=""></iframe></center>

In fact this algorithm has some quite interesting properties, depending on the radius of the distribution the behaviour can change from particle-like to field-like as shown in the simulation below(you need to unpause it). The distribution diameter oscillates between 0.75 and 1.25:
<center><iframe style="width:640px;height:360px;" frameborder="0" src="https://www.shadertoy.com/embed/tl2fWD?gui=true&amp;t=10&amp;paused=true" allowfullscreen=""></iframe></center>
### Using the SPH formulation instead of finite differences to compute forces

### Storing more properties inside a cell 
