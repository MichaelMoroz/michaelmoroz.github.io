
const canvas = document.body.appendChild(document.createElement('canvas'));
const regl = require('regl')({
  pixelRatio: Math.min(window.devicePixelRatio, 0.5),
  attributes: {
    antialias: false,
    stencil: false,
    alpha: false,
    depth: true
  },
  canvas: canvas,
  extensions: ['webgl_draw_buffers', 'oes_texture_float', 'oes_texture_float_linear']
});

const fit = require('canvas-fit');
window.addEventListener('resize', fit(canvas), false);

var mp = require('mouse-position')(canvas);
var mb = require('mouse-pressed')(canvas);
var mw = require('mouse-wheel');

var ControlKit = require('controlkit');

frame = 0;
scale = 1.0;
DX = 0.; DY = 0.;
aspect = 1.0;

var params = {
  dt : 0.75, dtrange:[0,1],
  N: 3, Nrange: [1, 50],
  frametime: 0,
  PN: 1,
  rho: 0.75, rhorange: [0., 1.5],
  solid: true,
  elastic_lambda: 5.0, elrange: [0., 20.],
  elastic_mu: 5.0, emrange: [0., 20.],
  fluid_p: 0.4, fprange: [0., 3.],
  diffusion: 0.13, dirange: [0., 3.]
};

var controlKit = new ControlKit();
	    controlKit.addPanel({label: 'Simulation parameters', width: 350})
	        .addGroup()
	            .addSubGroup()
                  .addSlider(params, 'dt', 'dtrange', {label: "Time step"})
                  .addSlider(params, 'N', 'Nrange', {step: 1, dp: 0, label: "Iterations per frame"})
                  .addSlider(params, 'rho', 'rhorange', {label: "Density"})
                  .addCheckbox(params,'solid', {label: "Is solid"})
                  .addSlider(params, 'elastic_lambda', 'elrange', {label: "Elastic Lambda"})
                  .addSlider(params, 'elastic_mu', 'emrange', {label: "Elastic Mu"})
                  .addSlider(params, 'fluid_p', 'fprange', {label: "Fluid Pressure"})
                  .addSlider(params, 'diffusion', 'dirange', {label: "Diffusion"})
                  .addButton('Restart',function(){frame = 0})
                  //.addValuePlotter(params, 'frametime', {label: 'Frame time', height: 100})
const glslify = require('glslify');

//array of float textures
function ColorArr(W, H, N)
{
  color = [];
  for(i = 0; i < N; i++)
    color.push(regl.texture({width: W, 
                             height: H, 
                             type: 'float32'}));
  return color;
}

const default_vertex = glslify.file('./shaders/vertex.glsl');


simW = 256; simH = 128; 

//mouse wheel scaling 
mw(function(dx, dy) {
  scale = scale*(1. + 0.3*dy/H);
})

fbo0 = regl.framebuffer({color: ColorArr(simW, simH, 3), depth: false});
fbo1 = regl.framebuffer({color: ColorArr(simW, simH, 3), depth: false});
fbo2 = regl.framebuffer({color: ColorArr(simW, simH, 2), depth: false});

const Reintegration = regl({
  frag: glslify.file('./shaders/reintegration.glsl'),
  vert: default_vertex,
  attributes: {position: [ -4, -4, 4, -4, 0, 4 ]},
  framebuffer: fbo0,
  uniforms: {
    iChannel0: fbo2.color[0],
    iChannel1: fbo2.color[1],
    iChannel2: fbo1.color[0],
    iChannel3: fbo1.color[1],
    iChannel4: fbo1.color[2],
    iFrame: function(){return frame},
    R: [simW, simH],
    dt: function() {return params.dt},
    rho: function() {return params.rho},
    diffusion: function() {return params.diffusion}
  },
  depth: { enable: false },
  count: 3
});

const DeformationGrad = regl({
  frag: glslify.file('./shaders/deformation.glsl'),
  vert: default_vertex,
  attributes: {position: [ -4, -4, 4, -4, 0, 4 ]},
  framebuffer: fbo1,
  uniforms: {
    iChannel0: fbo0.color[0],
    iChannel1: fbo0.color[1],
    iChannel2: fbo0.color[2],
    iChannel3: fbo2.color[0],
    iChannel4: fbo2.color[1],
    iFrame: function(){return frame},
    R: [simW, simH],
    dt: function() {return params.dt},
    rho: function() {return params.rho},
    solid: function() {return Boolean(params.solid)},
    elastic_lambda:  function() {return params.elastic_lambda},
    elastic_mu: function() {return params.elastic_mu},
    fluid_p: function() {return params.fluid_p}
  },
  depth: { enable: false },
  count: 3
});

const MomentumUpdate = regl({
  frag: glslify.file('./shaders/momentum.glsl'),
  vert: default_vertex,
  attributes: {position: [ -4, -4, 4, -4, 0, 4 ]},
  framebuffer: fbo2,
  uniforms: {
    iChannel0: fbo0.color[0],
    iChannel1: fbo0.color[1],
    iChannel2: fbo1.color[2],
    iFrame: function(){return frame},
    R: [simW, simH],
    dt: function() {return params.dt}
  },
  depth: { enable: false },
  count: 3
});
 
const Image = regl({
  vert: default_vertex,
  frag: glslify.file('./shaders/main.glsl'),
  uniforms: {
    iChannel0: fbo2.color[0],
    iChannel1: fbo2.color[1],
    iFrame: function(){return frame},
    scale: function(){return scale},
    iPos:  function(){return [DX,-DY]},
    aspect: function(){return aspect},
    R: [simW, simH]
  },
  attributes: {position: [ -4, -4, 4, -4, 0, 4 ]},
  depth: { enable: false },
  count: 3
});

time0 = new Date().getTime();
time1 = new Date().getTime();

regl.frame(() => {
  regl.clear({color: [0.1, 0.1, 0.1, 1]});
  for(i = 0; i < params.N; i++)
  {
    Reintegration();
    DeformationGrad();
    MomentumUpdate();
  }
  Image();
 
  if(mb.left)
  {
    DX = DX + scale*(mp[0] - mp.prev[0])/W;
    DY = DY + scale*(mp[1] - mp.prev[1])/H;
  }
  frame++;
  regl
  W = canvas.width;
  H = canvas.height;
  aspect = W/H;

  time0 = time1;
  time1 = new Date().getTime();
  params.frametime = time1 - time0; 
});