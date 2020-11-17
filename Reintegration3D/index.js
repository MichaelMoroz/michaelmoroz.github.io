
const canvas = document.body.appendChild(document.createElement('canvas'));

reglGL = require('regl');
regl = reglGL({
  pixelRatio: Math.min(window.devicePixelRatio, 0.5),
  attributes: {
    antialias: false,
    stencil: false,
    alpha: false,
    depth: true
  },
  canvas: canvas,
  extensions: ['webgl_draw_buffers', 'oes_texture_float', 'oes_texture_float_linear', 'OES_texture_half_float']
});


const fit = require('canvas-fit');
window.addEventListener('resize', fit(canvas), false);
var mp = require('mouse-position')(canvas);
var mb = require('mouse-pressed')(canvas);
var mw = require('mouse-wheel');
var kb = require('keyboard-key');
vec3 = require('gl-vec3');
const glslify = require('glslify');

var ControlKit = require('controlkit');

frame = 0;
scale = 1.0;
DX = 0.; DY = 0.;
aspect = 1.0;

//mouse wheel scaling 
mw(function(dx, dy) {
  scale = scale*(1. + 0.3*dy/H);
})

var params = {
  dt : 0.75, dtrange:[0,1],
  N: 2, Nrange: [0, 16],
  frametime: 0,
  PN: 1,
  rho: 0.75, rhorange: [0., 1.5],
  solid: true,
  elastic_lambda: 15.0, elrange: [0., 60.],
  elastic_mu: 25.0, emrange: [0., 60.],
  fluid_p: 0.4, fprange: [0., 3.],
  diffusion: 0.03, dirange: [0., 3.],
  options: ['64x64x16','64x64x64','128x128x16','128x128x64','256x256x16','256x256x64', '256x256x256'], 
  selection : null 
};

resolutions2d = [[256,256],[512,512],[512,512],[1024,1024],[1024,1024],[2048,2048],[4096,4096]];
resolutions3d = [[64,64,16],[64,64,64],[128,128,16],[128,128,64],[256,256,16],[256,256,64],[256,256,256]];
params.selection = params.options[1];
sim_resolution = resolutions2d[1];
sim_resolution3d = resolutions3d[1];


var regl;
var Reintegration;
var DeformationGrad;
var MomentumUpdate;
var Image;
var fbo0; var fbo1; var fbo2;

const default_vertex = glslify.file('./shaders/vertex.glsl');

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
            
function Initialize()
{
  frame = 0;

  Reintegration = regl({
    frag: glslify.file('./shaders/reintegration.glsl'),
    vert: default_vertex,
    attributes: {position: [ -4, -4, 4, -4, 0, 4 ]},
    framebuffer: regl.prop('out'),
    uniforms: {
      iChannel0: regl.prop('ch0'),
      iChannel1: regl.prop('ch1'),
      iChannel2: regl.prop('ch2'),
      iChannel3: regl.prop('ch3'),
      iChannel4: regl.prop('ch4'),
      iChannel5: regl.prop('ch5'),
      iFrame: function(){return frame},
      R: function(){return sim_resolution},
      R3D: function(){return sim_resolution3d},
      T: function(){return [sim_resolution[0]/sim_resolution3d[0], sim_resolution[1]/sim_resolution3d[1]]},
      dt: function() {return params.dt},
      rho: function() {return params.rho},
      diffusion: function() {return params.diffusion}
    },
    depth: { enable: false },
    count: 3
  });
  
  DeformationGrad = regl({
    frag: glslify.file('./shaders/deformation.glsl'),
    vert: default_vertex,
    attributes: {position: [ -4, -4, 4, -4, 0, 4 ]},
    framebuffer: regl.prop('out'),
    uniforms: {
      iChannel0: regl.prop('ch0'),
      iChannel1: regl.prop('ch1'),
      iChannel2: regl.prop('ch2'),
      iChannel3: regl.prop('ch3'),
      iChannel4: regl.prop('ch4'),
      iChannel5: regl.prop('ch5'),
      iFrame: function(){return frame},
      R: function(){return sim_resolution},
      R3D: function(){return sim_resolution3d},
      T: function(){return [sim_resolution[0]/sim_resolution3d[0], sim_resolution[1]/sim_resolution3d[1]]},
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
  
  MomentumUpdate = regl({
    frag: glslify.file('./shaders/momentum.glsl'),
    vert: default_vertex,
    attributes: {position: [ -4, -4, 4, -4, 0, 4 ]},
    framebuffer: regl.prop('out'),
    uniforms: {
      iChannel0: regl.prop('ch0'),
      iChannel1: regl.prop('ch1'),
      iChannel2: regl.prop('ch2'),
      iChannel3: regl.prop('ch3'),
      iChannel4: regl.prop('ch4'),
      iChannel5: regl.prop('ch5'),
      iFrame: function(){return frame},
      R: function(){return sim_resolution},
      R3D: function(){return sim_resolution3d},
      T: function(){return [sim_resolution[0]/sim_resolution3d[0], sim_resolution[1]/sim_resolution3d[1]]},
      dt: function() {return params.dt}
    },
    depth: { enable: false },
    count: 3
  });
   
  Image = regl({
    vert: default_vertex,
    frag: glslify.file('./shaders/main.glsl'),
    uniforms: {
      iChannel0: regl.prop('ch0'),
      iChannel1: regl.prop('ch1'),
      iChannel2: regl.prop('ch2'),
      iFrame: function(){return frame},
      scale: function(){return scale},
      iPos:  function(){return [DX,-DY]},
      aspect: function(){return aspect},
      R: function(){return sim_resolution},
      R3D: function(){return sim_resolution3d},
      T: function(){return [sim_resolution[0]/sim_resolution3d[0], sim_resolution[1]/sim_resolution3d[1]]}
    },
    attributes: {position: [ -4, -4, 4, -4, 0, 4 ]},
    depth: { enable: false },
    count: 3
  });
}

function InitializeFBOs()
{
  ww = sim_resolution[0]; hh = sim_resolution[1];
  fbo0 = regl.framebuffer({
    color: [
      regl.texture({width: ww, height: hh, 
        type: 'float32', format: 'rgb'}), //position
      regl.texture({width: ww, height: hh, 
        type: 'float32', format: 'rgb'}), //velocity
      regl.texture({width: ww, height: hh, 
        type: 'float32', format: 'rgb'}), //mass
      regl.texture({width: ww, height: hh, 
        type: 'float32', format: 'rgb'}), //dgrad0
      regl.texture({width: ww, height: hh, 
        type: 'float32', format: 'rgb'}), //dgrad1
      regl.texture({width: ww, height: hh, 
        type: 'float32', format: 'rgb'}) //dgrad2
    ],
    depth: false});
  fbo1 = regl.framebuffer({
    color: [
      regl.texture({width: ww, height: hh, 
        type: 'float16', format: 'rgb'}), //dgrad0
      regl.texture({width: ww, height: hh, 
        type: 'float16', format: 'rgb'}), //dgrad1
      regl.texture({width: ww, height: hh, 
        type: 'float16', format: 'rgb'}), //dgrad2
      regl.texture({width: ww, height: hh, 
        type: 'float16', format: 'rgb'}), //stress0
      regl.texture({width: ww, height: hh, 
        type: 'float16', format: 'rgb'}), //stress1
      regl.texture({width: ww, height: hh, 
        type: 'float16', format: 'rgb'}) //stress2
    ],
    depth: false});
  fbo2 = regl.framebuffer({
    color: [
      regl.texture({width: ww, height: hh, 
        type: 'float32', format: 'rgb'}), //position
      regl.texture({width: ww, height: hh, 
        type: 'float32', format: 'rgb'}), //velocity
      regl.texture({width: ww, height: hh, 
        type: 'float32', format: 'rgb'}), //mass
      regl.texture({width: ww, height: hh, 
        type: 'float32', format: 'rgb', mag: 'linear'}) //normals
    ],
    depth: false});
}

function ClearGL()
{
  Reintegration.destroy();
  DeformationGrad.destroy();
  MomentumUpdate.destroy();
  Image.destroy();
}

var controlKit = new ControlKit();
	    controlKit.addPanel({label: 'Simulation parameters', width: 350})
	        .addGroup()
              .addSubGroup()
                  .addSelect(params,'options',{target:'selection', label: 'Sim Resolution', 
                                            onChange:function(index){sim_resolution = resolutions2d[index];
                                                                     sim_resolution3d = resolutions3d[index];
                                                                     frame = 0;
                                                                     InitializeFBOs();
                                                                    }})
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


time0 = new Date().getTime();
time1 = new Date().getTime();

Initialize();
InitializeFBOs();

regl.frame(() => {
  regl.clear({color: [0.1, 0.1, 0.1, 1]});
  for(i = 0; i < params.N; i++)
  {
    Reintegration({out: fbo0, 
                   ch0: fbo2.color[0], ch1: fbo2.color[1], ch2: fbo2.color[2], 
                   ch3: fbo1.color[0], ch4: fbo1.color[1], ch5: fbo1.color[2]});
    DeformationGrad({out: fbo1, 
                     ch0: fbo0.color[0], ch1: fbo0.color[1], ch2: fbo0.color[2], 
                     ch3: fbo0.color[3], ch4: fbo0.color[4], ch5: fbo0.color[5]});
    MomentumUpdate({out: fbo2,
                    ch0: fbo0.color[0], ch1: fbo0.color[1], ch2: fbo0.color[2], 
                    ch3: fbo1.color[3], ch4: fbo1.color[4], ch5: fbo1.color[5]});
  }
  Image({ch0: fbo2.color[0], ch1: fbo2.color[2], ch2: fbo2.color[3]});
 
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