// Size of cells
int cellSize = 40;

// Array of cells
Particle[][] gridFRONT; 
// Buffer to record the state of the cells and use this while changing the others in the interations
Particle[][] gridBACK; 

int gridX, gridY;

// Variables for timer
int interval = 20;
int lastRecordedTime = 0;

void setup() {
  size (800, 800);
  gridX = width/cellSize;
  gridY = height/cellSize;
  
  // Instantiate arrays 
  gridFRONT = new Particle[gridX][gridY];
  gridBACK = new Particle[gridX][gridY];

  // This stroke will draw the background grid
  stroke(48);

  noSmooth();
  for(int i = 0; i<gridX; i++)
  {
    for(int j = 0; j<gridY; j++)
    {
      gridFRONT[i][j] = new Particle(float(i) + 0.5, float(j) + 0.5, 0, 0, 0);
    }
  }
  
  //set particle in center
  gridFRONT[gridX/2][gridY/2] = new Particle(float(gridX/2) + 0.5, float(gridY/2) + 0.5, 0.1, 0.1, 1);
  
  background(0); // Fill in black in case cells don't cover all the windows
}


void draw() 
{
  clear();
  //Draw grid
  for(int i = 0; i<gridX; i++)
  {
    for(int j = 0; j<gridY; j++)
    {
      fill(color(0));
      rect(i*cellSize, j*cellSize, cellSize, cellSize);
    }
  }
  
  //Draw particles
  for(int i = 0; i<gridX; i++)
  {
    for(int j = 0; j<gridY; j++)
    {
      float m = gridFRONT[i][j].m;
      fill(color(m*255)); 
      if(m > 1e-3)
      rect((gridFRONT[i][j].x - D*0.5)*cellSize, (gridFRONT[i][j].y - D*0.5)*cellSize, cellSize*D, cellSize*D);
    }
  }
  
  // Iterate if timer ticks
  if (millis()-lastRecordedTime>interval) {
    iteration();
    lastRecordedTime = millis();
  }
}


int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? (r + b) : r;
}

float mod(float a, float b)
{
    float r = a % b;
    return r < 0.0 ? (r + b) : r;
}

//neighbor check radius
int R = 1;
//time step
float dt = 1.0;
//diffusion radius
float D = 1.0;
Particle Reintegration(int i, int j)
{
  Particle buf = new Particle(0, 0, 0, 0, 0);
  for(int x = -R; x<=R; x++)
  {
    for(int y = -R; y<=R; y++)
    {
       float localx = float(i + x);
       float localy = float(j + y);
       
       //get particle in looped grid space
       Particle test = gridBACK[mod(i + x,gridX)][mod(j + y,gridY)];
       
       // put particle inside this cell(make sure the position are correct relatively to other border cells)
       float npx = localx + mod(test.x,1.0);
       float npy = localy + mod(test.y,1.0);
       
       // integrate particle position
       npx += test.vx*dt;
       npy += test.vy*dt;
       
       //check particle distribution intersection with this cell
       float xmin = max(min(npx - D*0.5, float(i) + 1.0), float(i));
       float xmax = max(min(npx + D*0.5, float(i) + 1.0), float(i));
       float ymin = max(min(npy - D*0.5, float(j) + 1.0), float(j));
       float ymax = max(min(npy + D*0.5, float(j) + 1.0), float(j));
       
       //fallen mass
       float mass = test.m*(xmax - xmin)*(ymax - ymin)/(D*D);
       //center of mass
       float cm_x = (xmax + xmin)*0.5;
       float cm_y = (ymax + ymin)*0.5;
       
       buf.m += mass;
       buf.x += cm_x*mass;
       buf.y += cm_y*mass;
       buf.vx += test.vx*mass;
       buf.vy += test.vy*mass;
    }
  }
  
  //normalize
  if(buf.m > 0.0)
  {
    buf.x /= buf.m;
    buf.y /= buf.m;
    buf.vx /= buf.m;
    buf.vy /= buf.m;
  }
  
  return buf;
}


void iteration() { // When the clock ticks
  // Save cells to buffer (so we opeate with one array keeping the other intact)
  float totalm = 0.;
  float vx = 0.;
  float vy = 0.;
  for(int i = 0; i<gridX; i++)
  {
    for(int j = 0; j<gridY; j++)
    {
      gridBACK[i][j] = gridFRONT[i][j];
    }
  }
    
  for(int i = 0; i<gridX; i++)
  {
    for(int j = 0; j<gridY; j++)
    {
      gridFRONT[i][j] = Reintegration(i, j);
      totalm += gridFRONT[i][j].m;
      vx += gridFRONT[i][j].m*gridFRONT[i][j].vx;
      vy += gridFRONT[i][j].m*gridFRONT[i][j].vy;
    }
  }
  vx /= totalm;
  vy /= totalm;
  println("Total mass:", totalm);
} 

void keyPressed() {
  if (key==' ') { 
   //iteration();
  }
}

class Particle {
  float x;
  float y;
  float vx;
  float vy;
  float m;
  
  Particle(float X, float Y, float VX, float VY, float M) {
    x = X;
    y = Y;
    vx = VX;
    vy = VY; 
    m = M;
  }
}  
