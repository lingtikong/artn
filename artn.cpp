/*---------------------------------------------------







----------------------------------------------------*/
#include "lmptype.h"
#include "mpi.h"
#include "math.h"
#include "string.h"
#include "min_cg.h"
#include "atom.h"
#include "update.h"
#include "output.h"
#include "timer.h"
#include "random_park.h"
#include "error.h"
#include "stdlib.h"
#include "minimize.h"
#include "artn.h"

using namespace LAMMPS_NS;

/*---------------------------------------------------
  Here I use clapack to help evaluate the lowest 
  eigenvalue of the matrix in lanczos.
---------------------------------------------------*/
extern "C"{
#include "f2c.h"
#include "clapack.h"
#include "blaswrap.h"
}


#define EPS_ENERGY 1.0e-8
#define INITIAL_STEPSIZE 0.02
#define SEED 998		// random seed
#define LOCAL_NUM 20
#define SADDLE_NUM 50
#define INCREMENT 0.01
#define HFTHRESHOLD 1.0e-3
#define HFTHRESHOLD2  (HFTHRESHOLD*HFTHRESHOLD)
#define MAKKPERP 4
#define MAXVEC 20	// maxvec for lanczos


Artn::Artn(LAMMPS *lmp):Min_linesearch(lmp){
  random = new RanPark(lmp, SEED)
}

/*---------------------------------------------------
  	destructor
---------------------------------------------------*/
Artn::~Artn(){
  if(initial_direction) delete []initial_direction;
  if(random) delete random;
  if(fperp) delete []fperp;
  if(eigenvector) delete []eigenvector;
  if(prefperp) delete []prefperp;
  if(prexvec) delete []prexvec;
  if(prefvec) delete []prefvec;
}

/*--------------------------------------------------- 
	setup before artn search
---------------------------------------------------*/ 
void Artn::setup(){
}

/*--------------------------------------------------- 
	initialization
---------------------------------------------------*/ 
void Artn::init(){
  linestyle = 0;
  einitial = efinal = eprevious = 0;
  alpha_final = 0;
  niter = neval = 0;
  ndottoal = 0;
  nstep = 0;
  nvec = 0;
  ecurrent = 0;
  xvec = fvec = NULL;
  x0 = g = h = NULL;
  ievent = 0;
  temperature = -1;
  newevent = 0;
  maxevent = 0;
  prefvec = new double[nvec];
  eigenvector =  new double[nvec];
}

/*--------------------------------------------------- 
	parase the artn command:
	
	artn maxevent maxsteps temperature etol ftol max_eval

---------------------------------------------------*/ 
void Artn::command(int narg, char ** arg){
  argc=arg;
  if (narg != 6 ) error->all(FLERR, "Illegal artn command!");
  if (domain->box_exist == 0)
    error->all(FLERR,"Artn command before simulation box is defined");
  maxevent = atof(arg[0]);
  update->nsteps = atoi(arg[1]);
  temperature = atof(arg[2]);
  update->etol = atof(arg[3]);
  update->ftol = atof(arg[4]);
  update->max_eval = atoi(arg[5]);
  if (update->etol < 0.0 || update->ftol < 0.0)
    error->all(FLERR,"Illegal Artn command");

  update->whichflag = 2;
  update->beginstep = update->firststep = update->ntimestep;		// ntimestep： the step now
  update->endstep = update->laststep = update->firststep + update->nsteps; //nsteps: the steps needed to be run
  if (update->laststep < 0 || update->laststep > MAXBIGINT)
    error->all(FLERR,"Too many iterations");

  lmp->init();
  update->minimize->setup();

  timer->init();
  timer->barrier_start(TIME_LOOP);
  search();
  timer->barrier_stop(TIME_LOOP);
  cleanup();
  Finish finish(lmp);
  finish.end(1);

  update->whichflag = 0;
  update->firststep = update->laststep = 0;
  update->beginstep = update->endstep = 0;
}

/*--------------------------------------------------- 
  	main artn loop
---------------------------------------------------*/ 
void Artn::search(){
  setup();
  // converge to the basin minmum.
  min_converge();

  // main event loop.
  for( ievent = 0; ievent < maxevent; ievent++){
    // print the current event.
    print_newevent();
    // find one saddle point
    while(find_saddle()){};
    if(!newevent) { ievent--; continue;}
    downhill();
    postcompute();
    output();
    // accept or reject the move based on Boltzman weight
    judgement();
  }
}

/*---------------------------------------------------
  	print the current event
---------------------------------------------------*/
void Artn::print_newevent(){
}

/*--------------------------------------------------- 
  	give a globely random △x
---------------------------------------------------*/ 
void Artn::globle_move(){
  double *delpos = new double[nvec];
  double norm;
  for(int i=0; i < nvec; i++){
    delpos[i] = 0.5-random->uniform();
  }
  center(delpos, nvec);
  for(int i=0, norm=0; i < nvec; i++){
    norm += delpos[i] * delpos[i]
  }
  double normall;
  MPI_Allreduce(norm,normall,1,MPI_DOUBLE,MPI_SUM,world);
  double norm_i = 1./normall;
  initial_direction = new double [nvec];
  for(int i=0; i < atom->nlocal; i++){
    initial_direction[i] = delpos[i] * norm_i;
    xvec[i] += INITIAL_STEPSIZE * initial_direction[i];
  }
  fix_minimize->reset_coords();
  reset_vectors();
}

/*--------------------------------------------------- 
  	give a local random △x around 
	a randomly chosen atom
---------------------------------------------------*/ 
void Artn::local_move(){}

/*--------------------------------------------------- 
  	converge to the minimum, here I use 
	the styles user defined or default
---------------------------------------------------*/ 
void Artn::min_converge(){
  update->minimize->run(update->nsteps);
}

/*--------------------------------------------------- 
  	To find the saddle point.
---------------------------------------------------*/
void Artn::find_saddle(){
  //----- random move the atoms in account -----
  globle_move();
  //----- move out the convex region -----
  fperp = new double[nvec];
  prefperp = new double[nvec];
  prexvec = new double [nvec];
  preenergy = ecurrent = energy_force(1);
  step = INCREMENT;
  for(int local_iter = 0; local_iter < LOCAL_NUM; local_iter++){
    min_perpendicular();
    min_along();
    if (local_iter > LOCAL_MIN){
      lanczos(-1, MAXVEC);
      if(eigen > 0) break;
    }
  }
  //----- now try to move close to the saddle point -----
  for(int saddle_iter = 0; saddle_iter < SADDLE_NUM; saddle_iter++){
    get_direction();
    saddle_min_perdicular();
    saddle_min_along();
    check_saddle();
  }
}

/*--------------------------------------------------- 
  	downhill from the saddle point to
	another minimum.
---------------------------------------------------*/ 
void Artn::downhill(){}

/*--------------------------------------------------- 
  	use lanczos method to fand the lowest 
	eigenvalue and the relevant vector
---------------------------------------------------*/ 
void Artn::lanczos(){}

/*--------------------------------------------------- 
  	accept or reject one move based on
       	Boltzman weight
---------------------------------------------------*/ 
void Artn::judgement(){}

/*--------------------------------------------------- 
	add to cleanup() if want to do sth.
---------------------------------------------------*/ 
void Artn::clean_up(){}

/*---------------------------------------------------
  	center the vector
---------------------------------------------------*/
void Artn::center(double * x, int n){
  double sum = 0;
  for(int i=0; i<n; i++){
    sum += x[i];
  }
  sum /= n;
  for(int i=0; i<n; i++){
    x[i] -= sum;
  }
}
/*---------------------------------------------------
  	minimize perpendicular to the 
	initial direction
---------------------------------------------------*/

void Artn::min_perpendicular(){
  double fperp2, fperpall2, fdotinit;
  int k=0, k_reject=0;
  fperp2 = 0., fperpall=0.;
  fdotinit = 0.;
  for(int i=0; i < nvec; i++) fdotinit += fvec[i]*initial_direction[i];
  for(int i=0; i < nvec; i++){
    fperp[i] = fvec[i] - fdotinit * initial_direction[i];
  }
  while(1){
    for(int i=0; i<nvec; i++){
      prefperp[i] = fperp[i];
      prexvec[i] = xvec[i];
      xvec[i] += step * prefperp[i];
    }
    ecurrent = energy_force(1);
    for(int i=0; i < nvec; i++) fdotinit += fvec[i]*initial_direction[i];
    for(int i=0; i < nvec; i++){
      fperp[i] = fvec[i] - fdotinit * initial_direction[i];
      fperp2 += fperp[i]*fperp[i];
    }
    MPI_Allreduce(fperp2,fperpall2,1,MPI_DOUBLE,MPI_SUM,world);
    if(ecurrent < preenergy){
      step = 1.2 * step;
      k += 1;
      k_reject = 0;
    }
    else{
      for(int i=0; i<nevc; i++){
	xvec[i] = prexvec[i];
	fperp[i] = prefperp[i];
	ecurrent = preenergy;
      }
      step *= 0.6;
      k_reject += 1;
    }
    if( fperpall2 < FTHRESH2 || k > MAKKPERP || k_rejected > 5) break;
  }
}
/*---------------------------------------------------
	walk along the initial direction
---------------------------------------------------*/
void Artn::min_along(){
  for(int i=0; i<nvec; i++){
    xvec[i] += INCREMENT * initial_direction[i];
  }
}

/*---------------------------------------------------
  The lanczos method to get the lowest eigenvalue
  and corresponding eigenvector
---------------------------------------------------*/
void lanczos(int flag, int maxvec){
  double *r0 = new double[nvec];
  if(flag > 0){
    for(int i = 0; i<nvec; i++) r0[i] = direction[i];
  }else{
    for(int i = 0; i<nvec; i++) r0[i] = 0.5 - random->uniform();
  }
  double rsum2 = 0, rsumall2;
  for(int i=0; i<nvec; i++) rsum += r0[i] * r0[i];
  MPI_Allreduce(rsum2,rsumall2,1,MPI_DOUBLE,MPI_SUM,world);
  b0 = rsumall2;
  rsumall2 = 1.0 / rsumall2;
  for(int i=0; i<nvec; i++) r0[i] *= rsumall2;
  const double XL = 0.001;
  const double iXL = 1.0 / XL;
  energy_force(1);
  for(int i=0; i<nvec; i++){
    prexvec[i] = xvec[i];
    prefvec[i] = fvec[i];
    xvec[i] += r0[i] * XL;
  }
  energy_force(1);
  double *u1 = new double[nvec];
  double *r1 = new double[nvec];
  double *d = new double[maxvec];
  double *e = new double[maxvec];
  for(int i=0; i <maxvec; i++){
    d[i] = e[i] =0.;
  }
  double a1 = 0.,b1 = 0.;
  for(int i=0; i<nvec; i++){
    u1[i] = (fvec[i] - prefvec[i]) * iXL;
    r1[i] = u1[i] - b0 * r0[i];
    a1 += r0[i] * r1[i];
  }
  double a1all = 0., b1all = 0.;
  MPI_Allreduce(a1,a1all,1,MPI_DOUBLE,MPI_SUM,world);
  a1 = a1all;
  for(int i=0; i<nvec; i++){
    r1[i] -= a1 * r0[i];
    b1 += r1[i] * r1[i];
  }
  MPI_Allreduce(b1,b1all,1,MPI_DOUBLE,MPI_SUM,world);
  b1 = b1all;
  d[0] = a1; e[0] = b1;
  double i_b1 = 1.0 / b1;
  double r1,r2;
  double *work, *z;
  long int ldz = maxvec, info;
  char jobs;
  if(flag > 0) jobs = 'V';
  else jobs = 'N';
  z = new double [ldz*maxvec];
  
  for(long int n=2; n<maxvec; n++){
    for(int i=0; i<nvec; i++)r0[i] = r1[i] * i_b1;
    for(int i=0; i<nvec; i++){
      prexvec[i] = xvec[i];
      prefvec[i] = fvec[i];
      xvec[i] += r0[i] * XL;
    }
    energy_force(1);
    a1 = b1 = a1all = b1all = 0.;
    for(int i=0; i<nvec; i++){
      u1[i] = (fvec[i] - prefvec[i]) * iXL;
      r1[i] = u1[i] - b0 * r0[i];
      a1 += r0[i] * r1[i];
    }
    MPI_Allreduce(a1,a1all,1,MPI_DOUBLE,MPI_SUM,world);
    a1 = a1all;
    for(int i=0; i<nvec; i++){
      r1[i] -= a1 * r0[i];
      b1 += r1[i] * r1[i];
    }
    MPI_Allreduce(b1,b1all,1,MPI_DOUBLE,MPI_SUM,world);
    b1 = b1all;
    i_b1 = 1.0 / b1;
    d[n-1] = a1; e[n-1] = b1;
    destev_(&jobs,&n,d,e,z,&ldz,work,&info);
    r1 = r2;
    r2 = d[0];
    if((r2-r1)/r2<0.1)eigenvalue=r2;
    for(int i=0; i<nvec; i++) eigenvector[i]=z[i];
  }
}
