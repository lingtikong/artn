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
const double E = 2.71828;
const double K_B = 1.3806505e-23/ 1.602e-19;


Artn::Artn(LAMMPS *lmp):Min_linesearch(lmp){
  random = new RanPark(lmp, SEED);
  init();
  setup();
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
  if(direction) delete []direction;
  if(pre_direction) delete []direction;
  if(prewell_x) delete []prewell_x;
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
  pre_direction = new double[nvec];
  direction = new double[nvec];
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
  prewell_energy = energy_force(1);
  prewell_x = new double[nvec];
  for(int i=0; i!= nvec; ++i) prewell_x[i] = xvec[i];
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
	initial_direction is obtained.
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
  delete []delpos;
  //fix_minimize->reset_coords();
  //reset_vectors();
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
  fpar = new double[nvec];
  prefperp = new double[nvec];
  prexvec = new double[nvec];
  direction = new double[nvec];
  preenergy = ecurrent = energy_force(1);
  step = INCREMENT;
  for(int local_iter = 0; local_iter < LOCAL_NUM; local_iter++){
    min_perpendicular();
    min_along();
    if (local_iter > LOCAL_MIN){
      lanczos(0,-1, MAXVEC);     // It is necessary to think about whether we need to calculate eigenvector here.
      if(eigenvalue > 0) break;
    }
  }
  //----- now try to move close to the saddle point -----
  int first_time = 1;
  for(int saddle_iter = 0; saddle_iter < SADDLE_NUM; saddle_iter++){
    // get the direction
    if(first_time){
      lanczos(0, 1, MAXVEC);
      first_time = 0;
      double sum = 0., sumall;
      for(int i = 0; i!= nvec; ++i){
	sum += eigenvector[i] * initial_direction[i];
      }
      MPI_Allreduce(sum,sumall,1,MPI_DOUBLE,MPI_SUM,world);
      if(sumall > 0){
	for(int i = 0; i != nvec; ++i){
	  direction[i] = pre_direction[i] = eigenvector[i];
	}
      }else{
	for(int i = 0; i != nvec; ++i){
	  direction[i] = pre_direction[i] = -eigenvector[i];
	}
      }
    }else{ 
      lanczos(1, 1, MAXVEC); 
      double sum = 0., sumall;
      for(int i = 0; i!= nvec; ++i){
	sum += eigenvector[i] * pre_direction[i];
      }
      MPI_Allreduce(sum,sumall,1,MPI_DOUBLE,MPI_SUM,world);
      if(sumall > 0){
	for(int i = 0; i != nvec; ++i){
	  direction[i] = pre_direction[i] = eigenvector[i];
	}
      }else{
	for(int i = 0; i != nvec; ++i){
	  direction[i] = pre_direction[i] = -eigenvector[i];
	}
      }
    }
    saddle_min_perdicular();
    saddle_min_along();
    if(eigenvalue > 0.) break; // drop into the harmonic well
    if(check_saddle()) break;
  }
}
/*---------------------------------------------------
  check if it is the saddle point
---------------------------------------------------*/
int Artn::check_saddle(){
  double fperpvalue = 0., fperpvalueall, fparvalue = 0., fparvalueall, fdotdirect = 0., fdotdirectall,
  for(int i = 0; i != nvec; ++i){
    fdotdirect += fvec[i] * direction[i];
  }
  MPI_Allreduce(fdotdirect,fdotdirectall,1,MPI_DOUBLE,MPI_SUM,world);
  for(int i = 0; i!= nvec; ++i){
    fpar[i] = fdotdirectall * direction[i];
    fperp[i] = fvec[i] - fpar[i];
    fparvalue += fpar[i] * fpar[i];
    fperpvalue += fperp[i] * fperp[i];
  }
  MPI_Allreduce(fparvalue,fparvalueall,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(fperpvalue,fperpvalueall,1,MPI_DOUBLE,MPI_SUM,world);
  fperpvalue = sqrt(fperpvalueall);
  fparvalue = sqrt(fparvalueall);
  if(fparvalue < 0.01 && fperpvalue < 0.1){
    return 1;
  }else return 0;
}

/*---------------------------------------------------
  minimize the energy perpendicular to the eigenvector
---------------------------------------------------*/
void Artn::saddle_min_perdicular(){
  double fperp2, fperpall2, fdotinit, fdotinitall;
  int k=0, k_reject=0;
  fperp2 = 0., fperpall=0.;
  fdotinit = 0.;
  for(int i=0; i < nvec; i++) fdotinit += fvec[i]*direction[i];
  MPI_Allreduce(fdotinit,fdotinitall,1,MPI_DOUBLE,MPI_SUM,world);
  for(int i=0; i < nvec; i++){
    fperp[i] = fvec[i] - fdotinitall * direction[i];
  }
  while(1){
    for(int i=0; i<nvec; i++){
      prefperp[i] = fperp[i];
      prexvec[i] = xvec[i];
      xvec[i] += step * prefperp[i];
    }
    ecurrent = energy_force(1);
    for(int i=0; i < nvec; i++) fdotinit += fvec[i]*direction[i];
    for(int i=0; i < nvec; i++){
      fperp[i] = fvec[i] - fdotinit * direction[i];
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
  minimize the energy along the eigenvector
---------------------------------------------------*/
void Artn::saddle_min_along{
  for(int i=0; i<nvec; i++){
    xvec[i] += INCREMENT * direction[i];
  }
}

/*--------------------------------------------------- 
  	downhill from the saddle point to
	another minimum.
---------------------------------------------------*/ 
void Artn::downhill(){
  for(int i=0; i<nvec; ++i){
    xvec[i] += INCREMENT * 3. * direction[i];
  }
  min_converge();
}

/*--------------------------------------------------- 
  	accept or reject one move based on
       	Boltzman weight
---------------------------------------------------*/ 
void Artn::judgement(){
  ecurrent = energy_force(1);
  if(ecurrent < prewell_energy){
    prewell_energy = ecurrent;
    for(int i = 0 i != nvec; ++i){
      prewell_x[i] = x[i];
    }
  }else if(random->uniform() < pow(E,-(ecurrent - prewell_energy)/K_B/temperature)){
    prewell_energy = ecurrent;
    for(int i = 0 i != nvec; ++i){
      prewell_x[i] = x[i];
    } 
  }else{
    for(int i = 0; i!= nvec; ++i){
      xvec[i] = prewell_x[i];
    }
  }
}

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
  double i_n = 1. / double(n);
  sum *= i_n;
  for(int i=0; i<n; i++){
    x[i] -= sum;
  }
}
/*---------------------------------------------------
  	minimize perpendicular to the 
	initial direction
---------------------------------------------------*/

void Artn::min_perpendicular(){
  double fperp2, fperpall2, fdotinit, fdotinitall;
  int k=0, k_reject=0;
  fperp2 = 0., fperpall=0.;
  fdotinit = 0.;
  for(int i=0; i < nvec; i++) fdotinit += fvec[i]*initial_direction[i];
  MPI_Allreduce(fdotinit,fdotinitall,1,MPI_DOUBLE,MPI_SUM,world);
  for(int i=0; i < nvec; i++){
    fperp[i] = fvec[i] - fdotinitall * initial_direction[i];
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
Ref: R.A. Olsen , G. J. Kroes, Comparison of methods for 
     fanding saddle points without knowledge of final states,
     121, 20(2004)
---------------------------------------------------*/
void Artn::lanczos(bool new_projection, int flag, int maxvec){
  // when flag > 0, we need the eigenvector
  /* 
     First we take the current posistion as the reference point 
     and will make a displacement in a random direction or using
     the previous direction as the starting point.
  */
  double *r0 = new double[nvec]; // r0 - random displacement
  /*
     If it is the first geometry iteration a random nonzero vector 
     is chosen, otherwise the eigenvector found in the previous 
     cycle is used.
  */
  double **lanc;
  if(flag>0)memory->create(lanc, maxvec, nvec, "lanc matrx");
  if(!new_projection){
    for(int i = 0; i<nvec; i++) r0[i] = eigenvector[i];
  }else{
    for(int i = 0; i<nvec; i++) r0[i] = 0.5 - random->uniform();
  }
  center(r0, nvec); // center of mass at zero, but is it necessary? And why?

  // normalize r0
  double rsum2 = 0., rsumall2=0.;
  for(int i=0; i<nvec; i++) rsum2 += r0[i] * r0[i];
  MPI_Allreduce(rsum2,rsumall2,1,MPI_DOUBLE,MPI_SUM,world);
  double b0 = rsumall2;
  rsumall2 = 1.0 / rsumall2;
  for(int i=0; i<nvec; i++){
    r0[i] *= rsumall2;		// r0 as q1
    if(flag>0)lanc[0][i] = r0[i];
  }

  // DEL_LANCZOS is the step we use in the finite differece approximation.
  const double DEL_LANCZOS = 0.001;
  const double IDEL_LANCZOS = 1.0 / DEL_LANCZOS;
  
  // refx and refff hold the referece point atom coordinates and force
  double *refx = new double [nvec];
  double *reff = new double [nvec];
  for(int i=0; i<nvec; i++){
    refx[i] = xvec[i];
    reff[i] = fvec[i];
    xvec[i] = refx[i] + r0[i] * IDEL_LANCZOS;
  }

  // caculate the new force
  energy_force(1);


  double *u1 = new double[nvec];
  double *r1 = new double[nvec];
  double *d_bak = new double [maxvec];
  double *e_bak = new double [maxvec];
  double *d = new double[maxvec]; // the diagonal elements of the matrix
  double *e = new double[maxvec]; // the subdiagonal elements
  for(int i=0; i <maxvec; i++){
    d[i] = e[i] =0.;
  }
  double a1 = 0.,b1 = 0.;
  // we get a1;
  for(int i=0; i<nvec; i++){
    u1[i] = (fvec[i] - reff[i]) * IDEL_LANCZOS;
    r1[i] = u1[i]; 
    a1 += r0[i] * u1[i];
  }
  double a1all = 0., b1all = 0.;
  MPI_Allreduce(a1,a1all,1,MPI_DOUBLE,MPI_SUM,world);
  a1 = a1all;


  // we get b1
  for(int i=0; i<nvec; i++){
    r1[i] -= a1 * r0[i];
    b1 += r1[i] * r1[i];
  }
  MPI_Allreduce(b1,b1all,1,MPI_DOUBLE,MPI_SUM,world);
  b1 = b1all;
  d[0] = a1; e[0] = b1;
  double i_b1 = 1.0 / b1;

  double eigen1 = 0., eigen2 = 0.;
  double *work, *z;
  long int ldz = maxvec, info;
  char jobs;
  if(flag > 0) jobs = 'V';
  else jobs = 'N';
  z = new double [ldz*maxvec];
  work = new double [2*maxvec-2];
  
  // now we repeat the game
  for(long int n=2; n<= maxvec; n++){
    for(int i=0; i<nvec; i++){
      r0[i] = r1[i] * i_b1;  // r0 as q2
      if(flag > 0)lanc[n-1][i] = r0[i];
    }
    for(int i=0; i<nvec; i++){
      xvec[i] = refx[i] + r0[i] * DEL_LANCZOS;
    }
    energy_force(1);
    a1 = b1 = a1all = b1all = 0.;
    for(int i=0; i<nvec; i++){
      u1[i] = (fvec[i] - reff[i]) * IDEL_LANCZOS; // u1 as u2
      r1[i] = u1[i] - b1 * r0[i];	// r1 as r2
      a1 += r0[i] * r1[i];
    }
    MPI_Allreduce(a1,a1all,1,MPI_DOUBLE,MPI_SUM,world);
    a1 = a1all;  	// a1 as a2
    for(int i=0; i<nvec; i++){
      r1[i] -= a1 * r0[i]; 
      b1 += r1[i] * r1[i];
    }
    MPI_Allreduce(b1,b1all,1,MPI_DOUBLE,MPI_SUM,world);
    b1 = b1all;
    i_b1 = 1.0 / b1;
    d[n-1] = a1; e[n-1] = b1;
    // Here destev_ is used. The latest version of Norman's code use
    // dgeev_, which may be better.
    for(int i = 0; i != maxvec; i++){
      d_bak[i] = d[i];
      e_bak[i] = e[i];
    }
    if(n >= 2){
      destev_(&jobs,&n,d_bak,e_bak,z,&ldz,work,&info);  
      if(info != 0) error->all(FLERR,"destev_ error in lanczos subroute");
      eigen1 = eigen2;
      eigen2 = d_bak[0];
    }
    if(n >= 3 && fabs((eigen2-eigen1)/eigen1) < 0.1){
      for(int i=0; i != nvec; i++) eigenvector[i] = 0.;
      eigenvalue = eigen2;
      if(flag > 0){
	for(int i=0; i<nvec; i++){
	  for(int j=0; j<n; j++){
	    eigenvector[i] += z[j]*lanc[j][i];
	  }
	}
      }
    }
  }
  // normalize eigenvector.
  double sum = 0., sumall;
  for(int i = 0; i != nvec; i++){
    xvec[i] = refx[i];
    fvec[i] = reff[i];
    sum += eigenvector[i] * eigenvector[i];
  }
  MPI_Allreduce(sum, sumall,1,MPI_DOUBLE,MPI_SUM,world);
  sumall = 1. / sumall;
  for(int i=0; i != nvec; ++i) eigenvector *= sumall;

  //get the dynamic memory I used back. May I could put them in the main class?
  delete []r0;
  delete []u1;
  delete []r1;
  delete []d;
  delete []e;
  delete []z;
  delete []work;
  delete []refx;
  delete []reff;
  if(flag>0 && lanc) memory->destrpy(lanc);
}
