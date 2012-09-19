/*---------------------------------------------------
  Some features of this code are writtern according to
  Norman's Code version 3.0 ARTn. The explanlation of 
  the parameters I used here can be found in the doc of
  his code.
  This code don't do minimizing include extra peratom dof or 
  extra global dof.
----------------------------------------------------*/
#include "lmptype.h"
#include "stdlib.h"
#include "minimize.h"
#include "mpi.h"
#include "math.h"
#include "string.h"
#include "min_cg.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "min.h"
#include "finish.h"
#include "output.h"
#include "timer.h"
#include "random_park.h"
#include "error.h"
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

enum{MAXITER,MAXEVAL,ETOL,FTOL,DOWNHILL,ZEROALPHA,ZEROFORCE,ZEROQUAD};

Artn::Artn(LAMMPS *lmp): MinLineSearch(lmp){
}

Artn::~Artn(){
}

/*---------------------------------------------------
  parase the artn command
---------------------------------------------------*/
void Artn::command(int narg, char ** arg){
  if (narg != 4 ) error->all(FLERR, "Illegal artn command!");
  if (domain->box_exist == 0)
    error->all(FLERR,"Artn command before simulation box is defined");
  update->etol = atof(arg[0]);
  update->ftol = atof(arg[1]);
  update->nsteps = atoi(arg[2]);
  update->max_eval = atoi(arg[3]);

  if (update->etol < 0.0 || update->ftol < 0.0)
    error->all(FLERR,"Illegal Artn command");

  update->whichflag = 2;
  update->beginstep = update->firststep = update->ntimestep;            
  update->endstep = update->laststep = update->firststep + update->nsteps; 
  if (update->laststep < 0 || update->laststep > MAXBIGINT)
    error->all(FLERR,"Too many iterations");

  lmp->init();
  update->minimize->setup();

  timer->init();
  timer->barrier_start(TIME_LOOP);

  mysetup();
  update->minimize->stop_condition = search(update->nsteps);		// check for revising
  update->minimize->stopstr = stopstrings(stop_condition);

  // output after search
  update->nsteps = niter;
  if (update->restrict_output == 0) {
    for (int idump = 0; idump < output->ndump; idump++)
      output->next_dump[idump] = update->ntimestep;
    output->next_dump_any = update->ntimestep;
    if (output->restart_every) output->next_restart = update->ntimestep;
  }
  output->next_thermo = update->ntimestep;

  modify->addstep_compute_all(update->ntimestep);
  ecurrent = energy_force(0);
  output->write(update->ntimestep);

  timer->barrier_stop(TIME_LOOP);

  update->minimize->cleanup();
  Finish finish(lmp);
  finish.end(1);

  update->whichflag = 0;
  update->firststep = update->laststep = 0;
  update->beginstep = update->endstep = 0;
}


/*---------------------------------------------------
  main loop
---------------------------------------------------*/

int Artn::search(int maxevent){
  myinit();
  return min_converge(maxevent);
  store_x();
  for(int ievent = 0; ievent < maxevent; ievent++){
    find_saddle();
    downhill();
    judgement();
  }
}

/*---------------------------------------------------
  setup default parameters
---------------------------------------------------*/
void Artn::mysetup(){
}

/*---------------------------------------------------
  initializing for ARTn
---------------------------------------------------*/
void Artn::myinit(){
}
/*---------------------------------------------------
---------------------------------------------------*/
void Artn::store_x(){
}

void Artn::find_saddle(){
}

void Artn::downhill(){
}

void Artn::judgement(){
}

/*---------------------------------------------------
  converge to minimum, here I use 
  conjuget gradient method.
---------------------------------------------------*/
int Artn::min_converge(int maxiter){
  int i,m,n,fail,ntimestep;
  double beta,gg,dot[2],dotall[2];

  // nlimit = max # of CG iterations before restarting
  // set to ndoftotal unless too big

  int nlimit = static_cast<int> (MIN(MAXSMALLINT,ndoftotal));

  // initialize working vectors

  for (i = 0; i < nvec; i++) h[i] = g[i] = fvec[i];

  gg = fnorm_sqr();

  for (int iter = 0; iter < maxiter; iter++) {
    ntimestep = ++update->ntimestep;
    niter++;

    // line minimization along direction h from current atom->x

    eprevious = ecurrent;
    fail = (this->*linemin)(ecurrent,alpha_final);
    if (fail) return fail;

    // function evaluation criterion

    if (neval >= update->max_eval) return MAXEVAL;

    // energy tolerance criterion

    if (fabs(ecurrent-eprevious) < update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
      return ETOL;
    // force tolerance criterion

    dot[0] = dot[1] = 0.0;
    for (i = 0; i < nvec; i++) {
      dot[0] += fvec[i]*fvec[i];
      dot[1] += fvec[i]*g[i];
    }
    MPI_Allreduce(dot,dotall,2,MPI_DOUBLE,MPI_SUM,world);

    if (dotall[0] < update->ftol*update->ftol)
      return FTOL;

    // update new search direction h from new f
    // = -Grad(x) and old g
    // this is Polak-Ribieri formulation
    // beta = dotall[0]/gg would be
    // Fletcher-Reeves
    // reinitialize CG every ndof iterations by
    // setting beta = 0.0

    beta = MAX(0.0,(dotall[0] - dotall[1])/gg);
    if ((niter+1) % nlimit == 0) beta = 0.0;
    gg = dotall[0];

    for (i = 0; i < nvec; i++) {
      g[i] = fvec[i];
      h[i] = g[i] + beta*h[i];
    }

    // reinitialize CG
    // if new search
    // direction h is
    // not downhill

    dot[0] = 0.0;
    for (i = 0; i < nvec; i++)
      dot[0] += g[i]*h[i];
    MPI_Allreduce(dot,dotall,1,MPI_DOUBLE,MPI_SUM,world);

    if (dotall[0] <= 0.0) {
      for (i = 0; i < nvec; i++) h[i] = g[i];
    }

    // output
    // for
    // thermo,
    // dump,
    // restart
    // files

    if (output->next == ntimestep) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(TIME_OUTPUT);
    }
  }

  return MAXITER;


}
