/*---------------------------------------------------
  Some features of this code are writtern according to
  Norman's Code version 3.0 ARTn. The explanlation of 
  the parameters I used here can be found in the doc of
  his code.
  This code don't do minimizing include extra peratom dof or 
  extra global dof.
----------------------------------------------------*/
#include "lmptype.h"
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
#include "modify.h"
#include <sstream>
#include "fix_minimize.h"
#include "lammps.h"
#include "memory.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "min.h"
#include "comm.h"
#include "modify.h"
#include "compute.h"
#include "neighbor.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "output.h"
#include "thermo.h"
#include "timer.h"
#include "memory.h"


using namespace LAMMPS_NS;
using namespace std;

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

/* -----------------------------------------------------------------------------
 * Constructor of ARTn
 * ---------------------------------------------------------------------------*/
ARTn::ARTn(LAMMPS *lmp): MinLineSearch(lmp){
  char **tmp;
  memory->create(tmp, 5, 10, "Artn string");
  tmp[0] = "ARTn";
  tmp[1] = "all";
  tmp[2] = "atom";
  tmp[3] = "1";
  tmp[4] = "min1000";
  mydump = new ARTn_dump(lmp, 5, tmp);
  memory->destroy(tmp);
}

/* -----------------------------------------------------------------------------
 * deconstructor of ARTn, close files, free memories
 * ---------------------------------------------------------------------------*/
ARTn::~ARTn()
{
  out_log.close();
  out_event_list.close();
  delete random;
  delete mydump;
}

/* -----------------------------------------------------------------------------
 * parase the artn command
 * ---------------------------------------------------------------------------*/
void ARTn::command(int narg, char ** arg)
{
  //delete update->minimize;
  //update->minimize = new ARTn(mylmp);
  if (narg != 4 ) error->all(FLERR, "Illegal ARTn command!");
  if (domain->box_exist == 0)
    error->all(FLERR,"ARTn command before simulation box is defined");

  update->etol = atof(arg[0]);
  update->ftol = atof(arg[1]);
  update->nsteps = atoi(arg[2]);
  update->max_eval = atoi(arg[3]);

  if (update->etol < 0.0 || update->ftol < 0.0) error->all(FLERR,"Illegal ARTn command");

  update->whichflag = 2;
  update->beginstep = update->firststep = update->ntimestep;            
  update->endstep = update->laststep = update->firststep + update->nsteps; 
  if (update->laststep < 0 || update->laststep > MAXBIGINT) error->all(FLERR,"Too many iterations");

  lmp->init();
  mydump->init();
  init();
  update->minimize->setup();
  setup();

  timer->init();
  timer->barrier_start(TIME_LOOP);

  run(update->nsteps);

  timer->barrier_stop(TIME_LOOP);

  //update->minimize->cleanup();
  cleanup();
  Finish finish(lmp);
  finish.end(1);

  update->whichflag = 0;
  update->firststep = update->laststep = 0;
  update->beginstep = update->endstep = 0;
}

/* -----------------------------------------------------------------------------
 * 
 * ---------------------------------------------------------------------------*/
int ARTn::iterate(int maxevent)
{
  return search(maxevent);
}

/* -----------------------------------------------------------------------------
 * 
 * ---------------------------------------------------------------------------*/
int ARTn::search(int maxevent)
{
  mysetup();
  myinit();
  if (me == 0)out_log << "Start to minimize the configuration before try to find the saddle point."<<endl;
  stop_condition = min_converge(max_converge_steps);
  stopstr = stopstrings(stop_condition);
  if (me == 0){
    out_log << "- Minimize stop condition: "<< stopstr<< endl;
    out_log << "- Current energy (reference energy): " << ecurrent << endl;
    out_log << "- Temperature: "<< temperature <<endl;
  }
  ostringstream strm;
  strm << "min"<<file_counter;
  config_file = strm.str();
  store_config(config_file);
  config_file.clear();
  if (me == 0)out_log << "Configuration stored in file: "<<config_file<<'\n'<<endl;
  for (int ievent = 0; ievent < maxevent; ievent++){
    while (!find_saddle());
    ostringstream strm(config_file);
    strm << "sad"<<file_counter;
    config_file = strm.str();
    store_config(config_file);
    config_file.clear();
    ++file_counter;
    downhill();
    judgement();
  }

return 1;
}

/* -----------------------------------------------------------------------------
 * 
 * ---------------------------------------------------------------------------*/
void ARTn::downhill()
{
  for(int i = 0; i < nvec; ++i) xvec[i] = (xvec[i] - x0[i]) * prefactor_push_over_saddle;


  if (me == 0) out_log << "Start to minimize the configuration to reach another minimal."<<endl;
  stop_condition = min_converge(max_converge_steps);
  stopstr = stopstrings(stop_condition);
  eref = ecurrent;

  if (me == 0){
    out_log << "- Minimize stop condition: "<< stopstr<< endl;
    out_log << "- Current energy : " << ecurrent << endl;
    out_log << "- Temperature: "<< temperature <<endl;
  }

  ostringstream strm;
  strm << "min"<<file_counter;
  config_file = strm.str();
  store_config(config_file);
  config_file.clear();
  if (me == 0) out_log << "Configuration stored in file: "<<config_file<<'\n'<<endl;

}

/* -----------------------------------------------------------------------------
 * decide whethe reject or accept the new configuration
 * ---------------------------------------------------------------------------*/
void ARTn::judgement()
{
  // KLT: do not use numbers, I believe you mean exp((eref-ecurrent)/temperature)?
  if (ecurrent < eref || random->uniform() < exp(eref - ecurrent)/temperature){
    eref = ecurrent;
    for (int i =0; i < nvec; ++i) x0[i] = xvec[i];

    out_log << "Accept this new configuration."<<endl;
    out_event_list << "accept" << endl;

  } else {

    for(int i = 0; i < nvec; ++i){
      xvec[i] = x0[i];
    }
    ecurrent = energy_force(0);
    out_log << "Reject this new configuration."<<endl;
    out_event_list << "reject" << endl;
  }
}

/* -----------------------------------------------------------------------------
 * store current configuration in file
 * ---------------------------------------------------------------------------*/
void ARTn::store_config(string file){
  out_event_list << file << '\t';
  mydump->modify_file(file);
  mydump->write();
}

/* -----------------------------------------------------------------------------
 * setup default parameters, these values come from Norman's code
 * ---------------------------------------------------------------------------*/
void ARTn::mysetup()
{
  max_converge_steps = 1000000;

  // for art
  temperature = -0.5;
  max_num_events = 1000;
  activation_maxiter = 300;
  increment_size = 0.09;
  force_threhold_perp_rel = 0.5;

  // for harmonic well
  initial_step_size = 0.05;
  basin_factor = 2.1;
  max_perp_moves_basin = 3;
  min_number_ksteps = 3;
  eigenvalue_threhold = -0.05;
  max_iter_basin = 12;

  // for lanczos
  number_lanczos_vectors_H = 13;
  number_lanczos_vectors_C = 12;
  delta_displ_lanczos = 0.01;

  // for convergence
  exit_force_threhold = 0.25;
  prefactor_push_over_saddle = 0.3;

  // for output
  event_list_file = "events.list";
  log_file = "log.file";
  file_counter = 1000;
}

/* -----------------------------------------------------------------------------
 * initializing for ARTn
 * ---------------------------------------------------------------------------*/
void ARTn::myinit()
{
  me =  MPI_Comm_rank(world,&me);
  out_event_list.open(event_list_file.c_str());
  if (!out_event_list) error->all(FLERR, "Open event list file error!");
  out_log.open(log_file.c_str());
  if (!out_log) error->all(FLERR, "open event log file error!");
  random = new RanPark(lmp, 12340);
  evalf = 0;


  // for peratom vector I use
  vec_count = 4;
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  x0tmp = fix_minimize->request_vector(vec_count);
  ++vec_count;
  h_old = fix_minimize->request_vector(vec_count);
  ++vec_count;
  eigenvector = fix_minimize->request_vector(vec_count);
  ++vec_count;
}


/* -----------------------------------------------------------------------------
 * Try to find saddle point. If failed, return 0, else return 1
 * ---------------------------------------------------------------------------*/
int ARTn::find_saddle()
{
  double first_time = 1;
  double ftot = 0., ftotall, fpar2, fperp2 = 0., fperp2all,  delr;
  double fdoth = 0., fdothall = 0.;
  double * fperp;
  double step = increment_size * 0.4, preenergy;
  double tmp;
  int m_perp = 0, trial = 0, nfail = 0;

  for (int i = 0; i < nvec; i++) x0[i] = xvec[i];
  global_random_move();
  ecurrent = energy_force(0);

  // try to leave harmonic well
  for (int local_iter = 0; local_iter < max_iter_basin; local_iter++){
    // do minimizing perpendicular
    preenergy = ecurrent;
    while (1){
      for (int i = 0; i < nvec; i++) fdoth += fvec[i] * h[i];
      MPI_Allreduce(&fdoth, &fdothall,1,MPI_DOUBLE,MPI_SUM,world);
      fperp = new double[nvec];
      fperp2 = ftot = 0.;

      for (int i = 0; i < nvec; i++){ 
        fperp[i] = fvec[i] - fdothall * h[i];
        fperp2 += fperp[i] * fperp[i];
      }
      MPI_Allreduce(&fperp2, &fperp2all,1,MPI_DOUBLE,MPI_SUM,world);
      fperp2 = sqrt(fperp2all);
      if (fperp2 < force_threhold_perp_rel || m_perp > max_perp_moves_basin || nfail > 5) break; // condition to break

      for (int i = 0; i < nvec; ++i){
	     x0tmp[i] = xvec[i];
	     xvec[i] += step * fperp[i];
      }
      ecurrent = energy_force(1);
      evalf++;
      reset_coords();

      delete []fperp;
      if (ecurrent < preenergy){
        step *= 1.2;
        m_perp++;
        nfail = 0;
        preenergy = ecurrent;

      } else {

        for (int i = 0; i < nvec; ++i) xvec[i] = x0tmp[i];
        step *= 0.6;
        nfail++;
        ecurrent = energy_force(1);
        evalf++;
      }
      trial++;
    }

    // push along the search direction
    for (int i = 0; i < nvec; ++i) {
      xvec[i] += basin_factor * increment_size * h[i];
      ftot += fvec[i] * fvec[i];
      tmp = xvec[i] - x0[0];
      delr += tmp * tmp;
    }
    tmp = delr;
    MPI_Allreduce(&ftot, &ftotall,1,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&tmp, &delr,1,MPI_DOUBLE,MPI_SUM,world);

    // output information
    out_log << local_iter << '\t' << ecurrent - eref << '\t';
    out_log << m_perp <<'\t'<< trial<<'\t' << sqrt(ftotall)<<'\t';
    out_log << fperp2 <<'\t'<< eigenvalue <<'\t'<< sqrt(delr)<<'\t' << evalf << endl;

    m_perp = trial = nfail = 0;
    delr = 0.;
    step = increment_size * 0.4;

    if (local_iter > min_number_ksteps){
      lanczos(first_time, 1, number_lanczos_vectors_H);
      first_time = 0;
      if (eigenvalue < eigenvalue_threhold){
        break;
        out_log << "Out of harmonic well, now search according to the eigenvector." << endl;
      }
    }
  }

  // now try to move close to the saddle point according to the eigenvector.
  for (int saddle_iter = 0; saddle_iter < activation_maxiter; ++saddle_iter){
    for (int i = 0; i < nvec; ++i) h_old[i] = h[i];

    lanczos(first_time, 1, number_lanczos_vectors_C);
    double sum , sumall, tmpsum, tmpsumall;
    sum = sumall = tmpsum = tmpsumall = 0.;
    for (int i =0; i < nvec; ++i) tmpsum += eigenvector[i] * fvec[i];

    MPI_Allreduce(&tmpsum,&tmpsumall,1,MPI_DOUBLE,MPI_SUM,world);
    if (tmpsumall < 0){
      for(int i = 0; i < nvec; ++i) h[i] = -eigenvector[i];
    }
    for(int i = 0; i <nvec; ++i) sum += h[i] * h_old[i];
    MPI_Allreduce(&sum,&sumall,1,MPI_DOUBLE,MPI_SUM,world);

    // do minimizing perpendicular
    preenergy = ecurrent;
    while (1){
      for (int i = 0; i < nvec; i++) fdoth += fvec[i] * h[i];
      MPI_Allreduce(&fdoth, &fdothall,1,MPI_DOUBLE,MPI_SUM,world);
      fperp = new double[nvec];
      fperp2 = ftot = 0.;
      for (int i = 0; i < nvec; i++){
        fperp[i] = fvec[i] - fdothall * h[i];
        fperp2 += fperp[i] * fperp[i];
      }
      MPI_Allreduce(&fperp2, &fperp2all,1,MPI_DOUBLE,MPI_SUM,world);
      fperp2 = sqrt(fperp2all);
      if (fperp2 < force_threhold_perp_rel || m_perp > max_perp_moves_basin || nfail > 5) break; // condition to break

      for (int i = 0; i < nvec; ++i){
        x0tmp[i] = xvec[i];
        xvec[i] += step * fperp[i];
      }
      ecurrent = energy_force(1);
      evalf++;
      reset_coords();

      delete []fperp;
      if (ecurrent < preenergy){
        step *= 1.2;
        m_perp++;
        nfail = 0;
        preenergy = ecurrent;

      } else {

        for (int i = 0; i < nvec; ++i) xvec[i] = x0tmp[i];

        step *= 0.6;
        nfail++;
        ecurrent = energy_force(1);
        evalf++;
      }
      trial++;
    }

    // push along the search direction
    for (int i = 0; i < nvec; ++i) {
      xvec[i] += basin_factor * increment_size * h[i];
      ftot += fvec[i] * fvec[i];
      tmp = xvec[i] - x0[0];
      delr += tmp * tmp;
    }
    tmp = delr;
    MPI_Allreduce(&ftot, &ftotall,1,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&tmp, &delr,1,MPI_DOUBLE,MPI_SUM,world);

    // output information
    fpar2 = ftotall - fperp2all;
    fpar2 = sqrt(fpar2);
    out_log << saddle_iter << '\t' << ecurrent - eref << '\t';
    out_log << m_perp <<'\t'<< trial<<'\t' << sqrt(ftotall)<<'\t'<< fpar2<<'\t';
    out_log << fperp2 <<'\t'<< eigenvalue <<'\t'<< sqrt(delr)<<'\t'<< evalf <<'\t'<< sumall<< endl;

    m_perp = trial = nfail = 0;
    delr = 0.;
    step = increment_size * 0.4;

    if (eigenvalue > 0.){
      out_log << "Failed to find the saddle point in lanczos steps, for eigenvalue > 0."<<'\n';
      out_log << "Reassign the random direction.";
      for(int i = 0; i < nvec; ++i) xvec[i] = x0[i];
      return 0;
    }

    if (sqrt(ftotall)<exit_force_threhold ){
      out_log << "Reach saddle point."<< endl;
      return 1;
    }
  }
  out_log << "Failed to find the saddle point in lanczos steps, for reaching the max lanczos steps."<<'\n';
  out_log << "Reassign the random direction.";
  for (int i = 0; i < nvec; ++i) xvec[i] = x0[i];

return 0;
}

/* -----------------------------------------------------------------------------
 *
 * ---------------------------------------------------------------------------*/
void ARTn::reset_coords()
{
  domain->set_global_box();

  double **x = atom->x;
  double *x0 = fix_minimize->request_vector(4);
  int nlocal = atom->nlocal;
  double dx,dy,dz,dx0,dy0,dz0;

  int n = 0;
  for (int i = 0; i < nlocal; i++) {
    dx = dx0 = x[i][0] - x0[n];
    dy = dy0 = x[i][1] - x0[n+1];
    dz = dz0 = x[i][2] - x0[n+2];
    domain->minimum_image(dx,dy,dz);

    if (dx != dx0) x0[n] = x[i][0] - dx;
    if (dy != dy0) x0[n+1] = x[i][1] - dy;
    if (dz != dz0) x0[n+2] = x[i][2] - dz;
    n += 3;
  }

  domain->set_global_box();
}

/* -----------------------------------------------------------------------------
 * give a global random delta_x 
 * ---------------------------------------------------------------------------*/
void ARTn::global_random_move()
{
  double *delpos = new double[nvec];
  double norm;
  for (int i=0; i < nvec; i++) delpos[i] = 0.5-random->uniform();

  center(delpos, nvec);
  for (int i=0, norm=0; i < nvec; i++) norm += delpos[i] * delpos[i];

  double normall;
  MPI_Allreduce(&norm,&normall,1,MPI_DOUBLE,MPI_SUM,world);

  double norm_i = 1./normall;
  for (int i=0; i < atom->nlocal; i++){
    h[i] = delpos[i] * norm_i;
    xvec[i] += initial_step_size * h[i];
  }
  delete []delpos;
  ecurrent = energy_force(0);
}

/* -----------------------------------------------------------------------------
 * converge to minimum, here I use conjuget gradient method.
 * ---------------------------------------------------------------------------*/
int ARTn::min_converge(int maxiter)
{
  int i,fail,ntimestep;
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
    for (i = 0; i < nvec; i++) dot[0] += g[i]*h[i];
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

/* -----------------------------------------------------------------------------
 * center the vector
 * ---------------------------------------------------------------------------*/
void ARTn::center(double * x, int n)
{
  double sum = 0.;
  for (int i=0; i<n; i++) sum += x[i];

  double i_n = 1. / double(n);
  sum *= i_n;
  for (int i=0; i<n; i++) x[i] -= sum;
}

/* -----------------------------------------------------------------------------
 * The lanczos method to get the lowest eigenvalue
 * and corresponding eigenvector
 * Ref: R.A. Olsen , G. J. Kroes, Comparison of methods for 
 *   fanding saddle points without knowledge of final states,
 *   121, 20(2004)
 * ---------------------------------------------------------------------------*/
void ARTn::lanczos(bool new_projection, int flag, int maxvec)
{
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
  double **lanc = NULL;
  if (flag>0) memory->create(lanc, maxvec, nvec, "lanc matrx");
  if (!new_projection){
    for (int i = 0; i<nvec; i++) r0[i] = eigenvector[i];
  } else {
    for (int i = 0; i<nvec; i++) r0[i] = 0.5 - random->uniform();
  }
  center(r0, nvec); // center of mass at zero, but is it necessary? And why?

  // normalize r0
  double rsum2 = 0., rsumall2=0.;
  for (int i=0; i<nvec; i++) rsum2 += r0[i] * r0[i];
  MPI_Allreduce(&rsum2,&rsumall2,1,MPI_DOUBLE,MPI_SUM,world);
  rsumall2 = 1.0 / rsumall2;

  for (int i=0; i<nvec; i++){
    r0[i] *= rsumall2;		// r0 as q1
    if (flag>0) lanc[0][i] = r0[i];
  }

  // DEL_LANCZOS is the step we use in the finite differece approximation.
  const double DEL_LANCZOS = delta_displ_lanczos;
  const double IDEL_LANCZOS = 1.0 / DEL_LANCZOS;
  
  // refx and refff hold the referece point atom coordinates and force

  for (int i=0; i<nvec; i++){
    x0tmp[i] = xvec[i];
    g[i] = fvec[i];
    xvec[i] += r0[i] * IDEL_LANCZOS;
  }

  // caculate the new force
  myenergy_force();
  ++evalf;
  reset_coords();

  double *u1 = new double[nvec];
  double *r1 = new double[nvec];
  double *d_bak = new double [maxvec];
  double *e_bak = new double [maxvec];
  double *d = new double[maxvec]; // the diagonal elements of the matrix
  double *e = new double[maxvec]; // the subdiagonal elements

  for (int i=0; i <maxvec; i++) d[i] = e[i] =0.;

  double a1 = 0.,b1 = 0.;
  // we get a1;
  for (int i=0; i<nvec; i++){
    u1[i] = (fvec[i] - g[i]) * IDEL_LANCZOS;
    r1[i] = u1[i]; 
    a1 += r0[i] * u1[i];
  }
  double a1all = 0., b1all = 0.;
  MPI_Allreduce(&a1,&a1all,1,MPI_DOUBLE,MPI_SUM,world);
  a1 = a1all;

  // we get b1
  for (int i=0; i<nvec; i++){
    r1[i] -= a1 * r0[i];
    b1 += r1[i] * r1[i];
  }
  MPI_Allreduce(&b1,&b1all,1,MPI_DOUBLE,MPI_SUM,world);
  b1 = b1all;
  d[0] = a1; e[0] = b1;
  double i_b1 = 1.0 / b1;

  double eigen1 = 0., eigen2 = 0.;
  double *work, *z;
  long int ldz = maxvec, info;
  char jobs;
  if (flag > 0) jobs = 'V';
  else jobs = 'N';
  z = new double [ldz*maxvec];
  work = new double [2*maxvec-2];
  
  // now we repeat the game
  for (long int n=2; n<= maxvec; n++){
    for (int i=0; i<nvec; i++){
      r0[i] = r1[i] * i_b1;  // r0 as q2
      if (flag > 0) lanc[n-1][i] = r0[i];
    }

    for (int i=0; i<nvec; i++) xvec[i] = x0tmp[i] + r0[i] * DEL_LANCZOS;

    myenergy_force();
    ++evalf;
    reset_coords();
    a1 = b1 = a1all = b1all = 0.;

    for (int i=0; i<nvec; i++){
      u1[i] = (fvec[i] - g[i]) * IDEL_LANCZOS; // u1 as u2
      r1[i] = u1[i] - b1 * r0[i];	// r1 as r2
      a1 += r0[i] * r1[i];
    }
    MPI_Allreduce(&a1,&a1all,1,MPI_DOUBLE,MPI_SUM,world);
    a1 = a1all;  	// a1 as a2

    for (int i=0; i<nvec; i++){
      r1[i] -= a1 * r0[i]; 
      b1 += r1[i] * r1[i];
    }
    MPI_Allreduce(&b1,&b1all,1,MPI_DOUBLE,MPI_SUM,world);
    b1 = b1all;
    i_b1 = 1.0 / b1;
    d[n-1] = a1; e[n-1] = b1;
    // Here destev_ is used. The latest version of Norman's code use
    // dgeev_, which may be better.
    for (int i = 0; i != maxvec; i++){
      d_bak[i] = d[i];
      e_bak[i] = e[i];
    }
    if (n >= 2){
      dstev_(&jobs,&n,d_bak,e_bak,z,&ldz,work,&info);  
      if (info != 0) error->all(FLERR,"destev_ error in lanczos subroute");
      eigen1 = eigen2;
      eigen2 = d_bak[0];
    }
    if (n >= 3 && fabs((eigen2-eigen1)/eigen1) < 0.1){
      for (int i = 0; i < nvec; ++i){
        xvec[i] = x0tmp[i];
        fvec[i] = g[i];
      }
      eigenvalue = eigen2;
      if (flag > 0){
        for (int i=0; i < nvec; i++) eigenvector[i] = 0.;
        for (int i=0; i<nvec; i++){
          for (int j=0; j<n; j++) eigenvector[i] += z[j]*lanc[j][i];
        }

        // normalize eigenvector.
        double sum = 0., sumall;
        for (int i = 0; i < nvec; i++) sum += eigenvector[i] * eigenvector[i];

        MPI_Allreduce(&sum, &sumall,1,MPI_DOUBLE,MPI_SUM,world);
        sumall = 1. / sumall;
        for (int i=0; i < nvec; ++i) eigenvector[i] *= sumall;
      }
    }
  }

  //get the dynamic memory I used back. May I could put them in the main class?
  delete []r0;
  delete []u1;
  delete []r1;
  delete []d;
  delete []e;
  delete []z;
  delete []work;
  if (flag>0 && lanc) memory->destroy(lanc);
}

/* -----------------------------------------------------------------------------
 * I use this function try to avoid atom communication 
 * in lanczos subroutine, for little change of atom cordinates.
 * ---------------------------------------------------------------------------*/
double ARTn::myenergy_force()
{
  ev_set(update->ntimestep);
  force_clear();
  if (modify->n_min_pre_force) modify->min_pre_force(vflag);

  timer->stamp();

  if (force->pair) {
    force->pair->compute(eflag,vflag);
    timer->stamp(TIME_PAIR);
  }

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
    timer->stamp(TIME_BOND);
  }

  if (force->kspace) {
    force->kspace->compute(eflag,vflag);
    timer->stamp(TIME_KSPACE);
  }

  if (force->newton) {
    comm->reverse_comm();
    timer->stamp(TIME_COMM);
  }
  // fixes that affect minimization

  if (modify->n_min_post_force) modify->min_post_force(vflag);

  // compute potential energy of system
  // normalize if thermo PE does

  double energy = pe_compute->compute_scalar();
  if (nextra_global) energy += modify->min_energy(fextra);
  if (output->thermo->normflag) energy /= atom->natoms;

return energy;
}
