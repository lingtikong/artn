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
#include "group.h"
#include <iomanip>

//#define TEST
#define TESTOUPUT

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
  strcpy(tmp[0],"ARTnmin");
  strcpy(tmp[1],"all");
  strcpy(tmp[2],"atom");
  strcpy(tmp[3],"1");
  strcpy(tmp[4],"min.lammpstrj");
  dumpmin = new DumpAtom(lmp, 5, tmp);
  //strcpy(tmp[0],"ARTnsadl");
  //strcpy(tmp[4],"saddle.lammpstrj");
  //dumpsadl = new DumpAtom(lmp, 5, tmp);
  memory->destroy(tmp);
  memory->create(tmp, 5, 10, "Artn string");
  strcpy(tmp[0],"ARTnsadl");
  strcpy(tmp[1],"all");
  strcpy(tmp[2],"atom");
  strcpy(tmp[3],"1");
  strcpy(tmp[4],"sadl.lammpstrj");
  dumpsadl = new DumpAtom(lmp, 5, tmp);
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
  delete dumpmin;
  delete dumpsadl;
}

/* -----------------------------------------------------------------------------
 * parase the artn command
 * ---------------------------------------------------------------------------*/
void ARTn::command(int narg, char ** arg)
{
#ifdef TEST 
  cout << "Entering command(), proc: " << me << endl;
#endif
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
  //mydump->init();
  dumpmin->init();
  dumpsadl->init();
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
 * This function try to fit min class function
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
#ifdef TEST
  cout << "Entering search(), proc: " << me << endl;
#endif
  mysetup();
  myinit();
  outlog("Start to minimize the configuration before try to find the saddle point.\n");
  stop_condition = min_converge(max_converge_steps);
  eref = ecurrent = energy_force(1);
#ifdef TEST
  double testsum = 0.;
  for (int i = 0; i < nvec; ++i) testsum += fvec[i] * fvec[i];
  cout << "Afer minimize, ftot = " << sqrt(testsum) << endl;
#endif
  stopstr = stopstrings(stop_condition);
  if (me == 0){
    out_log << "- Minimize stop condition: "<< stopstr<< endl;
    out_log << "- Current energy (reference energy): " << ecurrent << endl;
    out_log << "- Temperature: "<< temperature <<endl;
  }
  ostringstream strm;
  strm << "min"<<file_counter;
  config_file = strm.str();
  dumpmin->write();
  store_config(config_file);
  if (me == 0)out_log << "Configuration stored in file: "<<config_file<<'\n'<<endl;
  config_file.clear();
  for (int ievent = 0; ievent < maxevent; ievent++){
    while (!find_saddle());
    ostringstream strm(config_file);
    strm << "sad"<<file_counter;
    config_file = strm.str();
    dumpsadl->write();
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
  for(int i = 0; i < nvec; ++i) xvec[i] = x0[i] + (xvec[i] - x0[i]) * prefactor_push_over_saddle;
  energy_force(1);
  outlog("Start to minimize the configuration to reach another minimal.\n");
  stop_condition = min_converge(max_converge_steps);
  stopstr = stopstrings(stop_condition);

  ecurrent = energy_force(1);
  if (me == 0){
    out_log << "- Minimize stop condition: "<< stopstr<< endl;
    out_log << "- Current energy : " << ecurrent << endl;
    out_log << "- Temperature: "<< temperature <<endl;
  }

  ostringstream strm;
  strm << "min"<<file_counter;
  config_file = strm.str();
  dumpmin->write();
  store_config(config_file);
  if (me == 0) out_log << "Configuration stored in file: "<<config_file<<'\n'<<endl;
  config_file.clear();

}

/* -----------------------------------------------------------------------------
 * decide whether reject or accept the new configuration
 * ---------------------------------------------------------------------------*/
void ARTn::judgement()
{
  // KLT: do not use numbers, I believe you mean exp((eref-ecurrent)/temperature)?
  if (ecurrent < eref || random->uniform() < exp(eref - ecurrent)/temperature){
    eref = ecurrent;
    for (int i =0; i < nvec; ++i) x0[i] = xvec[i];

    outlog("Accept this new configuration.\n");
    outeven("accept\t");
    out_event_list<<ecurrent<<endl;

  } else {

    for(int i = 0; i < nvec; ++i){
      xvec[i] = x0[i];
    }
    ecurrent = energy_force(1);
    outlog("Reject this new configuration.\n");
    outeven("reject\t");
    out_event_list << ecurrent << endl;
  }
}

/* -----------------------------------------------------------------------------
 * store current configuration in file
 * ---------------------------------------------------------------------------*/
void ARTn::store_config(string file){
#ifdef TEST
  cout << "Entering store_config(), proc: " << me << endl;
#endif
  if (me == 0) out_event_list << file << '\t';
  //mydump->modify_file(file);
  //mydump->write();
#ifdef TEST
  cout << "Out of store_config(), proc: " << me << endl;
#endif
}

/* -----------------------------------------------------------------------------
 * setup default parameters, some values come from Norman's code
 * ---------------------------------------------------------------------------*/
void ARTn::mysetup()
{
  max_converge_steps = 1000000;

  // for art
  temperature = 0.5;
  max_num_events = 1000;
  activation_maxiter = 300;
  increment_size = 0.09;
  force_threhold_perp_rel = 0.05;
  group_random = true;

  // for harmonic well
  initial_step_size = 0.05;
  basin_factor = 2.1;
  max_perp_moves_basin = 8;		// 3
  min_number_ksteps = 2;		
  eigenvalue_threhold = -0.001;
  //max_iter_basin = 12;
  max_iter_basin = 100;
  force_threhold_perp_h = 0.5;

  // for lanczos
  //number_lanczos_vectors_H = 13;
  number_lanczos_vectors_H = 40;
  number_lanczos_vectors_C = 12;
  delta_displ_lanczos = 0.01;
  eigen_threhold = 0.1;

  // for convergence
  exit_force_threhold = 0.1;
  prefactor_push_over_saddle = 0.3;
  eigen_fail = 0.1;

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
  if (me == 0) out_event_list.open(event_list_file.c_str());
  if (!out_event_list) error->all(FLERR, "Open event list file error!");
  if (me == 0) out_log.open(log_file.c_str());
  if (!out_log) error->all(FLERR, "open event log file error!");
  random = new RanPark(lmp, 12340);
  evalf = 0;
  eigen_vector_exist = 0;


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
#ifdef TEST
  cout << "Enetering find_saddle(), proc: " << me << endl;
#endif
  int flag = 0;
  double ftot = 0., ftotall, fpar2, fperp2 = 0., fperp2all,  delr;
  double fdoth = 0., fdothall = 0.;
  double * fperp;
  double step, preenergy;
  double tmp;
  int m_perp = 0, trial = 0, nfail = 0;
  eigenvalue = 0.;
  eigen_vector_exist = 0;

  for (int i = 0; i < nvec; i++) x0[i] = xvec[i];
#ifdef TEST
  //global_random_move();
/*  outlog("Test: Start to minimize the configuration to reach another minimal.\n");
  stop_condition = min_converge(max_converge_steps);
  stopstr = stopstrings(stop_condition);

  if (me == 0){
    out_log << "- Minimize stop condition: "<< stopstr<< endl;
    out_log << "- Current energy : " << ecurrent << endl;
    out_log << "- Temperature: "<< temperature <<endl;
  }

  lanczos(1,1,30);
  cout << "After minimize , eigenvalue = " << eigenvalue << endl;
  */
#endif
  if (group_random) group_random_move();
  else global_random_move();
  ecurrent = energy_force(1);
  ++evalf;

#ifdef TESTOUPUT
  if ( me ==0 ) {
    out_log << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(5) << endl;
    out_log << setw(12) << "E-Eref"<< setw(12) << "m_perp" <<setw(12) << "trial" << setw(12) <<"ftot";
    out_log << setw(12) << "fperp" << setw(12) << "eigenvalue" << setw(12) << "delr" << setw(12) << "evalf\n";
  }
#else
  outlog("\tE-Eref\t\tm_perp\ttrial\tftot\t\tfperp\t\teigenvalue\tdelr\tevalf\n");
#endif
  // try to leave harmonic well
  for (int local_iter = 0; local_iter < max_iter_basin; local_iter++){
    // do minimizing perpendicular
    ecurrent = energy_force(1);
    ++evalf;
    m_perp = nfail = trial = 0;
    step = increment_size * 0.4;
    while (1){
      preenergy = ecurrent;
      fdoth = 0.;
      for (int i = 0; i < nvec; i++) fdoth += fvec[i] * h[i];
      MPI_Allreduce(&fdoth, &fdothall,1,MPI_DOUBLE,MPI_SUM,world);
      fperp = new double[nvec];
      fperp2 = 0.;

      for (int i = 0; i < nvec; i++){ 
	fperp[i] = fvec[i] - fdothall * h[i];
	fperp2 += fperp[i] * fperp[i];
      }
      MPI_Allreduce(&fperp2, &fperp2all,1,MPI_DOUBLE,MPI_SUM,world);
      fperp2 = sqrt(fperp2all);
      if (fperp2 < force_threhold_perp_h || m_perp > max_perp_moves_basin || nfail > 5) break; // condition to break

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

    ftot = delr = 0.;
    for (int i = 0; i < nvec; ++i) {
      ftot += fvec[i] * fvec[i];
      tmp = xvec[i] - x0[i];
      delr += tmp * tmp;
    }
    tmp = delr;
    double dot[2] = {ftot, tmp};
    double dotall[2];
    MPI_Allreduce(dot, dotall,2,MPI_DOUBLE,MPI_SUM,world);
    ftotall = dotall[0];
    delr = dotall[1];

    if (local_iter > min_number_ksteps){
      lanczos(!eigen_vector_exist, 1, number_lanczos_vectors_H);
    }
    // output information
#ifdef TESTOUTPUT
    if (me ==0){
      out_log << setw(12) << local_iter << setw(12) << ecurrent - eref << setw(12) <<  m_perp;
      out_log << setw(12) << trial<< setw(12) << sqrt(ftotall) << setw(12) << fperp2;
      out_log << setw(12) << eigenvalue << setw(12) << sqrt(delr)<< setw(12) << evalf << endl;
    }
#else
    if(me == 0){
      out_log << setiosflags(ios::fixed)<<setprecision(5);
      out_log << local_iter << '\t' << ecurrent - eref << '\t'<<'\t';
      out_log << m_perp <<'\t'<< trial<<'\t' << sqrt(ftotall)<<'\t' <<'\t';
      out_log << fperp2 <<'\t'<<'\t'<< eigenvalue <<'\t'<< sqrt(delr)<<'\t' << evalf << endl;
    }
#endif
    if (local_iter > min_number_ksteps){
      if (eigenvalue < eigenvalue_threhold){
	if(me == 0)out_log << "Out of harmonic well, now search according to the eigenvector." << endl;
	flag = 1;
	break;
      }
    }
    // push along the search direction
    for(int i = 0; i < nvec; ++i) xvec[i] += basin_factor * increment_size * h[i];

  }

  if(!flag){
    if(me == 0) out_log << "Reach  max_iter_basin. Could not get out of harmonic well."<<endl;
    return 0;
  }
  flag = 0;

  outlog("\tE-Eref\t\tm_perp\ttrial\tftot\t\tfpar\t\tfperp\t\teigenvalue\tdelr\tevalf\ta1\n");
  // now try to move close to the saddle point according to the eigenvector.
  for (int saddle_iter = 0; saddle_iter < activation_maxiter; ++saddle_iter){
    ecurrent = energy_force(1);
    for (int i = 0; i < nvec; ++i) h_old[i] = h[i];

    lanczos(!eigen_vector_exist, 1, number_lanczos_vectors_C);
    double hdot , hdotall, tmpsum, tmpsumall;
    hdot = hdotall = tmpsum = tmpsumall = 0.;
    for (int i =0; i < nvec; ++i) tmpsum += eigenvector[i] * fvec[i];

    MPI_Allreduce(&tmpsum,&tmpsumall,1,MPI_DOUBLE,MPI_SUM,world);
    if (tmpsumall < 0){
      for(int i = 0; i < nvec; ++i) h[i] = -eigenvector[i];
    }
    for(int i = 0; i <nvec; ++i) hdot += h[i] * h_old[i];
    MPI_Allreduce(&hdot,&hdotall,1,MPI_DOUBLE,MPI_SUM,world);
    // do minimizing perpendicular
    preenergy = ecurrent;
    m_perp = trial = nfail = 0;
    step = increment_size * 0.4;
    while (1){
      fdoth = 0.;
      for (int i = 0; i < nvec; i++) fdoth += fvec[i] * h[i];
      MPI_Allreduce(&fdoth, &fdothall,1,MPI_DOUBLE,MPI_SUM,world);
      fperp = new double[nvec];
      fperp2 = 0.;
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

    delr = ftot = 0.;
    for (int i = 0; i < nvec; ++i) {
      ftot += fvec[i] * fvec[i];
      tmp = xvec[i] - x0[i];
      delr += tmp * tmp;
    }
    tmp = delr;
    MPI_Allreduce(&ftot, &ftotall,1,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&tmp, &delr,1,MPI_DOUBLE,MPI_SUM,world);

    // output information
    fpar2 = ftotall - fperp2all;
    fpar2 = sqrt(fpar2);
    if (me == 0){
      out_log << saddle_iter << '\t' << ecurrent - eref << '\t' << '\t';
      out_log << m_perp <<'\t'<< trial<<'\t' << sqrt(ftotall)<<'\t' <<'\t'<< fpar2<<'\t' << '\t';
      out_log << fperp2 <<'\t'<< '\t'<< eigenvalue <<'\t'<< sqrt(delr)<<'\t'<< evalf <<'\t'<< hdotall<< endl;
    }


    if (eigenvalue > eigen_fail){
      if (me == 0){
	out_log << "Failed to find the saddle point in lanczos steps, for eigenvalue > 0."<<'\n';
	out_log << "Reassign the random direction.";
      }
      for(int i = 0; i < nvec; ++i) xvec[i] = x0[i];
      return 0;
    }

    if (sqrt(ftotall)<exit_force_threhold ){
      outlog("Reach saddle point.\n");
      return 1;
    }

    // push along the search direction
    for (int i = 0; i < nvec; ++i) xvec[i] += basin_factor * increment_size * h[i];
  }
  outlog("Failed to find the saddle point in lanczos steps, for reaching the max lanczos steps.\n");
  outlog("Reassign the random direction.\n");
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
#ifdef TEST
  cout << "Entering global_random_move(), proc: "<< me << endl;
#endif
  double *delpos = new double[nvec];
  double norm = 0.;
  for (int i=0; i < nvec; i++) delpos[i] = 0.5-random->uniform();

  center(delpos, nvec);
  for (int i=0; i < nvec; i++) norm += delpos[i] * delpos[i];

  double normall;
  MPI_Allreduce(&norm,&normall,1,MPI_DOUBLE,MPI_SUM,world);

  double norm_i = 1./sqrt(normall);
  for (int i=0; i < nvec; i++){
    h[i] = delpos[i] * norm_i;
    xvec[i] += initial_step_size * h[i];
  }
  delete []delpos;
  return;
}

/* ------------------------------------------------------------------------------
 * give a group random delta_x
 * ----------------------------------------------------------------------------*/
void ARTn::group_random_move()
{
  double *delpos = new double[nvec];
  double norm = 0.;
  for (int i = 0; i < nvec; ++i){
    int igroup = group->find("artn");
    if (igroup = -1) error->all(FLERR,"Could not find artn group!");
    int bit = group->bitmask[igroup];
    if (bit & atom->mask[i]) delpos[i] = 0.5 - random->uniform();
  }
  center(delpos, nvec);
  for (int i = 0; i < nvec; ++i) norm += delpos[i] *delpos[i];
  double normall;
  MPI_Allreduce(&norm,&normall,1,MPI_DOUBLE,MPI_SUM,world);

  double norm_i = 1./sqrt(normall);
  for (int i=0; i < nvec; i++){
    h[i] = delpos[i] * norm_i;
    xvec[i] += initial_step_size * h[i];
  }
  delete []delpos;
  return;
}
/* -----------------------------------------------------------------------------
 * converge to minimum, here I use conjugate gradient method.
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

  sum /= double(n);
  for (int i=0; i<n; i++) x[i] -= sum;
}

/* -----------------------------------------------------------------------------
 * The lanczos method to get the lowest eigenvalue and corresponding eigenvector
 * Ref: R.A. Olsen , G. J. Kroes, Comparison of methods for 
 *   fanding saddle points without knowledge of final states,
 *   121, 20(2004)
 * ---------------------------------------------------------------------------*/
void ARTn::lanczos(bool new_projection, int flag, int maxvec){
  FixMinimize * fix_lanczos;
  char **fixarg = new char*[3];
  fixarg[0] = (char *) "lanczos";
  fixarg[1] = (char *) "all";
  fixarg[2] = (char *) "MINIMIZE";
  modify->add_fix(3,fixarg);
  delete [] fixarg;
  fix_lanczos = (FixMinimize *) modify->fix[modify->nfix-1];

  fix_lanczos->add_vector(3);		// 0, for r_k_1
  fix_lanczos->add_vector(3);		// 1, for q_k_1
  fix_lanczos->add_vector(3);		// 2, for q_k
  fix_lanczos->add_vector(3);		// 3, for u_k
  fix_lanczos->add_vector(3);		// 4, for r_k
  double *r_k_1 = fix_lanczos->request_vector(0);
  double *q_k_1 = fix_lanczos->request_vector(1);
  for (int i = 0; i< nvec; ++i) q_k_1[i] = 0.;
  double *q_k = fix_lanczos->request_vector(2);
  double *u_k = fix_lanczos->request_vector(3);
  double *r_k = fix_lanczos->request_vector(4);
  double *d = new double [maxvec];
  double *e = new double [maxvec];
  double *d_bak = new double [maxvec];
  double *e_bak = new double [maxvec];
  double tmp;
  double beta_k_1 = 0., alpha_k, beta_k;
  int con_flag = 0;
  double ** lanc = new double *[maxvec];
  if (flag > 0) {
    for (int i = 0; i < maxvec; ++i){
      fix_lanczos->add_vector(3);
      lanc[i] = fix_lanczos->request_vector(i+5);
    }
  }
  //if (flag>0) memory->create(lanc, maxvec, nvec, "lanc matrx");
  // DEL_LANCZOS is the step we use in the finite differece approximation.
  const double DEL_LANCZOS = delta_displ_lanczos;
  const double IDEL_LANCZOS = 1.0 / DEL_LANCZOS;

  if (!new_projection){
    for (int i = 0; i<nvec; i++) r_k_1[i] = eigenvector[i];
  } else {
    for (int i = 0; i<nvec; i++) r_k_1[i] = 0.5 - random->uniform();
    //center(r0, nvec);
  }
  for (int i =0; i< nvec; ++i){
    beta_k_1 += r_k_1[i] * r_k_1[i];
  }
  MPI_Allreduce(&beta_k_1,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  beta_k_1 = sqrt(tmp);

  double eigen1 = 0., eigen2 = 0.;
  double *work, *z;
  long int ldz = maxvec, info;
  char jobs = 'V';
  z = new double [ldz*maxvec];
  work = new double [2*maxvec-2];

  for (int i=0; i<nvec; i++){
    x0tmp[i] = xvec[i];
    g[i] = fvec[i];
  }

  for ( long n = 1; n < maxvec; ++n){
    for (int i = 0; i < nvec; ++i){
      q_k[i] = r_k_1[i] / beta_k_1;
      lanc[n-1][i] = q_k[i];
    }
    for (int i = 0; i < nvec; ++i){
      xvec[i] = x0tmp[i] + q_k[i] * DEL_LANCZOS;
    }
    energy_force(1);
    reset_coords();
    alpha_k = 0.;
    for (int i = 0; i < nvec; ++i){
      u_k[i] = (g[i] - fvec[i]) * IDEL_LANCZOS;
      r_k[i] = u_k[i] - beta_k_1 * q_k_1[i];
      alpha_k += q_k[i] * r_k[i];
    }
    MPI_Allreduce(&alpha_k,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
    alpha_k = tmp;
    beta_k = 0.;
    for (int i = 0; i < nvec; ++i){
      r_k[i] -= alpha_k * q_k[i];
      beta_k += r_k[i] * r_k[i];
    }
    MPI_Allreduce(&beta_k,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
    beta_k = sqrt(tmp);
    d[n-1] = alpha_k;
    e[n-1] = beta_k;
    for (int i = 0; i != maxvec; i++){
      d_bak[i] = d[i];
      e_bak[i] = e[i];
    }
    if (n >= 2){
      dstev_(&jobs,&n,d_bak,e_bak,z,&ldz,work,&info);
      if (info != 0) error->all(FLERR,"destev_ error in lanczos subroute");
      eigen1 = eigen2;
      eigen2 = d_bak[0];
      //cout << "Current eigen = " << eigen2 <<endl;
    }
    if (n >= 3 && fabs((eigen2-eigen1)/eigen1) < eigen_threhold){
      con_flag = 1;
      for (int i = 0; i < nvec; ++i){
	xvec[i] = x0tmp[i];
	fvec[i] = g[i];
      }
      eigenvalue = eigen2;
      //cout << "eigenvalue = " << eigenvalue << endl;
      if (flag > 0){
	eigen_vector_exist = 1;
	for (int i=0; i < nvec; i++) eigenvector[i] = 0.;
	for (int i=0; i<nvec; i++){
	  for (int j=0; j<n; j++) eigenvector[i] += z[j]*lanc[j][i];
	}

	// normalize eigenvector.
	double sum = 0., sumall;
	for (int i = 0; i < nvec; i++) sum += eigenvector[i] * eigenvector[i];

	MPI_Allreduce(&sum, &sumall,1,MPI_DOUBLE,MPI_SUM,world);
	sumall = 1. / sqrt(sumall);
	for (int i=0; i < nvec; ++i) eigenvector[i] *= sumall;
      }
      break;
    }
    for (int i = 0; i < nvec; ++i){
      r_k_1[i] = r_k[i];
      q_k_1[i] = q_k[i];
    }
    beta_k_1 = beta_k;
  }
  if (con_flag == 0){
    outlog("WARNING, LNACZOS method not converged!\n");
    eigenvalue = eigen2;
    for (int i = 0; i < nvec; ++i){
      xvec[i] = x0tmp[i];
      fvec[i] = g[i];
    }
  }
  //delete []r_k_1;
  //delete []q_k_1;
  //delete []q_k;
  //delete []u_k;
  //delete []r_k;
  delete []d;
  delete []e;
  delete []d_bak;
  delete []e_bak;
  delete []z;
  delete []work;
  modify->delete_fix("lanczos");
}

/* -----------------------------------------------------------------------------
 * I use this function try to avoid atom communication 
 * in lanczos subroutine, for little change of atom cordinates.
 * ---------------------------------------------------------------------------*/
#ifdef TEST
double ARTn::myenergy_force(){
  return energy_force(1);
}
#else
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
#endif
