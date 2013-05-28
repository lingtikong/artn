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

#define TEST
#define TESTOUPUT
#define MAXLINE 256
#define MAX(A,B) ((A) > (B) ? (A):(B))
#define MIN(A,B) ((A) < (B) ? (A):(B))

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
  // two dumps are added.
  memory->create(tmp, 5, 15, "Artn string");
  strcpy(tmp[0],"ARTnmin");
  strcpy(tmp[1],"all");
  strcpy(tmp[2],"atom");
  strcpy(tmp[3],"1");
  strcpy(tmp[4],"min.lammpstrj");
  dumpmin = new DumpAtom(lmp, 5, tmp);
  memory->destroy(tmp);
  memory->create(tmp, 5, 15, "Artn string");
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
 * Main search function
 * ---------------------------------------------------------------------------*/
int ARTn::search(int maxevent)
{
#ifdef TEST
  cout << "Entering search(), proc: " << me << endl;
#endif
  mysetup();
  myinit();
  // minimize before searching saddle points.
  outlog("Start to minimize the configuration before trying to find the saddle point.\n");
  stop_condition = min_converge(max_converge_steps);
  eref = ecurrent = energy_force(0);
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
  for (int ievent = 0; ievent < max_num_events; ievent++){
    while (!find_saddle());
    if (me == 0 ) out_event_list<<ecurrent-eref<<'\t';
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
 * push the configuration downhill
 * ---------------------------------------------------------------------------*/
void ARTn::downhill()
{
  // push over the saddle point along the eigenvector direction
  double hdotxx0 = 0.;
  for (int i = 0; i < nvec; ++i) hdotxx0 += h[i] * (xvec[i] - x0[i]);
  if (hdotxx0 > 0) {
    for (int i = 0; i < nvec; ++i) xvec[i] = xvec[i] + h[i] * prefactor_push_over_saddle;
  } else {
    for (int i = 0; i < nvec; ++i) xvec[i] = xvec[i] - h[i] * prefactor_push_over_saddle;
  }
  //for (int i = 0; i < nvec; ++i) xvec[i] = x0[i] + prefactor_push_over_saddle * (xvec[i] - x0[i]);
  ecurrent = energy_force(1);
  myreset_vectors();
  if (me == 0) out_log << "After push over the saddle point, ecurrent = " << ecurrent << endl;
  outlog("Start to minimize the configuration to reach another minimal.\n");

  // do minimization with CG
  stop_condition = min_converge(max_converge_steps);
  stopstr = stopstrings(stop_condition);

  ecurrent = energy_force(1);
  myreset_vectors();

  // output minimization information
  if (me == 0){
    out_log << "- Minimize stop condition: "<< stopstr<< endl;
    out_log << "- Current energy : " << ecurrent << endl;
    out_log << "- Temperature: "<< temperature <<endl;
  }

  // store min configuration
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
  // output reference energy and current energy.
  if (me == 0) out_event_list << eref << '\t' << ecurrent << '\t' ;

  // jugement 
  if (temperature > 0. && (ecurrent < eref || random->uniform() < exp((eref - ecurrent)/temperature)) ){
    
    outlog("Accept this new configuration.\n");
    if (me == 0) out_event_list << ecurrent << '\t';
    outeven("accept\n");
    eref = ecurrent;
  } else {

    for(int i = 0; i < nvec; ++i){
      xvec[i] = x00[i];
    }
    ecurrent = energy_force(1);
    myreset_vectors();
    if (me == 0) out_event_list << ecurrent << '\t';
    outlog("Reject this new configuration.\n");
    outeven("reject\n");
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
  group_random = false;
  local_random = false;
  fire_on = false;

  // for harmonic well
  initial_step_size = 0.05;
  basin_factor = 2.1;
  max_perp_moves_basin = 8;		// 3
  min_number_ksteps = 0;		
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

  read_config();
}
/* -----------------------------------------------------------------------------
 * read parameters from file "config"
 * ---------------------------------------------------------------------------*/
void ARTn::read_config()
{
  char oneline[MAXLINE], *token1, *token2;
  FILE *fp = fopen("config", "r");
  if (fp == NULL){
    error->all(FLERR, "open config file error!");
  }
  while(1){
    fgets(oneline, MAXLINE, fp);
    if(feof(fp)) break;

    token1 = strtok(oneline," \t\n\r\f");
    if (token1 == NULL || strcmp(token1, "#") == 0) continue;
    token2 = strtok(NULL," \t\n\r\f");
    if (strcmp(token1, "max_converge_steps") == 0){
      max_converge_steps = atoi(token2);
    }else if (strcmp(token1, "temperature") == 0){
      temperature = atof(token2);
    }else if (strcmp(token1, "max_num_events") == 0){
      max_num_events = atoi(token2);
    }else if (strcmp(token1, "activation_maxiter") == 0){
      activation_maxiter = atoi(token2);
    }else if (strcmp(token1, "increment_size") == 0){
      increment_size = atof(token2);
    }else if (strcmp(token1, "force_threhold_perp_rel") == 0){
      force_threhold_perp_rel = atof(token2);
    }else if (strcmp(token1, "group_random") == 0){
      group_random = true;
    }else if (strcmp(token1, "initial_step_size") == 0){
      initial_step_size = atof(token2);
    }else if (strcmp(token1, "basin_factor") == 0){
      basin_factor = atof(token2);
    }else if (strcmp(token1, "max_perp_moves_basin") == 0){
      max_perp_moves_basin = atoi(token2);
    }else if (strcmp(token1, "min_number_ksteps") == 0){
      min_number_ksteps = atoi(token2);
    }else if (strcmp(token1, "eigenvalue_threhold") == 0){
      eigenvalue_threhold = atof(token2);
    }else if (strcmp(token1, "max_iter_basin") == 0){
      max_iter_basin = atoi(token2);
    }else if (strcmp(token1, "force_threhold_perp_h") == 0){
      force_threhold_perp_h = atof(token2);
    }else if (strcmp(token1, "number_lanczos_vectors_H") == 0){
      number_lanczos_vectors_H = atoi(token2);
    }else if (strcmp(token1, "number_lanczos_vectors_C") == 0){
      number_lanczos_vectors_C = atoi(token2);
    }else if (strcmp(token1, "delta_displ_lanczos") == 0){
      delta_displ_lanczos = atof(token2);
    }else if (strcmp(token1, "eigen_threhold") == 0){
      eigen_threhold = atof(token2);
    }else if (strcmp(token1, "exit_force_threhold") == 0){
      exit_force_threhold = atof(token2);
    }else if (strcmp(token1, "prefactor_push_over_saddle") == 0){
      prefactor_push_over_saddle = atof(token2);
    }else if (strcmp(token1, "eigen_fail") == 0){
      eigen_fail = atof(token2);
    }else if (strcmp(token1, "event_list_file") == 0){
      event_list_file = token2;
    }else if (strcmp(token1, "log_file") == 0){
      log_file = token2;
    }else if (strcmp(token1, "file_counter") == 0){
      file_counter = atoi(token2);
    }else if (strcmp(token1, "max_perp_moves_C") == 0){
      max_perp_moves_C = atof(token2);
    }else if (strcmp(token1, "force_threhold_perp_rel_C") == 0){
      force_threhold_perp_rel_C = atof(token2);
    }else if (strcmp(token1, "local_random") == 0){
      local_random = true;
    }else if (strcmp(token1, "fire") == 0){
      fire_on = true;
    }else error->all(FLERR, "Config file error! Command not found");    
  }
  fclose(fp);
  return;
}

/* -----------------------------------------------------------------------------
 * initializing for ARTn
 * ---------------------------------------------------------------------------*/
void ARTn::myinit()
{
  MPI_Comm_rank(world,&me);
  if (me == 0) out_event_list.open(event_list_file.c_str());
  if (!out_event_list) error->all(FLERR, "Open event list file error!");
  if (me == 0) out_log.open(log_file.c_str());
  if (!out_log) error->all(FLERR, "open event log file error!");
  random = new RanPark(lmp, 12340);
  evalf = 0;
  eigen_vector_exist = 0;
  if(nvec) vvec = atom->v[0];


  // for peratom vector I use
  vec_count = 3;
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  x0tmp = fix_minimize->request_vector(vec_count);
  ++vec_count;
  h_old = fix_minimize->request_vector(vec_count);
  ++vec_count;
  eigenvector = fix_minimize->request_vector(vec_count);
  ++vec_count;
  x00 = fix_minimize->request_vector(vec_count);
  ++vec_count;
  fperp = fix_minimize->request_vector(vec_count);
  ++vec_count;
}

/* ----------------------------------------------------------------------------
 * reset vectors
 * --------------------------------------------------------------------------*/
void ARTn::myreset_vectors()
{
  x0tmp = fix_minimize->request_vector(3);
  h_old = fix_minimize->request_vector(4);
  eigenvector = fix_minimize->request_vector(5);
  x00 = fix_minimize->request_vector(6);
  fperp = fix_minimize->request_vector(7);
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
  double ftot = 0., ftotall, fpar2=0.,fpar2all=0., fperp2 = 0., fperp2all,  delr;
  double fdoth = 0., fdothall = 0.;
  double step, preenergy;
  double tmp;
  int m_perp = 0, trial = 0, nfail = 0;
  eigenvalue = 0.;
  eigen_vector_exist = 0;

  for (int i = 0; i < nvec; i++){
    x0[i] = x00[i] = xvec[i];
  }

  // random move 
  if (group_random) group_random_move();
  else {if (local_random) local_random_move();
    else global_random_move();
  }
  ecurrent = energy_force(1);
  myreset_vectors();
  ++evalf;

#ifdef TESTOUPUT
  if ( me ==0 ) {
    out_log << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(5) << endl;
    out_log << setw(12) << "E-Eref"<< setw(12) << "m_perp" <<setw(12) << "trial" << setw(12) <<"ftot";
    out_log << setw(12) << "fpar";
    out_log << setw(12) << "fperp" << setw(12) << "eigenvalue" << setw(12) << "delr" << setw(12) << "evalf\n";
  }
#else
  outlog("\tE-Eref\t\tm_perp\ttrial\tftot\t\tfperp\t\teigenvalue\tdelr\tevalf\n");
#endif
  // try to leave harmonic well
  for (int local_iter = 0; local_iter < max_iter_basin; local_iter++){
    // do minimizing perpendicular
    // Here, I use Steepest Descent method 
    ecurrent = energy_force(1);
    myreset_vectors();
    ++evalf;
    m_perp = nfail = trial = 0;
    step = increment_size * 0.4;
    while (1){
      preenergy = ecurrent;
      fdoth = 0.;
      for (int i = 0; i < nvec; i++) fdoth += fvec[i] * h[i];
      MPI_Allreduce(&fdoth, &fdothall,1,MPI_DOUBLE,MPI_SUM,world);
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
      myreset_vectors();
      ++evalf;
      reset_coords();

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
	myreset_vectors();
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
    fpar2 = 0.;
    for (int i = 0; i < nvec; ++i){
      fpar2 += fvec[i] * h[i];
    }
    MPI_Allreduce(&fpar2, &fpar2all,1,MPI_DOUBLE,MPI_SUM,world);
    // output information
#ifdef TESTOUTPUT
    if (me ==0){
      out_log << setw(12) << local_iter << setw(12) << ecurrent - eref << setw(12) <<  m_perp;
      out_log << setw(12) << trial<< setw(12) << sqrt(ftotall) << setw(12) << fpar2all << setw(12) << fperp2;
      out_log << setw(12) << eigenvalue << setw(12) << sqrt(delr)<< setw(12) << evalf << endl;
    }
#else
    if(me == 0){
      out_log << setiosflags(ios::fixed)<<setprecision(4);
      out_log << local_iter << '\t' << ecurrent - eref << '\t'<<'\t';
      out_log << m_perp <<'\t'<< trial<<'\t' << sqrt(ftotall)<<'\t' <<'\t'<< fpar2all<< '\t';
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
    for (int i = 0; i < nvec; ++i) xvec[i] = x00[i];
    ecurrent = energy_force(1);
    myreset_vectors();
    return 0;
  }
  flag = 0;

  outlog("\tE-Eref\t\tm_perp\ttrial\tftot\t\tfpar\t\tfperp\t\teigenvalue\tdelr\tevalf\ta1\n");
  // now try to move close to the saddle point according to the eigenvector.
  int inc = 0;
  for (int saddle_iter = 0; saddle_iter < activation_maxiter; ++saddle_iter){
    ecurrent = energy_force(1);
    myreset_vectors();
    for (int i = 0; i < nvec; ++i) h_old[i] = h[i];

    // caculate eigenvector use lanczos
    lanczos(!eigen_vector_exist, 1, number_lanczos_vectors_C);

    // set search direction according to eigenvector
    double hdot , hdotall, tmpsum, tmpsumall;
    hdot = hdotall = tmpsum = tmpsumall = 0.;
    for (int i =0; i < nvec; ++i) tmpsum += eigenvector[i] * fvec[i];

    MPI_Allreduce(&tmpsum,&tmpsumall,1,MPI_DOUBLE,MPI_SUM,world);
    if (tmpsumall > 0){
      for(int i = 0; i < nvec; ++i) h[i] = -eigenvector[i];
    }else{
      for(int i = 0; i < nvec; ++i) h[i] = eigenvector[i];
    }
    for(int i = 0; i <nvec; ++i) hdot += h[i] * h_old[i];
    MPI_Allreduce(&hdot,&hdotall,1,MPI_DOUBLE,MPI_SUM,world);
    // do minimizing perpendicular use SD or FIRE
    if (fire_on) {
      min_perpendicular_fire(MIN(40, saddle_iter + 10));
      m_perp = trial = 0;
    }else{
      preenergy = ecurrent;
      m_perp = trial = nfail = 0;
      step = increment_size * 0.4;
      int max_perp = max_perp_moves_C + saddle_iter + inc;
      while (1){
	fdoth = 0.;
	for (int i = 0; i < nvec; i++) fdoth += fvec[i] * h[i];
	MPI_Allreduce(&fdoth, &fdothall,1,MPI_DOUBLE,MPI_SUM,world);
	fperp2 = 0.;
	for (int i = 0; i < nvec; i++){
	  fperp[i] = fvec[i] - fdothall * h[i];
	  fperp2 += fperp[i] * fperp[i];
	}
	MPI_Allreduce(&fperp2, &fperp2all,1,MPI_DOUBLE,MPI_SUM,world);
	fperp2 = sqrt(fperp2all);
	// condition to break
	if (fperp2 < force_threhold_perp_rel_C || m_perp > max_perp || nfail > 5) {
	  break; 
	}

	for (int i = 0; i < nvec; ++i){
	  x0tmp[i] = xvec[i];
	  xvec[i] += step * fperp[i];
	}
	ecurrent = energy_force(1);
	myreset_vectors();
	evalf++;
	reset_coords();

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
	  myreset_vectors();
	  reset_coords();
	  evalf++;
	}
	trial++;
      }
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
    fpar2 = 0.;
    for (int i = 0; i < nvec; ++i){
      fpar2 += fvec[i] * h[i];
    }
    MPI_Allreduce(&fpar2, &fpar2all,1,MPI_DOUBLE,MPI_SUM,world);
    if (fpar2all > -1.) inc = 10;
    fperp2 = 0.;
    for (int i = 0; i < nvec; ++i){
      tmp = fvec[i] - fpar2all * h[i];
      fperp2 += tmp * tmp;
    }
    MPI_Allreduce(&fperp2, &tmp,1,MPI_DOUBLE,MPI_SUM,world);
    fperp2 = tmp;
    if (me == 0){
      out_log << saddle_iter << '\t' << ecurrent - eref << '\t' << '\t';
      out_log << m_perp <<'\t'<< trial<<'\t' << sqrt(ftotall)<<'\t' <<'\t'<< fpar2all<<'\t' << '\t';
      out_log << sqrt(fperp2) <<'\t'<< '\t'<< eigenvalue <<'\t'<< sqrt(delr)<<'\t'<< evalf <<'\t'<< hdotall<< endl;
    }


    if (eigenvalue > eigen_fail){
      if (me == 0){
	out_log << "Failed to find the saddle point in lanczos steps, for eigenvalue > eigen_fail: "<< eigenvalue <<'>'
	  << eigen_fail << '\n';
	out_log << "Reassign the random direction.";
      }
      for(int i = 0; i < nvec; ++i) xvec[i] = x00[i];
#ifdef TEST
      ecurrent =  energy_force(0);
      cout << "After failed, eref = "<< eref << ", ecurrent = " << ecurrent << endl;
#endif
      return 0;
    }

    if (sqrt(ftotall)<exit_force_threhold ){
      outlog("Reach saddle point.\n");
      return 1;
    }

    // push along the search direction
    // E. Cances, et al. JCP, 130, 114711 (2009)
    double factor;
    factor = MIN(2. * increment_size, fabs(fpar2all)/MAX(fabs(eigenvalue),0.5));
    //factor = 0.3 * increment_size;
    for (int i = 0; i < nvec; ++i) xvec[i] += factor * h[i];
  }
  outlog("Failed to find the saddle point in lanczos steps, for reaching the max lanczos steps.\n");
  outlog("Reassign the random direction.\n");
  for (int i = 0; i < nvec; ++i) xvec[i] = x00[i];
  energy_force(1);
  myreset_vectors();
  return 0;
}

/* -----------------------------------------------------------------------------
 * reset coordinates x0tmp
 * ---------------------------------------------------------------------------*/
void ARTn::reset_coords()
{
  domain->set_global_box();

  double **x = atom->x;
  double *x0 = fix_minimize->request_vector(3);
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
  for (int i = 0; i < nvec; ++i) delpos[i] = 0.;
  double norm = 0.;
  int igroup = group->find("artn");
  if (igroup == -1) error->all(FLERR,"Could not find artn group!");
  int bit = group->bitmask[igroup];
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; ++i){
    if (bit & atom->mask[i]){
      delpos[i*3] = 0.5 - random->uniform();
      delpos[i*3+1] = 0.5 -random->uniform();
      delpos[i*3+2] = 0.5 - random->uniform();
    }
  }
  //center(delpos, nvec);
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
 * move one atom randomly
 * ---------------------------------------------------------------------------*/
void ARTn::local_random_move()
{
#ifdef TEST
  //cout << "before local_random_move(), energy = " << energy_force(0) << endl;
#endif
  double *delpos = new double[nvec];
  for (int i = 0; i < nvec; ++i) delpos[i] = 0.;
  double norm = 0.;
  int natoms = atom->natoms;
  int that_atom = floor(natoms * random->uniform());
  if (that_atom == natoms) that_atom -= 1;
  else that_atom += 1;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;
  int flag = 0, flagall = 0;
  for (int i = 0; i < nlocal; ++i){
    if (that_atom == tag[i]){
      flag = 1;
      delpos[i*3] = 0.5 - random->uniform();
      delpos[i*3+1] = 0.5 -random->uniform();
      delpos[i*3+2] = 0.5 - random->uniform();
    }
  }
  for (int i = 0; i < nvec; ++i) norm += delpos[i] * delpos[i];
  double normall;
  MPI_Allreduce(&norm,&normall,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&flag,&flagall,1,MPI_DOUBLE,MPI_SUM,world);
  if (flagall == 0) {
    out_log << "That atom: " << that_atom << endl;
    error->all(FLERR,"local random move error, atom not found.");
  }
  double norm_i = 1./sqrt(normall);
  for (int i=0; i < nvec; i++){
    h[i] = delpos[i] * norm_i;
    xvec[i] += initial_step_size * h[i];
  }
  delete []delpos;
#ifdef TEST
  //cout << "After local_random_move(), energy = " << energy_force(0) << endl;
#endif
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

  eprevious = ecurrent = energy_force(1);
  for (i = 0; i < nvec; i++) h[i] = g[i] = fvec[i];

  gg = fnorm_sqr();

  for (int iter = 0; iter < maxiter; iter++) {
    ntimestep = ++update->ntimestep;
    niter++;

    // line minimization along direction h from current atom->x

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
 * center the vector: Warining: no MPI_SUM is used.
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
  for (int i = 0; i < maxvec; ++i){
    fix_lanczos->add_vector(3);
    lanc[i] = fix_lanczos->request_vector(i+5);
  }
  // DEL_LANCZOS is the step we use in the finite differece approximation.
  const double DEL_LANCZOS = delta_displ_lanczos;
  const double IDEL_LANCZOS = 1.0 / DEL_LANCZOS;

  // set r(k-1) according to eigenvector or random vector
  if (!new_projection){
    for (int i = 0; i<nvec; i++) r_k_1[i] = eigenvector[i];
  } else {
    for (int i = 0; i<nvec; i++) r_k_1[i] = 0.5 - random->uniform();
  }
  for (int i =0; i< nvec; ++i){
    beta_k_1 += r_k_1[i] * r_k_1[i];
  }
  MPI_Allreduce(&beta_k_1,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  // beta(k-1) = |r(k-1)|
  beta_k_1 = sqrt(tmp);

  double eigen1 = 0., eigen2 = 0.;
  double *work, *z;
  long int ldz = maxvec, info;
  char jobs = 'V';
  z = new double [ldz*maxvec];
  work = new double [2*maxvec-2];

  // store origin configuration and force
  for (int i=0; i<nvec; i++){
    x0tmp[i] = xvec[i];
    g[i] = fvec[i];
  }

  // q(k) = r(k-1)/ beta(k-1)
  // lanc (n-1, :) = q(k)
  for ( long n = 1; n <= maxvec; ++n){
    for (int i = 0; i < nvec; ++i){
      q_k[i] = r_k_1[i] / beta_k_1;
      lanc[n-1][i] = q_k[i];
    }

    //for (int i = 0; i < n-1; i++){
    //  double tmp;
    //  tmp = 0.;
    //  for (int j = 0; j < nvec; j++){
    //    tmp += lanc[i][j] * q_k[j];
    //  }
    //  for (int j = 0; j < nvec; j++){
    //    q_k[j] -= tmp * lanc[i][j];
    //  }
    //}
    //for (int i = 0; i < nvec; ++i){
    //  lanc[n-1][i] = q_k[i];
    //}

    // random move to caculate u(k) with the finite difference approximation
    for (int i = 0; i < nvec; ++i){
      xvec[i] = x0tmp[i] + q_k[i] * DEL_LANCZOS;
    }
    energy_force(1);
    reset_coords();
    r_k_1 = fix_lanczos->request_vector(0);
    q_k_1 = fix_lanczos->request_vector(1);
    q_k = fix_lanczos->request_vector(2);
    u_k = fix_lanczos->request_vector(3);
    r_k = fix_lanczos->request_vector(4);
    myreset_vectors();
    for (int i = 0; i < maxvec; ++i){
      lanc[i] = fix_lanczos->request_vector(i+5);
    }
    alpha_k = 0.;
    // get u(k) and r(k)
    for (int i = 0; i < nvec; ++i){
      u_k[i] = (g[i] - fvec[i]) * IDEL_LANCZOS;
      r_k[i] = u_k[i] - beta_k_1 * q_k_1[i];
      alpha_k += q_k[i] * r_k[i];
    }
    MPI_Allreduce(&alpha_k,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
    alpha_k = tmp;
    beta_k = 0.;
    // update r(k) = r(k) - alpha(k) q(k)
    for (int i = 0; i < nvec; ++i){
      r_k[i] -= alpha_k * q_k[i];
      beta_k += r_k[i] * r_k[i];
    }
    MPI_Allreduce(&beta_k,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
    // beta(k) = |r(k)|

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
    if (n >= 3 && fabs((eigen2-eigen1)/eigen1) < eigen_threhold) {
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
	  for (int j=0; j<n; j++) eigenvector[i] += z[j] * lanc[j][i];
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
    //outlog("WARNING, LNACZOS method not converged!\n");
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
  delete []lanc;
}

/* ---------------------------------------------------------------------------
 *  FIRE: fast interial relaxation engine, here d_min is not considered.
 * -------------------------------------------------------------------------*/
int ARTn::min_perpendicular_fire(int maxiter){
  double dt = update->dt;
  const int n_min = 5;
  const double f_inc = 1.1;
  const double f_dec = 0.5;
  const double alpha_start = 0.1;
  const double f_alpha = 0.99;
  const double  TMAX = 10.;
  const double dtmax = TMAX * dt;
  double vdotf, vdotfall;
  double vdotv, vdotvall;
  double fdotf, fdotfall;
  double fdoth, fdothall;
  double scale1, scale2;
  double alpha;
  int last_negative = 0.;
  for (int i = 0; i < nvec; ++i) {
    vvec[i] = 0.;
  }
  alpha = alpha_start;
  for (int iter = 0; iter < maxiter; ++iter){
    fdoth = 0.;
    for (int i = 0; i < nvec; ++i) {
      fdoth += fvec[i] * h[i];
    }
    MPI_Allreduce(&fdoth,&fdothall,1,MPI_DOUBLE,MPI_SUM,world);
    for (int i = 0; i < nvec; ++i) {
      fperp[i] = fvec[i] - fdothall * h[i];
    }
    //double fperpmax = 0., fperpmaxall;
    //for (int i = 0; i < nvec; ++i) {
    //  fperpmax = MAX(fperpmax, abs(fperp[i]));
    //}
    //MPI_Allreduce(&fperpmax,&fperpmaxall,1,MPI_DOUBLE,MPI_MAX,world);
    //if (fperpmaxall < force_threhold_perp_rel_C) {
    //  return 1;
    //}
    vdotf = 0.;
    for (int i = 0; i < nvec; ++i) {
      vdotf += vvec[i] * fperp[i];
    }
    MPI_Allreduce(&vdotf,&vdotfall,1,MPI_DOUBLE,MPI_SUM,world);
    // if (v dot f) > 0:
    // v = (1-alpha) v + alpha |v| Fhat
    // |v| = length of v, Fhat = unit f
    // if more than DELAYSTEP since v dot f was negative:
    // increase timestep and decrease alpha
    if (vdotfall > 0.) {
      scale1 = 1. - alpha;
      vdotv = 0.;
      fdotf = 0.;
      for (int i = 0; i < nvec; ++i) {
	vdotv += vvec[i] * vvec[i];
	fdotf += fperp[i] * fperp[i];
      }
      MPI_Allreduce(&vdotv,&vdotvall,1,MPI_DOUBLE,MPI_SUM,world);
      MPI_Allreduce(&fdotf,&fdotfall,1,MPI_DOUBLE,MPI_SUM,world);

      if (sqrt(fdotf) < force_threhold_perp_rel_C) return 1;
      if (fdotfall == 0.) scale2 = 0.;
      else scale2 = alpha * sqrt(vdotvall/fdotfall);
      for (int i = 0; i < nvec; ++i){
	vvec[i] = scale1 * vvec[i] + scale2 * fperp[i];
      }
      if ((iter - last_negative) > n_min) {
	dt = MIN(dt * f_inc, dtmax);
	alpha = alpha * f_alpha;
      }
    }else{
      last_negative = iter;
      dt = dt * f_dec;
      for (int i = 0; i < nvec; ++i) vvec[i] = 0.;
      alpha = alpha_start;
    }
    double **x = atom->x;
    double **v = atom->v;
    double *rmass = atom->rmass;
    double *mass = atom->mass;
    int *type = atom->type;
    double dtfm;
    double dtf = dt * force->ftm2v;
    int nlocal = atom->nlocal;
    if (rmass) {
      for (int i = 0; i < nlocal; i++) {
	dtfm = dtf / rmass[i];
	x[i][0] += dt * v[i][0];
	x[i][1] += dt * v[i][1];
	x[i][2] += dt * v[i][2];
	v[i][0] += dtfm * fperp[3*i];
	v[i][1] += dtfm * fperp[3*i+1];
	v[i][2] += dtfm * fperp[3*i+2];
      }
    } else {
      for (int i = 0; i < nlocal; i++) {
	dtfm = dtf / mass[type[i]];
	x[i][0] += dt * v[i][0];
	x[i][1] += dt * v[i][1];
	x[i][2] += dt * v[i][2];
	v[i][0] += dtfm * fperp[3*i];
	v[i][1] += dtfm * fperp[3*i+1];
	v[i][2] += dtfm * fperp[3*i+2];
      }
    }
    eprevious = ecurrent;
    ecurrent = energy_force(1);
    myreset_vectors();
    ++evalf;
    
  }
  return 0;
}
