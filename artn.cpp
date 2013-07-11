/*---------------------------------------------------
  Some features of this code are writtern according to
  Norman's Code version 3.0 ARTn. The explanlation of 
  the parameters I used here can be found in the doc of
  his code.
  This code don't do minimizing include extra peratom dof or 
  extra global dof.
----------------------------------------------------*/
#include "minimize.h"
#include "string.h"
#include "min_cg.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "finish.h"
#include "timer.h"
#include "error.h"
#include "artn.h"
#include "modify.h"
#include "fix_minimize.h"
#include "memory.h"
#include "stdlib.h"
#include "compute.h"
#include "force.h"
#include "group.h"
#include "math.h"

#define MAXLINE 512

using namespace LAMMPS_NS;

/*---------------------------------------------------
  Here I use clapack to help evaluate the lowest 
  eigenvalue of the matrix in lanczos.
---------------------------------------------------*/
extern "C" {
#include "f2c.h"
#include "clapack.h"
}

#define EPS_ENERGY 1.0e-8

enum{MAXITER,MAXEVAL,ETOL,FTOL,DOWNHILL,ZEROALPHA,ZEROFORCE,ZEROQUAD};

/* -----------------------------------------------------------------------------
 * Constructor of ARTn
 * ---------------------------------------------------------------------------*/
ARTn::ARTn(LAMMPS *lmp): MinLineSearch(lmp)
{
  random = NULL;
  pressure = NULL;
  dumpmin = dumpsad = NULL;
  eigenvector = x0tmp = x00 = htmp = x_saddle = h_old = vvec = fperp = NULL;

  fp1 = fp2 = NULL;
  groupname = fctrl = flog = fevent = fconfg = NULL;

  char *id_press = new char[13];
  strcpy(id_press,"thermo_press");
  int pressure_compute = modify->find_compute(id_press); delete [] id_press;

  if (pressure_compute < 0) error->all(FLERR,"Could not find compute pressure ID");

  pressure = modify->compute[pressure_compute];

  // set default control parameters
  set_defaults();

  MPI_Comm_rank(world, &me);
return;
}

/* -----------------------------------------------------------------------------
 * deconstructor of ARTn, close files, free memories
 * ---------------------------------------------------------------------------*/
ARTn::~ARTn()
{
  if (fp1) fclose(fp1);
  if (fp2) fclose(fp2);
  if (flog)  delete [] flog;
  if (fctrl) delete [] fctrl;
  if (fevent) delete [] fevent;
  if (fconfg) delete [] fconfg;
  if (groupname) delete [] groupname;

  if (random) delete random;
  if (dumpmin) delete dumpmin;
  if (dumpsad) delete dumpsad;

return;
}

/* -----------------------------------------------------------------------------
 * parase the artn command
 * ---------------------------------------------------------------------------*/
void ARTn::command(int narg, char ** arg)
{
  // analyze the command line options
  // grammar: artn etol ftol max_for_eval control_file
  if (narg != 4 ) error->all(FLERR, "Illegal ARTn command!");
  if (domain->box_exist == 0) error->all(FLERR,"ARTn command before simulation box is defined");

  update->etol = atof(arg[0]);
  update->ftol = atof(arg[1]);
  //update->nsteps = atoi(arg[2]);
  update->max_eval = atoi(arg[2]);
  int n = strlen(arg[3]) + 1;
  fctrl = new char [n];
  strcpy(fctrl, arg[3]);

  // check the validity of the command line options
  if (update->etol < 0.0 || update->ftol < 0.0) error->all(FLERR,"Illegal ARTn command");

  // read the control parameters of artn
  read_control();

  update->whichflag = 2;
  update->beginstep = update->firststep = update->ntimestep;            
  update->endstep = update->laststep = update->firststep + update->nsteps; 
  if (update->laststep < 0 || update->laststep > MAXBIGINT) error->all(FLERR,"Too many iterations");

  lmp->init();
  dumpmin->init();
  dumpsad->init();
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
  artn_init();

  // minimize before searching saddle points.
  char str[MAXLINE];
  if (me == 0){
    sprintf(str, "\nMinimizing the initial configuration, id = %d ....\n", ref_id);
    fprintf(fp1, "\n%s", str);
    if (screen) fprintf(screen, str);
  }

  stop_condition = min_converge(max_conv_steps);
  eref = ecurrent = energy_force(0);
  stopstr = stopstrings(stop_condition);

  if (me == 0){
    fprintf(fp1, "  - Minimize stop condition   : %s\n",  stopstr);
    fprintf(fp1, "  - Current (ref) energy (eV) : %lg\n", ecurrent);
    fprintf(fp1, "  - Temperature               : %lg\n", temperature);
  }

  if (pressure_needed){
    pressure->compute_vector();
    double * press = pressure->vector;
    if (me == 0){
      fprintf(fp1, "  - Pressure tensor           :");
      for (int ii=0; ii<6; ii++) fprintf(fp1, " %g", press[ii]);
      fprintf(fp1, "\n");
    }
  }

  // dump the first stable configuration
  int idum = update->ntimestep;
  update->ntimestep = ref_id;
  dumpmin->write(); update->ntimestep = idum;

  // print header of event log file
  if (me == 0){
    if (pressure_needed){
      fprintf(fp2, "#       1           2      3     4     5    6       7   8   9  10  11  12   13     14         15        16\n");
      fprintf(fp2, "# energy_barrier ref_id sad_id min_id eref emin n_moved pxx pyy pzz pxy pxz pyz efinal accept_or_reject dr\n");
    } else {
      fprintf(fp2, "#       1           2      3    4      5     6       7       8        9         10\n");
      fprintf(fp2, "# energy_barrier ref_id sad_id min_id eref emin n_moved efinal accept_or_reject dr\n");
    }
  }

  int ievent = 0;
  while ( 1 ){
  //for (int ievent = 0; ievent < max_num_events; ievent++){
    while (! find_saddle() );
    if (flag_check_sad){
      if(! check_saddle_min() ) continue;
    }

    if (me == 0) fprintf(fp2, "%10.6f", ecurrent - eref);

    int idum = update->ntimestep;
    update->ntimestep = ++sad_id;
    dumpsad->write();
    update->ntimestep = idum;

    push_down();
    check_new_min();

    if (++ievent >= max_num_events) break;
  }

return 1;
}

/*-----------------------------------------------------------------------------
 * return 1, saddle connect with minimum, else return 0
 *---------------------------------------------------------------------------*/
int ARTn::check_saddle_min()
{
  for (int i = 0; i < nvec; ++i) {
    x_saddle[i] = xvec[i];
    htmp[i] = h[i];
    x0tmp[i] = x0[i];
  }
  // push over the saddle point along the eigenvector direction
  double hdotxx0 = 0., hdotxx0all;
  for (int i = 0; i < nvec; ++i) hdotxx0 += h[i] * (xvec[i] - x0[i]);
  MPI_Allreduce(&hdotxx0,&hdotxx0all,1,MPI_DOUBLE,MPI_SUM,world);
  if (hdotxx0all < 0) {
    for (int i = 0; i < nvec; ++i) xvec[i] = xvec[i] + h[i] * push_over_saddle;
  } else {
    for (int i = 0; i < nvec; ++i) xvec[i] = xvec[i] - h[i] * push_over_saddle;
  }
  //for (int i = 0; i < nvec; ++i) xvec[i] = x0[i] + push_over_saddle * (xvec[i] - x0[i]);
  ecurrent = energy_force(1);
  if (me == 0){
    fprintf(fp1, "\nNow to check if the found saddle is truly linked to config %d\n", ref_id);
    if (screen) fprintf(screen, "\nNow to check if the found saddle is truly linked to config %d\n", ref_id);
  }
  // do minimization with CG
  stop_condition = min_converge(max_conv_steps);
  stopstr = stopstrings(stop_condition);
  ecurrent = energy_force(1);
  myreset_vectors();
  // output minimization information
  if (me == 0){
    fprintf(fp1, "  - Minimize stop condition   : %s\n",  stopstr);
    fprintf(fp1, "  - Current (ref) energy (eV) : %lg\n", ecurrent);
    fprintf(fp1, "  - Temperature               : %lg\n", temperature);
  }
  double dr = 0., drall;
  double **x = atom->x;
  double *x0 = fix_minimize->request_vector(6);
  int nlocal = atom->nlocal;
  double dx,dy,dz;

  int n = 0;
  for (int i = 0; i < nlocal; ++i) {
    dx = x[i][0] - x0[n];
    dy = x[i][1] - x0[n+1];
    dz = x[i][2] - x0[n+2];
    domain->minimum_image(dx,dy,dz);
    dr += dx*dx + dy*dy + dz*dz;
    n += 3;
  }
  MPI_Allreduce(&dr,&drall,1,MPI_DOUBLE,MPI_SUM,world);
  drall = sqrt(drall);
  if (drall < max_disp_tol) {
    if (me == 0){
      fprintf(fp1, "  dr = %g < %g, accept the new saddle.\n", drall, max_disp_tol);
      if (screen) fprintf(screen, "  dr = %g < %g, accept the new saddle.\n", drall, max_disp_tol);
    }
    for (int i = 0; i < nvec; ++i){
      xvec[i] = x_saddle[i];
      h[i] = htmp[i];
      x0[i] = x0tmp[i];
    }
    ecurrent = energy_force(1);
    myreset_vectors();
    return 1;

  } else {

    if (me == 0){
      fprintf(fp1, "  dr = %g >= %g, reject this saddle point.\n", drall, max_disp_tol);;
      if (screen) fprintf(screen, "  dr = %g > %g, reject this saddle point.\n", drall, max_disp_tol);
    }

    for (int i = 0; i < nvec; ++i) xvec[i] = x00[i];

    ecurrent = energy_force(1);
    myreset_vectors();
  }

return 0;
}

/* -----------------------------------------------------------------------------
 * push the configuration push_down
 * ---------------------------------------------------------------------------*/
void ARTn::push_down()
{
  // push over the saddle point along the eigenvector direction
  double hdotxx0 = 0., hdotxx0all;
  for (int i = 0; i < nvec; ++i) hdotxx0 += h[i] * (xvec[i] - x0[i]);
  MPI_Allreduce(&hdotxx0,&hdotxx0all,1,MPI_DOUBLE,MPI_SUM,world);

  if (hdotxx0all > 0.) for (int i = 0; i < nvec; ++i) xvec[i] += h[i] * push_over_saddle;
  else for (int i = 0; i < nvec; ++i) xvec[i] -= h[i] * push_over_saddle;


  ecurrent = energy_force(1);
  if (me == 0){
    fprintf(fp1, "\nAfter push over the saddle point of %d, the current engergy becomes: %g\n", sad_id, ecurrent);
    fprintf(fp1, "Now begin to minimize the configuration to reach another minimium.\n");
    if (screen){
      fprintf(screen, "\nAfter push over the saddle point of %d, the current engergy becomes: %g\n", sad_id, ecurrent);
      fprintf(screen, "Now begin to minimize the configuration to reach another minimium.\n");
    }
  }

  // do minimization with CG
  stop_condition = min_converge(max_conv_steps);
  stopstr = stopstrings(stop_condition);

  ecurrent = energy_force(1);
  myreset_vectors();

  // output minimization information
  if (me == 0){
    fprintf(fp1, "  - Minimize stop condition   : %s\n",  stopstr);
    fprintf(fp1, "  - Current (ref) energy (eV) : %lg\n", ecurrent);
    fprintf(fp1, "  - Temperature               : %lg\n", temperature);
  }

  // store min configuration
  int idum = update->ntimestep;
  update->ntimestep = ++min_id;
  dumpmin->write();
  update->ntimestep = idum;

  if (me == 0){
    fprintf(fp1, "\nThe new minimum from min-%d cross sad-%d is stored as min-%d.\n", ref_id, sad_id, min_id);
    if (screen) fprintf(screen, "\nThe new minimum from min-%d cross sad-%d is stored as min-%d.\n", ref_id, sad_id, min_id);
  }

return;
}

/* -----------------------------------------------------------------------------
 * decide whether reject or accept the new configuration
 * ---------------------------------------------------------------------------*/
void ARTn::check_new_min()
{
  // output reference energy ,current energy and pressure.
  if (me == 0) fprintf(fp2, " %5d %4d %5d %12.6f %12.6f", ref_id, sad_id, min_id, eref, ecurrent);

  double dr = 0., drall;
  double **x = atom->x;
  double *x0 = fix_minimize->request_vector(6);
  int nlocal = atom->nlocal;
  double dx,dy,dz;

  int n = 0;
  // calculate dr
  // check minimum image
  double tmp;
  int n_moved = 0, n_movedall;
  for (int i = 0; i < nlocal; ++i) {
    dx = x[i][0] - x0[n];
    dy = x[i][1] - x0[n+1];
    dz = x[i][2] - x0[n+2];
    domain->minimum_image(dx,dy,dz);
    tmp = dx*dx + dy*dy + dz*dz;
    dr += tmp;
    if(sqrt(tmp) > atom_move_cutoff) n_moved += 1;
    n += 3;
  }
  MPI_Allreduce(&dr,&drall,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&n_moved,&n_movedall,1,MPI_DOUBLE,MPI_SUM,world);
  if (me == 0) fprintf(fp2, " %3d", n_movedall);

  drall = sqrt(drall);

  // set v = 0 to calculate pressure
  if (pressure_needed){
    for (int i = 0; i < nvec; ++i) vvec[i] = 0.;

    ++update->ntimestep;
    pressure->addstep(update->ntimestep);
    energy_force(0);
    pressure->compute_vector();
    double * press = pressure->vector;

    if (me == 0) for (int i = 0; i < 6; ++i) fprintf(fp2, " %8.4f", press[i]);
  }

  // Metropolis
  int acc = temperature > 0. && (ecurrent < eref || random->uniform() < exp((eref - ecurrent)/temperature));
  if (acc){
    ref_id = min_id; eref = ecurrent;
    if (me == 0){
      fprintf(fp1, "The new configuration %d is accepted.\n", min_id);
      if (screen) fprintf(screen, "The new configuration %d is accepted.\n", min_id);
    }

  } else {

    for (int i = 0; i < nvec; ++i) xvec[i] = x00[i];
    ecurrent = energy_force(1);
    myreset_vectors();
    if (me == 0){
      fprintf(fp1, "The new configuration %d is rejected.\n", min_id);
      if (screen) fprintf(screen, "The new configuration %d is rejected.\n", min_id);
    }
  }
  if (me == 0){
    fprintf(fp2, " %12.6f %2d %8.5f\n", ecurrent, acc, drall);
    fflush(fp2);
  }

return;
}

/* -----------------------------------------------------------------------------
 * setup default parameters, some values come from Norman's code
 * ---------------------------------------------------------------------------*/
void ARTn::set_defaults()
{
  seed = 12345;
  init_kick_mode    = 0;
  ref_id = min_id = sad_id    = 0;

  max_conv_steps    = 100000;

  // for art
  temperature       = 0.5;
  max_num_events    = 1000;
  max_activat_iter  = 300;
  increment_size    = 0.09;
  use_fire          = 0;
  flag_check_sad    = 0;
  max_disp_tol      = 0.1;
  pressure_needed   = 0;
  atom_move_cutoff  = 0.1;
  region_cutoff     = 5.0;

  // for harmonic well
  init_step_size    = 0.05;
  basin_factor      = 2.1;
  max_perp_move_h   = 10;
  min_num_ksteps    = 0;		
  eigen_th_well     = -0.001;
  max_iter_basin    = 100;
  force_th_perp_h   = 0.5;

  // for lanczos
  //num_lancz_vec_H = 13;
  num_lancz_vec_H   = 40;
  num_lancz_vec_C   = 12;
  del_disp_lancz    = 0.01;
  eigen_th_lancz    = 0.1;
  force_th_perp_sad = 0.05;

  // for convergence
  force_th_saddle   = 0.1;
  push_over_saddle  = 0.3;
  eigen_th_fail     = 0.1;

return;
}

/* -----------------------------------------------------------------------------
 * read parameters from file "config"
 * ---------------------------------------------------------------------------*/
void ARTn::read_control()
{
  char oneline[MAXLINE], str[MAXLINE], *token1, *token2;
  FILE *fp = fopen(fctrl, "r");
  if (fp == NULL){
    sprintf(str, "cannot open control parameter file: %s", fctrl);
    error->all(FLERR, str);
  }

  char *fmin, *fsad; fmin = fsad = NULL;
  while (1) {
    fgets(oneline, MAXLINE, fp);
    if(feof(fp)) break;

    if (token1 = strchr(oneline,'#')) *token1 = '\0';

    token1 = strtok(oneline," \t\n\r\f");
    if (token1 == NULL) continue;
    token2 = strtok(NULL," \t\n\r\f");
    if (token2 == NULL){
      sprintf(str, "Insufficient parameter for %s of ARTn!", token1);
      error->all(FLERR, str);
    }

    if (strcmp(token1, "max_conv_steps") == 0){
      max_conv_steps = atoi(token2);
      if (max_conv_steps < 1) error->all(FLERR, "max_conv_steps must be greater than 1.");

    } else if (strcmp(token1, "random_seed") == 0){
      seed = atoi(token2);
      if (seed < 1) error->all(FLERR, "seed must be greater than 1.");

    } else if (strcmp(token1, "temperature") == 0){
      temperature = atof(token2);

    } else if (strcmp(token1, "max_num_events") == 0){
      max_num_events = atoi(token2);
      if (max_num_events < 1) error->all(FLERR, "max_num_events must be greater than 1.");

    } else if (strcmp(token1, "max_activat_iter") == 0){
      max_activat_iter = atoi(token2);
      if (max_activat_iter < 1) error->all(FLERR, "max_activat_iter must be greater than 1.");

    } else if (strcmp(token1, "increment_size") == 0){
      increment_size = atof(token2);
      if (increment_size <= 0.) error->all(FLERR, "increment_size must be greater than 0.");

    } else if (strcmp(token1, "init_kick_mode") == 0){
      if (strcmp(token2, "global_random") == 0) init_kick_mode = 0;
      else if (strcmp(token2, "group_random") == 0) init_kick_mode = 1;
      else if (strcmp(token2, "local_random") == 0) init_kick_mode = 2;
      else init_kick_mode = 3;

    } else if (!strcmp(token1, "region_cutoff")){
      region_cutoff = atof(token2);
      if (region_cutoff <= 0.) error->all(FLERR, "region_cutoff must be greater than 0.");

    } else if (strcmp(token1, "group_name") == 0){
      if (groupname) delete [] groupname;
      groupname = new char [strlen(token2)+1];
      strcpy(groupname, token2);

    } else if (strcmp(token1, "init_step_size") == 0){
      init_step_size = atof(token2);
      if (init_step_size <= 0.) error->all(FLERR, "init_step_size must be greater than 0.");

    } else if (strcmp(token1, "basin_factor") == 0){
      basin_factor = atof(token2);
      if (basin_factor <= 0.) error->all(FLERR, "basin_factor must be greater than 0.");

    } else if (strcmp(token1, "max_perp_move_h") == 0){
      max_perp_move_h = atoi(token2);
      if (max_perp_move_h < 1) error->all(FLERR, "max_perp_move_h must be greater than 0.");

    } else if (strcmp(token1, "min_num_ksteps") == 0){
      min_num_ksteps = atoi(token2);
      if (min_num_ksteps < 1) error->all(FLERR, "min_num_ksteps must be greater than 0.");

    } else if (strcmp(token1, "eigen_th_well") == 0){
      eigen_th_well = atof(token2);
      if (eigen_th_well > 0.) error->all(FLERR, "eigen_th_well must be less than 0.");

    } else if (strcmp(token1, "max_iter_basin") == 0){
      max_iter_basin = atoi(token2);
      if (max_iter_basin < 1) error->all(FLERR, "max_iter_basin must be greater than 0.");

    } else if (strcmp(token1, "force_th_perp_h") == 0){
      force_th_perp_h = atof(token2);
      if (force_th_perp_h <= 0.) error->all(FLERR, "force_th_perp_h must be greater than 0.");

    } else if (strcmp(token1, "num_lancz_vec_H") == 0){
      num_lancz_vec_H = atoi(token2);
      if (num_lancz_vec_H < 1) error->all(FLERR, "num_lancz_vec_H must be greater than 0.");

    } else if (strcmp(token1, "num_lancz_vec_C") == 0){
      num_lancz_vec_C = atoi(token2);
      if (num_lancz_vec_C < 1) error->all(FLERR, "num_lancz_vec_C must be greater than 0.");

    } else if (strcmp(token1, "del_disp_lancz") == 0){
      del_disp_lancz = atof(token2);
      if (del_disp_lancz  <=  0.) error->all(FLERR, "del_disp_lancz must be greater than 0.");

    } else if (strcmp(token1, "eigen_th_lancz") == 0){
      eigen_th_lancz = atof(token2);
      if (eigen_th_lancz <=  0.) error->all(FLERR, "eigen_th_lancz must be greater than 0.");

    } else if (strcmp(token1, "force_th_saddle") == 0){
      force_th_saddle = atof(token2);
      if (force_th_saddle <=  0.) error->all(FLERR, "force_th_saddle must be greater than 0.");

    } else if (strcmp(token1, "push_over_saddle") == 0){
      push_over_saddle = atof(token2);
      if (push_over_saddle <=  0.) error->all(FLERR, "push_over_saddle must be greater than 0.");

    } else if (strcmp(token1, "eigen_th_fail") == 0){
      eigen_th_fail = atof(token2);
      if (eigen_th_fail <=  0.) error->all(FLERR, "eigen_th_fail must be greater than 0.");

    } else if (!strcmp(token1, "atom_move_cutoff")){
      atom_move_cutoff = atof(token2);
      if (atom_move_cutoff <= 0.) error->all(FLERR, "atom_move_cutoff must be greater than 0.");

    } else if (strcmp(token1, "max_perp_moves_C") == 0){
      max_perp_moves_C = atof(token2);
      if (max_perp_moves_C <= 0.) error->all(FLERR, "max_perp_moves_C must be greater than 0.");

    } else if (strcmp(token1, "force_th_perp_sad") == 0){
      force_th_perp_sad = atof(token2);
      if (force_th_perp_sad <= 0.) error->all(FLERR, "force_th_perp_sad must be greater than 0.");

    } else if (strcmp(token1, "use_fire") == 0){
      use_fire = atoi(token2);

    } else if (!strcmp(token1, "flag_check_sad")){
      flag_check_sad = atoi(token2);

    } else if (!strcmp(token1, "max_disp_tol")){
      max_disp_tol = atof(token2);
      if (max_disp_tol <= 0.) error->all(FLERR, "max_disp_tol must be greater than 0.");

    } else if (!strcmp(token1, "pressure_needed")){
      pressure_needed = atoi(token2);

    } else if (!strcmp(token1, "log_file")){
      if (flog) delete []flog;
      flog = new char [strlen(token2)+1];
      strcpy(flog, token2);

    } else if (strcmp(token1, "event_list_file") == 0){
      if (fevent) delete [] fevent;
      fevent = new char [strlen(token2)+1];
      strcpy(fevent, token2);

    } else if (strcmp(token1, "init_config_id") == 0){
      ref_id = atoi(token2);

    } else if (strcmp(token1, "dump_min_config") == 0){
      if (fmin) delete []fmin;
      fmin = new char [strlen(token2)+1];
      strcpy(fmin, token2);

    } else if (strcmp(token1, "dump_sad_config") == 0){
      if (fsad) delete []fsad;
      fsad = new char [strlen(token2)+1];
      strcpy(fsad, token2);

    } else {
      sprintf(str, "Unknown control parameter for ARTn: %s", token1);
      error->all(FLERR, str);
    }
  }
  fclose(fp);

  // set default output file names
  if (flog == NULL){
    flog = new char [9];
    strcpy(flog, "log.artn");
  }
  if (fevent == NULL){
    fevent = new char [10];
    strcpy(fevent, "log.event");
  }
  if (fmin == NULL){
    fmin = new char [14];
    strcpy(fmin, "min.lammpstrj");
  }
  if (fsad == NULL){
    fsad = new char [14];
    strcpy(fsad, "sad.lammpstrj");
  }
  min_id = ref_id;

  // if group info needed, check its validity
  if (init_kick_mode == 1){
    if (groupname == NULL){ groupname = new char [4]; strcpy(groupname, "all");}
    int igroup = group->find(groupname);
    if (igroup == -1){
      sprintf(str, "can not find ARTn group: %s", groupname);
      error->all(FLERR, str);
    }
    groupbit = group->bitmask[igroup];
  }

  // open files and output log file
  if (me == 0){
    fp1 = fopen(flog, "w");
    if (fp1 == NULL){
      sprintf(str, "can not open file %s for writing", flog);
      error->one(FLERR, str);
    }

    fp2 = fopen(fevent, "w");
    if (fp2 == NULL){
      sprintf(str, "can not open file %s for writing", fevent);
      error->one(FLERR, str);
    }

    fprintf(fp1, "\n============================== ARTn based on LAMMPS ==============================\n");
    fprintf(fp1, "max_conv_steps    %20d  # %s\n", max_conv_steps, "max_conv_steps for minimization");
    fprintf(fp1, "random_seed       %20d  # %s\n", seed, "seed for random generator");
    fprintf(fp1, "\n");
    fprintf(fp1, "temperature       %20g  # %s\n", temperature, "temperature for Metropolis acceptance, in eV");
    fprintf(fp1, "max_num_events    %20d  # %s\n", max_num_events,"max number of events");
    fprintf(fp1, "max_activat_iter  %20d  # %s\n", max_activat_iter, "Maximum # of iteraction for reaching the saddle point");
    fprintf(fp1, "increment_size    %20g  # %s\n", increment_size, "Overall scale for the increment moves");
    fprintf(fp1, "init_kick_mode    %20d  # %s\n", init_kick_mode, "global random; group random; local random; local region");
    fprintf(fp1, "region_cutoff     %20g  # %s\n", region_cutoff, "The region cutoff for local region random");
    if (init_kick_mode == 1){
      fprintf(fp1, "group_name        %20s  # %s\n", groupname, "The name of the group that will be kicked.");
    }
    fprintf(fp1, "\n");
    fprintf(fp1, "init_step_size    %20g  # %s\n", init_step_size, "Size of initial displacement");
    fprintf(fp1, "basin_factor      %20g  # %s\n", basin_factor, "Factor multiplying Increment_Size for leaving the basin");
    fprintf(fp1, "max_perp_move_h   %20d  # %s\n", max_perp_move_h, "Maximum number of perpendicular steps leaving basin");
    fprintf(fp1, "min_num_ksteps    %20d  # %s\n", min_num_ksteps, "Min. number of ksteps before calling lanczos");
    fprintf(fp1, "eigen_th_well     %20g  # %s\n", eigen_th_well, "Eigenvalue threshold for leaving basin");
    fprintf(fp1, "max_iter_basin    %20d  # %s\n", max_iter_basin, "Maximum number of iteration for leaving the basin");
    fprintf(fp1, "force_th_perp_h   %20g  # %s\n", force_th_perp_h, "Perpendicular force threshold in harmonic well");
    fprintf(fp1, "\n");
    fprintf(fp1, "num_lancz_vec_H   %20d  # %s\n", num_lancz_vec_H, "Number of vectors included in lanczos procedure in the Harmonic well");
    fprintf(fp1, "num_lancz_vec_C   %20d  # %s\n", num_lancz_vec_C, "Number of vectors included in lanczos procedure in convergence");
    fprintf(fp1, "del_disp_lancz    %20g  # %s\n", del_disp_lancz, "Step of the numerical derivative of forces in lanczos");
    fprintf(fp1, "eigen_th_lancz    %20g  # %s\n", eigen_th_lancz, "eigen_th_lanczos for lanczos convergence");
    fprintf(fp1, "\n");
    fprintf(fp1, "force_th_saddle   %20g  # %s\n", force_th_saddle, "Threshold for convergence at saddle point");
    fprintf(fp1, "push_over_saddle  %20g  # %s\n", push_over_saddle, "Fraction of displacement over the saddle");
    fprintf(fp1, "eigen_th_fail     %20g  # %s\n", eigen_th_fail, "The eigen cutoff for failure in searching saddle point");
    fprintf(fp1, "eigen_th_fail     %20g  # %s\n", eigen_th_fail, "The eigen cutoff for failure in searching saddle point");
    fprintf(fp1, "max_perp_moves_C  %20g  # %s\n", max_perp_moves_C, "Maximum number of perpendicular steps approaching saddle point");
    fprintf(fp1, "force_th_perp_sad %20g  # %s\n", force_th_perp_sad, "Perpendicular force threshold approaching saddle point");
    fprintf(fp1, "\n");
    fprintf(fp1, "use_fire          %20d  # %s\n", use_fire, "FIRE minimization method is used in the perpendicular steps approaching saddle point.");
    fprintf(fp1, "flag_check_sad    %20d  # %s\n", flag_check_sad, "Push back saddle point to check if it connect with the minimum.");
    fprintf(fp1, "max_disp_tol      %20g  # %s\n", max_disp_tol, "Tolerance of displacement to claim the found saddle is linked to the original min.");
    fprintf(fp1, "pressure_needed   %20d  # %s\n", pressure_needed, "The pressure will be calculated.");
    fprintf(fp1, "atom_move_cutoff  %20g  # %s\n", atom_move_cutoff, "The cutoff to decide whether an atom is moved");
    fprintf(fp1, "\n");
    fprintf(fp1, "log_file          %20s  # %s\n", flog, "name of the log file");
    fprintf(fp1, "event_list_file   %20s  # %s\n", fevent, "name of the file to write event info");
    fprintf(fp1, "init_config_id    %20d  # %s\n", min_id, "initial configuration ID");
    fprintf(fp1, "dump_min_config   %20s  # %s\n", fmin, "filename to write the atomic dump for stable configurations");
    fprintf(fp1, "dump_sad_config   %20s  # %s\n", fsad, "filename to write the atomic dump for saddle configurations");
    fprintf(fp1, "==================================================================================\n");
  }

  // open dump files
  char **tmp;
  memory->create(tmp, 5, 15, "ARTn string");

  strcpy(tmp[0],"ARTnmin");
  strcpy(tmp[1],"all");
  strcpy(tmp[2],"atom");
  strcpy(tmp[3],"1");
  strcpy(tmp[4],fmin);
  dumpmin = new DumpAtom(lmp, 5, tmp);

  strcpy(tmp[0],"ARTnsad");
  strcpy(tmp[1],"all");
  strcpy(tmp[2],"atom");
  strcpy(tmp[3],"1");
  strcpy(tmp[4],fsad);
  dumpsad = new DumpAtom(lmp, 5, tmp);

  memory->destroy(tmp);

  delete []fmin;
  delete []fsad;

return;
}

/* -----------------------------------------------------------------------------
 * initializing for ARTn
 * ---------------------------------------------------------------------------*/
void ARTn::artn_init()
{
  random = new RanPark(lmp, seed);

  evalf = 0;
  eigen_vec_exist = 0;
  if (nvec) vvec = atom->v[0];


  // peratom vector I use
  vec_count = 3;
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  x0tmp = fix_minimize->request_vector(vec_count);	//3
  ++vec_count;
  h_old = fix_minimize->request_vector(vec_count);	//4
  ++vec_count;
  eigenvector = fix_minimize->request_vector(vec_count);//5
  ++vec_count;
  x00 = fix_minimize->request_vector(vec_count);	//6
  ++vec_count;
  fperp = fix_minimize->request_vector(vec_count);	//7
  ++vec_count;
  htmp = fix_minimize->request_vector(vec_count);	//8
  ++vec_count;
  x_saddle =fix_minimize->request_vector(vec_count);    //9
  ++vec_count;

return;
}

/* ----------------------------------------------------------------------------
 * reset vectors
 * --------------------------------------------------------------------------*/
void ARTn::myreset_vectors()
{
  x0tmp = fix_minimize->request_vector(3);
  h_old = fix_minimize->request_vector(4);
  eigenvector = fix_minimize->request_vector(5);

  x00      = fix_minimize->request_vector(6);
  fperp    = fix_minimize->request_vector(7);
  htmp     = fix_minimize->request_vector(8);
  x_saddle =fix_minimize->request_vector(9);

return;
}

/* -----------------------------------------------------------------------------
 * Try to find saddle point. If failed, return 0, else return 1
 * ---------------------------------------------------------------------------*/
int ARTn::find_saddle( )
{
  int flag = 0;
  double ftot = 0., ftotall, fpar2 = 0.,fpar2all = 0., fperp2 = 0., fperp2all,  delr;
  double fdoth = 0., fdothall = 0.;
  double step, preenergy;
  double tmp;
  int m_perp = 0, trial = 0, nfail = 0;
  eigenvalue = 0.;
  eigen_vec_exist = 0;

  for (int i = 0; i < nvec; i++) x0[i] = x00[i] = xvec[i];

  // random move choose
  if      (init_kick_mode == 0) global_random_move();
  else if (init_kick_mode == 1) group_random_move();
  else if (init_kick_mode == 2) local_random_move();
  else local_region_random();

  ecurrent = energy_force(1);
  myreset_vectors();
  ++evalf;

  if (me == 0){
    fprintf(fp1, "\nBegin to search for the saddle point from configuration %d\n", ref_id);
    fprintf(fp1, "Steps  E-Eref m_perp trial ftot       fpar        fperp     eigen       delr       evalf\n");
    if (screen){
      fprintf(screen, "\nBegin to search for the saddle point from configuration %d\n", ref_id);
      fprintf(screen, "Steps  E-Eref m_perp trial ftot       fpar        fperp     eigen       delr       evalf\n");
    }
  }

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
      if (fperp2 < force_th_perp_h || m_perp > max_perp_move_h || nfail > 5) break; // condition to break

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

    if (local_iter > min_num_ksteps) lanczos(!eigen_vec_exist, 1, num_lancz_vec_H);

    fpar2 = 0.;
    for (int i = 0; i < nvec; ++i) fpar2 += fvec[i] * h[i];
    MPI_Allreduce(&fpar2, &fpar2all,1,MPI_DOUBLE,MPI_SUM,world);

    if (me == 0){
      fprintf(fp1, "%4d %10.5f %3d %3d %10.5f %10.5f %10.5f %10.5f %10.5f %8d\n", local_iter,
      ecurrent-eref, m_perp, trial, sqrt(ftotall), fpar2all, fperp2, eigenvalue, sqrt(delr), evalf);
      if (screen) fprintf(screen, "%4d %10.5f %3d %3d %10.5f %10.5f %10.5f %10.5f %10.5f %8d\n", local_iter,
      ecurrent-eref, m_perp, trial, sqrt(ftotall), fpar2all, fperp2, eigenvalue, sqrt(delr), evalf);
    }

    if (local_iter > min_num_ksteps && eigenvalue < eigen_th_well){
	   if (me == 0){
        fprintf(fp1, "\nOut of harmonic well of %d, continue to search based on the eigenvector\n", ref_id);
        if (screen) fprintf(screen, "\nOut of harmonic well of %d, continue to search based on the eigenvector\n", ref_id);
      }

      flag = 1;
      break;
    }
    // push along the search direction
	 for(int i = 0; i < nvec; ++i) xvec[i] += basin_factor * increment_size * h[i];
  }

  if (! flag){
    if (me == 0){
      fprintf(fp1, "\nReach  max_iter_basin = %d. Could not get out of the harmonic well.\n\n", max_iter_basin);
      if (screen) fprintf(screen, "\nReach  max_iter_basin = %d. Could not get out of the harmonic well.\n\n", max_iter_basin);
    }
    for (int i = 0; i < nvec; ++i) xvec[i] = x00[i];
    ecurrent = energy_force(1);
    myreset_vectors();
    return 0;
  }
  flag = 0;

  if (me == 0){
    fprintf(fp1, "\nNow to converge to the saddle point via Lanczos ...\n");
    fprintf(fp1, "Iter   E-Eref m_perp trial  ftot      fpar        fperp      eigen      delr       evalf    a1\n");
    if (screen){
      fprintf(screen, "\nNow to converge to the saddle point via Lanczos ...\n");
      fprintf(screen, "Iter   E-Eref m_perp trial  ftot      fpar        fperp      eigen      delr       evalf    a1\n");
    }
  }

  // now try to move close to the saddle point according to the eigenvector.
  int inc = 0;
  for (int saddle_iter = 0; saddle_iter < max_activat_iter; ++saddle_iter){
    ecurrent = energy_force(1);
    myreset_vectors();
    for (int i = 0; i < nvec; ++i) h_old[i] = h[i];

    // caculate eigenvector use lanczos
    lanczos(!eigen_vec_exist, 1, num_lancz_vec_C);

    // set search direction according to eigenvector
    double hdot , hdotall, tmpsum, tmpsumall;
    hdot = hdotall = tmpsum = tmpsumall = 0.;
    for (int i =0; i < nvec; ++i) tmpsum += eigenvector[i] * fvec[i];

    MPI_Allreduce(&tmpsum,&tmpsumall,1,MPI_DOUBLE,MPI_SUM,world);
    if (tmpsumall > 0) for (int i = 0; i < nvec; ++i) h[i] = -eigenvector[i];
    else for (int i = 0; i < nvec; ++i) h[i] = eigenvector[i];

    for(int i = 0; i <nvec; ++i) hdot += h[i] * h_old[i];
    MPI_Allreduce(&hdot,&hdotall,1,MPI_DOUBLE,MPI_SUM,world);

    // do minimizing perpendicular use SD or FIRE
    if (use_fire) {
      min_perpendicular_fire(MIN(50, saddle_iter + 18));
      m_perp = trial = 0;

    } else {
      preenergy = ecurrent;
      m_perp = trial = nfail = 0;
      step = increment_size * 0.4;
      int max_perp = max_perp_moves_C + saddle_iter + inc;
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

        // condition to break
        if (fperp2 < force_th_perp_sad || m_perp > max_perp || nfail > 5) break; 
        
        for (int i = 0; i < nvec; ++i){
          x0tmp[i] = xvec[i];
          xvec[i] += step * fperp[i];
        }
        ecurrent = energy_force(1);
        myreset_vectors();
        evalf++;
        
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
    for (int i = 0; i < nvec; ++i) fpar2 += fvec[i] * h[i];
  
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
      fprintf(fp1, "%4d %10.5f %3d %3d %10.5f %10.5f %10.5f %10.5f %10.5f %8d %10.5f\n",
      saddle_iter, ecurrent - eref, m_perp, trial, sqrt(ftotall), fpar2all, sqrt(fperp2), eigenvalue, sqrt(delr), evalf, hdotall);
      if (screen) fprintf(screen, "%4d %10.5f %3d %3d %10.5f %10.5f %10.5f %10.5f %10.5f %8d %10.5f\n",
      saddle_iter, ecurrent - eref, m_perp, trial, sqrt(ftotall), fpar2all, sqrt(fperp2), eigenvalue, sqrt(delr), evalf, hdotall);
    }
   
   
    if (eigenvalue > eigen_th_fail){
      if (me == 0){
        fprintf(fp1, "\nFailed to find the saddle point in Lanczos, as eigenvalue = %g > eigen_threshold = %g\n", eigenvalue, eigen_th_fail);
        fprintf(fp1, "Restart the search in another random direction...\n");
        if (screen){
          fprintf(screen, "\nFailed to find the saddle point in Lanczos, as eigenvalue = %g > eigen_threshold = %g\n", eigenvalue, eigen_th_fail);
          fprintf(screen, "Restart the search in another random direction...\n");
        }
      }
      for(int i = 0; i < nvec; ++i) xvec[i] = x00[i];
  
      return 0;
     }
  
    if (sqrt(ftotall) < force_th_saddle){
      if (me == 0){
        fprintf(fp1, "\nA new saddle point is reached.\n");
        if (screen) fprintf(screen, "\nA new saddle point is reached.\n");
      }
      return 1;
    }
  
    // push along the search direction; E. Cances, et al. JCP, 130, 114711 (2009)
    double factor = MIN(2. * increment_size, fabs(fpar2all)/MAX(fabs(eigenvalue),0.5));
    for (int i = 0; i < nvec; ++i) xvec[i] += factor * h[i];
  }

  if (me == 0){
    fprintf(fp1, "Failed to find the saddle point in Lanczos steps till reaching the max Lanczos steps.\n");
    fprintf(fp1, "Restart the search in another random direction...\n");
    if (screen){
      fprintf(screen, "Failed to find the saddle point in Lanczos steps till reaching the max Lanczos steps.\n");
      fprintf(screen, "Restart the search in another random direction...\n");
    }
  }

  for (int i = 0; i < nvec; ++i) xvec[i] = x00[i];
  energy_force(1); myreset_vectors();

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

    // KLT: it is quite impractical to compare = for floats
    if (dx != dx0) x0[n] = x[i][0] - dx;
    if (dy != dy0) x0[n+1] = x[i][1] - dy;
    if (dz != dz0) x0[n+2] = x[i][2] - dz;
    n += 3;
  }

  domain->set_global_box();

return;
}

/* -----------------------------------------------------------------------------
 * give a global random delta_x 
 * ---------------------------------------------------------------------------*/
void ARTn::global_random_move()
{
  double *delpos = new double [nvec];
  double norm = 0.;
  for (int i=0; i < nvec; i++) delpos[i] = 0.5-random->uniform();

  for (int i=0; i < nvec; i++) norm += delpos[i] * delpos[i];

  double normall;
  MPI_Allreduce(&norm,&normall,1,MPI_DOUBLE,MPI_SUM,world);

  double norm_i = 1./sqrt(normall);
  for (int i=0; i < nvec; i++){
    h[i] = delpos[i] * norm_i;
    xvec[i] += init_step_size * h[i];
  }

  delete []delpos;
return;
}

/* ------------------------------------------------------------------------------
 * give a group random delta_x
 * ----------------------------------------------------------------------------*/
void ARTn::group_random_move()
{
  double *delpos = new double [nvec];
  for (int i = 0; i < nvec; ++i) delpos[i] = 0.;
  double norm = 0.;
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; ++i){
    if (groupbit & atom->mask[i]){
      delpos[i*3]   = 0.5 - random->uniform();
      delpos[i*3+1] = 0.5 - random->uniform();
      delpos[i*3+2] = 0.5 - random->uniform();
    }
  }

  for (int i = 0; i < nvec; ++i) norm += delpos[i] * delpos[i];
  double normall;
  MPI_Allreduce(&norm,&normall,1,MPI_DOUBLE,MPI_SUM,world);

  double norm_i = 1./sqrt(normall);
  for (int i=0; i < nvec; i++){
    h[i] = delpos[i] * norm_i;
    xvec[i] += init_step_size * h[i];
  }
  delete []delpos;

return;
}

/* -----------------------------------------------------------------------------
 * move one atom randomly
 * ---------------------------------------------------------------------------*/
void ARTn::local_random_move()
{
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
      delpos[i*3]   = 0.5 - random->uniform();
      delpos[i*3+1] = 0.5 - random->uniform();
      delpos[i*3+2] = 0.5 - random->uniform();
    }
  }

  for (int i = 0; i < nvec; ++i) norm += delpos[i] * delpos[i];
  double normall;
  MPI_Allreduce(&norm,&normall,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&flag,&flagall,1,MPI_DOUBLE,MPI_SUM,world);

  if (flagall == 0) error->all(FLERR,"local random move error, atom not found.");

  double norm_i = 1./sqrt(normall);
  for (int i=0; i < nvec; i++){
    h[i] = delpos[i] * norm_i;
    xvec[i] += init_step_size * h[i];
  }
  delete []delpos;

return;
}

/* -----------------------------------------------------------------------------
 * local region random
 * ---------------------------------------------------------------------------*/
void ARTn::local_region_random()
{
  double *delpos = new double[nvec];
  for (int i = 0; i < nvec; ++i) delpos[i] = 0.;
  double norm = 0., normall;
  double coords[3] = {0., 0., 0.};
  double coordsall[3];
  double **x = atom->x;
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
      coords[0] = x[i][0];
      coords[1] = x[i][1];
      coords[2] = x[i][2];
    }
  }
  MPI_Allreduce(&flag,&flagall,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(coords,coordsall,3,MPI_DOUBLE,MPI_SUM,world);
  if (flagall == 0) {
    error->all(FLERR,"Local region random error, atom not found.");
  } else if (flagall > 1){
    error->all(FLERR,"Local region random error, too many atom found.");
  }

  double delta_r = 0.;
  double cutoff2 = region_cutoff * region_cutoff;
  for (int i = 0; i < nlocal; ++i){
    delta_r = (x[i][0] - coordsall[0]) * (x[i][0] - coordsall[0])
            + (x[i][1] - coordsall[1]) * (x[i][1] - coordsall[1])
            + (x[i][2] - coordsall[2]) * (x[i][2] - coordsall[2]);

    if (delta_r < cutoff2) {
      delpos[i*3]   = 0.5 - random->uniform();
      delpos[i*3+1] = 0.5 - random->uniform();
      delpos[i*3+2] = 0.5 - random->uniform();
    }
  }
  for (int i = 0; i < nvec; ++i) norm += delpos[i] * delpos[i];
  MPI_Allreduce(&norm,&normall,1,MPI_DOUBLE,MPI_SUM,world);
  double norm_i = 1./sqrt(normall);

  for (int i=0; i < nvec; i++){
    h[i] = delpos[i] * norm_i;
    xvec[i] += init_step_size * h[i];
  }
  delete []delpos;

return;
}

/* -----------------------------------------------------------------------------
 * converge to minimum, here I use conjugate gradient method.
 * ---------------------------------------------------------------------------*/
int ARTn::min_converge(int maxiter)
{
  neval = 0;
  int i,fail,ntimestep;
  double beta,gg,dot[2],dotall[2];

  // nlimit = max # of CG iterations before restarting
  // set to ndoftotal unless too big
  int nlimit = static_cast<int> (MIN(MAXSMALLINT,ndoftotal));

  // initialize working vectors

  eprevious = ecurrent = energy_force(1);
  for (i = 0; i < nvec; i++) h[i] = g[i] = fvec[i];

  gg = fnorm_sqr();
  ntimestep = 0;

  for (int iter = 0; iter < maxiter; iter++) {
    //ntimestep = ++update->ntimestep;
    ++ ntimestep;
    niter++;

    // line minimization along direction h from current atom->x

    fail = (this->*linemin)(ecurrent,alpha_final);
    if (fail) return fail;

    // function evaluation criterion

    if (neval >= update->max_eval) return MAXEVAL;

    // energy tolerance criterion

    if (fabs(ecurrent-eprevious) < update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY)) return ETOL;
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
    // not push_down

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

    //if (output->next == ntimestep) {
    //  timer->stamp();
    //  //output->write(ntimestep);
    //  timer->stamp(TIME_OUTPUT);
    //}
  }

return MAXITER;
}

/* -----------------------------------------------------------------------------
 * center the vector: Warining: no MPI_SUM is used.
 * ---------------------------------------------------------------------------*/
/*
void ARTn::center(double * x, int n)
{
  double sum = 0.;
  for (int i=0; i<n; i++) sum += x[i];

  sum /= double(n);
  for (int i=0; i<n; i++) x[i] -= sum;
}
*/

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
  const double DEL_LANCZOS = del_disp_lancz;
  const double IDEL_LANCZOS = 1.0 / DEL_LANCZOS;

  // set r(k-1) according to eigenvector or random vector
  if (! new_projection) for (int i = 0; i<nvec; i++) r_k_1[i] = eigenvector[i];
  else for (int i = 0; i<nvec; i++) r_k_1[i] = 0.5 - random->uniform();
  for (int i =0; i< nvec; ++i) beta_k_1 += r_k_1[i] * r_k_1[i];

  MPI_Allreduce(&beta_k_1,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  beta_k_1 = sqrt(tmp);

  double eigen1 = 0., eigen2 = 0.;
  char jobs = 'V';
  double *work, *z;
  long ldz = maxvec, info;
  z = new double [ldz*maxvec];
  work = new double [2*maxvec];

  // store origin configuration and force
  for (int i=0; i<nvec; i++){
    x0tmp[i] = xvec[i];
    g[i] = fvec[i];
  }

  for (long n = 1; n <= maxvec; ++n){
    for (int i = 0; i < nvec; ++i){
      q_k[i] = r_k_1[i] / beta_k_1;
      lanc[n-1][i] = q_k[i];
    }

    // random move to caculate u(k) with the finite difference approximation
    for (int i = 0; i < nvec; ++i) xvec[i] = x0tmp[i] + q_k[i] * DEL_LANCZOS;

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
      dstev_(&jobs, &n, d_bak, e_bak, z, &ldz, work, &info);

      //for (int ii=0; ii<n; ii++) printf(" %g", d_bak[ii]); printf("\n");
      if ((int)info != 0) error->all(FLERR,"destev_ error in Lanczos subroute");

      eigen1 = eigen2; eigen2 = d_bak[0];
    }
    if (n >= 3 && fabs((eigen2-eigen1)/eigen1) < eigen_th_lancz) {
      con_flag = 1;
      for (int i = 0; i < nvec; ++i){
	     xvec[i] = x0tmp[i];
	     fvec[i] = g[i];
      }
      eigenvalue = eigen2;
      if (flag > 0){
        eigen_vec_exist = 1;
        for (int i=0; i < nvec; i++) eigenvector[i] = 0.;
        for (int i=0; i<nvec; i++)
        for (int j=0; j<n; j++) eigenvector[i] += z[j] * lanc[j][i];

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

  delete []d;
  delete []e;
  delete []d_bak;
  delete []e_bak;
  delete []z;
  delete []work;
  modify->delete_fix("lanczos");
  delete []lanc;

return;
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
    for (int i = 0; i < nvec; ++i) fdoth += fvec[i] * h[i];

    MPI_Allreduce(&fdoth,&fdothall,1,MPI_DOUBLE,MPI_SUM,world);
    for (int i = 0; i < nvec; ++i) fperp[i] = fvec[i] - fdothall * h[i];

    //double fperpmax = 0., fperpmaxall;
    //for (int i = 0; i < nvec; ++i) {
    //  fperpmax = MAX(fperpmax, abs(fperp[i]));
    //}
    //MPI_Allreduce(&fperpmax,&fperpmaxall,1,MPI_DOUBLE,MPI_MAX,world);
    //if (fperpmaxall < force_th_perp_sad) {
    //  return 1;
    //}
    vdotf = 0.;
    for (int i = 0; i < nvec; ++i) vdotf += vvec[i] * fperp[i];
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

      if (sqrt(fdotf) < force_th_perp_sad) return 1;
      if (fdotfall == 0.) scale2 = 0.;
      else scale2 = alpha * sqrt(vdotvall/fdotfall);
      for (int i = 0; i < nvec; ++i) vvec[i] = scale1 * vvec[i] + scale2 * fperp[i];

      if ((iter - last_negative) > n_min) {
        dt = MIN(dt * f_inc, dtmax);
        alpha = alpha * f_alpha;
      }
    } else {
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

/* ---------------------------------------------------------------------- */
