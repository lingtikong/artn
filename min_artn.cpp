/* -------------------------------------------------------------------------------------------------
 * ARTn: Activation Relaxation Technique nouveau
------------------------------------------------------------------------------------------------- */
#include "min_artn.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "timer.h"
#include "error.h"
#include "modify.h"
#include "fix_minimize.h"
#include "memory.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "compute.h"
#include "force.h"
#include "group.h"
#include "math.h"
#include "output.h"

#define MAXLINE 512
#define ZERO  1.e-10

using namespace LAMMPS_NS;

/* -------------------------------------------------------------------------------------------------
 * lapack or MKL-lapack is used to evaluate the lowest eigenvalue of the matrix in Lanczos.
------------------------------------------------------------------------------------------------- */
#ifdef MKL
#include "mkl.h"
#define dstev_  dstev
#else
extern "C" {
extern void dstev_(char *, int*, double *, double *, double *, int *, double *, int *);
};
#endif

#define EPS_ENERGY 1.e-8

enum{MAXITER,MAXEVAL,ETOL,FTOL,DOWNHILL,ZEROALPHA,ZEROFORCE,ZEROQUAD};

/* -------------------------------------------------------------------------------------------------
 * Constructor of ARTn
------------------------------------------------------------------------------------------------- */
MinARTn::MinARTn(LAMMPS *lmp): MinLineSearch(lmp)
{
  random = NULL;
  pressure = NULL;
  dumpmin = dumpsad = NULL;
  egvec = x0tmp = x00 = vvec = fperp = NULL;

  fp1 = fp2 = NULL;
  glist = NULL;
  groupname = flog = fevent = fconfg = NULL;

  char *id_press = new char [13];
  strcpy(id_press,"thermo_press");
  int pressure_compute = modify->find_compute(id_press); delete [] id_press;

  if (pressure_compute < 0) error->all(FLERR,"Could not find compute pressure ID");

  pressure = modify->compute[pressure_compute];

  // set default control parameters
  set_defaults();

  MPI_Comm_rank(world, &me);
  MPI_Comm_size(world, &np);
return;
}

/* -------------------------------------------------------------------------------------------------
 * The main loops of ARTn
------------------------------------------------------------------------------------------------- */
int MinARTn::iterate(int maxevent)
{
  // read in control parameters
  read_control();
  artn_init();

  max_conv_steps = update->nsteps;

  // minimize before searching saddle points.
  if (me == 0) print_info(0);

  if(!min_fire)stop_condition = min_converge(max_conv_steps,0);
  else stop_condition = min_converge_fire(max_conv_steps); evalf += neval;
  eref = ecurrent;
  stopstr = stopstrings(stop_condition);

  if (me == 0) print_info(1);
  if (flag_press){
    pressure->compute_vector();
    double * press = pressure->vector;
    if (me == 0 && fp1){
      fprintf(fp1, "  - Pressure tensor           :");
      for (int ii = 0; ii < 6; ++ii) fprintf(fp1, " %g", press[ii]);
      fprintf(fp1, "\n");
    }
  }

  // dump the first stable configuration
  if (dumpmin){
    int idum = update->ntimestep;
    update->ntimestep = ref_id;
    dumpmin->write();
    update->ntimestep = idum;
  }

  // print header of event log file
  if (me == 0 && fp2){
    if (flag_press) print_info(2);
    else print_info(3);
  }

  // main loop of ARTn
  int ievent = 0;
  while (ievent < max_num_events){
    // activation
    stage = 1; ++nattempt;
    while (find_saddle() == 0) {stage = 1; ++nattempt;}

    // confirm saddle
    ++sad_found;
    if (check_sad2min()) continue;

    if (flag_push_back) if (push_back_sad() == 0) continue;
    ++sad_id;

    if (flag_relax_sad) sad_converge(max_conv_steps);

    ++ievent; ++stage;
    if (me == 0 && fp2) fprintf(fp2, "%5d %9.6f %7.3f", ievent, delE, egval);

    if (dumpsad){
      int idum = update->ntimestep;
      update->ntimestep = sad_id;
      if(sad_id % dump_sad_every == 0) dumpsad->write();
      update->ntimestep = idum;
    }

    // Relaxation & Metropolis
    push_down();
    metropolis();
  }

  // finalize ARTn
  artn_final();

return MAXITER;
}

/* -------------------------------------------------------------------------------------------------
 * return 0 if distance between new saddle and original min is greater than initial kick.
------------------------------------------------------------------------------------------------- */
int MinARTn::check_sad2min()
{
  lanczos(flag_egvec, 1, num_lancz_vec_c);

  reset_x00();
  // check current center-of-mass
  group->xcm(groupall, masstot, com);
  double dxcm[3];
  dxcm[0] = com[0] - com0[0];
  dxcm[1] = com[1] - com0[1];
  dxcm[2] = com[2] - com0[2];

  // check the distance between new saddle and original min
  // fperp now stores the displacement vector, will be used by push_back and push_down
  double tmp_me[2], tmp_all[2];
  tmp_me[0] = tmp_me[1] = 0.;

  for (int i = 0; i < nvec; ++i){
    double dx = xvec[i] - x00[i] - dxcm[i%3];
    tmp_me[0] += dx * dx;
    tmp_me[1] += dx * egvec[i];
  }
  MPI_Allreduce(tmp_me, tmp_all, 2, MPI_DOUBLE, MPI_SUM, world);
  ddum = sqrt(tmp_all[0]);

  int status = 0;
  if (ddum >= disp_sad2min_thr){
    if (me == 0) print_info(20);

    if (tmp_all[1] > 0.) for (int i = 0; i < nvec; ++i) fperp[i] = egvec[i] * push_over_saddle;
    else for (int i = 0; i < nvec; ++i) fperp[i] = -egvec[i] * push_over_saddle;

  } else {
    if (me == 0) print_info(21);
    
    for (int i = 0; i < nvec; ++i) xvec[i] = x00[i];
    status = 1;
  }

return status;
}

/* -------------------------------------------------------------------------------------------------
 * return 1, saddle connect with starting minimum, else return 0
------------------------------------------------------------------------------------------------- */
int MinARTn::push_back_sad()
{
  ++stage;
  for (int i = 0; i < nvec; ++i) x0tmp[i] = xvec[i];  // x0tmp stores the saddle position

  // push back the saddle; fperp carries the direction vector, set by check_sad2min
  for (int i = 0; i < nvec; ++i) xvec[i] -= fperp[i];

  if (me == 0) print_info(30);

  // minimization using CG
  if(!min_fire){
    stop_condition = min_converge(max_conv_steps,2); 
  }else{
    stop_condition = min_converge_fire(max_conv_steps);
  }
  evalf += neval;
  stopstr = stopstrings(stop_condition); artn_reset_vec();
  reset_coords();

  // output minimization information
  if (me == 0) print_info(31);
  ddum = fabs(ecurrent - eref);

  if (ddum > max_ener_tol) {
    if (me == 0) print_info(32);

    for (int i = 0; i < nvec; ++i) xvec[i] = x00[i];
    return 0;
  }

  // check current center-of-mass
  double dxcm[3];
  group->xcm(groupall, masstot, com);
  dxcm[0] = com[0] - com0[0];
  dxcm[1] = com[1] - com0[1];
  dxcm[2] = com[2] - com0[2];

  double dr = 0., drall;
  for (int i = 0; i < nvec; i++){
    double dx = xvec[i] - x00[i] - dxcm[i%3];
    dr += dx * dx;
  }
  MPI_Allreduce(&dr,&drall,1,MPI_DOUBLE,MPI_SUM,world);
  ddum = sqrt(drall);

  if (ddum < max_disp_tol) {
    if (me == 0) print_info(33);

    for (int i = 0; i < nvec; ++i) xvec[i] = x0tmp[i];
    return 1;

  } else {

    if (me == 0) print_info(34);
    for (int i = 0; i < nvec; ++i) xvec[i] = x00[i];
  }

return 0;
}

/* -------------------------------------------------------------------------------------------------
 * push the configuration push_down
------------------------------------------------------------------------------------------------- */
void MinARTn::push_down()
{
  // calculate the moved atoms at saddle point
  group->xcm(groupall, masstot, com);
  double **x = atom->x;
  double dx, dy, dz;
  double dxcm[3];
  dxcm[0] = com[0] - com0[0];
  dxcm[1] = com[1] - com0[1];
  dxcm[2] = com[2] - com0[2];

  int n_moved = 0, n_movedall, n = 0;
  double tmp, disp_thr2 = atom_disp_thr*atom_disp_thr;
  for (int i = 0; i < atom->nlocal; ++i) {
    dx = x[i][0] - x00[n]   - dxcm[0];
    dy = x[i][1] - x00[n+1] - dxcm[1];
    dz = x[i][2] - x00[n+2] - dxcm[2];

    tmp = dx*dx + dy*dy + dz*dz;
    n += 3;

    if (tmp > disp_thr2) ++n_moved;
  }
  MPI_Reduce(&n_moved, &n_movedall, 1, MPI_INT, MPI_SUM, 0, world);
  if (me == 0 && fp2) fprintf(fp2, " %5d", n_movedall);


  // push down the saddle; fperp carries the direction vector, set by check_sad2min

  for (int i = 0; i < nvec; ++i) xvec[i] += fperp[i];

  ecurrent = energy_force(1); ++evalf;
  if (me == 0) print_info(50);

  // minimization using CG or FIRE
  if(!min_fire){SD_min_converge(SD_steps,1); evalf += neval;
  stop_condition = min_converge(max_conv_steps,1); evalf += neval;
  }else{stop_condition = min_converge_fire(max_conv_steps); evalf += neval;}
  stopstr = stopstrings(stop_condition);
  artn_reset_vec();
  reset_x00();

  // output minimization information
  if (me == 0) print_info(51);

  // store min configuration
  ++min_id;
  if (dumpmin){
    int idum = update->ntimestep;
    update->ntimestep = min_id;
    if(min_id % dump_min_every == 0) dumpmin->write();
    update->ntimestep = idum;
  }

  return;
}

/* -------------------------------------------------------------------------------------------------
 * decide whether reject or accept the new configuration
------------------------------------------------------------------------------------------------- */
void MinARTn::metropolis()
{
  // output reference energy ,current energy and pressure.
  if (me == 0 && fp2) fprintf(fp2, " %5d %4d %5d %7d %10.3f %10.3f", ref_id, sad_id, min_id, that, eref, ecurrent);

  double dr = 0., drall;
  double **x = atom->x;
  double dx,dy,dz;
  double disp_me[2], disp_all[2];
  disp_all[0] = disp_all[1] = 0.;

  // check current center-of-mass
  group->xcm(groupall, masstot, com);
  double dxcm[3];
  dxcm[0] = com[0] - com0[0];
  dxcm[1] = com[1] - com0[1];
  dxcm[2] = com[2] - com0[2];

  // calculate displacment w.r.t. ref
  double tmp, disp_thr2 = atom_disp_thr*atom_disp_thr;
  int n_moved = 0, n_movedall, n = 0;
  for (int i = 0; i < atom->nlocal; ++i) {
    dx = x[i][0] - x00[n]   - dxcm[0];
    dy = x[i][1] - x00[n+1] - dxcm[1];
    dz = x[i][2] - x00[n+2] - dxcm[2];

    tmp = dx*dx + dy*dy + dz*dz;
    dr += tmp; n += 3;

    if (tmp > disp_thr2) ++n_moved;
  }
  disp_me[0] = dr; disp_me[1] = double(n_moved);
  MPI_Reduce(disp_me, disp_all, 2, MPI_DOUBLE, MPI_SUM, 0, world);
  drall = sqrt(disp_all[0]); n_movedall = int(disp_all[1]);

  if (me == 0 && fp2) fprintf(fp2, " %5d", n_movedall);

  // set v = 0 to calculate pressure
  if (flag_press){
    for (int i = 0; i < nvec; ++i) vvec[i] = 0.;

    ++update->ntimestep;
    pressure->addstep(update->ntimestep);
    energy_force(0); ++evalf; reset_x00();
    pressure->compute_vector();
    double * press = pressure->vector;

    if (me == 0 && fp2) for (int i = 0; i < 6; ++i) fprintf(fp2, " %10g", press[i]);
  }

  // Metropolis
  int acc = 0;
  if (me == 0){
    ddum = drall;
    print_info(60);

    if (temperature > 0. && (ecurrent < eref || random->uniform() < exp((eref - ecurrent)/temperature))) acc = 1;
  }
  MPI_Bcast(&acc, 1, MPI_INT, 0, world);

  if (acc){
    if (me == 0) print_info(61);
    ref_id = min_id; eref = ecurrent;

  } else {
    if (me == 0) print_info(62);

    for (int i = 0; i < nvec; ++i) xvec[i] = x00[i];
    ecurrent = energy_force(1); ++evalf;
    artn_reset_vec();
  }

  if (me == 0 && fp2){
    for (int i = 0; i < 3; ++i) dxcm[i] *= double(atom->natoms);
    fprintf(fp2, " %12.5f %2d %9.5f %9.5f %9.5f %8.5f\n", ecurrent, acc, dxcm[0], dxcm[1], dxcm[2], drall);
    fflush(fp2);
  }

return;
}

/* -------------------------------------------------------------------------------------------------
 * setup default parameters, some values come from Norman's code
------------------------------------------------------------------------------------------------- */
void MinARTn::set_defaults()
{
  // global
  seed = 12345;
  nattempt = ref_id = min_id = sad_id = 0;
  sad_found = 0;
  max_num_events   = 1000;
  flag_press       = 0;
  min_fire         = 0;

  // activation, harmonic well escape
  cluster_radius   = 5.0;
  init_step_size   = 0.1;
  basin_factor     = 2.5;
  max_perp_move_h  = 20;
  max_iter_basin   = 30;
  min_num_ksteps   = 0;		
  increment_size   = 0.09;
  force_th_perp_h  = 0.5;
  eigen_th_well    = -0.01;

  // activation, converge to saddle
  max_activat_iter  = 100;
  use_fire          = 0;
  force_th_saddle   = 0.005;
  eigen_th_fail     = 0.000;
  conv_perp_inc     = 40;
  max_perp_moves_c  = 15;
  force_th_perp_sad = 0.001;

  // confirmatom of new saddle
  disp_sad2min_thr = -1.;
  flag_push_back   = 0;
  max_disp_tol     = 0.2;
  max_ener_tol     = 0.001;
  flag_relax_sad   = 0;

  // convergence to new minimum
  push_over_saddle = 0.2;
  atom_disp_thr    = 0.2;
  temperature      = 0.1;
  SD_steps         = 5;

  // for lanczos
  num_lancz_vec_h  = 30;
  num_lancz_vec_c  = 20;
  del_disp_lancz   = 0.001;
  eigen_th_lancz   = 0.01;

  // output
  log_level        = 1;
  print_freq       = 1;
  dump_min_every   = 1;
  dump_sad_every   = 1;

return;
}

/* -------------------------------------------------------------------------------------------------
 * read ARTn control parameters from file "artn.control"
------------------------------------------------------------------------------------------------- */
void MinARTn::read_control()
{
  char oneline[MAXLINE], str[MAXLINE], *token1, *token2;
  FILE *fp = fopen("artn.control", "r");
  char *fmin, *fsad; fmin = fsad = NULL;
  if (fp == NULL){
    error->warning(FLERR, "Cannot open ARTn control parameter file. Default parameters will be used.\n");

  } else {
    while ( 1 ) {
      fgets(oneline, MAXLINE, fp);
      if (feof(fp)) break;

      if (token1 = strchr(oneline,'#')) *token1 = '\0';

      token1 = strtok(oneline," \t\n\r\f");
      if (token1 == NULL) continue;
      token2 = strtok(NULL," \t\n\r\f");
      if (token2 == NULL){
        sprintf(str, "Insufficient parameter for %s of ARTn!", token1);
        error->all(FLERR, str);
      }

      if (strcmp(token1, "random_seed") == 0){
        seed = force->inumeric(FLERR, token2);
        if (seed < 1) error->all(FLERR, "ARTn: seed must be greater than 0");

      } else if (strcmp(token1, "temperature") == 0){
        temperature = force->numeric(FLERR, token2);

      } else if (strcmp(token1, "max_num_events") == 0){
        max_num_events = force->inumeric(FLERR, token2);
        if (max_num_events < 1) error->all(FLERR, "ARTn: max_num_events must be greater than 0");

      } else if (strcmp(token1, "max_activat_iter") == 0){
        max_activat_iter = force->inumeric(FLERR, token2);
        if (max_activat_iter < 1) error->all(FLERR, "ARTn: max_activat_iter must be greater than 0");

      } else if (strcmp(token1, "increment_size") == 0){
        increment_size = force->numeric(FLERR, token2);
        if (increment_size <= 0.) error->all(FLERR, "ARTn: increment_size must be greater than 0.");

      } else if (!strcmp(token1, "cluster_radius")){
        cluster_radius = force->numeric(FLERR, token2);

      } else if (strcmp(token1, "group_4_activat") == 0){
        if (groupname) delete [] groupname;
        groupname = new char [strlen(token2)+1];
        strcpy(groupname, token2);

      } else if (strcmp(token1, "init_step_size") == 0){
        init_step_size = force->numeric(FLERR, token2);
        if (init_step_size <= 0.) error->all(FLERR, "ARTn: init_step_size must be greater than 0.");

      } else if (strcmp(token1, "basin_factor") == 0){
        basin_factor = force->numeric(FLERR, token2);
        if (basin_factor <= 0.) error->all(FLERR, "ARTn: basin_factor must be greater than 0.");

      } else if (strcmp(token1, "max_perp_move_h") == 0){
        max_perp_move_h = force->inumeric(FLERR, token2);
        if (max_perp_move_h < 1) error->all(FLERR, "ARTn: max_perp_move_h must be greater than 0.");

      } else if (strcmp(token1, "min_num_ksteps") == 0){
        min_num_ksteps = force->inumeric(FLERR, token2);
        if (min_num_ksteps < 1) error->all(FLERR, "ARTn: min_num_ksteps must be greater than 0");

      } else if (strcmp(token1, "eigen_th_well") == 0){
        eigen_th_well = force->numeric(FLERR, token2);
        if (eigen_th_well > 0.) error->all(FLERR, "ARTn: eigen_th_well must be less than 0.");

      } else if (strcmp(token1, "max_iter_basin") == 0){
        max_iter_basin = force->inumeric(FLERR, token2);
        if (max_iter_basin < 1) error->all(FLERR, "ARTn: max_iter_basin must be greater than 0");

      } else if (strcmp(token1, "force_th_perp_h") == 0){
        force_th_perp_h = force->numeric(FLERR, token2);
        if (force_th_perp_h <= 0.) error->all(FLERR, "ARTn: force_th_perp_h must be greater than 0.");

      } else if (strcmp(token1, "num_lancz_vec_h") == 0){
        num_lancz_vec_h = force->inumeric(FLERR, token2);
        if (num_lancz_vec_h < 1) error->all(FLERR, "ARTn: num_lancz_vec_h must be greater than 0");

      } else if (strcmp(token1, "num_lancz_vec_c") == 0){
        num_lancz_vec_c = force->inumeric(FLERR, token2);
        if (num_lancz_vec_c < 1) error->all(FLERR, "ARTn: num_lancz_vec_c must be greater than 0");

      } else if (strcmp(token1, "del_disp_lancz") == 0){
        del_disp_lancz = force->numeric(FLERR, token2);
        if (del_disp_lancz  <=  0.) error->all(FLERR, "ARTn: del_disp_lancz must be greater than 0.");

      } else if (strcmp(token1, "eigen_th_lancz") == 0){
        eigen_th_lancz = force->numeric(FLERR, token2);
        if (eigen_th_lancz <=  0.) error->all(FLERR, "ARTn: eigen_th_lancz must be greater than 0.");

      } else if (strcmp(token1, "force_th_saddle") == 0){
        force_th_saddle = force->numeric(FLERR, token2);
        if (force_th_saddle <=  0.) error->all(FLERR, "ARTn: force_th_saddle must be greater than 0.");

      } else if (strcmp(token1, "disp_sad2min_thr") == 0){
        disp_sad2min_thr = force->numeric(FLERR, token2);
        if (disp_sad2min_thr <=  0.) error->all(FLERR, "ARTn: disp_sad2min_thr must be greater than 0.");

      } else if (strcmp(token1, "push_over_saddle") == 0){
        push_over_saddle = force->numeric(FLERR, token2);
        if (push_over_saddle <=  0.) error->all(FLERR, "ARTn: push_over_saddle must be greater than 0.");

      } else if (strcmp(token1, "eigen_th_fail") == 0){
        eigen_th_fail = force->numeric(FLERR, token2);

      } else if (!strcmp(token1, "atom_disp_thr")){
        atom_disp_thr = force->numeric(FLERR, token2);
        if (atom_disp_thr <= 0.) error->all(FLERR, "ARTn: atom_disp_thr must be greater than 0.");

      } else if (strcmp(token1, "max_perp_moves_c") == 0){
        max_perp_moves_c = force->inumeric(FLERR, token2);
        if (max_perp_moves_c < 1) error->all(FLERR, "ARTn: max_perp_moves_c must be greater than 0.");

      } else if (strcmp(token1, "force_th_perp_sad") == 0){
        force_th_perp_sad = force->numeric(FLERR, token2);
        if (force_th_perp_sad <= 0.) error->all(FLERR, "ARTn: force_th_perp_sad must be greater than 0.");

      } else if (strcmp(token1, "use_fire") == 0){
        use_fire = force->inumeric(FLERR, token2);

      } else if (!strcmp(token1, "flag_push_back")){
        flag_push_back = force->inumeric(FLERR, token2);

      } else if (!strcmp(token1, "flag_relax_sad")){
        flag_relax_sad = force->inumeric(FLERR, token2);

      } else if (!strcmp(token1, "max_disp_tol")){
        max_disp_tol = force->numeric(FLERR, token2);
        if (max_disp_tol <= 0.) error->all(FLERR, "ARTn: max_disp_tol must be greater than 0.");

      } else if (!strcmp(token1, "max_ener_tol")){
        max_ener_tol = force->numeric(FLERR, token2);
        if (max_ener_tol <= 0.) error->all(FLERR, "ARTn: max_ener_tol must be greater than 0.");

      } else if (!strcmp(token1, "flag_press")){
        flag_press = force->inumeric(FLERR, token2);

      } else if (!strcmp(token1, "log_file")){
        if (flog) delete []flog;
        flog = new char [strlen(token2)+1];
        strcpy(flog, token2);

      } else if (!strcmp(token1, "log_level")){
        log_level = force->inumeric(FLERR, token2);

      } else if (!strcmp(token1, "print_freq")){
        print_freq = force->inumeric(FLERR, token2);

      } else if (strcmp(token1, "event_list_file") == 0){
        if (fevent) delete [] fevent;
        fevent = new char [strlen(token2)+1];
        strcpy(fevent, token2);

      } else if (strcmp(token1, "init_config_id") == 0){
        ref_id = force->inumeric(FLERR, token2);

      } else if (strcmp(token1, "dump_min_config") == 0){
        if (fmin) delete []fmin;
        fmin = new char [strlen(token2)+1];
        strcpy(fmin, token2);

      } else if (strcmp(token1, "dump_sad_config") == 0){
        if (fsad) delete []fsad;
        fsad = new char [strlen(token2)+1];
        strcpy(fsad, token2);

      } else if (strcmp(token1, "conv_perp_inc") == 0){
        conv_perp_inc = force->inumeric(FLERR, token2);

      } else if (strcmp(token1, "SD_steps") == 0){
	SD_steps = force->inumeric(FLERR, token2);

      } else if (strcmp(token1, "dump_min_every") == 0){
	dump_min_every = force->inumeric(FLERR, token2);

      } else if (strcmp(token1, "dump_sad_every") == 0){
	dump_sad_every = force->inumeric(FLERR, token2);

      } else if (strcmp(token1, "min_fire") == 0){
	min_fire = force->inumeric(FLERR, token2);

      } else {
        sprintf(str, "Unknown control parameter for ARTn: %s", token1);
        error->all(FLERR, str);
      } 

    }
    fclose(fp);
  }

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
  min_id = ref_0 = ref_id;

  // if disp_sad2min_thr not set, set as twice init_step_size
  if (disp_sad2min_thr <= 0.) disp_sad2min_thr = init_step_size + init_step_size;

  // default group name is all
  if (groupname == NULL) {groupname = new char [4]; strcpy(groupname, "all");}

  int igroup = group->find(groupname);

  if (igroup == -1){
    sprintf(str, "Cannot find ARTn group: %s", groupname);
    error->all(FLERR, str);
  }
  groupbit = group->bitmask[igroup];
  ngroup = group->count(igroup);
  if (ngroup < 1) error->all(FLERR, "No atom is found in your desired group for activation!");

  // group info for all
  groupall = group->find("all");
  masstot = group->mass(groupall);

  // open log file and output control parameter info
  if (me == 0 && strcmp(flog, "NULL") != 0){
    fp1 = fopen(flog, "w");
    if (fp1 == NULL){
      sprintf(str, "Cannot open ARTn log file: %s for writing", flog);
      error->one(FLERR, str);
    }

    fprintf(fp1, "\n#===================================== ARTn based on LAMMPS ========================================\n");
    fprintf(fp1, "# global control parameters\n");
    fprintf(fp1, "max_num_events      %-18d  # %s\n", max_num_events,"Max number of events");
    fprintf(fp1, "flag_press          %-18d  # %s\n", flag_press, "Flag whether the pressure info will be monitored");
    fprintf(fp1, "random_seed         %-18d  # %s\n", seed, "Seed for random generator");
    fprintf(fp1, "init_config_id      %-18d  # %s\n", min_id, "ID of the initial stable configuration");
    fprintf(fp1, "\n# activation, harmonic well escape\n");
    fprintf(fp1, "group_4_activat     %-18s  # %s\n", groupname, "The lammps group ID of the atoms that can be activated");
    fprintf(fp1, "cluster_radius      %-18g  # %s\n", cluster_radius, "The radius of the cluster that will be activated");
    fprintf(fp1, "init_step_size      %-18g  # %s\n", init_step_size, "Norm of the initial displacement (activation)");
    fprintf(fp1, "basin_factor        %-18g  # %s\n", basin_factor, "Factor multiplying Increment_Size for leaving the basin");
    fprintf(fp1, "min_num_ksteps      %-18d  # %s\n", min_num_ksteps, "Min # of k-steps before calling Lanczos");
    fprintf(fp1, "max_perp_move_h     %-18d  # %s\n", max_perp_move_h, "Max # of perpendicular steps leaving basin");
    fprintf(fp1, "max_iter_basin      %-18d  # %s\n", max_iter_basin, "Maximum # of iteration for leaving the basin");
    fprintf(fp1, "increment_size      %-18g  # %s\n", increment_size, "Overall scale for the increment moves");
    fprintf(fp1, "force_th_perp_h     %-18g  # %s\n", force_th_perp_h, "Perpendicular force threshold in harmonic well");
    fprintf(fp1, "eigen_th_well       %-18g  # %s\n", eigen_th_well, "Eigenvalue threshold for leaving basin");
    fprintf(fp1, "\n# activation, converging to saddle\n");
    fprintf(fp1, "max_activat_iter    %-18d  # %s\n", max_activat_iter, "Maximum # of iteraction to approach the saddle");
    fprintf(fp1, "use_fire            %-18d  # %s\n", use_fire, "Use FIRE for perpendicular steps approaching the saddle");
    fprintf(fp1, "min_fire            %-18d  # %s\n", min_fire, "use FIRE to do minimization both in push back & push forward");
    fprintf(fp1, "eigen_th_fail       %-18g  # %s\n", eigen_th_fail, "Eigen threshold for failure in searching the saddle");
    fprintf(fp1, "force_th_saddle     %-18g  # %s\n", force_th_saddle, "Force threshold for convergence at saddle point");
    fprintf(fp1, "conv_perp_inc       %-18d  # %s\n", conv_perp_inc, "Increment of max # of perpendicular steps when fpar > -1.0");
    fprintf(fp1, "max_perp_moves_c    %-18d  # %s\n", max_perp_moves_c, "Maximum # of perpendicular steps approaching the saddle");
    fprintf(fp1, "force_th_perp_sad   %-18g  # %s\n", force_th_perp_sad, "Perpendicular force threshold approaching saddle point");
    fprintf(fp1, "\n# confirmation of new found saddle\n");
    fprintf(fp1, "disp_sad2min_thr    %-18g  # %s\n", disp_sad2min_thr, "Minimum distance between saddle and original minimum");
    fprintf(fp1, "flag_push_back      %-18d  # %s\n", flag_push_back, "Push back the saddle to check its linkage to the start");
    fprintf(fp1, "push_over_saddle    %-18g  # %s\n", push_over_saddle, "Scale of displacement when pushing over the saddle");
    fprintf(fp1, "max_disp_tol        %-18g  # %s\n", max_disp_tol, "Tolerance displacement to claim the saddle is linked");
    fprintf(fp1, "max_ener_tol        %-18g  # %s\n", max_ener_tol, "Tolerance displacement to claim the saddle is linked");
    fprintf(fp1, "flag_relax_sad      %-18d  # %s\n", flag_relax_sad, "Further relax the newly found saddle via SD algorithm");
    fprintf(fp1, "SD_steps            %-18d  # %s\n", SD_steps, "Steepest Descent steps before CG minimizationm");
    fprintf(fp1, "\n# Lanczos related parameters\n");
    fprintf(fp1, "num_lancz_vec_h     %-18d  # %s\n", num_lancz_vec_h, "Num of vectors included in Lanczos for escaping well");
    fprintf(fp1, "num_lancz_vec_c     %-18d  # %s\n", num_lancz_vec_c, "Num of vectors included in Lanczos for convergence");
    fprintf(fp1, "del_disp_lancz      %-18g  # %s\n", del_disp_lancz, "Step of the numerical derivative of forces in Lanczos");
    fprintf(fp1, "eigen_th_lancz      %-18g  # %s\n", eigen_th_lancz, "Eigenvalue threshold for Lanczos convergence");
    fprintf(fp1, "\n# Metropolis\n");
    fprintf(fp1, "temperature         %-18g  # %s\n", temperature, "Temperature for Metropolis algorithm, in eV");
    fprintf(fp1, "atom_disp_thr       %-18g  # %s\n", atom_disp_thr, "Displacement threshold to identify an atom as displaced");
    fprintf(fp1, "\n# Output related parameters\n");
    fprintf(fp1, "log_file            %-18s  # %s\n", flog, "File to write ARTn log info; NULL to skip");
    fprintf(fp1, "log_level           %-18d  # %s\n", log_level, "Level of ARTn log ouput: 1, high; 0, low.");
    fprintf(fp1, "print_freq          %-18d  # %s\n", print_freq, "Print ARTn log ouput frequency, if log_level is 1.");
    fprintf(fp1, "event_list_file     %-18s  # %s\n", fevent, "File to record the event info; NULL to skip");
    fprintf(fp1, "dump_min_config     %-18s  # %s\n", fmin, "File for atomic dump of stable configurations; NULL to skip");
    fprintf(fp1, "dump_sad_config     %-18s  # %s\n", fsad, "file for atomic dump of saddle configurations; NULL to skip");
    fprintf(fp1, "dump_min_every      %-18d  # %s\n", dump_min_every, "Dump min configuration every # step (if 0, no dump)");
    fprintf(fp1, "dump_sad_every      %-18d  # %s\n", dump_sad_every, "Dump sad configuration every # step (if 0, no dump)");
    fprintf(fp1, "#====================================================================================================\n");
  }

  if (me == 0 && strcmp(fevent, "NULL") != 0){
    fp2 = fopen(fevent, "w");
    if (fp2 == NULL){
      sprintf(str, "Cannot open file: %s for writing", fevent);
      error->one(FLERR, str);
    }
  }

  // open dump files
  char **tmp;
  memory->create(tmp, 5, MAX(MAX(10,strlen(fmin)+1),strlen(fsad)+1), "ARTn string");

  if (strcmp(fmin, "NULL") != 0){
    strcpy(tmp[0],"ARTnmin");
    strcpy(tmp[1],"all");
    strcpy(tmp[2],"atom");
    strcpy(tmp[3],"1");
    strcpy(tmp[4],fmin);
    if(dump_min_every) dumpmin = new DumpAtom(lmp, 5, tmp);
  }

  if (strcmp(fsad, "NULL") != 0){
    strcpy(tmp[0],"ARTnsad");
    strcpy(tmp[1],"all");
    strcpy(tmp[2],"atom");
    strcpy(tmp[3],"1");
    strcpy(tmp[4],fsad);
    if(dump_sad_every) dumpsad = new DumpAtom(lmp, 5, tmp);
  }

  memory->destroy(tmp);

  delete []fmin;
  delete []fsad;

return;
}

/* -------------------------------------------------------------------------------------------------
 * initializing ARTn
------------------------------------------------------------------------------------------------- */
void MinARTn::artn_init()
{
  random = new RanPark(lmp, seed+me);

  evalf = 0;
  flag_egvec = 0;
  if (nvec) vvec = atom->v[0];

  // peratom vector I use
  fix_minimize->store_box();
  fix_minimize->add_vector(3);			//3
  fix_minimize->add_vector(3);			//4
  fix_minimize->add_vector(3);			//5
  fix_minimize->add_vector(3);			//6
  x0tmp = fix_minimize->request_vector(3);  //3
  egvec = fix_minimize->request_vector(4);  //4
  x00   = fix_minimize->request_vector(5);  //5
  fperp = fix_minimize->request_vector(6);  //6

  // group list
  int *tag  = atom->tag;
  int *mask = atom->mask;
  int *llist; memory->create(llist, MAX(1, atom->nlocal), "llist");

  int n = 0;
  for (int i = 0; i < atom->nlocal; ++i) if (mask[i] & groupbit) llist[n++] = tag[i];

  int nsingle = n, nall;
  MPI_Allreduce(&nsingle,&nall,1,MPI_INT,MPI_SUM,world);

  if (nall != ngroup) error->all(FLERR, "# of atoms in group mismatch!");
  if (me == 0) memory->create(glist, ngroup, "glist");

  int *disp = new int [np];
  int *recv = new int [np];
  for (int i = 0; i < np; ++i) disp[i] = recv[i] = 0;
  MPI_Gather(&nsingle,1,MPI_INT,recv,1,MPI_INT,0,world);
  for (int i = 1; i < np; ++i) disp[i] = disp[i-1] + recv[i-1];

  MPI_Gatherv(llist,nsingle,MPI_INT,glist,recv,disp,MPI_INT,0,world);
  delete [] disp;
  delete [] recv;
  memory->destroy(llist);

  if (dumpmin) dumpmin->init();
  if (dumpsad) dumpsad->init();

return;
}

/* -------------------------------------------------------------------------------------------------
 * reset vectors
------------------------------------------------------------------------------------------------- */
void MinARTn::artn_reset_vec()
{
  x0tmp = fix_minimize->request_vector(3);
  egvec = fix_minimize->request_vector(4);
  x00   = fix_minimize->request_vector(5);
  fperp = fix_minimize->request_vector(6);

return;
}

/* -------------------------------------------------------------------------------------------------
 * Try to find saddle point. If failed, return 0; else return 1
------------------------------------------------------------------------------------------------- */
int MinARTn::find_saddle( )
{
  int flag = 0;
  int nlanc = 0;
  double ftot = 0., ftotall, fpar2 = 0.,fpar2all = 0., fperp2 = 0., fperp2all = 0.,  delr;
  double fdoth = 0., fdothall = 0.;
  double step, preenergy;
  double tmp;
  int m_perp = 0, trial = 0, nfail = 0;
  double tmp_me[3], tmp_all[3];

  double force_thh_p2 = force_th_perp_h * force_th_perp_h;
  double force_thc_p2 = force_th_perp_sad * force_th_perp_sad;

  // record center-of-mass
  group->xcm(groupall, masstot, com0);

  egval = 0.;
  flag_egvec = 0;

  // record the original atomic positions
  for (int i = 0; i < nvec; ++i) x0[i] = x00[i] = xvec[i];

  // randomly displace the desired atoms: activation
  random_kick();

  if (me == 0) print_info(10);

  int nmax_perp = max_perp_move_h;

  // try to leave harmonic well
  for (int it = 0; it < max_iter_basin; ++it){
    // minimizing perpendicularly by using SD method
    ecurrent = energy_force(1); ++evalf;
    artn_reset_vec(); reset_x00();

    m_perp = nfail = trial = 0;
    step = increment_size * 0.4;
    preenergy = ecurrent;
    while ( 1 ){
      //preenergy = ecurrent;
      fdoth = 0.;
      for (int i = 0; i < nvec; ++i) fdoth += fvec[i] * h[i];
      MPI_Allreduce(&fdoth, &fdothall,1,MPI_DOUBLE,MPI_SUM,world);
      fperp2 = 0.;
      
      for (int i = 0; i < nvec; ++i){ 
        fperp[i] = fvec[i] - fdothall * h[i];
        fperp2 += fperp[i] * fperp[i];
      }
      MPI_Allreduce(&fperp2, &fperp2all,1,MPI_DOUBLE,MPI_SUM,world);
      if (fperp2all < force_thh_p2 || m_perp > nmax_perp || nfail > 5) break; // condition to break
      
      for (int i = 0; i < nvec; ++i){
        x0tmp[i] = xvec[i];
        xvec[i] += MIN(dmax, MAX(-dmax, step * fperp[i]));
      }
      ecurrent = energy_force(1); ++evalf;
      artn_reset_vec(); reset_coords();
      
      if (ecurrent < preenergy){
        step *= 1.2;
        ++m_perp; nfail = 0;
        preenergy = ecurrent;

      } else {
        for (int i = 0; i < nvec; ++i) xvec[i] = x0tmp[i];
        step *= 0.5; ++nfail;
        ecurrent = energy_force(1); ++evalf;
        artn_reset_vec(); reset_x00();
      }
      ++trial;
    }
    
    ftot = delr = 0.;
    for (int i = 0; i < nvec; ++i) {
      ftot += fvec[i] * fvec[i];
      tmp = xvec[i] - x0[i];
      delr += tmp * tmp;
    }
    tmp_me[0] = ftot; tmp_me[1] = delr;
    MPI_Reduce(tmp_me, tmp_all,2,MPI_DOUBLE,MPI_SUM,0,world);
    ftotall = tmp_all[0]; delr = tmp_all[1];
    
    if (it > min_num_ksteps) nlanc = lanczos(flag_egvec, 1, num_lancz_vec_h);
    
    fpar2 = 0.;
    for (int i = 0; i < nvec; ++i) fpar2 += fvec[i] * h[i];
    MPI_Allreduce(&fpar2, &fpar2all,1,MPI_DOUBLE,MPI_SUM,world);
    
    delE = ecurrent-eref;
    if (me == 0){
      fperp2 = sqrt(fperp2all);
      ftot   = sqrt(ftotall);
      delr   = sqrt(delr);
      if (fp1 && log_level && it%print_freq == 0) fprintf(fp1, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %10.5f %10.5f " BIGINT_FORMAT "\n", it,
      delE, m_perp, trial, nlanc, ftot, fpar2all, fperp2, egval, delr, evalf);
      if (screen && it%print_freq == 0) fprintf(screen, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %10.5f %10.5f " BIGINT_FORMAT "\n", it,
      delE, m_perp, trial, nlanc, ftot, fpar2all, fperp2, egval, delr, evalf);
    }
    
    if (it > min_num_ksteps && egval < eigen_th_well){
      idum = it;
      if (me == 0){
        if (fp1 && log_level && it%print_freq) fprintf(fp1, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %10.5f %10.5f " BIGINT_FORMAT "\n", it,
        delE, m_perp, trial, nlanc, ftot, fpar2all, fperp2, egval, delr, evalf);
        if (screen && it%print_freq) fprintf(screen, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %10.5f %10.5f " BIGINT_FORMAT "\n", it,
        delE, m_perp, trial, nlanc, ftot, fpar2all, fperp2, egval, delr, evalf);
      
        print_info(11);
      }

      flag = 1;
      break;
    }

    // push along the search direction
    step = basin_factor * increment_size;
    for(int i = 0; i < nvec; ++i) xvec[i] += MIN(dmax, MAX(-dmax, step * h[i]));
  }

  delE = ecurrent-eref;
  if (flag == 0){
    if (me == 0){
      if (fp1 && log_level && (max_iter_basin-1)%print_freq) fprintf(fp1, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %10.5f %10.5f " BIGINT_FORMAT "\n", (max_iter_basin-1),
      delE, m_perp, trial, nlanc, ftot, fpar2all, fperp2, egval, delr, evalf);
      if (screen && (max_iter_basin-1)%print_freq) fprintf(screen, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %10.5f %10.5f " BIGINT_FORMAT "\n", (max_iter_basin-1),
      delE, m_perp, trial, nlanc, ftot, fpar2all, fperp2, egval, delr, evalf);

      print_info(12);
    }
    reset_x00();
    for (int i = 0; i < nvec; ++i) xvec[i] = x00[i];

    return 0;
  }

  flag = 0; ++stage;
  if (me == 0) print_info(13);
  double hdot, hdotall;

  // now try to move close to the saddle point according to the egvec.
  int inc = 0;
  for (int it_s = 0; it_s < max_activat_iter; ++it_s){

    // g store old h
    for (int i = 0; i < nvec; ++i) g[i] = h[i];

    // caculate egvec use lanczos
    nlanc = lanczos(flag_egvec, 1, num_lancz_vec_c);
    for (int i = 0; i < nvec; ++i) h[i] = egvec[i];

    hdot = hdotall = 0.;
    for(int i = 0; i < nvec; ++i) hdot += h[i] * g[i];
    MPI_Reduce(&hdot,&hdotall,1,MPI_DOUBLE,MPI_SUM,0,world);

    // do minimizing perpendicular use SD or FIRE
    if (use_fire) {
      m_perp = trial = min_perp_fire(it_s + max_perp_moves_c + inc);
    } else {
      m_perp = trial = nfail = 0;
      step = increment_size * 0.25;
      int max_perp = max_perp_moves_c + it_s + inc;
      preenergy = ecurrent;
      while ( 1 ){
        fdoth = 0.;
        for (int i = 0; i < nvec; ++i) fdoth += fvec[i] * h[i];
        MPI_Allreduce(&fdoth, &fdothall,1,MPI_DOUBLE,MPI_SUM,world);
        fperp2 = 0.;
        for (int i = 0; i < nvec; ++i){
          fperp[i] = fvec[i] - fdothall * h[i];
          fperp2 += fperp[i] * fperp[i];
        }
        MPI_Allreduce(&fperp2, &fperp2all,1,MPI_DOUBLE,MPI_SUM,world);

        // condition to break
        if (fperp2all < force_thc_p2 || m_perp > max_perp || nfail > 5) break; 
        
        for (int i = 0; i < nvec; ++i){
          x0tmp[i] = xvec[i];
          xvec[i] += MIN(dmax, MAX(-dmax, step * fperp[i]));
        }
        ecurrent = energy_force(1); ++evalf;
        artn_reset_vec(); reset_coords(); 

        // debug
        if (fp1 && that == 5720) fprintf(fp1,"%8d %10.5f %10.5f %10.5f %3d %3d %5d\n", it_s, ecurrent, eref, ecurrent-eref,m_perp, trial, nlanc);
        
        if (ecurrent < preenergy){
          step *= 1.2;
          ++m_perp; nfail = 0;
          preenergy = ecurrent;
        
        } else {
          for (int i = 0; i < nvec; ++i) xvec[i] = x0tmp[i];
          step *= 0.6; ++nfail;
          ecurrent = energy_force(1); ++evalf;
          artn_reset_vec(); reset_x00();
        }
        ++trial;
      }
    }
        
    tmp_me[0] = tmp_me[1] = tmp_me[2] = 0.;
    for (int i = 0; i < nvec; ++i) {
      tmp_me[0] += fvec[i] * fvec[i];
      delr = xvec[i] - x0[i];
      tmp_me[1] += delr * delr;
      tmp_me[2] += fvec[i] * h[i];
    }
    MPI_Allreduce(tmp_me, tmp_all, 3, MPI_DOUBLE, MPI_SUM, world);
    ftotall = sqrt(tmp_all[0]); delr = sqrt(tmp_all[1]); fpar2all = tmp_all[2];
    if (fpar2all > -1.) inc = conv_perp_inc;
   
    // output information
    fperp2 = 0.;
    for (int i = 0; i < nvec; ++i){
      tmp = fvec[i] - fpar2all * h[i];
      fperp2 +=  tmp * tmp;
    }
    MPI_Reduce(&fperp2, &fperp2all,1,MPI_DOUBLE,MPI_SUM,0,world);
  
    delE = ecurrent - eref;
    if (me == 0){
      fperp2 = sqrt(fperp2all);
      if (fp1 && log_level && it_s%print_freq==0) fprintf(fp1, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %8.4f %8.4f %6.3f " BIGINT_FORMAT "\n",
      it_s, delE, m_perp, trial, nlanc, ftotall, fpar2all, fperp2, egval, delr, hdotall, evalf);
      if (screen && it_s%print_freq==0) fprintf(screen, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %8.4f %8.4f %6.3f " BIGINT_FORMAT "\n",
      it_s, delE, m_perp, trial, nlanc, ftotall, fpar2all, fperp2, egval, delr, hdotall, evalf);
    }
   
    if (egval > eigen_th_fail){
      if (me == 0){
        if (fp1 && log_level && it_s%print_freq) fprintf(fp1, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %8.4f %8.4f %6.3f " BIGINT_FORMAT "\n",
        it_s, delE, m_perp, trial, nlanc, ftotall, fpar2all, fperp2, egval, delr, hdotall, evalf);
        if (screen && it_s%print_freq) fprintf(screen, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %8.4f %8.4f %6.3f " BIGINT_FORMAT "\n",
        it_s, delE, m_perp, trial, nlanc, ftotall, fpar2all, fperp2, egval, delr, hdotall, evalf);
 
        print_info(14);
      }

      reset_x00();
      for(int i = 0; i < nvec; ++i) xvec[i] = x00[i];
  
      return 0;
    }

    if (delr < disp_sad2min_thr){
      if (me == 0){
        if (fp1 && log_level && it_s%print_freq) fprintf(fp1, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %8.4f %8.4f %6.3f " BIGINT_FORMAT "\n",
        it_s, delE, m_perp, trial, nlanc, ftotall, fpar2all, fperp2, egval, delr, hdotall, evalf);
        if (screen && it_s%print_freq) fprintf(screen, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %8.4f %8.4f %6.3f " BIGINT_FORMAT "\n",
        it_s, delE, m_perp, trial, nlanc, ftotall, fpar2all, fperp2, egval, delr, hdotall, evalf);
 
        ddum = delr;
        print_info(15);
      }

      reset_x00();
      for(int i = 0; i < nvec; ++i) xvec[i] = x00[i];
  
      return 0;
    }
  
    if (ftotall < force_th_saddle){
      if (me == 0){
        if (fp1 && log_level && it_s%print_freq) fprintf(fp1, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %8.4f %8.4f %6.3f " BIGINT_FORMAT "\n",
        it_s, delE, m_perp, trial, nlanc, ftotall, fpar2all, fperp2, egval, delr, hdotall, evalf);
        if (screen && it_s%print_freq) fprintf(screen, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %8.4f %8.4f %6.3f " BIGINT_FORMAT "\n",
        it_s, delE, m_perp, trial, nlanc, ftotall, fpar2all, fperp2, egval, delr, hdotall, evalf);

        idum = it_s;
        print_info(16);
      }

      return 1;
    }
  
    // caculate egvec use lanczos
    double sign = 1.;
    nlanc = lanczos(flag_egvec, 1, num_lancz_vec_c);

    double tmpsum = 0., tmpsumall;
    for (int i = 0; i < nvec; ++i) tmpsum += egvec[i] * fvec[i];
    MPI_Allreduce(&tmpsum,&tmpsumall,1,MPI_DOUBLE,MPI_SUM,world);

    if (tmpsumall > 0.) sign = -1.;

#define MinEGV 0.5 // was 0.5
    // push along the search direction; E. Cances, et al. JCP, 130, 114711 (2009)
    double factor = sign * MIN(2.*increment_size, fabs(fpar2all)/MAX(fabs(egval), MinEGV));
    for (int i = 0; i < nvec; ++i) xvec[i] += MIN(dmax, MAX(-dmax, factor * egvec[i]));
    ecurrent = energy_force(1); ++evalf;
    artn_reset_vec(); reset_x00();
  }

  delE = ecurrent - eref;
  if (me == 0){
    if (fp1 && log_level && (max_activat_iter-1)%print_freq) fprintf(fp1, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %8.4f %8.4f %6.3f " BIGINT_FORMAT "\n",
    (max_activat_iter-1), delE, m_perp, trial, nlanc, ftotall, fpar2all, fperp2, egval, delr, hdotall, evalf);
    if (screen && (max_activat_iter-1)%print_freq) fprintf(screen, "%8d %10.5f %3d %3d %5d %10.5f %10.5f %10.5f %8.4f %8.4f %6.3f " BIGINT_FORMAT "\n",
    (max_activat_iter-1), delE, m_perp, trial, nlanc, ftotall, fpar2all, fperp2, egval, delr, hdotall, evalf);

    print_info(17);
  }

  reset_x00();
  for (int i = 0; i < nvec; ++i) xvec[i] = x00[i];

return 0;
}

/* -----------------------------------------------------------------------------
 * reset coordinates x0tmp
 * ---------------------------------------------------------------------------*/
void MinARTn::reset_coords()
{
  domain->set_global_box();

  double **x = atom->x;
  int nlocal = atom->nlocal;
  double dx,dy,dz,dx0,dy0,dz0;

  int n = 0;
  for (int i = 0; i < nlocal; ++i) {
    dx = dx0 = x[i][0] - x0tmp[n];
    dy = dy0 = x[i][1] - x0tmp[n+1];
    dz = dz0 = x[i][2] - x0tmp[n+2];
    domain->minimum_image(dx,dy,dz);

    if (dx != dx0) x0tmp[n]   = x[i][0] - dx;
    if (dy != dy0) x0tmp[n+1] = x[i][1] - dy;
    if (dz != dz0) x0tmp[n+2] = x[i][2] - dz;

    n += 3;
  }

  domain->set_global_box();

  reset_x00();

return;
}

/* -----------------------------------------------------------------------------
 * reset coordinates x00
 * ---------------------------------------------------------------------------*/
void MinARTn::reset_x00()
{
  domain->set_global_box();

  double **x = atom->x;
  int nlocal = atom->nlocal;
  double dx,dy,dz,dx0,dy0,dz0;

  int n = 0;
  for (int i = 0; i < nlocal; ++i) {
    dx = dx0 = x[i][0] - x00[n];
    dy = dy0 = x[i][1] - x00[n+1];
    dz = dz0 = x[i][2] - x00[n+2];
    domain->minimum_image(dx,dy,dz);

    if (dx != dx0) x00[n]   = x[i][0] - dx;
    if (dy != dy0) x00[n+1] = x[i][1] - dy;
    if (dz != dz0) x00[n+2] = x[i][2] - dz;

    n += 3;
  }

  domain->set_global_box();

return;
}
/* ------------------------------------------------------------------------------
 * random kick atoms in the defined group and radius
 * ----------------------------------------------------------------------------*/
void MinARTn::random_kick()
{
  // define the central atom that will be activated
  if (me == 0){
    int index = int(random->uniform()*double(ngroup))%ngroup;
    that = glist[index];

    print_info(18);
  }
  MPI_Bcast(&that, 1, MPI_INT, 0, world);

  double cord[3];

  double *delpos = fix_minimize->request_vector(4);
  for (int i = 0; i < nvec; ++i) delpos[i] = 0.;
  int nlocal = atom->nlocal;
  int *tag   = atom->tag;
  int nhit = 0;

  if (fabs(cluster_radius) < ZERO){ // only the cord atom will be kicked
    for (int i = 0; i < nlocal; ++i){
      if (tag[i] == that){
        int n = 3*i;
        delpos[n  ] = 0.5 - random->uniform();
        delpos[n+1] = 0.5 - random->uniform();
        if(domain->dimension == 3)delpos[n+2] = 0.5 - random->uniform();
        else delpos[n+2] = 0.;

        ++nhit; break;
      }
    }

  } else if (cluster_radius < 0.){ // all atoms in group will be kicked
    for (int i = 0; i < nlocal; ++i){
      if (groupbit & atom->mask[i]){
        int n = 3*i;
        delpos[n  ] = 0.5 - random->uniform();
        delpos[n+1] = 0.5 - random->uniform();
        if(domain->dimension == 3)delpos[n+2] = 0.5 - random->uniform();
        else delpos[n+2] = 0.;
        ++nhit;
      }
    }

  } else { // only atoms within a radius to the central atom will be kicked
    double one[3]; one[0] = one[1] = one[2] = 0.;
    for (int i = 0; i < nlocal; ++i){
      if (tag[i] == that){
        one[0] = atom->x[i][0];
        one[1] = atom->x[i][1];
        one[2] = atom->x[i][2];
      }
    }
    MPI_Allreduce(&one[0], &cord[0], 3, MPI_DOUBLE, MPI_SUM, world);
    double rcut2 = cluster_radius * cluster_radius;
    for (int i = 0; i < nlocal; ++i){
      //if (groupbit & atom->mask[i]){
      double dx = atom->x[i][0] - cord[0];
      double dy = atom->x[i][1] - cord[1];
      double dz = atom->x[i][2] - cord[2];
      domain->minimum_image(dx, dy, dz);
      double r2 = dx*dx + dy*dy + dz*dz;
      if (r2 <= rcut2){
        int n = 3*i;
        delpos[n  ] = 0.5 - random->uniform();
        delpos[n+1] = 0.5 - random->uniform();
        if(domain->dimension == 3)delpos[n+2] = 0.5 - random->uniform();
        else delpos[n+2] = 0.;
        ++nhit;
      }
      //}
    }
  }

  MPI_Reduce(&nhit,&idum,1,MPI_INT,MPI_SUM,0,world);
  if (me == 0 && idum < 1) error->one(FLERR, "No atom to kick!");
  if (me == 0) print_info(19);

  // now normalize and apply the kick to the selected atom(s)
  double norm = 0., normall;
  for (int i = 0; i < nvec; ++i) norm += delpos[i] * delpos[i];
  MPI_Allreduce(&norm,&normall,1,MPI_DOUBLE,MPI_SUM,world);

  double norm_i = 1./sqrt(normall);
  for (int i = 0; i < nvec; ++i){
    h[i] = delpos[i] * norm_i;
    xvec[i] += MIN(dmax, MAX(-dmax, init_step_size * h[i]));
  }

return;
}

/* -----------------------------------------------------------------------------
 * converge to minimum, here I use conjugate gradient method.
 * ---------------------------------------------------------------------------*/
int MinARTn::min_converge(int maxiter, const int flag)
{
  neval = 0;
  int i,fail;
  double beta,gg,dot[2],dotall[2];

  // nlimit = max # of CG iterations before restarting
  // set to ndoftotal unless too big
  int nlimit = static_cast<int> (MIN(MAXSMALLINT,ndoftotal));

  // initialize working vectors

  eprevious = ecurrent = energy_force(1); ++neval;
  for (i = 0; i < nvec; ++i) h[i] = g[i] = fvec[i];

  gg = fnorm_sqr();
  double ftol_sq = update->ftol*update->ftol;

  niter = 0;
  for (int iter = 0; iter < maxiter; ++iter) {

    ++niter;

    // line minimization along direction h from current atom->x
    fail = (this->*linemin)(ecurrent,alpha_final);
    if (fail) return fail;

    // function evaluation criterion
    if (neval >= update->max_eval) return MAXEVAL;

    // energy tolerance criterion
    if (fabs(ecurrent-eprevious) < update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY)) return ETOL;

    // force tolerance criterion

    dot[0] = dot[1] = 0.;
    for (i = 0; i < nvec; ++i) {
      dot[0] += fvec[i]*fvec[i];
      dot[1] += fvec[i]*g[i];
    }
    MPI_Allreduce(dot,dotall,2,MPI_DOUBLE,MPI_SUM,world);

    if (dotall[0] < ftol_sq) return FTOL;

    // update new search direction h from new f
    // = -Grad(x) and old g
    // this is Polak-Ribieri formulation
    // beta = dotall[0]/gg would be
    // Fletcher-Reeves
    // reinitialize CG every ndof iterations by
    // setting beta = 0.0

    beta = MAX(0.,(dotall[0] - dotall[1])/gg);
    if ((niter+1) % nlimit == 0) beta = 0.;
    gg = dotall[0];

    for (i = 0; i < nvec; ++i) {
      g[i] = fvec[i];
      h[i] = g[i] + beta*h[i];
    }

    // reinitialize CG
    // if new search
    // direction h is
    // not push_down

    dot[0] = 0.;
    for (i = 0; i < nvec; ++i) dot[0] += g[i]*h[i];
    MPI_Allreduce(dot,dotall,1,MPI_DOUBLE,MPI_SUM,world);

    if (dotall[0] <= 0.) for (i = 0; i < nvec; ++i) h[i] = g[i];
    if (flag == 2) reset_coords();
    else if (flag == 1) reset_x00();
  }

return MAXITER;
}

/* -----------------------------------------------------------------------------
 * The lanczos method to get the lowest eigenvalue and corresponding eigenvector
 * Ref: R.A. Olsen , G. J. Kroes, Comparison of methods for 
 *   fanding saddle points without knowledge of final states,
 *   121, 20(2004)
 * ---------------------------------------------------------------------------*/
int MinARTn::lanczos(bool egvec_exist, int flag, int maxvec){
  FixMinimize * fix_lanczos;
  char **fixarg = new char*[3];
  fixarg[0] = (char *) "artn_lanczos";
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
  fix_lanczos->add_vector(3);		// 5, for f0;
  double *r_k_1 = fix_lanczos->request_vector(0);
  double *q_k_1 = fix_lanczos->request_vector(1);
  for (int i = 0; i < nvec; ++i) q_k_1[i] = 0.;
  double *q_k = fix_lanczos->request_vector(2);
  double *u_k = fix_lanczos->request_vector(3);
  double *r_k = fix_lanczos->request_vector(4);
  double *f0  = fix_lanczos->request_vector(5);
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
    lanc[i] = fix_lanczos->request_vector(i+6);
  }
  // DEL_LANCZOS is the step we use in the finite differece approximation.
  const double DEL_LANCZOS = del_disp_lancz;
  const double IDEL_LANCZOS = 1.0 / DEL_LANCZOS;

  // set r(k-1) according to egvec or random vector
  int nlocal = atom->nlocal;
  if (egvec_exist) for (int i = 0; i < nvec; ++i) r_k_1[i] = egvec[i];
  else for (int i = 0; i < nlocal; ++i) {
    int n = i*3;
    r_k_1[n]   = 0.5 - random->uniform();
    r_k_1[n+1] = 0.5 - random->uniform();
    if(domain->dimension == 3)r_k_1[n+2] = 0.5 - random->uniform();
    else r_k_1[n+2] = 0.;
  }
  for (int i =0; i < nvec; ++i) beta_k_1 += r_k_1[i] * r_k_1[i];

  MPI_Allreduce(&beta_k_1,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  beta_k_1 = sqrt(tmp);

  double eigen1 = 0., eigen2 = 0.;
  char jobs = 'V';
  double *work, *z;
  int ldz = maxvec, info;
  z = new double [ldz*maxvec];
  work = new double [2*maxvec];

  // store origin configuration and force
  for (int i = 0; i < nvec; ++i){
    x0tmp[i] = xvec[i];
    f0[i] = fvec[i];
  }
  int n;
  for (n = 1; n <= maxvec; ++n){
    for (int i = 0; i < nvec; ++i){
      q_k[i] = r_k_1[i] / beta_k_1;
      lanc[n-1][i] = q_k[i];
    }

    //reset_coords();
    // random move to caculate u(k) with the finite difference approximation
    for (int i = 0; i < nvec; ++i) xvec[i] = x0tmp[i] + q_k[i] * DEL_LANCZOS;

    energy_force(1); ++evalf;
    reset_coords();

    r_k_1 = fix_lanczos->request_vector(0);
    q_k_1 = fix_lanczos->request_vector(1);
    q_k = fix_lanczos->request_vector(2);
    u_k = fix_lanczos->request_vector(3);
    r_k = fix_lanczos->request_vector(4);
    f0  = fix_lanczos->request_vector(5);
    artn_reset_vec();
    for (int i = 0; i < maxvec; ++i){
      lanc[i] = fix_lanczos->request_vector(i+6);
    }
    alpha_k = 0.;

    for (int i = 0; i < nvec; ++i){
      u_k[i] = (f0[i] - fvec[i]) * IDEL_LANCZOS;
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
    for (int i = 0; i != maxvec; ++i){
      d_bak[i] = d[i];
      e_bak[i] = e[i];
    }
    if (n >= 2){
      dstev_(&jobs, &n, d_bak, e_bak, z, &ldz, work, &info);

      if (info != 0) error->all(FLERR,"ARTn: dstev_ error in Lanczos subroute");

      eigen1 = eigen2; eigen2 = d_bak[0];
    }
    if (n >= 3 && fabs((eigen2-eigen1)/eigen1) < eigen_th_lancz) {
      con_flag = 1;
      for (int i = 0; i < nvec; ++i){
	     xvec[i] = x0tmp[i];
	     fvec[i] = f0[i];
      }
      egval = eigen2;
      if (flag > 0){
        flag_egvec = 1;
        for (int i = 0; i < nvec; ++i) egvec[i] = 0.;
        for (int i = 0; i < nvec; ++i)
        for (int j = 0; j < n; ++j) egvec[i] += z[j] * lanc[j][i];

        // normalize egvec.
        double sum = 0., sumall;
        for (int i = 0; i < nvec; ++i) sum += egvec[i] * egvec[i];
        
        MPI_Allreduce(&sum, &sumall,1,MPI_DOUBLE,MPI_SUM,world);
        sumall = 1. / sqrt(sumall);
        for (int i = 0; i < nvec; ++i) egvec[i] *= sumall;
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
    egval = eigen2;
    if (flag > 0){
      flag_egvec = 1;
      for (int i = 0; i < nvec; ++i) egvec[i] = 0.;
      for (int i = 0; i < nvec; ++i)
      for (int j = 0; j < n-1; ++j) egvec[i] += z[j] * lanc[j][i];

      // normalize egvec.
      double sum = 0., sumall;
      for (int i = 0; i < nvec; ++i) sum += egvec[i] * egvec[i];

      MPI_Allreduce(&sum, &sumall,1,MPI_DOUBLE,MPI_SUM,world);
      sumall = 1. / sqrt(sumall);
      for (int i = 0; i < nvec; ++i) egvec[i] *= sumall;
    }

    for (int i = 0; i < nvec; ++i){
      xvec[i] = x0tmp[i];
      fvec[i] = f0[i];
    }
  }

  delete []d;
  delete []e;
  delete []d_bak;
  delete []e_bak;
  delete []z;
  delete []work;
  delete []lanc;

  modify->delete_fix("artn_lanczos");

return MIN(int(n),maxvec);
}

/* ---------------------------------------------------------------------------
 *  FIRE: fast interial relaxation engine, return iteration number
 * -------------------------------------------------------------------------*/
int MinARTn::min_perp_fire(int maxiter)
{
  double dt = update->dt;
  const int n_min = 5;
  const double f_inc = 1.1;
  const double f_dec = 0.5;
  const double alpha_start = 0.1;
  const double f_alpha = 0.99;
  const double  TMAX = 10.;
  const double dtmax = TMAX * dt;
  double vdotf, vdotfall;
  double vdotvall;
  double fdotfall;
  double fdoth, fdothall;
  double scale1, scale2;
  double alpha;
  int last_negative = 0;

  double force_thr2 = force_th_perp_sad*force_th_perp_sad;

  for (int i = 0; i < nvec; ++i) vvec[i] = 0.;

  alpha = alpha_start;
  for (int iter = 0; iter < maxiter; ++iter){

    fdoth = 0.;
    for (int i = 0; i < nvec; ++i) fdoth += fvec[i] * h[i];
    MPI_Allreduce(&fdoth,&fdothall,1,MPI_DOUBLE,MPI_SUM,world);

    for (int i = 0; i < nvec; ++i) fperp[i] = fvec[i] - fdothall * h[i];

    vdotf = 0.;
    for (int i = 0; i < nvec; ++i) vdotf += vvec[i] * fperp[i];
    MPI_Allreduce(&vdotf,&vdotfall,1,MPI_DOUBLE,MPI_SUM,world);
    
    // if (v dot f) > 0:
    // v = (1-alpha) v + alpha |v| Fhat
    // |v| = length of v, Fhat = unit f
    // if more than DELAYSTEP since v dot f was negative:
    // increase timestep and decrease alpha
    if (vdotfall > 0.) {
      double tmp_me[2], tmp_all[2];
      scale1 = 1. - alpha;
      tmp_me[0] = tmp_me[1] = 0.;
      for (int i = 0; i < nvec; ++i) {
        tmp_me[0] += vvec[i] * vvec[i];
        tmp_me[1] += fperp[i] * fperp[i];
      }
      MPI_Allreduce(tmp_me, tmp_all,2,MPI_DOUBLE,MPI_SUM,world);
      vdotvall = tmp_all[0]; fdotfall = tmp_all[1];
      
      if (fdotfall < force_thr2) return iter;

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
    // limit timestep so no particle moves further than dmax
    double dtvone = dt;
    double vmax = 0.;
    double dtv;
    for (int i = 0; i < atom->nlocal; i++) {
      vmax = MAX(fabs(v[i][0]),fabs(v[i][1]));
      vmax = MAX(vmax,fabs(v[i][2]));
      if (dtvone*vmax > dmax) dtvone = dmax/vmax;
    }
    MPI_Allreduce(&dtvone,&dtv,1,MPI_DOUBLE,MPI_MIN,world);

    double dtf = dtv * force->ftm2v;
    int n = 0;
    if (rmass) {
      for (int i = 0; i < atom->nlocal; ++i) {
        dtfm = dtf / rmass[i];
        x[i][0] += dt * v[i][0];
        x[i][1] += dt * v[i][1];
        x[i][2] += dt * v[i][2];
        v[i][0] += dtfm * fperp[n++];
        v[i][1] += dtfm * fperp[n++];
        v[i][2] += dtfm * fperp[n++];
      }
    } else {
      for (int i = 0; i < atom->nlocal; ++i) {
        dtfm = dtf / mass[type[i]];
        x[i][0] += dt * v[i][0];
        x[i][1] += dt * v[i][1];
        x[i][2] += dt * v[i][2];
        v[i][0] += dtfm * fperp[n++];
        v[i][1] += dtfm * fperp[n++];
        v[i][2] += dtfm * fperp[n++];
      }
    }
    eprevious = ecurrent;
    ecurrent = energy_force(1); ++evalf;
    artn_reset_vec();
  }

return maxiter;
}

/* ---------------------------------------------------------------------------
 *  A little bit statistics on exit
 * -------------------------------------------------------------------------*/
void MinARTn::artn_final()
{
  if (me == 0){
    if (fp1){
      fprintf(fp1, "\n");
      fprintf(fp1, "#=========================================================================================\n");
      fprintf(fp1, "# Total number of ARTn attempts : %d\n", nattempt);
      fprintf(fp1, "# Number of new found saddle    : %d (%4.1f%% success)\n", sad_found, double(sad_found)/double(MAX(1,nattempt))*100.);
      fprintf(fp1, "# Number of accepted new saddle : %d (%4.1f%% acceptance)\n", sad_id, double(sad_id)/double(MAX(1,sad_found))*100.);
      fprintf(fp1, "# Overall mission success rate  : %4.1f%%\n", double(sad_id)/double(MAX(1,nattempt))*100.);
      fprintf(fp1, "# Number of new minimumi found  : %d\n", min_id-ref_0);
      fprintf(fp1, "# Number of accepted minima     : %d (%g%% acceptance)\n", ref_id-ref_0, double(ref_id-ref_0)/double(MAX(1,min_id-ref_0))*100.);
      fprintf(fp1, "# Number of force evaluation    : " BIGINT_FORMAT "\n", evalf);
      fprintf(fp1, "#=========================================================================================\n");
      fclose(fp1); fp1 = NULL;
    }
   
    if (fp2) fclose(fp2);

    if (screen){
      fprintf(screen, "\n");
      fprintf(screen, "#=========================================================================================\n");
      fprintf(screen, "# Total number of ARTn attempts : %d\n", nattempt);
      fprintf(screen, "# Number of new found saddle    : %d (%.1f%% success)\n", sad_found, double(sad_found)/double(MAX(1,nattempt))*100.);
      fprintf(screen, "# Number of accepted new saddle : %d (%.1f%% acceptance)\n", sad_id, double(sad_id)/double(MAX(1,sad_found))*100.);
      fprintf(screen, "# Overall mission success rate  : %.1f%%\n", double(sad_id)/double(MAX(1,nattempt))*100.);
      fprintf(screen, "# Number of new minimumi found  : %d\n", min_id-ref_0);
      fprintf(screen, "# Number of accepted minima     : %d (%g%% acceptance)\n", ref_id-ref_0, double(ref_id-ref_0)/double(MAX(1,min_id-ref_0))*100.);
      fprintf(screen, "# Number of force evaluation    : " BIGINT_FORMAT "\n", evalf);
      fprintf(screen, "#=========================================================================================\n");
    }
  }

  if (flog)   delete [] flog;
  if (fevent) delete [] fevent;
  if (fconfg) delete [] fconfg;
  if (groupname) delete [] groupname;

  if (glist)  delete [] glist;

  if (random)  delete random;
  if (dumpmin) delete dumpmin;
  if (dumpsad) delete dumpsad;

return;
}

/* -------------------------------------------------------------------------------------------------
 *  Write out related info
------------------------------------------------------------------------------------------------- */
void MinARTn::print_info(const int flag)
{
  if (flag == 0){
    if (fp1) fprintf(fp1, "\nMinimizing the initial configuration, id = %d ....\n", ref_id);
    if (screen) fprintf(screen, "\nMinimizing the initial configuration, id = %d ....\n", ref_id);

  } else if (flag == 1){
    if (fp1){
      if (log_level) fprintf(fp1, "  - Minimizer stop condition  : %s\n",  stopstr);
      fprintf(fp1, "  - Current (ref) energy (eV) : %.6f\n", ecurrent);
      fprintf(fp1, "  - Temperature   (eV)        : %.6f\n", temperature);
    }
    if (screen){
      if (log_level) fprintf(screen, "  - Minimizer stop condition  : %s\n",  stopstr);
      fprintf(screen, "  - Current (ref) energy (eV) : %.6f\n", ecurrent);
      fprintf(screen, "  - Temperature   (eV)        : %.6f\n", temperature);
    }

  } else if (flag == 2){
      fprintf(fp2, "#  1       2        3       4    5     6      7       8      9        10      11        12         13         14         15         16         17       18     19        20        21        22       23\n");
      fprintf(fp2, "#Event   del-E   egv-sad   nsadl ref  sad   min   center   Eref      Emin     nMove    pxx        pyy        pzz        pxy        pxz         pyz     Efinal   status disp-x    disp-y    disp-z     dr\n");
      fprintf(fp2, "#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");

  } else if (flag == 3){
      fprintf(fp2, "#  1       2        3       4    5     6      7    8         9        10     11         12     13       14        15     16       17\n");
      fprintf(fp2, "#Event   del-E   egv-sad   nsadl ref  sad   min   center   Eref      Emin     nMove  Efinal    status disp-x   disp-y    disp-z   dr\n");
      fprintf(fp2, "#------------------------------------------------------------------------------------------------------------------------------------\n");

  } else if (flag == 10){
    if (fp1){
      fprintf(fp1, "  Stage %d, search for the saddle from configuration %d\n", stage, ref_id);
      if (log_level){
        fprintf(fp1, "  ----------------------------------------------------------------------------------------------------\n");
        fprintf(fp1, "    Steps  E-Eref m_perp trial nlanc ftot       fpar        fperp     eigen       delr   evalf\n");
        fprintf(fp1, "  ----------------------------------------------------------------------------------------------------\n");
      }
      fflush(fp1);
    }
    if (screen){
      fprintf(screen, "  Stage %d, search for the saddle from configuration %d\n", stage, ref_id);
      fprintf(screen, "  ----------------------------------------------------------------------------------------------------\n");
      fprintf(screen, "    Steps  E-Eref m_perp trial nlanc ftot       fpar        fperp     eigen       delr   evalf\n");
      fprintf(screen, "  ----------------------------------------------------------------------------------------------------\n");
    }

  } else if (flag == 11){
    if (fp1){
      if (log_level) fprintf(fp1, "  ----------------------------------------------------------------------------------------------------\n");
      fprintf(fp1, "  Stage %d succeeded after %d iterations, continue searching based on eigen-vector.\n", stage, idum);
    }
    if (screen){
      fprintf(screen, "  ----------------------------------------------------------------------------------------------------\n");
      fprintf(screen, "  Stage %d succeeded after %d iterations, continue searching based on eigen-vector.\n", stage, idum);
    }

  } else if (flag == 12){
    if (fp1){
      if (log_level) fprintf(fp1, "  ----------------------------------------------------------------------------------------------------\n");
      fprintf(fp1, "  Stage %d failed, cannot get out of the harmonic well after %d steps.\n\n", stage, max_iter_basin);
    }
    if (screen){
      fprintf(screen, "  ----------------------------------------------------------------------------------------------------\n");
      fprintf(screen, "  Stage %d failed, cannot get out of the harmonic well after %d steps.\n\n", stage, max_iter_basin);
    }

  } else if (flag == 13){
    if (fp1){
      fprintf(fp1, "  Stage %d, converge to the saddle by using Lanczos\n", stage);
      if (log_level){
        fprintf(fp1, "  ----------------------------------------------------------------------------------------------------\n");
        fprintf(fp1, "    Iter   E-Eref m_perp trial nlanc ftot      fpar        fperp     eigen     delr    h.h' evalf\n");
        fprintf(fp1, "  ----------------------------------------------------------------------------------------------------\n");
      }
    }
    if (screen){
      fprintf(screen, "  Stage %d, converge to the saddle by using Lanczos\n", stage);
      fprintf(screen, "  ----------------------------------------------------------------------------------------------------\n");
      fprintf(screen, "    Iter   E-Eref m_perp trial nlanc ftot      fpar        fperp     eigen     delr    h.h' evalf\n");
      fprintf(screen, "  ----------------------------------------------------------------------------------------------------\n");
    }

  } else if (flag == 14){
    if (fp1){
      if (log_level) fprintf(fp1, "  ----------------------------------------------------------------------------------------------------\n");
      fprintf(fp1, "  Stage %d failed, the smallest eigen value is %g > %g\n", stage, egval, eigen_th_fail);
    }
    if (screen){
      fprintf(screen, "  ----------------------------------------------------------------------------------------------------\n");
      fprintf(screen, "  Stage %d failed, the smallest eigen value is %g > %g\n", stage, egval, eigen_th_fail);
    }

  } else if (flag == 15){
    if (fp1){
      if (log_level) fprintf(fp1, "  ----------------------------------------------------------------------------------------------------\n");
      fprintf(fp1, "  Stage %d failed, the distance to min-%d is %g < %g\n", stage, ref_id, ddum, disp_sad2min_thr);
    }
    if (screen){
      fprintf(screen, "  ----------------------------------------------------------------------------------------------------\n");
      fprintf(screen, "  Stage %d failed, the distance to min-%d is %g < %g\n", stage, ref_id, ddum, disp_sad2min_thr);
    }

  } else if (flag == 16){
    if (fp1){
      if (log_level) fprintf(fp1, "  ----------------------------------------------------------------------------------------------------\n");
      fprintf(fp1, "  Stage %d converged at a new saddle after %d iterations, dE = %g\n", stage, idum, ecurrent-eref);
    }
    if (screen){
      fprintf(screen, "  ----------------------------------------------------------------------------------------------------\n");
      fprintf(screen, "  Stage %d converged at a new saddle after %d iterations, dE = %g\n", stage, idum, ecurrent-eref);
    }

  } else if (flag == 17){
    if (fp1){
      if (log_level) fprintf(fp1, "  ----------------------------------------------------------------------------------------------------\n");
      fprintf(fp1, "  Stage %d failed, the max Lanczos steps %d reached.\n", stage, max_activat_iter);
    }
    if (screen){
      fprintf(screen, "  ----------------------------------------------------------------------------------------------------\n");
      fprintf(screen, "  Stage %d failed, the max Lanczos steps %d reached.\n", stage, max_activat_iter);
    }

  } else if (flag == 18){
    if (fp1) fprintf(fp1, "\nAttempt %d, new activation centered on atom %d", nattempt, that);
    if (screen) fprintf(screen, "\nAttempt %d, new activation centered on atom %d", nattempt, that);

  } else if (flag == 19){
    if (fp1) fprintf(fp1, " with total %d atoms. %d success till now.\n", idum, sad_id);
    if (screen) fprintf(screen, " with total %d atoms. %d success till now.\n", idum, sad_id);

  } else if (flag == 20){
    if (fp1) fprintf(fp1, "    The distance between new found saddle and min-%d is %g, > %g, acceptable.\n", ref_id, ddum, disp_sad2min_thr);
    if (screen) fprintf(screen, "    The distance between new found saddle and min-%d is %g, > %g, acceptable.\n", ref_id, ddum, disp_sad2min_thr);

  } else if (flag == 21){
    if (fp1) fprintf(fp1, "    The distance between new saddle and min-%d is %g, < %g, rejected.\n", ref_id, ddum, disp_sad2min_thr);
    if (screen) fprintf(screen, "    The distance between new saddle and min-%d is %g, < %g, rejected.\n", ref_id, ddum, disp_sad2min_thr);

  } else if (flag == 30){
    if (fp1) fprintf(fp1, "  Stage %d, push back the saddle to confirm if it is linked with min-%d\n", stage, ref_id);
    if (screen) fprintf(screen, "  Stage %d, push back the saddle to confirm if it is linked with min-%d\n", stage, ref_id);

  } else if (flag == 31){
    if (fp1) {
      fprintf(fp1, "    - Current   energy (eV)     : %.6f\n", ecurrent);
      fprintf(fp1, "    - Reference energy (eV)     : %.6f\n", eref);
      if (log_level) fprintf(fp1, "    - Minimizer stop condition  : %s\n",  stopstr);
      if (log_level) fprintf(fp1, "    - # of force evaluations    : %d\n", neval);
    }
    if (screen) {
      fprintf(screen, "    - Current   energy (eV)     : %.6f\n", ecurrent);
      fprintf(screen, "    - Reference energy (eV)     : %.6f\n", eref);
      if (log_level) fprintf(screen, "    - Minimizer stop condition  : %s\n",  stopstr);
      if (log_level) fprintf(screen, "    - # of force evaluations    : %d\n", neval);
    }

  } else if (flag == 32){
    if (fp1) fprintf(fp1, "  Stage %d failed, |Ecurrent - Eref| = %g > %g, reject the new saddle.\n", stage, ddum, max_ener_tol);
    if (screen) fprintf(screen, "  Stage %d failed, |Ecurrent - Eref| = %g > %g, reject the new saddle.\n", stage, ddum, max_ener_tol);

  } else if (flag == 33){
    if (fp1) fprintf(fp1, "  Stage %d succeeded, dr = %g < %g, accept the new saddle.\n", stage, ddum, max_disp_tol);
    if (screen) fprintf(screen, "  Stage %d succeeded, dr = %g < %g, accept the new saddle.\n", stage, ddum, max_disp_tol);

  } else if (flag == 34){
    if (fp1) fprintf(fp1, "  Stage %d failed, dr = %g >= %g, reject the new saddle.\n", stage, ddum, max_disp_tol);
    if (screen) fprintf(screen, "  Stage %d failed, dr = %g >= %g, reject the new saddle.\n", stage, ddum, max_disp_tol);

  } else if (flag == 40){
    if (fp1) fprintf(fp1, "  Stage %d, further relax the newly found sad-%d ...\n", stage, sad_id);
    if (screen) fprintf(screen, "  Stage %d, further relax the newly found sad-%d ...\n", stage, sad_id);

  } else if (flag == 41){
    if (fp1){
      fprintf(fp1, "    The new sad-%d is now converged as:\n", sad_id);
      fprintf(fp1, "      - Current energy  (eV)      : %.6f\n", ecurrent);
      fprintf(fp1, "      - Energy  barrier (eV)      : %.6f\n", delE);
      if (log_level) fprintf(fp1, "      - Norm2  of total force     : %lg\n", sqrt(ddum));
      if (log_level) fprintf(fp1, "      - Minimizer stop condition  : %s\n",  stopstr);
      if (log_level) fprintf(fp1, "      - # of force evaluations    : %d\n", neval);
    }
    if (screen){
      fprintf(screen, "    The new sad-%d is now converged as:\n", sad_id);
      fprintf(screen, "      - Current energy  (eV)      : %.6f\n", ecurrent);
      fprintf(screen, "      - Energy  barrier (eV)      : %.6f\n", delE);
      if (log_level) fprintf(screen, "      - Norm2  of total force     : %lg\n", sqrt(ddum));
      if (log_level) fprintf(screen, "      - Minimizer stop condition  : %s\n",  stopstr);
      if (log_level) fprintf(screen, "      - # of force evaluations    : %d\n", neval);
    }

  } else if (flag == 50){
    if (fp1) fprintf(fp1, "  Stage %d, push over the new saddle, Enew: %g; Eref= %g. Relaxing...\n", stage, ecurrent, eref);
    if (screen) fprintf(screen, "  Stage %d, push over the new saddle, Enew: %g; Eref= %g. Relaxing...\n", stage, ecurrent, eref);

  } else if (flag == 51){
    if (fp1){
      fprintf(fp1, "    Relaxed to a nearby minimum to sad-%d\n", sad_id);
      fprintf(fp1, "      - Current  min  energy (eV) : %.6f\n", ecurrent);
      if (log_level)fprintf(fp1, "      - Reference     energy (eV) : %.6f\n", eref);
      if (log_level) fprintf(fp1, "      - Minimizer stop condition  : %s\n",  stopstr);
    }
    if (screen){
      fprintf(screen, "    Relaxed to a nearby minimum to sad-%d\n", sad_id);
      fprintf(screen, "      - Current  min  energy (eV) : %.6f\n", ecurrent);
      if (log_level) fprintf(screen, "      - Reference     energy (eV) : %.6f\n", eref);
      if (log_level) fprintf(screen, "      - Minimizer stop condition  : %s\n",  stopstr);
    }

  } else if (flag == 60){
    if (fp1 && log_level) fprintf(fp1, "      - Distance to min-%8.8d  : %g\n", ref_id, ddum);
    if (screen && log_level) fprintf(screen, "      - Distance to min-%8.8d  : %g\n", ref_id, ddum);

  } else if (flag == 61){
    if (fp1) fprintf(fp1, "  Stage %d done, the new min (E= %g) of ID  %d is accepted by Metropolis.\n", stage, ecurrent, min_id);
    if (screen) fprintf(screen, "  Stage %d done, the new min (E= %g) of ID %d is accepted by Metropolis.\n", stage, ecurrent, min_id);

  } else if (flag == 62){
    if (fp1) fprintf(fp1, "  Stage %d done, the new min (E= %g) of ID %d is rejected by Metropolis.\n", stage, ecurrent, min_id);
    if (screen) fprintf(screen, "  Stage %d done, the new min (E= %g) of ID %d is rejected by Metropolis.\n", stage, ecurrent, min_id);

  }
return;
}

/* -------------------------------------------------------------------------------------------------
 * Converge the saddle point by using SD method; the force parallel to the eigenvector corresponding
 * to the smallest egval is reversed.
------------------------------------------------------------------------------------------------- */
void MinARTn::sad_converge(int maxiter)
{
  ++stage;
  if (me == 0) print_info(40);

  neval = 0;
  int i,fail;
  double edf, edf_all;

  lanczos(flag_egvec, 1, num_lancz_vec_c);
  // initialize working vectors
  edf = 0.;
  for (i =0; i < nvec; ++i) edf += egvec[i] * fvec[i];
  MPI_Allreduce(&edf, &edf_all,1,MPI_DOUBLE,MPI_SUM,world);
  for (i = 0; i < nvec; ++i) h[i] = fvec[i] - 2.*edf_all*egvec[i];

  stop_condition = MAXITER;

  for (int iter = 0; iter < maxiter; ++iter) {
    // line minimization along h from current position x
    // h = downhill gradient direction
    eprevious = ecurrent;
    fail = (this->*linemin)(ecurrent,alpha_final);
    if (fail) {stop_condition = fail; break;}

    // function evaluation criterion
    if (neval >= update->max_eval) {stop_condition = MAXEVAL; break;}

    // energy tolerance criterion
    if (fabs(ecurrent-eprevious) < update->etol * 0.5*(fabs(ecurrent)
    + fabs(eprevious) + EPS_ENERGY)) {stop_condition = ETOL; break;}

    // force tolerance criterion
    double fdotf = fnorm_sqr();
    if (fdotf < update->ftol*update->ftol) {stop_condition = FTOL; break;}

    // set new search direction h to f = -Grad(x)
    lanczos(flag_egvec, 1, num_lancz_vec_c);
    edf = 0.;
    for (i = 0; i < nvec; ++i) edf += egvec[i] * fvec[i];
    MPI_Allreduce(&edf, &edf_all,1,MPI_DOUBLE,MPI_SUM,world);

    for (i = 0; i < nvec; ++i) h[i] = fvec[i] - 2.*edf_all*egvec[i];
  }

  evalf += neval;
  stopstr = stopstrings(stop_condition);

  // output minimization information
  delE = ecurrent-eref;
  ddum = fnorm_sqr();
  if (me == 0) print_info(41);

return;
}
/* ---------------------------------------------------------------------------------------------- */
int MinARTn::SD_min_converge(int maxiter, const int flag)
{
  neval = 0;
  int i,m,n,fail,ntimestep;
  double fdotf;
  double *fatom,*hatom;

  // initialize working vectors

  for (i = 0; i < nvec; i++) h[i] = fvec[i];
  if (nextra_atom)
    for (m = 0; m < nextra_atom; m++) {
      fatom = fextra_atom[m];
      hatom = hextra_atom[m];
      n = extra_nlen[m];
      for (i = 0; i < n; i++) hatom[i] = fatom[i];
    }
  if (nextra_global)
    for (i = 0; i < nextra_global; i++) hextra[i] = fextra[i];

  for (int iter = 0; iter < maxiter; iter++) {
    ntimestep = ++update->ntimestep;
    niter++;

    // line minimization along h from current position x
    // h = downhill gradient direction

    eprevious = ecurrent = energy_force(1); ++neval;
    fail = (this->*linemin)(ecurrent,alpha_final);
    if (fail) return fail;

    // function evaluation criterion

    if (neval >= update->max_eval) return MAXEVAL;

    // energy tolerance criterion

    if (fabs(ecurrent-eprevious) <
        update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
      return ETOL;

    // force tolerance criterion

    fdotf = fnorm_sqr();
    if (fdotf < update->ftol*update->ftol) return FTOL;

    // set new search direction h to f = -Grad(x)

    for (i = 0; i < nvec; i++) h[i] = fvec[i];
    if (nextra_atom)
      for (m = 0; m < nextra_atom; m++) {
        fatom = fextra_atom[m];
        hatom = hextra_atom[m];
        n = extra_nlen[m];
        for (i = 0; i < n; i++) hatom[i] = fatom[i];
      }
    if (nextra_global)
      for (i = 0; i < nextra_global; i++) hextra[i] = fextra[i];
    if (flag == 2) reset_coords();
    else if (flag == 1) reset_x00();
  }
  return MAXITER;
   
}
int MinARTn::min_converge_fire(int maxiter){
  double dt = update->dt;
  const int n_min = 5;
  const double f_inc = 1.1;
  const double f_dec = 0.5;
  const double alpha_start = 0.1;
  const double f_alpha = 0.99;
  const double  TMAX = 10.;
  const double dtmax = TMAX * dt;
  double vdotf, vdotfall;
  double vdotvall;
  double fdotfall;
  double fdoth, fdothall;
  double scale1, scale2;
  double alpha;
  int last_negative = 0;
  neval = 0;

  double force_thr2 = force_th_perp_sad*force_th_perp_sad;

  for (int i = 0; i < nvec; ++i) vvec[i] = 0.;

  alpha = alpha_start;
  for (int iter = 0; iter < maxiter; ++iter){
    vdotf = 0.;
    for (int i = 0; i < nvec; ++i) vdotf += vvec[i] * fvec[i];
    MPI_Allreduce(&vdotf,&vdotfall,1,MPI_DOUBLE,MPI_SUM,world);

    // if (v dot f) > 0:
    // v = (1-alpha) v + alpha |v| Fhat
    // |v| = length of v, Fhat = unit f
    // if more than DELAYSTEP since v dot f was negative:
    // increase timestep and decrease alpha
    if (vdotfall > 0.) {
      double tmp_me[2], tmp_all[2];
      scale1 = 1. - alpha;
      tmp_me[0] = tmp_me[1] = 0.;
      for (int i = 0; i < nvec; ++i) {
	tmp_me[0] += vvec[i] * vvec[i];
	tmp_me[1] += fvec[i] * fvec[i];
      }
      MPI_Allreduce(tmp_me, tmp_all,2,MPI_DOUBLE,MPI_SUM,world);
      vdotvall = tmp_all[0]; fdotfall = tmp_all[1];

      if (fdotfall == 0.) scale2 = 0.;
      else scale2 = alpha * sqrt(vdotvall/fdotfall);
      for (int i = 0; i < nvec; ++i) vvec[i] = scale1 * vvec[i] + scale2 * fvec[i];

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
    int n = 0;
    if (rmass) {
      for (int i = 0; i < atom->nlocal; ++i) {
	dtfm = dtf / rmass[i];
	x[i][0] += dt * v[i][0];
	x[i][1] += dt * v[i][1];
	x[i][2] += dt * v[i][2];
	v[i][0] += dtfm * fvec[n++];
	v[i][1] += dtfm * fvec[n++];
	v[i][2] += dtfm * fvec[n++];
      }
    } else {
      for (int i = 0; i < atom->nlocal; ++i) {
	dtfm = dtf / mass[type[i]];
	x[i][0] += dt * v[i][0];
	x[i][1] += dt * v[i][1];
	x[i][2] += dt * v[i][2];
	v[i][0] += dtfm * fvec[n++];
	v[i][1] += dtfm * fvec[n++];
	v[i][2] += dtfm * fvec[n++];
      }
    }
    eprevious = ecurrent;
    ecurrent = energy_force(1); ++neval;
    artn_reset_vec();
    // energy tolerance criterion
    // only check after DELAYSTEP elapsed since velocties reset to 0

    if (update->etol > 0.0 && iter-last_negative > n_min) {
      if (fabs(ecurrent-eprevious) <
	  update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
	return ETOL;
    }
    // force tolerance criterion

    if (update->ftol > 0.0) {
      double fdotf = fnorm_sqr();
      if (fdotf < update->ftol*update->ftol) return FTOL;
    }


  }

  return MAXITER;

}
