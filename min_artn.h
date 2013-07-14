#ifdef MINIMIZE_CLASS

MinimizeStyle(artn,MinARTn)

#else

#ifndef ARTN_H
#define ARTN_H

#include "dump_atom.h"
#include "random_park.h"
#include "min_linesearch.h"

using namespace std;

namespace LAMMPS_NS {

class MinARTn: public MinLineSearch {
public:
  MinARTn(class LAMMPS *);
  int iterate(int);

private:
  void set_defaults();			
  void artn_init();
  void read_control();

  void random_kick();

  int find_saddle();
  int min_converge(int);
  int sad_converge(int);

  int push_back_sad();
  void push_down();
  void check_new_min();

  void artn_reset_vec();
  void reset_coords();
  void lanczos(bool, int, int);
  int min_perp_fire(int);

  int sad_found;
  int ref_id, min_id, sad_id, ref_0;
  DumpAtom * dumpmin;
  DumpAtom * dumpsad;
  Compute *pressure;

  int me, np;
  bigint evalf;
  bool flag_egvec;

  double eref;

  double egval;
  double *egvec;
  double *x0tmp;
  double *x00;
  double *htmp;
  double *x_sad;
  double *h_old;
  double *vvec;
  double *fperp;

  int seed;
  RanPark *random;

  // global control
  int max_conv_steps;
  double temperature;      // Fictive temperature, if negative always reject the event

  // for art
  int nattempt, stage;
  int max_num_events;      // Maximum number of events
  int max_activat_iter;    // Maximum number of iteractions for reaching the saddle point
  double increment_size;   // Overall scale for the increment moves
  int use_fire;            // use FIRE to do minimuzation in the perpendicular direction
  int flag_push_back;      // if 1, will push back saddle point to check if it connect with the minimum
  int flag_relax_sad;      // further relax to the newly found saddle
  double max_disp_tol;     // tolerance displacement between ref and push-back that can claim saddle is indeed linked to ref
  int flag_press;          // Pressure will be calculated.
  double atom_disp_thr;    // threshold to identify whether an atom is displaced or not

  int groupbit, ngroup;    // group bit & # of atoms for initial kick
  int that, *glist;        // ID and list for kick
  char *groupname;         // group name for initial kick
  double cluster_radius;   // radius for kick; <0, all; ==0, single; >0, cluster
  double *delpos;          // initial kick

  // for harmonic well
  double init_step_size;   // Size of initial displacement
  double basin_factor;     // Factor multiplying Increment_Size for leaving the basin
  int max_perp_move_h;     // Maximum number of perpendicular steps leaving basin
  int min_num_ksteps;      // Min. number of ksteps before calling lanczos
  double eigen_th_well;    // Eigenvalue threshold for leaving basin
  int max_iter_basin;      // Maximum number of iteraction for leaving the basin (kter)
  double force_th_perp_h;  // Perpendicular force threhold in harmonic well

  // for lanczos
  int num_lancz_vec_h;     // Number of vectors included in lanczos procedure in the Harmonic well
  int num_lancz_vec_c;     // Number of vectors included in lanczos procedure in convergence
  double del_disp_lancz;   // Step of the numerical derivative of forces in lanczos
  double eigen_th_lancz;   // Eigen_threhold for lanczos convergence

  // for convergence
  double force_th_saddle;  // Threshold for convergence at saddle point
  double push_over_saddle; // Fraction of displacement over the saddle
  double eigen_th_fail;    // the eigen cutoff for failing in searching saddle point
  int    max_perp_moves_c; // Maximum number of perpendicular steps approaching saddle point
  double force_th_perp_sad;// Perpendicular force threhold approaching saddle point

  // for output
  FILE *fp1, *fp2;
  char *flog, *fevent, *fconfg;
  int log_level;           // 1, all; 0, main
  int print_freq;          // default 1

  void artn_final();
  void write_header(const int);

};

}
#endif
#endif
