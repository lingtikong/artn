#ifdef COMMAND_CLASS

CommandStyle(artn,ARTn)

#else

#ifndef ARTN_H
#define ARTN_H

#include "min_linesearch.h"
#include "random_park.h"
#include "dump_atom.h"

using namespace std;

namespace LAMMPS_NS {

class ARTn: public MinLineSearch {
public:
  ARTn(class LAMMPS *);
  ~ARTn();

  int search(int );
  void command(int, char **);

private:
  void set_defaults();			
  void artn_init();

  int iterate(int );
  int min_converge(int);
  int find_saddle();
  void read_control();

  void random_kick();

  int check_saddle_min();
  void push_down();
  void check_new_min();

  void myreset_vectors();
  void reset_coords();
  void lanczos(bool , int , int);
  int min_perpendicular_fire(int );

  int ref_id, min_id, sad_id;
  DumpAtom * dumpmin;
  DumpAtom * dumpsad;
  Compute *pressure;

  int me, np;
  int vec_count;
  int evalf;
  bool eigen_vec_exist;

  double eref;

  double eigenvalue;
  double *eigenvector;
  double *x0tmp;
  double *x00;
  double *htmp;
  double *x_saddle;
  double *h_old;
  double *vvec;
  double *fperp;

  int seed;
  RanPark *random;

  int max_conv_steps;

  // for art
  double temperature;      // Fictive temperature, if negative always reject the event
  int max_num_events;      // Maximum number of events
  int max_activat_iter;    // Maximum number of iteractions for reaching the saddle point
  double increment_size;   // Overall scale for the increment moves
  int use_fire;            // use FIRE to do minimuzation in the perpendicular direction
  int flag_check_sad;      // Push back saddle point to check if it connect with the minimum
  double max_disp_tol;     // tolerance to claim as linked saddle
  int pressure_needed;     // Pressure will be calculated.
  double atom_move_cutoff; // cutoff to decide whether an atom is moved

  int groupbit, ngroup;    // group bit & # of atoms for initial kick
  int that, *glist;        // ID and list for kick
  char *groupname;         // group name for initial kick
  double kick_radius;      // radius for kick; <0, all; ==0, single; >0, cluster
  double *delpos;

  // for harmonic well
  double init_step_size;   // Size of initial displacement
  double basin_factor;     // Factor multiplying Increment_Size for leaving the basin
  int max_perp_move_h;     // Maximum number of perpendicular steps leaving basin
  int min_num_ksteps;      // Min. number of ksteps before calling lanczos
  double eigen_th_well;    // Eigenvalue threshold for leaving basin
  int max_iter_basin;      // Maximum number of iteraction for leaving the basin (kter)
  double force_th_perp_h;  // Perpendicular force threhold in harmonic well

  // for lanczos
  int num_lancz_vec_H;     // Number of vectors included in lanczos procedure in the Harmonic well
  int num_lancz_vec_C;     // Number of vectors included in lanczos procedure in convergence
  double del_disp_lancz;   // Step of the numerical derivative of forces in lanczos
  double eigen_th_lancz;   // Eigen_threhold for lanczos convergence

  // for convergence
  double force_th_saddle;  // Threshold for convergence at saddle point
  double push_over_saddle; // Fraction of displacement over the saddle
  double eigen_th_fail;    // the eigen cutoff for failing in searching saddle point
  double max_perp_moves_C; // Maximum number of perpendicular steps approaching saddle point
  double force_th_perp_sad;// Perpendicular force threhold approaching saddle point

  // for input
  char *fctrl;

  // for output
  FILE *fp1, *fp2;
  char *flog, *fevent, *fconfg;
};

}
#endif
#endif
