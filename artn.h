/*---------------------------------------------------

---------------------------------------------------*/
#ifdef COMMAND_CLASS

CommandStyle(artn,Artn)

#else

#ifndef ARTN_H
#define ARTN_H

#include "min_linesearch.h"
#include <fstream>
#include <string>
#include <sstream>
#include "random_park.h"
using namespace std;
namespace LAMMPS_NS{
class Artn: public MinLineSearch{
  public:
    Artn(class LAMMPS *);
    ~Artn();
    int search(int );
    void command(int, char **);
  private:
    void mysetup();
    void myinit();
    int iterate(int );
    int min_converge(int);
    void store_config(string);
    void store_x();
    int find_saddle();
    void global_random_move();
    void downhill();
    void judgement();
    void myreset_vectors();
    void center(double *, int );
    void reset_coords();
    double myenergy_force();
    void lanczos(bool , int , int);

    int me;
    int vec_count;
    int evalf;

    double eref;
    double eigenvalue;
    double *eigenvector;
    double *x0tmp;
    double *h_old;
    RanPark *random;

    int max_converge_steps;
    // for art
    double temperature;			// Fictive temperature, if negative always reject the event
    int max_num_events;			// Maximum number of events
    int activation_maxiter;		// Maximum number of iteraction for reaching the saddle point
    double increment_size;		// Overall scale for the increment moves
    double force_threhold_perp_rel;	// Threshold for perpendicular relaxation

    // for harmonic well
    double initial_step_size;		// Size of initial displacement
    double basin_factor;		// Factor multiplying Increment_Size for leaving the basin
    int max_perp_moves_basin;		// Maximum number of perpendicular steps leaving basin
    int min_number_ksteps;		// Min. number of ksteps before calling lanczos
    double eigenvalue_threhold;		// Eigenvalue threshold for leaving basin
    int max_iter_basin;			// Maximum number of iteraction for leaving the basin (kter)

    // for lanczos
    int number_lanczos_vectors_H;	// Number of vectors included in lanczos procedure in the Harmonic well
    int number_lanczos_vectors_C;	// Number of vectors included in lanczos procedure in convergence
    double delta_displ_lanczos;		// Step of the numerical derivative of forces in lanczos

    // for convergence
    double exit_force_threhold;		// Threshold for convergence at saddle point
    double prefactor_push_over_saddle;	// Fraction of displacement over the saddle

    // for output
    ofstream out_event_list;
    ofstream out_log;
    string event_list_file;
    string log_file;
    string config_file;
    int file_counter;
};
    
}
#endif
#endif
