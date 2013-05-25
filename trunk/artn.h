/*---------------------------------------------------

---------------------------------------------------*/
#ifdef COMMAND_CLASS

CommandStyle(artn,ARTn)

#else

#ifndef ARTN_H
#define ARTN_H

#include "min_linesearch.h"
#include <fstream>
#include <string>
#include <sstream>
#include "random_park.h"
#include "dump_atom.h"
using namespace std;
namespace LAMMPS_NS{
//class ARTn_dump: public DumpAtom{
//  public:
//    ARTn_dump(LAMMPS *lmp, int narg, char**arg):DumpAtom(lmp, narg, arg){};
//    void modify_file(string file){
//      if (me == 0){
//	fclose(fp);
//	fp = fopen(file.c_str(),"w");
//      }
//    };
//};

class ARTn: public MinLineSearch{
public:
    ARTn(class LAMMPS *);
    ~ARTn();
    int search(int );
    void command(int, char **);

private:
    void mysetup();			
    void myinit();
    int iterate(int );
    int min_converge(int);
    void store_config(string);
    int find_saddle();
    void read_config();
    void global_random_move();
    void group_random_move();
    void local_random_move();
    void downhill();
    void judgement();
    void myreset_vectors();
    void center(double *, int );
    void reset_coords();
    void lanczos(bool , int , int);
    int min_perpendicular_fire(int );
    //ARTn_dump * mydump;
    DumpAtom * dumpmin;
    DumpAtom * dumpsadl;
    inline void outlog(char *tmp){if (me == 0) out_log << tmp << flush ;}
    inline void outeven(char *tmp){if (me == 0) out_event_list << tmp << flush;}

    int me;
    int vec_count;
    int evalf;
    bool eigen_vector_exist;

    double eref;
  
    double eigenvalue;
    double *eigenvector;
    double *x0tmp;
    double *x00;
    double *h_old;
    double *vvec;
    double *fperp;

    RanPark *random;

    int max_converge_steps;
    // for art
    double temperature;			// Fictive temperature, if negative always reject the event
    int max_num_events;			// Maximum number of events
    int activation_maxiter;		// Maximum number of iteraction for reaching the saddle point
    double increment_size;		// Overall scale for the increment moves
    double force_threhold_perp_rel;	// Threshold for perpendicular relaxation
    bool group_random;			// do group random move away from minimum
    bool local_random;			// do local random move away from minimum
    bool fire_on;			// use FIRE to do minimuzation in the perpendicular direction

    // for harmonic well
    double initial_step_size;		// Size of initial displacement
    double basin_factor;		// Factor multiplying Increment_Size for leaving the basin
    int max_perp_moves_basin;		// Maximum number of perpendicular steps leaving basin
    int min_number_ksteps;		// Min. number of ksteps before calling lanczos
    double eigenvalue_threhold;		// Eigenvalue threshold for leaving basin
    int max_iter_basin;			// Maximum number of iteraction for leaving the basin (kter)
    double force_threhold_perp_h;

    // for lanczos
    int number_lanczos_vectors_H;	// Number of vectors included in lanczos procedure in the Harmonic well
    int number_lanczos_vectors_C;	// Number of vectors included in lanczos procedure in convergence
    double delta_displ_lanczos;		// Step of the numerical derivative of forces in lanczos
    double eigen_threhold;

    // for convergence
    double exit_force_threhold;		// Threshold for convergence at saddle point
    double prefactor_push_over_saddle;	// Fraction of displacement over the saddle
    double eigen_fail;			// the eigen cutoff for failing in searching saddle point
    double max_perp_moves_C;		
    double force_threhold_perp_rel_C;

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

