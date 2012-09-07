/*--------------------------------------------------
  the command for artn like this:

  artn maxevent temperature etol ftol  maxeval

--------------------------------------------------*/
#ifdef COMMAND_CLASS

CommandStyle(artn,Artn)

#else

#ifndef ARTN_H
#define ARTN_H

#include "pointer.h"

namespace LAMMPS_NS {

class Artn: public MinLineSearch{
  public:
    Artn(class LAMMPS *);
    ~Artn();
    
    int ievent;
    int nstep;
    double temperature;
    double prewell_energy;
    bool newevent;
    int maxevent;
    double *initial_direction;
    double *direction;
    double *pre_direction;
    double *prewell_x;

    void modify_params(int, char **);
    void command(int, char **);
    void search(int);
    RanPark *random;

  protected:
    double *fperp;
    double *fpar;
    double *eigenvector;
    double eigenvalue;
    double *prefperp;
    double *prexvec;
    double *prefvec;
    double preenergy;
    double step;
    void lanczos();
    void findsaddle();
    void output();
    void local_move();
    void globle_move();
    void downhill();
    void center(double *, int);
}

}
#endif
#endif
