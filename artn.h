/*---------------------------------------------------

---------------------------------------------------*/
#ifdef COMMAND_CLASS

CommandStyle(artn,Artn)

#else

#ifndef ARTN_H
#define ARTN_H

#include "min_linesearch.h"
namespace LAMMPS_NS{
class Artn: public MinLineSearch{
  public:
    Artn(class LAMMPS *);
    ~Artn();
    void command(int , char **);
    int search(int );
  private:
    void mysetup();
    void myinit();
    int min_converge(int);
    void store_x();
    void find_saddle();
    void downhill();
    void judgement();

};
    
}
#endif
#endif
