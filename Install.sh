# Install/unInstall package classes in LAMMPS

if (test $1 = 1) then

  cp -p artn.h ..
  cp -p artn.cpp ..

elif (test $1 = 0) then

  rm ../artn.h
  rm ../artn.cpp

fi
