/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef DUMP_CLASS

DumpStyle(atom/adios,DumpAtomADIOS)

#else

#ifndef LMP_DUMP_ATOM_ADIOS_H
#define LMP_DUMP_ATOM_ADIOS_H

#include "dump_atom.h"
#include <stdlib.h>
#include <stdint.h>
#include "adios.h"

namespace LAMMPS_NS {

class DumpAtomADIOS : public DumpAtom {

 public:
  DumpAtomADIOS(class LAMMPS *, int, char **);
  virtual ~DumpAtomADIOS();

 protected:

  int64_t fh, gh;         // adios file handle and group handle
  static const char * groupname;   // adios needs a group of variables and a group name
  uint64_t groupsize; // pre-calculate # of bytes written per processor in a step before writing anything
  char *filecurrent;  // name of file for this round (with % and * replaced)

  virtual void openfile();
  virtual void write();
  virtual void init_style();

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot open dump file %s

The output file for the dump command cannot be opened.  Check that the
path and name are correct.

E: Too much per-proc info for dump

Number of local atoms times number of columns must fit in a 32-bit
integer for dump.

*/
