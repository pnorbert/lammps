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

DumpStyle(custom/adios,DumpCustomADIOS)

#else

#ifndef LMP_DUMP_CUSTOM_ADIOS_H
#define LMP_DUMP_CUSTOM_ADIOS_H

#include "dump_custom.h"
#include <stdlib.h>
#include <stdint.h>
#include "adios.h"

namespace LAMMPS_NS {

class DumpCustomADIOS : public DumpCustom {
 public:
  DumpCustomADIOS(class LAMMPS *, int, char **);
  virtual ~DumpCustomADIOS();

 protected:

  int64_t fh, gh;         // adios file handle and group handle
  static const char * groupname;   // adios needs a group of variables and a group name
  uint64_t groupsize; // pre-calculate # of bytes written per processor in a step before writing anything
  uint64_t groupTotalSize; // ADIOS buffer size returned by adios_group_size(), valid only if size is > default 16MB ADIOS buffer
  char *filecurrent;  // name of file for this round (with % and * replaced)
  char **columnNames; // list of column names for the atom table (individual list of 'columns' string)

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

E: Dump_modify format string is too short

There are more fields to be dumped in a line of output than your
format string specifies.

E: Could not find dump custom compute ID

Self-explanatory.

E: Could not find dump custom fix ID

Self-explanatory.

E: Dump custom and fix not computed at compatible times

The fix must produce per-atom quantities on timesteps that dump custom
needs them.

E: Could not find dump custom variable name

Self-explanatory.

E: Region ID for dump custom does not exist

Self-explanatory.

*/
