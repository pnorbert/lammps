/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Coffman (IBM)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "dump_custom_adios.h"
#include "atom.h"
#include "force.h"
#include "domain.h"
#include "region.h"
#include "group.h"
#include "input.h"
#include "variable.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "fix.h"
#include "universe.h"
#include "memory.h"
#include "error.h"
#include <stdlib.h>

using namespace LAMMPS_NS;

#define MAX_TEXT_HEADER_SIZE 4096
#define DUMP_BUF_CHUNK_SIZE 16384
#define DUMP_BUF_INCREMENT_SIZE 4096

enum{ID,MOL,TYPE,ELEMENT,MASS,
     X,Y,Z,XS,YS,ZS,XSTRI,YSTRI,ZSTRI,XU,YU,ZU,XUTRI,YUTRI,ZUTRI,
     XSU,YSU,ZSU,XSUTRI,YSUTRI,ZSUTRI,
     IX,IY,IZ,
     VX,VY,VZ,FX,FY,FZ,
     Q,MUX,MUY,MUZ,MU,RADIUS,DIAMETER,
     OMEGAX,OMEGAY,OMEGAZ,ANGMOMX,ANGMOMY,ANGMOMZ,
     TQX,TQY,TQZ,SPIN,ERADIUS,ERVEL,ERFORCE,
     COMPUTE,FIX,VARIABLE};
enum{LT,LE,GT,GE,EQ,NEQ};
enum{INT,DOUBLE,STRING,BIGINT};    // same as in DumpCustom

const char * DumpCustomADIOS::groupname = "custom";   // adios needs a group of variables and a group name
/* ---------------------------------------------------------------------- */

DumpCustomADIOS::DumpCustomADIOS(LAMMPS *lmp, int narg, char **arg) :
  DumpCustom(lmp, narg, arg)
{
    fh = 0;
    gh = 0;
    groupsize = 0;
    groupTotalSize = 0;
    filecurrent = NULL;
    //if (screen) fprintf(screen, "DumpCustomADIOS constructor: nvariable=%d id_variable=%p, variables=%p, nfield=%d, earg=%p\n", nvariable, id_variable, variable, nfield, earg);
    columnNames = new char*[nfield];
    for (int i = 0; i < nfield; ++i) {
        columnNames[i] = strndup(earg[i], 128);
        //if (screen) fprintf(screen, "earg[%d] = '%s'\n", i, earg[i]);
    }
}

/* ---------------------------------------------------------------------- */

DumpCustomADIOS::~DumpCustomADIOS()
{
    adios_free_group(gh);
    for (int i = 0; i < size_one; ++i) {
        delete [] columnNames[i];
    }
    delete [] columnNames;
}

/* ---------------------------------------------------------------------- */

void DumpCustomADIOS::openfile()
{
    // if one file per timestep, replace '*' with current timestep
    filecurrent = filename;

    if (multifile) {
        char *filestar = filecurrent;
        filecurrent = new char[strlen(filestar) + 16];
        char *ptr = strchr(filestar,'*');
        *ptr = '\0';
        if (padflag == 0)
            sprintf(filecurrent,"%s" BIGINT_FORMAT "%s",
                    filestar,update->ntimestep,ptr+1);
        else {
            char bif[8],pad[16];
            strcpy(bif,BIGINT_FORMAT);
            sprintf(pad,"%%s%%0%d%s%%s",padflag,&bif[1]);
            sprintf(filecurrent,pad,filestar,update->ntimestep,ptr+1);
        }
        *ptr = '*';
    }
    char fmode[] = "w";
    if ( append_flag || (!multifile && singlefile_opened)) {
        fmode[0] = 'a';
    }
    int err = adios_open (&fh, groupname, filecurrent, fmode, world);
    if (err != MPI_SUCCESS) {
        char str[128];
        sprintf(str,"Cannot open dump file %s",filecurrent);
        error->one(FLERR,str);
    }
    adios_group_size(fh, groupsize, &groupTotalSize);
    if (multifile == 0) singlefile_opened = 1; // to append to file at next open
}

/* ---------------------------------------------------------------------- */

void DumpCustomADIOS::write()
{
    if (domain->triclinic == 0) {
        boxxlo = domain->boxlo[0];
        boxxhi = domain->boxhi[0];
        boxylo = domain->boxlo[1];
        boxyhi = domain->boxhi[1];
        boxzlo = domain->boxlo[2];
        boxzhi = domain->boxhi[2];
    } else {
        boxxlo = domain->boxlo_bound[0];
        boxxhi = domain->boxhi_bound[0];
        boxylo = domain->boxlo_bound[1];
        boxyhi = domain->boxhi_bound[1];
        boxzlo = domain->boxlo_bound[2];
        boxzhi = domain->boxhi_bound[2];
        boxxy = domain->xy;
        boxxz = domain->xz;
        boxyz = domain->yz;
    }

    // nme = # of dump lines this proc contributes to dump

    nme = count();

    // ntotal = total # of atoms in snapshot
    // atomOffset = sum of # of atoms up to this proc (exclusive prefix sum)

    bigint bnme = nme;
    MPI_Allreduce(&bnme,&ntotal,1,MPI_LMP_BIGINT,MPI_SUM,world);

    bigint atomOffset; // sum of all atoms on processes 0..me-1
    MPI_Scan (&bnme, &atomOffset, 1, MPI_LMP_BIGINT, MPI_SUM, world);
    atomOffset -= nme; // exclusive prefix sum needed

    // insure filewriter proc can receive everyone's info
    // limit nmax*size_one to int since used as arg in MPI_Rsend() below
    // pack my data into buf
    // if sorting on IDs also request ID list from pack()
    // sort buf as needed

    if (nme > maxbuf) {
        if ((bigint) nme * size_one > MAXSMALLINT)
            error->all(FLERR,"Too much per-proc info for dump");
        maxbuf = nme;
        memory->destroy(buf);
        memory->create(buf,(maxbuf*size_one),"dump:buf");
    }
    if (sort_flag && sortcol == 0 && nme > maxids) {
        maxids = nme;
        memory->destroy(ids);
        memory->create(ids,maxids,"dump:ids");
    }

    if (sort_flag && sortcol == 0) pack(ids);
    else pack(NULL);
    if (sort_flag) sort();

    // Calculate data size written by this process
    groupsize = nme * size_one * sizeof(double); // size of atoms data on this process
    groupsize += 3*sizeof(uint64_t) + 1*sizeof(int); // scalars written by each process
    if (me == 0) {
        groupsize += 1*sizeof(uint64_t) + 1*sizeof(int) + 6*sizeof(double); // scalars
        if (domain->triclinic) {
            groupsize += 3*sizeof(double); // boxxy, boxxz, boxyz
        }
    }

    openfile();

    // write info on data as scalars (by me==0)
    if (me == 0) {
        adios_write (fh, "ntimestep",   &update->ntimestep);
        adios_write (fh, "nprocs",      &nprocs);

        adios_write (fh, "boxxlo", &boxxlo);
        adios_write (fh, "boxxhi", &boxxhi);
        adios_write (fh, "boxylo", &boxylo);
        adios_write (fh, "boxyhi", &boxyhi);
        adios_write (fh, "boxzlo", &boxzlo);
        adios_write (fh, "boxzhi", &boxzhi);

        if (domain->triclinic) {
            adios_write (fh, "boxxy", &boxxy);
            adios_write (fh, "boxxz", &boxxz);
            adios_write (fh, "boxyz", &boxyz);
        }
    }
    // Everyone needs to write scalar variables that are used as dimensions and offsets of arrays
    adios_write (fh, "natoms",   &ntotal);
    adios_write (fh, "ncolumns", &size_one);
    adios_write (fh, "nme",      &bnme);
    adios_write (fh, "offset",   &atomOffset);
    // now write the atoms
    adios_write (fh, "atoms",    buf);

    adios_close(fh); // I/O will happen now...
    if (multifile) delete [] filecurrent;
}

/* ---------------------------------------------------------------------- */

void DumpCustomADIOS::init_style()
{

    // setup boundary string

    domain->boundary_string(boundstr);

    // remove % from filename since ADIOS always writes a global file with data/metadata
    filecurrent = filename;
    int len = strlen(filename);
    char *ptr = strchr(filename,'%');
    if (ptr) {
        *ptr = '\0';
        char *s = new char[len-1];
        sprintf(s,"%s%s",filename,ptr+1);
        strncpy(filename,s,len);
    }

    /* The next four loops are copied from dump_custom_mpiio, but nothing is done with them.
     * It is unclear why we need them here.
     * For metadata, variable[] will be written out as an ADIOS attribute if nvariable>0
     */
    // find current ptr for each compute,fix,variable
    // check that fix frequency is acceptable
    int icompute;
    for (int i = 0; i < ncompute; i++) {
        icompute = modify->find_compute(id_compute[i]);
        if (icompute < 0) error->all(FLERR,"Could not find dump custom compute ID");
        compute[i] = modify->compute[icompute];
    }

    int ifix;
    for (int i = 0; i < nfix; i++) {
        ifix = modify->find_fix(id_fix[i]);
        if (ifix < 0) error->all(FLERR,"Could not find dump custom fix ID");
        fix[i] = modify->fix[ifix];
        if (nevery % modify->fix[ifix]->peratom_freq)
            error->all(FLERR,"Dump custom and fix not computed at compatible times");
    }

    int ivariable;
    for (int i = 0; i < nvariable; i++) {
        ivariable = input->variable->find(id_variable[i]);
        if (ivariable < 0)
            error->all(FLERR,"Could not find dump custom variable name");
        variable[i] = ivariable;
    }

    // set index and check validity of region
    if (iregion >= 0) {
        iregion = domain->find_region(idregion);
        if (iregion == -1)
            error->all(FLERR,"Region ID for dump custom does not exist");
    }

    /* Define the group of variables for the atom style here since it's a fixed set */
    adios_declare_group(&gh, groupname, NULL, adios_flag_yes);
    if (multiproc == nprocs || multiproc == 0) {
        adios_select_method (gh, "MPI", "", "");
        if (me==0 && screen) fprintf(screen, "ADIOS method for %s is n-to-1 (MPI method)\n", filename);
    } else {
        int num_aggregators = multiproc;
        if (num_aggregators == 0)
            num_aggregators = 1;
        char opts[128];
        sprintf (opts, "num_aggregators=%d;num_ost=%d;striping=0", multiproc,multiproc);
        adios_select_method (gh, "MPI_AGGREGATE", opts, "");
        if (me==0 && screen) fprintf(screen, "ADIOS method for %s is n-to-m (aggregation with %d writers)\n", filename, multiproc);
    }

    adios_define_var (gh, "ntimestep","", adios_long, NULL, NULL, NULL);
    adios_define_var (gh, "natoms","",    adios_long, NULL, NULL, NULL);

    adios_define_var (gh, "nprocs","",    adios_integer, NULL, NULL, NULL);
    adios_define_var (gh, "ncolumns","",  adios_integer, NULL, NULL, NULL);

    adios_define_var (gh, "boxxlo","", adios_double, NULL, NULL, NULL);
    adios_define_var (gh, "boxxhi","", adios_double, NULL, NULL, NULL);
    adios_define_var (gh, "boxylo","", adios_double, NULL, NULL, NULL);
    adios_define_var (gh, "boxyhi","", adios_double, NULL, NULL, NULL);
    adios_define_var (gh, "boxzlo","", adios_double, NULL, NULL, NULL);
    adios_define_var (gh, "boxzhi","", adios_double, NULL, NULL, NULL);

    adios_define_var (gh, "boxxy","",  adios_double, NULL, NULL, NULL);
    adios_define_var (gh, "boxxz","",  adios_double, NULL, NULL, NULL);
    adios_define_var (gh, "boxyz","",  adios_double, NULL, NULL, NULL);

    adios_define_attribute_byvalue (gh, "triclinic", "",  adios_integer, 1,  &domain->triclinic);
    adios_define_attribute_byvalue (gh, "boundary", "",   adios_integer, 6, domain->boundary);
    adios_define_attribute_byvalue (gh, "columns", "",  adios_string_array, size_one, columnNames);
    //if (screen) fprintf(screen, "ADIOS nvariable=%d id_variable=%p, variables=%p, nfield=%d, earg=%p\n", nvariable, id_variable, variable, nfield, earg);
    if (nvariable) {
        adios_define_attribute_byvalue (gh, "variable_names", "",  adios_string_array, nvariable, id_variable);
        adios_define_attribute_byvalue (gh, "variable_ids", "",    adios_integer, nvariable, variable);
    }

    adios_define_attribute (gh, "columnstr", "",         adios_string, columns, NULL);
    adios_define_attribute (gh, "boundarystr", "",       adios_string, boundstr, NULL);
    adios_define_attribute (gh, "LAMMPS/dump_style", "", adios_string, "custom", NULL);
    adios_define_attribute (gh, "LAMMPS/version", "",    adios_string, universe->version, NULL);
    adios_define_attribute (gh, "LAMMPS/num_ver", "",    adios_string, universe->num_ver, NULL);

    adios_define_var (gh, "nme","",    adios_long, NULL, NULL, NULL); // local dimension variable
    adios_define_var (gh, "offset","", adios_long, NULL, NULL, NULL); // local dimension variable
    adios_define_var (gh, "atoms","",  adios_double, "nme,ncolumns", "natoms,ncolumns", "offset,0");

}
