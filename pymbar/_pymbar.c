/*
  C helper module for time-critical MBAR routines.
#=============================================================================================
# VERSION CONTROL INFORMATION
#=============================================================================================
__version__ = "$Revision: $ $Date: $"
# $Date: $
# $Revision: $
# $LastChangedBy: $
# $HeadURL: $
# $Id: $

#=============================================================================================

#=============================================================================================
# COPYRIGHT NOTICE
#
# Written by Michael R. Shirts <mrshirts@gmail.com> and John D. Chodera <jchodera@gmail.com>.
#
# Copyright (c) 2006-2007 The Regents of the University of California.  All Rights Reserved.
# Portions of this software are Copyright (c) 2007-2008 Stanford University and Columbia University.
# Portions of this software are Copyright (c) 2014 University of Virginia
# Portions of this software are Copyright (c) 2017 University of Colorado Boulder, Memorial Sloan Kettering Cancer Center
#
# This program is free software; you can redistribute it and/or modify it under the terms of
# the MIT License.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the MIT License for more details.
# 
# You should have received a copy of the MIT License along with this program;
#=============================================================================================

#=============================================================================================
# INSTALLATION INSTRUCTIONS
#=============================================================================================

  To compile on Mac OS X: 

  gcc -O3 -lm -bundle -I(directory with Python.h) -I(directory with numpy/arrayobject.h) _pymbar.c -o _pymbar.so -undefined dynamic_lookup

  For Python 2.6 and numpy installed via MacPorts:

  gcc -O3 -lm -bundle -I/opt/local/Library/Frameworks/Python.framework/Versions/2.6/include/python2.6 -I/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/numpy/core/include _pymbar.c -o _pymbar.so -undefined dynamic_lookup

  For Mac OS X system Python 2.5 with system numpy:
 
  gcc -O3 -lm -bundle -I/System/Library/Frameworks/Python.framework/Versions/2.5/include/python2.5 -I/System/Library/Frameworks/Python.framework/Versions/2.5/Extras/lib/python/numpy/core/include/ _pymbar.c -o _pymbar.so -undefined dynamic_lookup

  For example, for Python 2.4 installed via fink:

  gcc -O3 -lm -bundle -I/sw/include/python2.4 -I/sw/lib/python2.4/site-packages/numpy/core/include/ _pymbar.c -o _pymbar.so -undefined dynamic_lookup

  For Python 2.4 and numpy installed by MacPorts:

  gcc -O3 -lm -bundle -I/opt/local/Library/Frameworks/Python.framework/Versions/2.4/include/python2.4/ -I/opt/local/var/macports/software/py-numpy/1.0.3_0/opt/local/lib/python2.4/site-packages/numpy/core/include/ _pymbar.c -o _pymbar.so -undefined dynamic_lookup

  For Python 2.7 installed in anaconda

  gcc -O3 -lm -bundle -I${HOME}/anaconda/include/python2.7/ -I${HOME}/anaconda/lib/python2.7/site-packages/numpy/core/include/ _pymbar.c -o _pymbar.so -undefined dynamic_lookup
  * * *

  To compile on Linux:

  gcc -O3 -lm -fPIC -shared -I(directory with Python.h) -I(directory with numpy/arrayobject.h) -o _pymbar.so _pymbar.c

  For example, for a default installation of python 2.5:

  gcc -O3 -lm -fPIC -shared -I/usr/local/include/python2.5 -I/usr/local/lib/python2.5/site-packages/numpy/core/include -o _pymbar.so _pymbar.c

  On NCSA Lincoln:

  gcc -O3 -lm -fPIC -shared -I/usr/local/Python-2.5.2/include/python2.5/ -I/usr/local/Python-2.5.2/lib/python2.5/site-packages//numpy/core/include/ -o _pymbar.so _pymbar.c

  On NCSA Forge:

  gcc -O3 -lm -fPIC -shared -I$HOME/epd-7.1-2-rh5-x86_64/include/python2.7/ -I$HOME/epd-7.1-2-rh5-x86_64/lib/python2.7/site-packages/numpy/core/include/ -o _pymbar.so _pymbar.c

*/

#include "Python.h"
#include "numpy/arrayobject.h"

PyObject *_pymbar_computeUnnormalizedLogWeightsCpp(PyObject *self, PyObject *args) {
  
  int i,k,n,K,N,nonzero_N;
  int s0,s1;
  npy_intp dim1[1];
  int *nonzero_N_k,*N_k;
  double *FlogN,*f_k,*log_term,*u_n,*log_w_n;
  double u_k, max_log_term, term_sum, log_sum;

  PyArrayObject *array_nonzero_N_k, *array_N_k, *array_f_k, *array_u_kn, *array_u_n, *array_log_w_n;

  if (!PyArg_ParseTuple(args, "iiiOOOOO",
			&K, &N, &nonzero_N, &array_nonzero_N_k,
			&array_N_k, &array_f_k, 
			&array_u_kn, &array_u_n)) {
    return NULL;
  }

  //-------------------------------------------
  //Set the dimensions Python array of log_w_n
  //-------------------------------------------

  dim1[0] = N;

  //-------------------------------------------
  //Create Python array of log_w_n
  //-------------------------------------------

  array_log_w_n = (PyArrayObject *) PyArray_SimpleNew(1, dim1, PyArray_DOUBLE);

  //-------------------------------------------
  //Make C arrays from python single-dimension numeric arrays
  //-------------------------------------------
  
  nonzero_N_k  = (int *) array_nonzero_N_k->data;
  N_k          = (int *) array_N_k->data;
  f_k          = (double *) array_f_k->data;
  u_n          = (double *) array_u_n->data;
  log_w_n      = (double *) array_log_w_n->data;

  //-------------------------------------------
  // Allocate space for helper arrays
  //-------------------------------------------

  FlogN = malloc(nonzero_N*sizeof(double));
  log_term = malloc(nonzero_N*sizeof(double));  
  
  //-------------------------------------------
  // Precalculate some constant terms
  //-------------------------------------------

  for (i=0;i<nonzero_N;i++) {
      k = nonzero_N_k[i];
      FlogN[i] = log((double)N_k[k])+f_k[k];
  }

  s0 = array_u_kn->strides[0];
  s1 = array_u_kn->strides[1];

  //-------------------------------------------
  // The workhorse double loop
  //-------------------------------------------  

  for (n=0;n<N;n++) {
      term_sum = 0;
      max_log_term = -1e300; // very low number
      //-------------------------------------------
      // Sum over only nonzero terms
      //------------------------------------------- 
      for (i=0;i<nonzero_N;i++) {
          k = nonzero_N_k[i];
          u_k = *((double *)(array_u_kn->data + k*s0 + n*s1));
          //------------------------------------------------------------------------
          // Heart of the calculation -- sum_k over log(N_k) + f_k - (u_kn + u_n)
          //------------------------------------------------------------------------
          log_term[i] = FlogN[i] - (u_k - u_n[n]);
          if (log_term[i] > max_log_term) {max_log_term = log_term[i];}
      }
      //----------------------------------------------
      // subtract off the maximum, to prevent overflow
      //----------------------------------------------
      for (i=0;i<nonzero_N;i++) {
          term_sum += exp(log_term[i]-max_log_term);
      }
      log_sum = log(term_sum) + max_log_term;
      log_w_n[n] = -log_sum;
  }
  
  //-------------------------------------------
  // Free the temporary helper arrays
  //-------------------------------------------  

  free(FlogN);
  free(log_term);

  return PyArray_Return(array_log_w_n);
}

static PyMethodDef _pymbar_methods[] = {
  {"computeUnnormalizedLogWeightsCpp", (PyCFunction)_pymbar_computeUnnormalizedLogWeightsCpp, METH_VARARGS, "Computes unnormalized log weights via compiled C++, computeUnnormalizedLogWeightsCpp(K,N,nonzero_N,f_k,u_kn)"},
  {NULL, NULL, 0, NULL}
};

DL_EXPORT(void) init_pymbar(void)
{
  Py_InitModule3("_pymbar", _pymbar_methods, "Computes unnormalized log weights via compiled C++.\n");   
  import_array();
}
