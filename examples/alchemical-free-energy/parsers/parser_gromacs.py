import numpy
import os                       # for os interface
import re                       # for regular expressions
from glob import glob           # for pathname matching
from collections import Counter # for counting elements in an array

import unixlike                 # some implemented unixlike commands

#===================================================================================================
# FUNCTIONS: This is the Gromacs dhdl.xvg file parser.
#===================================================================================================

def readDataGromacs(P):
   """Read in .xvg files; return nsnapshots, lv, dhdlt, and u_klt."""
   
   class F:
      """This is the object to be built on the filename."""
 
      def __init__(self, filename):
         self.filename = filename
 
      def sortedHelper(self):
         """This function will assist the built-in 'sorted' to sort filenames.
            Returns a tuple whose first element is an integer while others are strings."""
         meat = os.path.basename(self.filename).replace(P.prefix, '').replace(P.suffix, '')
         l = [i for i in re.split('\.|-|_', meat) if i]
         try:
            self.state = l[0] = int(l[0]) # Will be of use for selective MBAR analysis.
         except:
            print "\nERROR!\nFile's prefix should be followed by a numerical character. Cannot sort the files.\n"
            raise
         return tuple(l)
 
      def readHeader(self):
         self.skip_lines = 0  # Number of lines from the top that are to be skipped.
         self.lv_names   = () # Lambda type names, e.g. 'coul', 'vdw'.
         snap_size       = [] # Time from first two snapshots to determine snapshot's size.
         self.lv         = [] # Lambda vectors, e.g. (0, 0), (0.2, 0), (0.5, 0).
 
         self.bEnergy    = False
         self.bPV        = False
         self.bExpanded  = False
 
         print "Reading metadata from %s..." % self.filename
         with open(self.filename,'r') as infile:
            for line in infile:
 
               if line.startswith('#'):
                  self.skip_lines += 1
 
               elif line.startswith('@'):
                  self.skip_lines += 1
                  elements = unixlike.trPy(line).split()
                  if not 'legend' in elements:
                     continue
 
                  if 'Energy' in elements:
                     self.bEnergy = True
                  if 'pV' in elements:
                     self.bPV = True
                  if 'state' in elements:
                     self.bExpanded = True
 
                  if 'dH' in elements:
                     self.lv_names += elements[7],
                  if 'xD' in elements:
                     self.lv.append(elements[-len(self.lv_names):])
 
               else:
                  snap_size.append(float(line.split()[0]))
                  if len(snap_size) > 1:
                     self.snap_size = numpy.diff(snap_size)[0]
                     P.snap_size.append(self.snap_size)
                     break
         return self.lv
 
      def iter_loadtxt(self, state):
         """Houstonian Joe Kington claims it is faster than numpy.loadtxt:
         http://stackoverflow.com/questions/8956832/python-out-of-memory-on-large-csv-file-numpy"""
         
         def iter_func():
            with open(self.filename, 'r') as infile:
               for _ in range(self.skip_lines):
                  next(infile)
               for line in infile:
                  line = line.split()
                  for item in line:
                     yield item
 
         def slice_data(data, state=state):
            # Where the dE columns should be stored.
            if (len(ndE_unique)>1 and ndE[state]<4):
               # If BAR, store shifted 2/3 arrays.
               s1, s2 = numpy.array((0, ndE[state])) + state-(state>0)
            else:
               # If MBAR or selective MBAR or BAR/MBAR, store all.
               s1, s2 = (0, K)
            # Which dhdl columns are to be read.
            read_dhdl_sta = 1+self.bEnergy+self.bExpanded
            read_dhdl_end = read_dhdl_sta + n_components
  
            data = data.T
            dhdlt[state, :, :nsnapshots[state]] = data[read_dhdl_sta : read_dhdl_end, :]
  
            if not bSelective_MBAR:
               r1, r2 = (read_dhdl_end, read_dhdl_end+K)
               if bPV:
                  u_klt[state, s1:s2, :nsnapshots[state]] = P.beta * ( data[r1:r2, :] + data[-1,:] )
               else:
                  u_klt[state, s1:s2, :nsnapshots[state]] = P.beta * data[r1:r2, :]
            else: # can't do slicing; prepare a mask (slicing is thought to be faster/less memory consuming than masking)
               mask_read_uklt = numpy.array( [0]*read_dhdl_end + [1 if (k in sel_states) else 0 for k in range(ndE[0])] + ([0] if bPV else []), bool )
               if bPV:
                  u_klt[state, s1:s2, :nsnapshots[state]] = P.beta * ( data[mask_read_uklt, :] + data[-1,:] )
               else:
                  u_klt[state, s1:s2, :nsnapshots[state]] = P.beta * data[mask_read_uklt, :]
            return
 
         print "Loading in data from %s (%s) ..." % (self.filename, "all states" if self.bExpanded else 'state %d' % state)
         data = numpy.fromiter(iter_func(), dtype=float)
         if not self.len_first == self.len_last:
            data = data[: -self.len_last]
         data = data.reshape((-1, self.len_first))
 
         if self.bExpanded:
            for k in range(K):
               mask_k = (data[:, 1] == k)
               data_k = data[mask_k]
               slice_data(data_k, k)
         else:
            slice_data(data)
 
      def parseLog(self):
         """By parsing the .log file of the expanded-ensemble simulation
         find out the time in ps when the WL equilibration has been reached.
         Return the greater of WLequiltime and equiltime."""
         if not(P.bIgnoreWL):
            logfilename = self.filename.replace('.xvg', '.log')
            if not os.path.isfile(logfilename):
               raise SystemExit("\nERROR!\nThe .log file '%s' is needed to figure out when the Wang-Landau weights have been equilibrated, and it was not found.\nYou may rerun with the -x flag and the data will be discarded to 'equiltime', not bothering\nwith the extraction of the information on when the WL weights equilibration was reached.\nOtherwise, put the proper log file into the directory which is subject to the analysis." % logfilename)
            try:
               with open(logfilename, 'r') as infile:
                  dt = float(unixlike.grepPy(infile, s='delta-t').split()[-1])
                  WLstep = int(unixlike.grepPy(infile, s='equilibrated').split()[1].replace(':', ''))
            except:
               print "\nERROR!\nThe Wang-Landau weights haven't equilibrated yet.\nIf you comprehend the consequences,\nrerun with the -x flag and the data\nwill be discarded to 'equiltime'.\n"
               raise
            WLtime = WLstep * dt
         else:
            WLtime = -1
         return max(WLtime, P.equiltime)

   #===================================================================================================
   # Preliminaries I: Sort the dhdl.xvg files; read in the @-header.
   #===================================================================================================
   
   datafile_tuple = P.datafile_directory, P.prefix, P.suffix
   fs = [ F(filename) for filename in glob( '%s/%s*%s' % datafile_tuple ) ]
   n_files = len(fs)
   
   if not n_files:
      raise SystemExit("\nERROR!\nNo files found within directory '%s' with prefix '%s' and suffix '%s': check your inputs." % datafile_tuple)
   if n_files > 1:
      fs = sorted(fs, key=F.sortedHelper)
   
   if P.bSkipLambdaIndex:
      try:
         lambdas_to_skip = [int(l) for l in unixlike.trPy(P.bSkipLambdaIndex, '-').split()]
      except:
         print '\nERROR!\nDo not understand the format of the string that follows -k.\nIt should be a string of lambda indices linked by "-".\n'
         raise
      fs = [f for f in fs if not f.state in lambdas_to_skip]
      n_files = len(fs)
   
   lv = []  # *** 
   P.snap_size = []
   for nf, f in enumerate(fs):
   
      lv.append(f.readHeader())
   
      if nf>0:
   
         if not f.lv_names == lv_names:
            if not len(f.lv_names) == n_components:
               raise SystemExit("\nERROR!\nFiles do not contain the same number of lambda gradient components; I cannot combine the data.")
            else:
               raise SystemExit("\nERROR!\nThe lambda gradient components have different names; I cannot combine the data.")
         if not f.bPV == bPV:
            raise SystemExit("\nERROR!\nSome files contain the PV energies, some do not; I cannot combine the files.")
   
      else:
   
         P.lv_names = lv_names = f.lv_names
         n_components = len(lv_names)
         bPV = f.bPV
         P.bExpanded = f.bExpanded

   #===================================================================================================
   # Preliminaries II: Analyze data for validity; build up proper 'lv' and count up lambda states 'K'.
   #===================================================================================================
   
   ndE = [len(i) for i in lv]     # ***
   ndE_unique = numpy.unique(ndE) # ***
   
   # Scenario #1: Each file has all the dE columns -- can use MBAR.
   if len(ndE_unique) == 1: # [K]
      if not numpy.array([i == lv[0] for i in lv]).all():
         raise SystemExit("\nERROR!\nArrays of lambda vectors are different; I cannot combine the data.")
      else:
         lv = lv[0]
         # Handle the case when only some particular files/lambdas are given.
         if 1 < n_files < len(lv):
            bSelective_MBAR = True
            sel_states = [f.state for f in fs]
            lv = [lv[i] for i in sel_states]
         else:
            bSelective_MBAR = False
   
   elif len(ndE_unique) <= 3:
      bSelective_MBAR = False
      # Scenario #2: Have the adjacent states only; 2 dE columns for the terminal states, 3 for inner ones.
      if ndE_unique.tolist() == [2, 3]:
         lv  = [l[i>0]  for i,l in enumerate(lv)]
      # Scenario #3: Have a mixture of formats (adjacent and all): either [2,3,K], or [2,K], or [3,K].
      else:
         lv = lv[ndE_unique.argmax()]
      if 'MBAR' in P.methods:
         print "\nNumber of states is NOT the same for all simulations; I'm assuming that we only evaluate"
         print "nearest neighbor states, and so cannot use MBAR, removing the method."
         P.methods.remove('MBAR')
      print "\nStitching together the dhdl files. I am assuming that the files are numbered in order of"
      print "increasing lambda; otherwise, results will not be correct."
   
   else:
      print "The files contain the number of the dE columns I cannot deal with; will terminate.\n\n%-10s %s " % ("# of dE's", "File")
      for nf, f in enumerate(fs):
         print "%6d     %s" % (ndE[nf], f.filename)
      raise SystemExit("\nERROR!\nThere are more than 3 groups of files (%s, to be exact) each having different number of the dE columns; I cannot combine the data." % len(ndE_unique))
   
   lv = numpy.array(lv, float) # *** Lambda vectors.
   K  = len(lv)                # *** Number of lambda states.

   #===================================================================================================
   # Preliminaries III: Count up the equilibrated snapshots.
   #===================================================================================================
   
   equiltime = P.equiltime
   nsnapshots = numpy.zeros(K, int)
   
   for nf, f in enumerate(fs):
   
      f.len_first, f.len_last = (len(line.split()) for line in unixlike.tailPy(f.filename, 2))
      bLenConsistency = (f.len_first != f.len_last)
         
      if f.bExpanded:
   
         equiltime       = f.parseLog()
         equilsnapshots  = int(round(equiltime/f.snap_size))
         f.skip_lines   += equilsnapshots
   
         extract_states  = numpy.genfromtxt(f.filename, dtype=int, skiprows=f.skip_lines, skip_footer=1*bLenConsistency, usecols=1)
         nsnapshots     += numpy.array(Counter(extract_states).values())
   
      else:
         equilsnapshots  = int(equiltime/f.snap_size)
         f.skip_lines   += equilsnapshots
         nsnapshots[nf] += unixlike.wcPy(f.filename) - f.skip_lines - 1*bLenConsistency
   
      print "First %s ps (%s snapshots) will be discarded due to equilibration from file %s..." % (equiltime, equilsnapshots, f.filename)
   
   #===================================================================================================
   # Preliminaries IV: Load in equilibrated data.
   #===================================================================================================   
   
   maxn  = max(nsnapshots)                                   # maximum number of the equilibrated snapshots from any state
   dhdlt = numpy.zeros([K,n_components,int(maxn)], float)    # dhdlt[k,n,t] is the derivative of energy component n with respect to state k of snapshot t
   u_klt = numpy.zeros([K,K,int(maxn)], numpy.float64)       # u_klt[k,m,t] is the reduced potential energy of snapshot t of state k evaluated at state m
   
   for nf, f in enumerate(fs):
      f.iter_loadtxt(nf)

   return nsnapshots, lv, dhdlt, u_klt
