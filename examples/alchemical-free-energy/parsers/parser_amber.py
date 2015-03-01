import numpy
import os             # for os interface
from glob import glob # for pathname matching

import unixlike       # some implemented unixlike commands

#===================================================================================================
# FUNCTIONS: Miscellanea.
#===================================================================================================

def amputareFile(filename):
   """Specific to MDOUT.
   Returns True if the last line of the file is a float entry."""
   amp = True
   try:
      _ = float(unixlike.tailPy(filename, 1)[0])
   except ValueError:
      amp = False
   return amp

def readFileMDOUT(f, skip_lines, up_to):
   """Reads in the 'Summary of dvdl values over' section."""
   for i in range(skip_lines):
      f.next()
   # Assumption for the iterator below: one field per line.
   return numpy.fromiter((l for l in f), dtype=float, count=up_to)

@unixlike.workingOnFileObject
def readFileMDEN(f):
   def iter_func():
      for line in f:
         els = line.split()
         for el in els:
            yield el
   A = numpy.fromiter(iter_func(), dtype='|S16')
   for i,el in enumerate(A[:200]):
      try:
         _ = float(el)
         ind = i-1
         break
      except:
         pass
   A = A.reshape(ind, -1)
   n = A[0]
   names = n.tolist()
   indices = range(ind)
   LL = ['L%d' % i for i in range(10)]
   for i, el in enumerate(n):
      if el=='Nsteps' or el=='time(ps)' or any(el[:2]==L for L in LL):
         indices.remove(i)
         names.remove(el)
   A = A[1:, numpy.array(indices)].astype(float)
   AVE = A.mean(axis=0)
   RMS = A.std(axis=0)
   return names, AVE, RMS

def getG(A_n, mintime=3):
   """Computes 'g', the statistical inefficiency, as defined in eq (20) of Ref [1].
   [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
   histogram analysis method for the analysis of simulated and parallel tempering simulations.
   JCTC 3(1):26-41, 2007."""
         
   N = A_n.size
   dA_n = A_n - A_n.mean()
 
   g = denominator = N*dA_n.var()/2
   if g == 0:
      raise SystemExit('The variance is 0; cannot proceed.')
 
   for t in range(1, mintime+1):
      N_t = N - t
      g += N_t * (dA_n[0:N_t]*dA_n[t:N]).mean()
   t += 1
   while t < N-1:
     N_t = N - t
     C = (dA_n[0:N_t]*dA_n[t:N]).mean()
     if C <= 0:
        break
     
     g += C * N_t
     t += 1
 
   g /= denominator
   return g if g > 1 else 1

def uncorrelateAmber(dhdl_k, uncorr_threshold):
   """Retain every 'g'th sample of the original array 'dhdl_k'."""

   g = getG(dhdl_k)

   if g==1:
      return dhdl_k

   N = dhdl_k.size  # Number of correlated samples.
   N_k = 1+int(N/g) # Number of uncorrelated samples.
   if int(round(N_k*g-g)) > N-1:
      N_k -= 1

   if N_k < uncorr_threshold: # Return the original array if N_k is too low.
      print "WARNING:\nOnly %s uncorrelated samples found;\nproceeding with analysis using correlated samples..." % N_k
      return dhdl_k
   indices = numpy.rint(g*numpy.arange(N_k)).astype(int)
   return dhdl_k[indices]

def displayMDENstats(fn_ene):
   print "\nThe MDEN averages:"
   for i,f in enumerate(fn_ene):
      print 64*'-'+'\n'+("  State %s (file %s)  " % (i, f)).center(64, '-')+'\n'
      if not os.path.isfile(f):
         print "\nThere is no such file.\n"
         continue
      print "%6s  %18s %18s %18s\n" % ('', 'Quantity', 'Average', 'RMS fluct.')
      for num, (nam, a, s) in enumerate(zip(*readFileMDEN(f))):
         print "%6s) %18s %18.8G %18.8G" % (num+1, nam, a, s)
      print "\n\n"
   return

def displayDVDLcomponents(DVDL, dvdl_components):
   K, clen = DVDL.shape
   Z, X = (6, 10)
   print "\nThe DV/DL components:"
   print ("%s " % (Z*'-')) + ' '.join(clen*[X*'-'])
   print ("%*s " % (Z, 'State'))  + ' '.join(["%*s" % (X, name) for name in dvdl_components])
   print ("%s " % (Z*'-')) + ' '.join(clen*[X*'-'])
   for i in range(K):
      print ("%*d " % (Z, i)) + ' '.join(["%*.4f" % (X, DVDL[i, n]) for n in range(clen)])
   return

#===================================================================================================
# FUNCTIONS: This is the Amber MDOUT file parser.
#===================================================================================================

def readDataAmber(P):

   # To suppress unwanted calls in __main__.
   P.lv_names = ['']

   # A tuple (not exhaustive) of names of the DV/DL components in the MDOUT file.
   dvdl_components = ('DV/DL', 'EPtot', ' BOND', 'ANGLE', 'DIHED', '1-4 NB', '1-4 EEL', 'VDWAALS', 'EELEC', 'EHBOND', 'RESTRAINT')

   def parseFile(fn):
      """Read in the dvdl data from file (string) 'fn'."""

      # Check whether the file is cut off.
      amp = amputareFile(fn)

      print "Loading in data from %s..." % fn
      with open(fn, 'r') as f:

         # Search for the MDEN file name, time step, lambda value, dvdl output frequency,
	 # dvdl components, and number of the dvdl entries.
         fn_ene,          = unixlike.grepFromSection(f, 'File Assignments', 'MDEN')[1]
         fn_ene = "%s/%s" % (fn.rpartition('/')[0], fn_ene)
         dt,              = unixlike.grepFromSection(f, 'Molecular', 'dt')[1]
         logdvdl, clambda = unixlike.grepFromSection(f, 'Free energy', 'logdvdl', 'clambda')[1]
         DVDL             = unixlike.grepFromSection(f, 'DV/DL,', *dvdl_components)[1]
         total            = unixlike.grepFromSection(f, 'Summary')[1]

         # How many dvdl entries are to be skipped.
         sta_fro = P.equiltime/(int(logdvdl)*float(dt))
         up_to = (total - sta_fro) if not amp else -1

         # Read in data.
         dhdl_k = readFileMDOUT(f, int(sta_fro), int(up_to))
         N = dhdl_k.size

      # The autocorrelation analysis.
      if P.uncorr_threshold:
         dhdl_k = uncorrelateAmber(dhdl_k, P.uncorr_threshold)

      # Compute the average and standard error of the mean.
      N_k = dhdl_k.size
      ave_dhdl = numpy.average(dhdl_k)
      std_dhdl = numpy.std(dhdl_k)/numpy.sqrt(N_k-1)

      return (clambda, ave_dhdl, std_dhdl, N, N_k, DVDL, fn, fn_ene)

   # List the files of interest and count them.
   datafile_tuple = P.datafile_directory, P.prefix, P.suffix
   fs = glob('%s/%s*%s' % datafile_tuple) # will be sorted later
   K = len(fs)
   if not K:
      raise SystemExit("\nERROR!\nNo files found within directory '%s' with prefix '%s' and suffix '%s': check your inputs." % datafile_tuple)

   # Get a list of tuples and sort them by clambda.
   fs = [ parseFile(filename) for filename in fs ]
   fs = sorted(fs)

   print "\nThe average and standard error of the mean in raw data units:"
   print "%6s %12s %12s %12s %12s %12s    %s" % ('State', 'Lambda', 'N', '(Total N)', '<dv/dl>', 'SEM', 'Filename')
   for i, (lam, ave, std, n1, n2, z, f1, f2) in enumerate(fs):
      print "%6s %12s %12s %12s %12.6f %12.6f    %s" % (i, lam, n2, '('+str(n1)+')', ave, std, f1)

   # Extract the lists/arrays.
   lv, ave_dhdl, std_dhdl, N, N_k, DVDL, fn, fn_ene = zip(*fs)

   # Display the DV/DL components as they appear in the MDOUT file.
   displayDVDLcomponents(numpy.array(DVDL, dtype=float), dvdl_components)

   # Load in the MDEN files and compute mean and std for each quantity.
   if P.verbose:
      displayMDENstats(fn_ene)

   # Build proper (conformant to the __main__ format) lv, ave_dhdl, and std_dhdl.
   lv = numpy.array(lv, float).reshape(K,1) # 1 is n_components
   ave_dhdl = P.beta*numpy.array(ave_dhdl).reshape(K,1)
   std_dhdl = P.beta*numpy.array(std_dhdl).reshape(K,1)
   return lv, ave_dhdl, std_dhdl
