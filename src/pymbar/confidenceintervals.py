#!/usr/bin/python 

import pdb
import numpy 
import scipy
import scipy.special
import scipy.stats

def generateConfidenceIntervals(replicates,K):
   # inputs:
   #      replicates: list of replicates
   #      K: number of replicates
   #=============================================================================================
   # Analyze data.
   #=============================================================================================
   #
   # By Chebyshev's inequality, we should have
   #   P(error >= alpha sigma) <= 1 / alpha^2
   # so that a lower bound will be
   #   P(error < alpha sigma) > 1 - 1 / alpha^2
   # for any real multiplier 'k', where 'sigma' represents the computed uncertainty (as one standard deviation).
   #
   # If the error is normal, we should have
   #   P(error < alpha sigma) = erf(alpha / sqrt(2))

   print "The uncertainty estimates are tested in this section."
   print "If the error is normally distributed, the actual error will be less than a" 
   print "multiplier 'alpha' times the computed uncertainty 'sigma' a fraction of" 
   print "time given by:"
   print "P(error < alpha sigma) = erf(alpha / sqrt(2))"
   print "For example, the true error should be less than 1.0 * sigma" 
   print "(one standard deviation) a total of 68% of the time, and"
   print "less than 2.0 * sigma (two standard deviations) 95% of the time."
   print "The observed fraction of the time that error < alpha sigma, and its" 
   print "uncertainty, is given as 'obs' (with uncertainty 'obs err') below."
   print "This should be compared to the column labeled 'normal'."
   print "A weak lower bound that holds regardless of how the error is distributed is given"
   print "by Chebyshev's inequality, and is listed as 'cheby' below."
   print "Uncertainty estimates are tested for both free energy differences and expectations."
   print ""

   # error bounds

   min_alpha = 0.1
   max_alpha = 4.0
   nalpha = 40
   alpha_values = numpy.linspace(min_alpha, max_alpha, num = nalpha)
   Pobs = numpy.zeros([nalpha], dtype = numpy.float64)
   dPobs = numpy.zeros([nalpha], dtype = numpy.float64)
   Plow = numpy.zeros([nalpha], dtype = numpy.float64)
   Phigh = numpy.zeros([nalpha], dtype = numpy.float64)
   nreplicates = len(replicates)
   dim = len(numpy.shape(replicates[0]['estimated']))
   for alpha_index in range(0,nalpha):
      # Get alpha value.
      alpha = alpha_values[alpha_index]
      # Accumulate statistics across replicates
      a = 1.0
      b = 1.0
      # how many dimensions in the data?


      for (replicate_index,replicate) in enumerate(replicates):
         # Compute fraction of free energy differences where error <= alpha sigma
         # We only count differences where the analytical difference is larger than a cutoff, so that the results will not be limited by machine precision.
         if (dim==0): 
            if numpy.isnan(replicate['error']) or numpy.isnan(replicate['destimated']):
               print "replicate %d" % replicate_index
               print "error"
               print replicate['error']
               print "destimated"
               print replicate['destimated']
               raise "isnan"
            else:                                                             
               if abs(replicate['error']) <= alpha * replicate['destimated']:
                  a += 1.0
               else:
                  b += 1.0
                  
         elif (dim==1): 
            for i in range(0,K):
               if numpy.isnan(replicate['error'][i]) or numpy.isnan(replicate['destimated'][i]):
                  print "replicate %d" % replicate_index
                  print "error"
                  print replicate['error']
                  print "destimated"
                  print replicate['destimated']
                  raise "isnan"
               else:                                                             
                  if abs(replicate['error'][i]) <= alpha * replicate['destimated'][i]:
                     a += 1.0
                  else:
                     b += 1.0
                    
         elif (dim==2):   
            for i in range(0,K):
               for j in range(0,i):
                  if numpy.isnan(replicate['error'][i,j]) or numpy.isnan(replicate['destimated'][i,j]):
                     print "replicate %d" % replicate_index
                     print "ij_error"
                     print replicate['error']
                     print "ij_estimated"
                     print replicate['destimated']
                     raise "isnan"
                  else:                                                             
                     if abs(replicate['error'][i,j]) <= alpha * replicate['destimated'][i,j]:
                        a += 1.0
                     else:
                        b += 1.0
                       
      Pobs[alpha_index] = a / (a+b)
      Plow[alpha_index] = scipy.stats.beta.ppf(0.025,a,b)
      Phigh[alpha_index] = scipy.stats.beta.ppf(0.975,a,b)
      dPobs[alpha_index] = numpy.sqrt( a*b / ((a+b)**2 * (a+b+1)) )

   # Write error as a function of sigma.
   print "Error vs. alpha"
   print "%5s %10s %10s %16s %17s" % ('alpha', 'cheby', 'obs', 'obs err', 'normal')
   Pnorm = scipy.special.erf(alpha_values / numpy.sqrt(2.))
   for alpha_index in range(0,nalpha):
     alpha = alpha_values[alpha_index]
     print "%5.1f %10.6f %10.6f (%10.6f,%10.6f) %10.6f" % (alpha, 1. - 1./alpha**2, Pobs[alpha_index], Plow[alpha_index], Phigh[alpha_index],Pnorm[alpha_index])

   # compute bias, average, etc - do it by replicate, not by bias
   if dim==0:
      vals = numpy.zeros([nreplicates],dtype = numpy.float64)
      vals_error = numpy.zeros([nreplicates],dtype = numpy.float64)
      vals_std = numpy.zeros([nreplicates],dtype = numpy.float64)
   elif dim==1:  
      vals = numpy.zeros([nreplicates,K],dtype = numpy.float64)
      vals_error = numpy.zeros([nreplicates,K],dtype = numpy.float64)
      vals_std = numpy.zeros([nreplicates,K],dtype = numpy.float64)
   elif dim==2:
      vals = numpy.zeros([nreplicates,K,K],dtype = numpy.float64)
      vals_error = numpy.zeros([nreplicates,K,K],dtype = numpy.float64)
      vals_std = numpy.zeros([nreplicates,K,K],dtype = numpy.float64)

   rindex = 0
   for replicate in replicates:    
      if dim==0:
            vals[rindex] = replicate['estimated']      
            vals_error[rindex] = replicate['error']      
            vals_std[rindex] = replicate['destimated']
      elif dim==1:
         for i in range(0,K):
            vals[rindex,:] = replicate['estimated']      
            vals_error[rindex,:] = replicate['error']      
            vals_std[rindex,:] = replicate['destimated']
      elif dim==2:
         for i in range(0,K):
            for j in range(0,i):
               vals[rindex,:,:] = replicate['estimated']      
               vals_error[rindex,:,:] = replicate['error']      
               vals_std[rindex,:,:] = replicate['destimated']
      rindex += 1   

   aveval = numpy.average(vals,axis=0)
   standarddev = numpy.std(vals,axis=0)
   bias = numpy.average(vals_error,axis=0)
   aveerr = numpy.average(vals_error,axis=0)
   d2 = vals_error**2
   rms_error = (numpy.average(d2,axis=0))**(1.0/2.0)
   d2 = vals_std**2
   ave_std = (numpy.average(d2,axis=0))**(1.0/2.0)

   # for now, just print out the data at the end for each 
   print ""
   print "     i      average    bias      rms_error     stddev  ave_analyt_std";
   print "---------------------------------------------------------------------";
   if dim == 0:
      pave = aveval
      pbias = bias
      prms = rms_error
      pstdev = standarddev
      pavestd = ave_std
   elif dim==1:
      for i in range(0,K):
         pave = aveval[i]
         pbias = bias[i]
         prms = rms_error[i]
         pstdev = standarddev[i]
         pavestd = ave_std[i]
         print "%7d %10.4f  %10.4f  %10.4f  %10.4f %10.4f" % (i,pave,pbias,prms,pstdev,pavestd)          
   elif dim==2: 
      for i in range(0,K):
         pave = aveval[0,i]
         pbias = bias[0,i]
         prms = rms_error[0,i]
         pstdev = standarddev[0,i]
         pavestd = ave_std[0,i]
         print "%7d %10.4f  %10.4f  %10.4f  %10.4f %10.4f" % (i,pave,pbias,prms,pstdev,pavestd)          

   print "Totals: %10.4f  %10.4f  %10.4f  %10.4f %10.4f" % (pave,pbias,prms,pstdev,pavestd)          

   return alpha_values,Pobs,Plow,Phigh,dPobs,Pnorm
