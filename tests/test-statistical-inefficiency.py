#!/usr/bin/env python

import sys
import numpy
import math

from pymbar import timeseries

N=10000
K=10

if __name__ == "__main__" :        
   var=numpy.ones(N)
   for replica in xrange(2,K+1):
      var=numpy.concatenate((var,numpy.ones(N)))      
   X=numpy.random.normal(numpy.zeros(K*N), var).reshape((K,N))/10.0
   Y=numpy.random.normal(numpy.zeros(K*N), var).reshape((K,N))

#   X=numpy.random.normal(numpy.zeros(K*N), var).reshape((K,N))
#   Y=numpy.random.normal(numpy.zeros(K*N), var).reshape((K,N))

#   print "X.shape = "
#   print X.shape
   energy = 10*(X**2)/2.0 + (Y**2)/2.0

   print "statisticalInefficiencyMultiple(X)"
   print timeseries.statisticalInefficiencyMultiple(X)
   print "statisticalInefficiencyMultiple(X**2)"
   print timeseries.statisticalInefficiencyMultiple(X**2)
   print "statisticalInefficiencyMultiple(X[0,:]**2)"
   print timeseries.statisticalInefficiencyMultiple(X[0,:]**2)
   print "statisticalInefficiencyMultiple(X[0:2,:]**2)"
   print timeseries.statisticalInefficiencyMultiple(X[0:2,:]**2)      
   print "statisticalInefficiencyMultiple(energy)"
   print timeseries.statisticalInefficiencyMultiple(energy)
