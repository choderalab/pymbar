"""
setup.py: Install pymbar.  
"""
from distutils.sysconfig import get_config_var
from distutils.core import setup, Extension
import numpy
import glob

CMBAR = Extension('_pymbar',
                  sources = ["_pymbar.c"],
                  extra_compile_args=["-std=c99","-O2","-shared","-msse2","-msse3"],
                  include_dirs = [numpy.get_include(),numpy.get_include()+"/numpy/"]
                  )

def buildKeywordDictionary():
    from distutils.core import Extension
    setupKeywords = {}
    setupKeywords["name"]              = "pymbar"
    setupKeywords["version"]           = "2.0beta"
    setupKeywords["author"]            = "Michael R. Shirts and John D. Chodera"
    setupKeywords["author_email"]      = "michael.shirts@virginia.edu, jchodera@gmail.com"
    setupKeywords["license"]           = "GPL 2.0"
    setupKeywords["url"]               = "https://simtk.org/home/pymbar"
    setupKeywords["download_url"]      = "https://simtk.org/home/pymbar"
    setupKeywords["py_modules"]        = ["pymbar", "timeseries", "testsystems", "confidenceintervals"]
    setupKeywords["data_files"]        = []
    setupKeywords["ext_modules"]       = [CMBAR]
    setupKeywords["platforms"]         = ["Linux", "Mac OS X", "Windows"]
    setupKeywords["description"]       = "Python Code for performing ."
    setupKeywords["long_description"]  = """
    Pymbar (https://simtk.org/home/pymbar) is a library
    that provides tools for optimally combining simulations 
    from multiple thermodynamic states using maximum likelihood 
    methods to compute free energies (normalization constants) 
    and expectation values from all of the samples simultaneously.
    """
    outputString=""
    firstTab     = 40
    secondTab    = 60
    for key in sorted( setupKeywords.iterkeys() ):
         value         = setupKeywords[key]
         outputString += key.rjust(firstTab) + str( value ).rjust(secondTab) + "\n"
    
    print "%s" % outputString

    get_config_var(None)  # this line is necessary to fix the imports Mac OS X
    return setupKeywords
    

def main():
    setupKeywords = buildKeywordDictionary()
    setup(**setupKeywords)

if __name__ == '__main__':
    main()




