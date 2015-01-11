import re

#===================================================================================================
# FUNCTIONS: The unix-like helpers.
#===================================================================================================

def trPy(s, l='[,\\\\"/()-]', char=' '):
   """In string 's' replace all the charachters from 'l' with 'char'."""
   return re.sub(l, char, s)

def workingOnFileObject(function):
   """A decorator that ensures that function 'function' is called upon a file object rather than a string."""
   def convert_if_needed(f, *args, **kwargs):
      if isinstance(f, str):
         with open(f, 'r') as infile:
            return function(infile, *args, **kwargs)
      return function(f, *args, **kwargs)
   return convert_if_needed

@workingOnFileObject
def wcPy(f):
   """Count up lines in file 'f'."""
   return sum(1 for l in f)

@workingOnFileObject
def grepPy(f, s):
   """From file 'f' extract the (first occurence of the) line that contains string 's'."""
   for line in f:
      if s in line:
         return line
   return ''

@workingOnFileObject
def tailPy(f, nlines, LENB=1024):
   f.seek(0, 2)
   sizeb = f.tell()
   n_togo = nlines
   i = 1
   excerpt = []
   while n_togo > 0 and sizeb > 0:
      if (sizeb - LENB > 0):
         f.seek(-i*LENB, 2)
         excerpt.append(f.read(LENB))
      else:
         f.seek(0,0)
         excerpt.append(f.read(sizeb))
      ll = excerpt[-1].count('\n')
      n_togo -= ll
      sizeb -= LENB
      i += 1
   return ''.join(excerpt).splitlines()[-nlines:]

def grepFromSection(f, section, *stringhe, **kwargs):
   """From section 'section' of file 'f' extract the values of strings 'stringhe'."""
   n = 0 if not 'n' in kwargs else kwargs['n']
   f_tell_0 = f.tell()==0 # Will need later to adjust 'i'.
   for i, line in enumerate(f):
      if section in line:
         if not stringhe:
            i += (1 if f_tell_0 else 2)
            # Assumption: will extract the first number found.
            nsteps = [int(el) for el in line.split() if el.isdigit()][0]
            return (i+n, nsteps)
         found = {}
         try:
            line = f.next() # The first line after the line with 'section'.
            i += 1
	    while not line.strip('\n '): # Skip blank lines at the beginning.
               line = f.next()
               i += 1
            if not f_tell_0:
               i += 1 # To compensate for the uncounted line.
	    while line.strip('\n '): # Search until a blank line is encountered.
               line = trPy(line, '[,:=]')
               for s in stringhe:
                  if s in line:
                     found[s] = line.split(s)[1].split()[0]
               line = f.next()
               i += 1 # Will account for the initial 0 when leaving the loop.
         except StopIteration:
            pass
         if not len(found)==len(stringhe):
            raise SystemExit("\nERROR!\nSection '%s' of file '%s' does not contain \
string(s): %s." % (section, f.name, ', '.join([s for s in stringhe if s not in found])))
         return i+n, list(found[s] for s in stringhe)
   raise SystemExit("\nERROR!\nThere is no section '%s' in file '%s'." % (section, f.name))
