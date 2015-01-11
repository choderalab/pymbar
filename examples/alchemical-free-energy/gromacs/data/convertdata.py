# Program to convert data files in 
# pymbar/trunk/examples/alchemical-free-energy/data to format of data files 
# produced by GROMACS 4.6 so that data files will work with updated
# alchemical-analysis.py

import commands
from os import remove
from shutil import move

# This program must be called from the folder that contains the files to be
# updated.  THE FOLDER SHOULD ONLY CONTAIN DHDL FILES TO BE UPDATED!

# Get all filenames for files to be updated
filenames = commands.getoutput('ls' %vars()).split()
n_files = len(filenames)

print "The number of files read in for processing is: ",n_files

# for loop to open and edit each file in the folder
for nf in range(n_files):
	#open file with read write access
	infile = open(filenames[nf], 'r')
	outfile = open("tempfile", 'w')
	#read all of the lines of the file
	lines = infile.readlines()
	#for each file read
	for line in lines:
		#if the first element of the line is @ then it MAY BE a line that should be changed
		if (line[0] == '@'):
			#if the third element of the line is an s then it IS a line to be changed
			if (line[2] == 's'):
				#changed line with replacements to the line adding word 'to' and a space
				cline = line.replace("{} (", "{} to (").replace(",", ", ")
				#write changed line over original line
				outfile.write(cline)
			else:
				outfile.write(line)
		else:
			outfile.write(line)
	
	#close infile and outfile
	infile.close
	outfile.close
	#remove original file
	remove(filenames[nf])
	#move new file to original file location
	move("tempfile", filenames[nf])
	print "Updated file: ", filenames[nf]
