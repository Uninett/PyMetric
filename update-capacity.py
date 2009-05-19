#!/usr/bin/env python

import sys

if not len(sys.argv) > 1:
   print "Usage: update-capacity.py <load-file> <metricfile>"
   sys.exit(1)
   
loadfile = sys.argv[1]
metricfile = sys.argv[2]
outfile = metricfile + "-new"

lfh = open(loadfile, 'r')
mfh = open(metricfile, 'r')
ofh = open(outfile, 'w')

capacity = {}

for line in lfh.readlines():
   descr, l1, l2, l3, l4, cap = line.split()

   descr = descr.strip()
   cap = int(cap.strip())

   capacity[descr] = cap

for line in mfh.readlines():
   tokens = line.split()
   desc = tokens[-1].replace('"', '')
   if desc in capacity:
      line = line.strip()
      idx = len(line)
      try:
         idx = line.rindex('c ')
      except ValueError:
         pass
      
      cstr = "c %d" % capacity[desc]
      ofh.write(line[:idx] + "%s%s" % (' '*(50-len(line)+2), cstr + "\n"))
   else:
      ofh.write(line)

lfh.close()
mfh.close()
ofh.close()

print "Written to", outfile
