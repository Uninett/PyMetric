#!/usr/bin/python
"""
PyMetric is a network simulation and visualisation tool. It
allows the user to trace paths through his network, given a
set of nodes and links with metrics (costs).

It allows the simulation of metric changes, router failures and
link outages, and provides various information relating to the
changes in topology and routing.

PyMetric can also show various statistics on the topology, such
as equal-cost paths, links with assymetric costs, longest paths
and lots more. It can even suggest some metric changes to help
the user with traffic engineering.

It still has some assumptions that are IS-IS specific, but they
are minor and will go away at some point. PyMetric should perform
its job regardless of the routing protocol used (as long as it
is a LSP)

"""
__version__   = "0.9"
__author__    = """Morten Knutsen (morten.knutsen@uninett.no)"""
__copyright__ = """Copyright (C) 2009-2010
Morten Knutsen <morten.knutsen@uninett.no>
"""
import matplotlib, scripting

if __name__ == '__main__':

   import sys

   if not len(sys.argv) > 1:
      print "Please specify topology-file"
      sys.exit(1)

   infile = sys.argv[1]   
   
   if not len(sys.argv) > 2:
      matplotlib.use("TkAgg")
      exec("from command import MetricShell")
      cli = MetricShell(filename=infile)
      cli.version = __version__
      matplotlib.interactive(True)
      cli.cmdloop()
   else:
      scriptFile = None
      outpng = None
      if sys.argv[2] == "-s":
         if not len(sys.argv) == 4:
             print "-s requires a script-file"
             sys.exit(1)
         else:
            scriptFile = sys.argv[3]
      else:
         outpng = sys.argv[2]

      matplotlib.use("cairo")
      matplotlib.interactive(False)
      old_stdout = sys.stdout
      sys.stdout = open("/dev/null", "w")
      exec("from command import MetricShell")
      cli = MetricShell(filename=infile)
      cli.version = __version__

      if outpng:
         cli.onecmd("png %s" % outpng)
         sys.exit(0)

      if scriptFile:
         se = scripting.ScriptEngine(cli)
         sys.exit(se.run(scriptFile))

