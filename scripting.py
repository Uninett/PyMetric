import os.path
import sys

class ScriptEngine():

   def __init__(self, cli):

      self.STATE_UNKNOWN = -1
      self.STATE_OPEN = 0
      self.STATE_BEGIN = 1
      self.STATE_SIM = 2
      self.STATE_END = 4
      self.STATE_ERROR = 8
      self.STATE_FAIL = 16

      assert cli.model
      assert cli.simulation
      self.cli = cli
      self.model = self.cli.model
      self.simulation = self.cli.simulation
      self.state = self.STATE_UNKNOWN
      self.current_line = 0
      self.current_script = None

   def run(self, script):
      if not os.path.isfile(script):
         print >> sys.stderr, "No such file: %s" % script
         return 1
      try:
         fh = open(script, "r")
      except:
         print >> sys.stderr, "Something went wrong when trying to open scriptfile"
         return 1
      self.state = self.STATE_OPEN
      self.current_script = script
      self.savedata = {}
      
      for line in fh.readlines():
         line = line.strip()
         if self.state in [self.STATE_ERROR, self.STATE_END]: break 
         self.current_line += 1
         if line.startswith('#'): continue
         if not line: continue
         elif line.startswith('begin'): self._do_begin()
         elif line.startswith('reset'): self._do_reset()
         elif line.startswith('linkfail'): self._do_linkfail(line)
         elif line.startswith('assert'): self._do_assert(line)
         elif line.startswith('save'): self._do_save(line)
         elif line.startswith('end'): self._do_end()
         else:
            self.state = self.STATE_ERROR
            print >> sys.stderr, "Syntax error at line %d: Unknown keyword" % self.current_line

      if self.state == self.STATE_END:
         self.current_script = None
         self.current_line = 0
         return 0
      elif self.state == self.STATE_FAIL:
         self.current_script = None
         self.current_line = 0
         return 1
      else:
         self.current_script = None
         self.current_line = 0
         return 1

   def _do_begin(self):
      if self.state > self.STATE_OPEN:
         print >> sys.stderr, "Syntax error at line %d: 'begin' allready specified" % self.current_line
         self.state = self.STATE_ERROR
         return
      self.cli.onecmd("sim")
      assert self.simulation.is_active()
      self.state = self.STATE_BEGIN

   def _do_reset(self):
      if self.state <= self.STATE_BEGIN:
         print >> sys.stderr, "Warning: 'reset' without changes ignored (line %d)" % self.current_line
         return
      self.cli.onecmd("stop")
      assert not self.simulation.is_active()
      self.cli.onecmd("sim")
      assert self.simulation.is_active()
      self.state = self.STATE_BEGIN
      return

   def _do_linkfail(self, line):
      args = self._get_args(line)
      if not len(args) == 2:
         print >> sys.stderr, "Syntax error at line %d: Wrong number of arguments for 'linkfail'" % self.current_line
         self.state = self.STATE_ERROR
         return
      if not self.state > self.STATE_OPEN:
         print >> sys.stderr, "Syntax error at line %d: Missing 'begin' statement before 'linkfail'" % self.current_line
         self.state = self.STATE_ERROR
         return
      self.cli.onecmd("linkfail %s %s" % (args[0], args[1]))
      assert self.simulation.changes
      self.state = self.STATE_SIM

   def _do_assert(self, line):
      args = self._get_args(line)
      if not len(args) >= 5:
         print >> sys.stderr, "Syntax error at line %d: Wrong number of arguments for 'assert'" % self.current_line
         self.state = self.STATE_ERROR
         return
      if not self.state > self.STATE_OPEN:
         print >> sys.stderr, "Syntax error at line %d: Missing 'begin' statement before 'assert'" % self.current_line
         self.state = self.STATE_ERROR
         return
      if not self.state == self.STATE_SIM:
         print >> sys.stderr, "Warning: assertion not computed, nothing has been changed! (line %d)" % self.current_line
         return
      if args[3] == 'eq' and args[4] == 'saved-path':
         self.loaddata = self._read_savedata()
         link_key = "%s###%s" % (args[1], args[2])
         if not self.loaddata or link_key not in self.loaddata:
            print >> sys.stderr, "Warning: assertion not computed, no save data to compare with (line %d)" % self.current_line
            return
         else:
            simpaths = self.simulation.path(args[1], args[2])[1]
            if not self.loaddata[link_key] == simpaths:
               print >> sys.stderr, "     FAIL: Assertion failed (line %d)" %  self.current_line
               print >> sys.stderr, "      Got: %s" % ("\n           ".join(map(str, simpaths)))
               print >> sys.stderr, " Expected: %s" % ("\n           ".join(map(str, self.loaddata[link_key])))
               self.state = self.STATE_FAIL
               return
      elif args[3] == 'eq':
         expected = eval(" ".join(args[4:]))
         simpaths = self.simulation.path(args[1], args[2])[1]
         if not expected == simpaths:
            print >> sys.stderr, "     FAIL: Assertion failed (line %d)" % self.current_line
            print >> sys.stderr, "      Got: %s" % ("\n           ".join(map(str, simpaths)))
            print >> sys.stderr, " Expected: %s" % ("\n           ".join(map(str, expected)))
            self.state = self.STATE_FAIL
            return
      else:
         print >> sys.stderr, "Error: Not implemented.... (line %d)" % (self.current_line)
         return

   def _do_save(self, line):
      args = self._get_args(line)
      if not len(args) == 3:
         print >> sys.stderr, "Syntax error at line %d: Wrong number of arguments for 'save'" % self.current_line
         self.state = self.STATE_ERROR
         return
      if not self.state == self.STATE_SIM:
         print >> sys.stderr, "Warning: Not saving (line %d)" % self.current_line
         return
      simpaths = self.simulation.path(args[1], args[2])[1]
      if not simpaths:
         print >> sys.stderr, "Warning: Ignoring 'save' for non-existant path at line %d" % self.current_line
         return
      link_key = "%s###%s" % (args[1], args[2])
      self.savedata[link_key] = simpaths
      
   def _do_end(self):
      if not self.state >= self.STATE_BEGIN:
         print >> sys.stderr, "Syntax error at line %d: Missing 'begin' before 'end'" % self.current_line
         self.state = self.STATE_ERROR
         return
      if self.savedata:
         self._write_savedata()
      self.cli.onecmd("stop")
      assert not self.simulation.is_active()
      self.state = self.STATE_END
      return
      
   def _read_savedata(self):
      try:
         fh = open(self.current_script + ".save", "r")
      except IOError:
         print >> sys.stderr, "Warning: Could not open savefile"
         return {}
      retdata = {}
      for line in fh.readlines():
         line.strip()
         tokens = line.split()
         retdata[tokens[0]] = eval(" ".join(tokens[1:]))
      return retdata

   def _write_savedata(self):
      try:
         fh = open(self.current_script + ".save", "w")
      except IOError:
         print >> sys.stderr, "Warning: Could not open savefile for writing"
         return
      for key in self.savedata:
         fh.write("%s %s\n" % (key, self.savedata[key].__repr__()))
      fh.close()

   def _get_args(self, line):
      line.strip()
      tokens = line.split()
      return tokens[1:]
         
