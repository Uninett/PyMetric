from cmd import Cmd
import readline
import networkx as nx
import sys
import os.path
from pajek import read_pajek
from model import Simulation, Model
from config import Config
from plotting import PlotUI
from textwrap import TextWrapper
from termcolor import colored, colored2, colored3
import utils

class MetricShell(Cmd):

   def __init__(self, model=None, filename="model.net",
                linkloads=False, debug=False):
      self.histfile = os.path.join(os.environ["HOME"], ".pymetric-hist")
      try:
         readline.read_history_file(self.histfile)
      except IOError:
         pass

      self.config = Config()

      readline.set_completer_delims(" ")
      
      Cmd.__init__(self)

      self.termcolor = True
      self._colormode(self.termcolor)

      self.uiwait = False
      self.defaultprompt = ">>> "
      self.debug = debug
      self.has_plotted = False
      self.filename = filename
      print
      print "Initializing model...."
      if not model:
         self.model = Model(nx.Graph(), self.config)
         self.model.refresh_from_file(self.filename)
         if linkloads:
            self.model.refresh_linkloads()
            
      self.simulation = Simulation(self.model)
      self.gui = PlotUI(self)
      self.tw = TextWrapper(initial_indent=' '*4,
                            subsequent_indent=' '*4,
                            width=80)
      
      self.intro = self.bt("PyMetric interactive shell, type 'help' for help")

      self.prompt = self.defaultprompt

   def cmdloop(self):
      while 1:
         try:
            Cmd.cmdloop(self)
         except KeyboardInterrupt:
            self.intro = "\n"
            pass

   def fromui(self, node):
      if not self.uiwait:
         self.do_info(node)
   
   def completenames(self, text, *ignored):
      dotext = 'do_'+text
      return [a[3:] + " " for a in self.get_names() if a.startswith(dotext)]         

   def postcmd(self,stop,line):
      if self.simulation.is_active():
         promptstring = "(sim) "
         if self.simulation.has_changes():
            effects = self.simulation.get_effects()
            multinodes = filter(lambda x: len(effects[x].keys()) >= 5,
                                effects.keys())
            multimulti = filter(lambda x: len(effects[x].keys()) >= 20,
                                effects.keys())                               
            difflen = utils.mean_shortest_path_length(self.simulation.graph)\
                    - utils.mean_shortest_path_length(self.model.G)

            components = nx.connected_component_subgraphs(
                           self.simulation.graph.to_undirected())
            cnodes = 0
            if len(components) > 1:
               cnodes = sum([len(g.nodes()) for g in components[1:]])

            try:
               diffrad = nx.radius(self.simulation.graph)\
                       - nx.radius(self.model.G)
            except:
               diffrad = None

            try:
               diffdia = nx.diameter(self.simulation.graph)\
                       - nx.diameter(self.model.G)
            except:
               diffdia = None

            uzs = None
            uzs_50_cnt = 0
            uzs_75_cnt = 0
            uzs_95_cnt = 0
            drop_warning = False
            if self.model.has_linkloads():
               uzs = self.simulation.get_link_utilizations()
               for (u,v) in uzs:
                  if uzs[(u,v)] >= 0.95:
                     uzs_95_cnt += 1
                     if uzs[(u,v)] >= 1:
                        drop_warning = True
                  elif uzs[(u,v)] >= 0.75:
                     uzs_75_cnt += 1
                  elif uzs[(u,v)] >= 0.5:
                     uzs_50_cnt += 1
                     
            promptstring += "(" \
                         + self.pbblt("%dc" % self.simulation.no_changes()) \
                         + "/" \
                         + self.pblt("%d:" % len(effects.keys())) \
                         + self.pblt("%d:" % len(multinodes)) \
                         + self.pblt("%dn" % len(multimulti))
            if uzs:
               promptstring += "/"
               promptstring += self.pblt("%d:" % uzs_50_cnt) \
                            +  self.pblt("%d:" % uzs_75_cnt)
               if uzs_95_cnt:
                  promptstring +=  self.prt("%du" % uzs_95_cnt)
               else:
                  promptstring += self.pblt("%du" % uzs_95_cnt)            
            if difflen and difflen >= 0.01:
               promptstring += "/"
               promptstring += self.pblt("%.2fL" % difflen)            
            if diffrad:
               promptstring += "/"
               promptstring += self.pblt("%sr" % diffrad)
            if diffdia:
               promptstring += "/"
               promptstring += self.pblt("%sd" % diffdia)
            if len(components) > 1:
               promptstring += "/"
               promptstring += self.prt("%d:%dp" % (len(components), cnodes))
               print self.bt("Warning:") + " Network is partitioned"
            promptstring += ") "

            if drop_warning:
               print self.bt("Warning:") + " There are traffic drops"

         acnodes = self.simulation.get_anycast_nodes()

         if acnodes:
            no_acgroups = len(acnodes)
            groupcounts = [str(len(self.simulation.get_anycast_group(x)))
                           for x in sorted(acnodes)]
            promptstring += "(" \
                         + self.pbgrt("%da" % no_acgroups) \
                         + "/" \
                         + self.pgrt(":".join(groupcounts) + "m") \
                         + ") "
         promptstring += ">>> "
         self.prompt = promptstring

      else:
         self.prompt = self.defaultprompt
      return stop

   def emptyline(self):
      return

   #
   # Global commands
   #
   def do_version(self, args):
      if self.version:
         print "PyMetric %s" % self.version
      else:
         print "Unknown version"
      return
   
   def do_colors(self, args):
      color = self.termcolor
      if args:
         if args == 'on':
            color = True
         elif args == 'off':
            color = False
         else:
            print "Unknown argument:", args
            return
      else:
         print "Colormode is currently %s" % self.termcolor
         return
      self._colormode(color)
      print "Colormode is now %s" % self.termcolor
      return
         
   def do_reload(self, args):
      if self.simulation.is_active():
         print "Please end simulation before reload."
         return
      self.model.refresh_from_file(self.filename)
      
      if self.gui.has_plotted:
         self.do_plot("")
      return

   def do_linkloads(self, args):
      if not self.config.get('use_linkloads'):
         print "Link loads are not enabled"
         return
      if self.model.refresh_linkloads():
         print "OK, loads refreshed, use 'plot with-load' to plot"
         if not self.simulation.linkloads:
            self.simulation.linkloads = self.model.linkloads
            if self.simulation.has_changes():
               self.simulation._refresh_linkload()
         return
      print "Couldn't refresh load data"
      return
      
   def do_load(self, args):
      if self.simulation.is_active():
         print "Please end simulation before loading new file."
         return
      if not args:
         print "Must specify filename to load model from."
         return
      confirm = raw_input('Are you sure you want to load new model [Y/N]? ')
      if not confirm.lower() == 'y':
         return
      try:
         self.model.refresh_from_file(args)
         self.filename = args
      except IOError:
         print "ERROR: Could not read file, does it exist?"
         self.model.refresh_from_file(self.filename)         
      except:
         import traceback         
         traceback.print_exc()
         print
         print "ERROR: Load failed, model not changed."         
         self.model.refresh_from_file(self.filename)
      return         

   def do_utilizations(self, args):
      if self.simulation.is_active():
         model = self.simulation
      else:
         model = self.model

      if not self.model.has_linkloads():
         print "No link load information available. Use 'linkloads' to fetch."
         return

      utils = model.get_link_utilizations()

      sorted_utils = sorted(utils,
                          cmp=lambda x, y: cmp(utils[x], utils[y]))
      sorted_utils.reverse()

      ab75_utils = filter(lambda x: utils[x] >= 0.75, sorted_utils)

      print "Utilizations, type 'linkinfo source dest' for more details"
      print
      if ab75_utils:
         print ">75% utilization:"
         for (u,v) in ab75_utils:
            print "  * %s->%s: %.2f%%" % (u,v, utils[(u,v)]*100)
         print
      print "Top 10:"
      for (u,v) in sorted_utils[:10]:
         print "  * %s->%s: %.2f%%" % (u,v, utils[(u,v)]*100)
      print
      
   def do_list(self, args):
      print "List of nodes:"
      print sorted(self.model.get_nodes())

   def do_asymmetric(self, args):
      model = self.model
      G = model.G
      if self.simulation.is_active():
         model = self.simulation
         G = model.graph
      uneven = model.uneven_metrics()
      printed = {}
      if not uneven:
         print "All link metrics are symmetric"
         return
      print "Links with asymmetric metrics (%d):" \
            % (len(uneven)/2)
      print
      for (u,v,w) in uneven:
         x = G.get_edge(v,u)
         if (v,u) in printed: continue
         print "%-15s -> %s: %s" % (u,v,w)
         print "%-15s -> %s: %s" % (v,u,x)
         printed[(u,v)] = True
         print
      return

   def do_eval(self, args):
      if not args:
         return

      retstr = ""
      evalstr = "retstr=%s" % args
      try:
         exec(evalstr)
      except:
         import traceback
         print "An error occured:"
         traceback.print_exc()
         return

      print retstr

      return

   def do_simplot(self, args):
      if not self.simulation.is_active():
         print "No simulation is active, type 'simulation start' to start one"
         return

      subargs = args.split()
      cmap=False
      capa=False
      if 'with-load' in subargs:
         if self.simulation.linkloads:
            cmap = {}
            capa = {}
            for (u,v,d) in self.simulation.graph.edges(data=True):
               cmap[(u,v)] = self.simulation.get_link_utilization(u,v)
               capa[(u,v)] = self.model.get_link_capacity(u,v)            
         else:
            print "Warning: No linkloads are defined. Use 'linkloads' to update."
      elif self.simulation.get_anycast_nodes():
         self.do_anycast("")
         return
      
      self.gui.clear()
      graphdata = {}
      graphdata['nodegroups'] = self.simulation.get_node_groups()
      graphdata['edgegroups'] = self.simulation.get_edge_groups()

      G = self.simulation.graph
      
      graphdata['labels'] = utils.short_names(G.nodes())
      graphdata['edgelabels'] = utils.edge_labels(G.edges(data=True),
                                                  graphdata['edgegroups'])
      graphdata['pos'] = self.model.get_positions(G.nodes())

      graphdata['title'] = "Simulated topology"
      if cmap:
         graphdata['title'] += " - Loads and Capacity view"      
      self.gui.plot(G, graphdata, edge_cmap=cmap, edge_capa=capa)      

   def do_png(self, args):
      fname = "isis-metrics.png"
      subargs = args.split()
      plotarg = ""
      load = False
      if len(subargs) >= 1: fname = subargs[0]
      if len(subargs) == 2 and subargs[1] == 'with-load':
         plotarg = "with-load"
         load = True

      if self.simulation.is_active():
         self.do_simplot(plotarg)
      else:
         self.do_plot(plotarg)
      self.gui.savefig(fname, load)
      return
      
   def do_plot(self, args):
      self.gui.clear()
      subargs = args.split()
      if "simulation" in subargs and self.simulation.is_active():
         self.do_simplot(args)
         return

      cmap = False
      capa = False
      if 'with-load' in subargs:
         if self.model.linkloads:
            cmap = {}
            capa = {}
            for (u,v,d) in self.model.graph.edges(data=True):
               cmap[(u,v)] = self.model.get_link_utilization(u,v)
               capa[(u,v)] = self.model.get_link_capacity(u,v)            
         else:
            print "Warning: No linkloads are defined. Use 'linkloads' to update."
      
      graphdata = {}
      
      G = self.model.G
      graphdata['nodegroups'] = self.model.get_node_groups()
      graphdata['edgegroups'] = self.model.get_edge_groups()
      graphdata['labels'] = utils.short_names(G.nodes())
      graphdata['edgelabels'] = utils.edge_labels(G.edges(data=True),
                                                  graphdata['edgegroups'])
      graphdata['pos'] = self.model.get_positions(G.nodes())
      graphdata['title'] = "Current topology"
      if cmap:
         graphdata['title'] += " - Loads and Capacity view"
      self.gui.plot(G, graphdata, edge_cmap=cmap, edge_capa=capa)

   def do_areaplot(self, args):
      if not self.config.get('use_areas'):
         print "IS-IS areas are not enabled"
         return

      G = self.model.G
      areas = self.model.get_areas(G.nodes())
      if not areas:
         print "No IS-IS areas known"
         return

      self.gui.clear()

      graphdata = {}
      
      graphdata['nodegroups'] = self.model.get_node_groups()      
      graphdata['areagroups'] = areas
      graphdata['edgegroups'] = self.model.get_edge_groups()
      graphdata['labels'] = utils.short_names(G.nodes())
      graphdata['edgelabels'] = utils.edge_labels(G.edges(data=True),
                                                  graphdata['edgegroups'])
      graphdata['pos'] = self.model.get_positions(G.nodes())
      graphdata['title'] = "Current topology with IS-IS areas"
      self.gui.plot(G, graphdata, areagroups=True)
      
   def do_stats(self, args):
      model = self.model
      stats = self.model.get_stats()
      if self.simulation.is_active():
         model = self.simulation         
      stats = model.get_stats()         
      self.tw.initial_indent=''
      self.tw.subsequent_indent=' '*18
      self.tw.width=80-18
      for (name, value) in sorted(stats.items()):
         if type(value) == type([]):
            value = ", ".join(value)
         value = str(value)

         print "%s: %s" % (name.rjust(16), self.tw.fill(value))

      self.tw.width=80

   def do_linkinfo(self, args):
      model = self.model
      if self.simulation.is_active():
         model = self.simulation
      subargs = args.split()
      if not len(subargs) == 2:
         print "Must specify two nodes"
         return
         self.help_linkinfo()

      (u,v) = subargs[:2]
      if not model.graph.has_edge(u,v):
         print "Model has no link (%s,%s)" % (u,v)
         return

      self.tw.initial_indent = ''
      self.tw.subsequent_indent = ' '*17
      self.tw.width = 80 - 17         
      print "Information for link (%s,%s):" % (u,v)
      infohash = model.get_link_info(u,v)
      for key in ['name', 'betweenness', 
                  'capacity', 'load', 'utilization']:
         if key not in infohash:
            continue
         info = infohash[key]
         if type(info) == type([]):
            info = ', '.join(info)
         info = str(info)
         print "%-15s: %s" % (key, self.tw.fill(info))
      print

      self.tw.width = 80
      
   def do_info(self, args):
      model = self.model
      if self.simulation.is_active():
         model = self.simulation               
      if not args:
         print "Must specify a node"
         return
         self.help_info()
      if args not in model.graph.nodes():
         print "%s is not a valid node name" % args
         return

      self.tw.initial_indent = ''
      self.tw.subsequent_indent = ' '*17
      self.tw.width = 80 - 17         
      print "Information for node %s:" % (args)
      infohash = model.get_node_info(args)
      for key in ['name', 'degree', 'eccentricity',
                  'betweenness', 'neighbors', 'links',
                  'longest paths', 'anycast group']:

         if key not in infohash:
            continue
         info = infohash[key]
         if type(info) == type([]):
            info = ', '.join(info)
         info = str(info)
         print "%-15s: %s" % (key, self.tw.fill(info))
      print

      self.tw.width = 80
      
   def do_simulation(self, args):
      if args == 'stop':
         if self.simulation.is_active():
            print self.bt("Simulation ended")
            self.simulation.stop()
         else:
            print "No simulation is active, type 'simulation start' to start one"
      else:
         if self.simulation.is_active():
            print "Simulation allready in progress, type 'stop' to end it."
            return
         print self.bt("Simulation mode active, type 'stop' or 'simulation stop' to end it")
         self.simulation.start()

   def do_undo(self, args):
      if not self.simulation.is_active():
         print "No simulation is active, type 'simulation start' to start one"
         return
      if not args:
         print "Must supply a change (number) or 'all'"
         return
      if args and args == 'all':
         while self.simulation.has_changes():
            self.simulation.undo(1)
         return
      try:
         change_no = int(args)
      except:
         print "Please supply a valid change number (integer)"
         return
      if not self.simulation.undo(change_no):
         print "No such change, type 'changes' to see all changes"
         return
      print "Done"
      return
   
   def do_stop(self, args):
      if not self.simulation.is_active():
         print "No simulation is active, type 'simulation start' to start one"
         return
      self.do_simulation("stop")
      
   def do_changes(self, args):
      if not self.simulation.is_active():
         print "No simulation is active, type 'simulation start' to start one"
         return
      if not self.simulation.has_changes():
         print "No changes"
         return      
      print "Simulated changes:"

      for (i, change) in enumerate(self.simulation.get_changes_strings()):
         print "  %d: %s" % (i+1, change)
      print

      subargs = args.split()
      if 'as-commands' in subargs:
         print "As commands:"
         for cmd in self.simulation.get_changes_strings(commands=True):
            print "  %s" % (cmd)
         print

      if 'no-effects' in subargs:
         return
      
      if not self.simulation.has_effects():
         print "No effect on model"
         return

      print "Effects of changes:"
      for arg in subargs:
         if arg not in self.simulation.get_nodes(): continue
         nodechanges = self.simulation.get_effects_node(arg)
         if not nodechanges.keys(): print " * No changes for %s " % (arg)
         print " * Details for %s " % (arg)
         print



         self.tw.initial_indent=''
         self.tw.subsequent_indent=' '*17
         self.tw.width=80-17
         for dest in nodechanges.keys():
            ddiffs = nodechanges[dest]
            for diff in ddiffs:
               if not diff['new']:
                  print "  - %s now unreachable" % (dest)
                  print "    Was reachable via %s" % ("/".join(diff['old']))
               else:
                  print "  - Path to %s now via %s" % (dest,
                                                       "/".join(diff['new']))
                  print "    instead of %s" % ("/".join(diff['old']))
                  print
         return

      print "  * Affects %d nodes total" % (len(self.simulation.get_effects()))
      print
      print "  * Summary:"

      srcsummary, dstsummary = self.simulation.get_effects_summary()
      
      for src in sorted(srcsummary):
         count = len(srcsummary[src])
         if count > 3:
            for dest in dstsummary:
               if src in dstsummary[dest]:
                  del dstsummary[dest][dstsummary[dest].index(src)]
            print "    %s changed path to %d destinations" % (src, count)
            if count < 6:
               print "      => %s" % (sorted(srcsummary[src]))
               print
            else:
               print
            
      for dest in sorted(dstsummary):
         count = len(dstsummary[dest])
         if count == 0: continue
         if count < 3:
            print "    %s changed path to %s" \
                % (" and ".join(sorted(dstsummary[dest])), dest)
         else:
            print "    %d sources changed path to %s" % (count, dest)
         if count > 3 and count < 6:
            print "      => %s" % (sorted(dstsummary[dest]))
            print
         else:
            print

   def do_anycast(self, args):
      if not self.simulation.is_active():
         print "Must be in simulation mode to model anycast"
         return
      subargs = args.split()
      if not subargs:
         acnodes = self.simulation.get_anycast_nodes()
         if not acnodes:
            print "No anycast nodes configured in current simulation."
         else:
            print "Current anycast nodes:"
            for node in acnodes:
               members = self.simulation.get_anycast_group(node)
               print "  * %-15s (%s members)" \
                   % (node, str(len(members)).rjust(2))

            self.gui.clear()

            graphdata = {}
            G = self.simulation.graph
            acgroups = self.simulation.get_anycast_groups_by_source()
         
            graphdata['nodegroups'] = self.simulation.get_node_groups()      
            graphdata['acnodes'] = acnodes
            graphdata['acgroups'] = acgroups
            graphdata['edgegroups'] = self.simulation.get_edge_groups()
            graphdata['labels'] = utils.short_names(G.nodes())
            graphdata['edgelabels'] = utils.edge_labels(G.edges(data=True),
                                                        graphdata['edgegroups'])
            graphdata['pos'] = self.model.get_positions(G.nodes())
            graphdata['title'] = "Simulated topology with anycast groups"
            self.gui.plot(G, graphdata, anycast=True)
            return

      elif len(subargs) == 1:
         if subargs[0] == 'clear':
            self.simulation.remove_anycast_nodes(self.simulation.get_anycast_nodes())
            return
         else:
            print "Invalid input"
            return
            self.help_anycast()

      elif len(subargs) > 1:
         if subargs[0] == 'add':
            for node in subargs[1:]:
               if not node in self.simulation.graph.nodes():
                  print "Invalid node: %s" % node
                  return            
            self.simulation.add_anycast_nodes(subargs[1:])
         elif subargs[0] == 'remove':
            for node in subargs[1:]:
               if not node in self.simulation.graph.nodes():
                  print "Invalid node: %s" % node
                  return            
            self.simulation.remove_anycast_nodes(subargs[1:])         
         else:
            print "Invalid input"
            return
            self.help_anycast()
      
   def do_reroute(self, args):
      if not self.simulation.is_active():
         print "Must be in simulation mode to run reroute"
         return
      subargs = args.split()
      if not len(subargs) >= 3:
         print "Invalid input"
         self.help_reroute()
         return

      equal = False
      start, end, via = subargs[:3]

      if len(subargs) == 4:
         if subargs[3] == 'equal-path':
            equal = True
         else:
            print "Warning: Last argument (%s) ignored" % subargs[3]
            
      ret = self.simulation.reroute(start, end, via, equal)
      
      if not ret[0]:
         print "No solution could be found.."
         return

      if not ret[1]:
         print "The path allready goes through %s" % via
         return
      
      print "The following metric changes are suggested:"
      
      G = self.simulation.graph
      shown = {}
      for e in sorted(ret[1]):
         u,v,w = e[0], e[1], ret[1][e]
         if (u,v) in shown: continue
         w_old = G.get_edge(u,v)
         if w_old != w:
            linkstr = "%s <-> %s" % (u,v)
            oldstr = "%s" % int(w_old)
            newstr = "%s" % int(w)

            print "%-40s %s -> %s" % (linkstr, oldstr.rjust(2),
                                              newstr.rjust(2))
            shown[(u,v)] = True
            shown[(v,u)] = True
            
      apply = raw_input("Apply changes to current simulation (Y/N)? ")

      applied = {}
      if apply.lower() == 'y':
         for e in ret[1]:
            u,v,w = e[0], e[1], ret[1][e]            
            if (u,v) in applied: continue
            w_old = G.get_edge(u,v)
            if w_old != w:
               self.simulation.change_metric(u,v,w, True)
               applied[(u,v)] = True
               applied[(v,u)] = True
      else:
         print "Not applied."
      return
   

      
   def do_minimize(self, args):
      if not self.simulation.is_active():
         print "Must be in simulation mode to run minimize"
         return

      G = self.simulation.graph

      print "Please wait, this can take a little while..."
      
      H = self.simulation.minimal_link_costs()
      
      shown = {}
      header = False
      for (u,v,w) in sorted(H.edges(data=True)):
         if (u,v) in shown: continue
         w_old = G.get_edge(u,v)
         if w_old != w:
            if not header:
               print "The following metric changes are suggested:"
               header = True
            linkstr = "%s <-> %s" % (u,v)
            oldstr = "%s" % int(w_old)
            newstr = "%s" % int(w)

            print "%-40s %s -> %s" % (linkstr, oldstr.rjust(2),
                                              newstr.rjust(2))
            shown[(u,v)] = True
            shown[(v,u)] = True
            
      apply = raw_input("Apply changes to current simulation (Y/N)? ")

      applied = {}
      if apply.lower() == 'y':
         for (u,v,w) in H.edges(data=True):
            if (u,v) in applied: continue
            w_old = G.get_edge(u,v)
            if w_old != w:
               self.simulation.change_metric(u,v,w, True)
               applied[(u,v)] = True
               applied[(v,u)] = True
      else:
         print "Not applied."
      return
      
   def do_metric(self, args):
      if not self.simulation.is_active():
         print "Must be in simulation mode to run metric changes"
         return
      if not args:
         print "Invalid input"
         self.help_metric()
         return
      subargs = args.split()
      if not len(subargs) >= 3:
         print "Invalid input"
         self.help_metric()
         return
      bidir = None
      (n1, n2, metric) = subargs[0:3]
      if len(subargs) == 4:
         if subargs[3] == 'one-way':
            bidir = False
         elif subargs[3] == 'two-way':
            bidir = True
         else:
            print "Warning: last argument ignored: %s" % (subargs[3])
      if self.simulation.changes:
         for i, change in enumerate(self.simulation.changes):
            if change['type'] == Simulation.SC_METRIC \
                   and change['pair'] == (n1,n2):
               self.simulation.undo(i+1)
      if not self.simulation.change_metric(n1, n2, metric,bidir=bidir):
         print "No link from %s to %s" % (n1, n2)
      
   
   def do_linkfail(self, args):
      if not self.simulation.is_active():
         print "Must be in simulation mode to run link changes"
         return
      if not args:
         print "Invalid input"
         self.help_linkfail()
         return
      subargs = args.split()
      if not len(subargs) == 2:
         print "Invalid input"
         self.help_linkfail()
         return
      (n1, n2) = subargs[0:2]

      if not self.simulation.linkfail(n1, n2):
         print "No link from %s to %s" % (n1, n2)

   def do_routerfail(self, args):
      if not self.simulation.is_active():
         print "Must be in simulation mode to run router changes"
         return
      if not args:
         print "Invalid input"
         self.help_routerfail()
         return
      subargs = args.split()
      if not len(subargs) == 1:
         print "Invalid input"
         self.help_routerfail()      
         return
      n1 = subargs[0]

      if not self.simulation.routerfail(n1):
         print "No node %s in current topology."
      
   def do_simpath(self, args):
      if not self.simulation.is_active():
         print "No simulation active, type 'simulation' to start one"
         return
      subargs = args.split()
      if not len(subargs) == 2:
         self.help_simpath()
         return
      a, b = subargs[0], subargs[1]
      if a not in self.simulation.get_nodes():
         print "%s not a valid node, type 'list' to see nodes"\
               % (a)
         return
      if b not in self.simulation.get_nodes():
         print "%s not a valid node, type 'list' to see nodes"\
               % (b)

      length, paths = self.simulation.path(a,b)

      if not length and not paths:
         print "No valid path from %s to %s in model"
         return
      
      print "Path from %s to %s:"\
            % (a, b)
      print "  * Cost: %d" % (length)
      if len(paths) > 1:
         print "  * %d paths total (equal cost):" % len(paths)
         print

      self.tw.subsequent_indent = ' '*5
      for path in paths:
         print "  * Hops: %d" % (len(path))
         print "  * Path:\n", self.tw.fill(" => ".join(path))
         print "  * Slowest link: ", self.model.get_path_capacity(path,
                                                                  True,
                                                                  True)
         print

      graphdata = {}
      self.gui.clear()
      G = self.simulation.graph

      path = paths[0]
      hops = str(len(paths[0]))
      if len(paths) > 1:
         path = reduce(lambda x,y: x+y, paths)
         hops = "/".join(map(lambda x: str(len(x)), paths))
      
      graphdata['nodegroups'] = self.simulation.get_node_groups(path=path)
      graphdata['edgegroups'] = self.simulation.get_edge_groups(path=paths)
      graphdata['labels'] = utils.short_names(G.nodes())
      graphdata['edgelabels'] = utils.edge_labels(G.edges(data=True),
                                                  graphdata['edgegroups'])
      graphdata['pos'] = self.model.get_positions(G.nodes())
      graphdata['title'] = "Simulated path from %s to %s (cost: %d, %s hops)" \
                           % (a, b, length, hops)
      
      self.gui.plot(G, graphdata, 0.7)
      
   def do_diffpath(self, args):
      if not self.simulation.is_active():
         print "No simulation active, type 'simulation' to start one"
         return
      subargs = args.split()
      if not len(subargs) == 2:
         self.help_diffpath()
         return
      a, b = subargs[0], subargs[1]
      if a not in self.simulation.get_nodes():
         print "%s not a valid node, type 'list' to see nodes"\
               % (a)
         return
      if b not in self.simulation.get_nodes():
         print "%s not a valid node, type 'list' to see nodes"\
               % (b)

      slength, spaths = self.simulation.path(a,b)
      length, paths = self.model.path(a,b)      

      print "Path from %s to %s:"\
            % (a, b)

      if not length:
         print "Path does not exist in model."
         return
      if not slength:
         print "Path no longer possible in simulation."
         print "Type 'path %s %s' to see original path." % (a,b)
         return

      shops = str(len(spaths[0]))
      hops = str(len(paths[0]))
      spath = spaths[0]
      path = paths[0]
      if len(spaths) > 1:
         spath = reduce(lambda x,y: x+y, spaths)
         shops = "/".join(map(lambda x: str(len(x)), spaths))
      if len(paths) > 1:
         path = reduce(lambda x,y: x+y, paths)
         hops = "/".join(map(lambda x: str(len(x)), paths))   
      
      if len(spaths) > 1 or len(paths) > 1:
         print "  * %d vs %d paths total:" % (len(spaths), len(paths))
         print
      print "  * Cost: %d vs %d" % (slength, length)         
      print "  * Hops: %s vs %s" % (shops, hops)

      self.tw.initial_indent = ' '*5
      self.tw.subsequent_indent = ' '*5      
      for i in range(max(len(paths), len(spaths))):

         if i < len(spaths):
            print "  * Path:\n", self.tw.fill(" => ".join(spaths[i]))
            print "  * Slowest link: ", self.model.get_path_capacity(spaths[i],
                                                                  True,
                                                                  True)            
         else:
            print "  *    NA"
         print "    vs."

         if i < len(paths):
            print "%s" % self.tw.fill(" => ".join(paths[i]))
            print "  * Slowest link: ", self.model.get_path_capacity(paths[i],
                                                                  True,
                                                                  True)            
         else:
            print "      NA"
         print
               
      graphdata = {}
      
      self.gui.clear()
      G = self.simulation.graph
      H = self.model.G

      gng   = self.simulation.get_diff_node_groups(path,spath)
      geg   = self.simulation.get_diff_edge_groups(paths,spaths)

      lb   = utils.short_names(H.nodes())
      elb  = utils.edge_labels(H.edges(data=True), geg)
      pos = self.model.get_positions(H.nodes())
      graphdata['title'] = "Simulated path from %s to %s (cost: %d, %s hops)" \
                           % (a, b, slength, shops)

      graphdata['nodegroups'] = gng
      graphdata['edgegroups'] = geg
      graphdata['labels'] = lb
      graphdata['edgelabels'] = elb
      graphdata['pos'] = pos

      self.gui.plot(H, graphdata, 0.7)
      
   def do_listequal(self, args):
      model = self.model
      if self.simulation.is_active():
         model = self.simulation
         
      nodes = model.get_nodes()

      equal = {}
      for source in nodes:
         for dest in nodes:
            if source == dest: continue
            length, paths = model.path(source, dest)
            if len(paths) > 1:
               if source in equal:
                  equal[source].append((dest, len(paths)))
               else:
                  equal[source] = [(dest, len(paths))]

      if not equal:
         print "No equal-cost paths found in model."
         return

      print "Equal-cost paths ('path source dest' for details):"
      self.tw.initial_indent = ' '*2
      self.tw.subsequent_indent = ' '*25
      self.tw.width=80-23
      for source,dests in sorted(equal.items()):
         dststr = []
         for d in sorted(dests):
            appstr = ""
            if d[1] > 2:
               appstr = " (%s)" % d[1]
            dststr.append("%s%s" % (d[0], appstr))
         print "  * %-15s -> %s" % (source, self.tw.fill(", ".join(dststr)))
      return                        
         
   def do_path(self, args):
      subargs = args.split()
      if not len(subargs) == 2:
         self.help_path()
         return
      a, b = subargs[0], subargs[1]
      if a not in self.model.get_nodes():
         print "%s not a valid node, type 'list' to see nodes"\
               % (a)
         return
      if b not in self.model.get_nodes():
         print "%s not a valid node, type 'list' to see nodes"\
               % (b)
         return

      length, paths = self.model.path(a,b)

      if not length and not paths:
         print "No valid path from %s to %s in model"
         return
      
      print "Path from %s to %s:"\
            % (a, b)
      print "  * Cost: %d" % (length)
      if len(paths) > 1:
         print "  * %d paths total (equal cost):" % len(paths)
         #if len(paths) != len(selection):
         #   print "    => %d path(s) preferred" % (len(selection))
         print
      self.tw.initial_indent = ' '*5
      self.tw.subsequent_indent = ' '*6
      for path in paths:
         print "  * Hops: %d" % (len(path))
         print "  * Path:\n", self.tw.fill(" => ".join(path))
         print "  * Slowest link: ", self.model.get_path_capacity(path,
                                                                  True,
                                                                  True)
         #if len(paths) != len(selection) and path in selection:
         #   print "     <<preferred path>>"
         print

      self.gui.clear()
      graphdata = {}

      path = paths[0]
      hops = str(len(paths[0]))
      if len(paths) > 1:
         path = reduce(lambda x,y: x+y, paths)
         hops = "/".join(map(lambda x: str(len(x)), paths))
      G = self.model.G
      graphdata['nodegroups'] = self.model.get_node_groups(path=path)
      graphdata['edgegroups'] = self.model.get_edge_groups(path=paths)
      graphdata['labels'] = utils.short_names(G.nodes())
      graphdata['edgelabels'] = utils.edge_labels(G.edges(data=True),
                                                  graphdata['edgegroups'])
      graphdata['pos'] = self.model.get_positions(G.nodes())
      graphdata['title'] = "Path(s) from %s to %s (cost: %d, %s hops)" \
                           % (a, b, length, hops)
      
      self.gui.plot(G, graphdata, 0.7)

   def do_sim(self, args):
      return self.do_simulation(args)
      
   def do_help(self, args):
      if args:
         Cmd.do_help(self, args)
         return
      cmdlist = map(lambda x: x.ljust(12),
                map(lambda x: x.replace('help_', ''),
             filter(lambda x: x.startswith('help_'), dir(self))))
      
      print """
The program allows you to view topology and metrics, and to
simulate various changes to investigate their impact on the
routing.

To view the current topology use 'plot'. To trace the path
between two nodes use 'path A B'.

In order to make changes, make sure you enter
simulation mode first. This is done with the 'simulation'
(or just 'sim') command.

When in simulation mode you can view the current topology and trace paths
with 'simplot' and 'simpath' respectivly. You can view the
effects of the simulation with the 'changes' command.

If you want a graphical comparison of paths, use the 'diffpath' command
while in simulation mode.

More help is available per command, just type 'help <command>'.

Available commands:
=================="""

      cmdstring = " ".join(sorted(cmdlist))

      self.tw.initial_indent = ''
      self.tw.subsequent_indent = ''
      print self.tw.fill(cmdstring)
      print
      
   #
   # Completions
   #
   def complete_path(self, text, line, begidx, endidx):
      if self.debug:
         print "text:", text
         print "line:", line
      tokens = line.split()
      length = len(tokens)
      model = self.model
      if self.simulation.is_active():
         model = self.simulation
      if tokens[0] in ['metric', 'linkfail', 'linkinfo'] and \
             ((length == 2 and not text) or length >= 3):
         startnode = tokens[1]
         return filter(lambda x: x.startswith(text),
                       map(lambda x: x+" ",
                           model.graph.neighbors(startnode)))
      return filter(lambda x: x.startswith(text),
                    map(lambda x: x+" ",
                        model.graph.nodes()))

   def complete_colors(self, text, line, begidx, endidx):
      return ['on', 'off']
   
   def complete_simpath(self, text, line, begidx, endidx):
      return self.complete_path(text, line, begidx, endidx)

   def complete_diffpath(self, text, line, begidx, endidx):
      return self.complete_path(text, line, begidx, endidx)
   
   def complete_metric(self, text, line, begidx, endidx):
      length = len(line.split())
      if length == 3 and not text:
         return []
      elif length == 4 and not text:
         return ['one-way', 'two-way']
      elif length == 5:
         return filter(lambda x: x.startswith(text), ['one-way', 'two-way'])
      else:
         return self.complete_path(text, line, begidx, endidx)

   def complete_linkfail(self, text, line, begidx, endidx):
      return self.complete_path(text, line, begidx, endidx)

   def complete_routerfail(self, text, line, begidx, endidx):
      return self.complete_path(text, line, begidx, endidx)

   def complete_changes(self, text, line, begidx, endidx):
      length = len(line.split())
      return filter(lambda x: x.startswith(text), ['as-commands', 'no-effects']) \
            + self.complete_path(text, line, begidx, endidx)      

   def complete_stats(self, text, line, begidx, endidx):
      return []

   def complete_plot(self, text, line, begidx, endidx):
      return filter(lambda x: x.startswith(text), ['with-load'])

   def complete_simplot(self, text, line, begidx, endidx):
      return self.complete_plot(text, line, begidx, endidx)
   
   def complete_info(self, text, line, begidx, endidx):
      return self.complete_path(text, line, begidx, endidx)

   def complete_linkinfo(self, text, line, begidx, endidx):
      return self.complete_path(text, line, begidx, endidx)   
   
   def complete_reroute(self, text, line, begidx, endidx):
      length = len(line.split())
      if length == 4 and not text:
         return ['equal-path']
      elif length == 5:
         return filter(lambda x: x.startswith(text), ['equal-path'])
      else:
         return self.complete_path(text, line, begidx, endidx)

   def complete_anycast(self, text, line, begidx, endidx):
      tokens = line.split()
      length = len(tokens)      
      if length == 1 and not text:
         return ['add ', 'remove ', 'clear']
      elif length == 2 and text:
         return filter(lambda x: x.startswith(text),
                         ['add ', 'remove ', 'clear'])
      elif length >= 2:
         if tokens[1] == 'clear':
            return []
         if tokens[1] == 'remove':
            return filter(lambda x: x.startswith(text),
                          map(lambda x: x+" ",
                              self.simulation.get_anycast_nodes()))
         return self.complete_path(text, line, begidx, endidx)

   #
   # Help-methods
   #
   def help_version(self):
      print """
            Usage: version

            Display the program version.
            """

   def help_colors(self):
      print """
            Usage: colors (on|off)

            Display or set current color-setting.
            When set to off no ANSI colors will be used
            in the terminal output.
            """
   
   def help_path(self):
      print """
            Usage: path A B

            Display shortest path from node A to node B.
            Alternate equal-cost paths will be shown
            with dashed lines.
            """

   def help_simpath(self):
      print """
            Usage: simpath A B

            Display shortest path from node A to node B
            given the current simulated changes.
            Alternate equal-cost paths will be shown
            with dashed lines.
            """

   def help_diffpath(self):
      print """
            Usage: diffpath A B

            Display shortest path from node A to node B
            given the current simulated changes, and show
            the difference compared to the original path.

            The original path will be drawn in light
            yellow, while nodes and paths common to both
            will be drawn with a dark blue.

            Alternate equal-cost paths will be shown
            with dashed lines.
            """

   def help_reroute(self):
      print """
            Usage: reroute A B C (equal-path)

            Try to find a suitable set of metric changes
            to make the shortest path from A to B go
            through C. Works on the current simulated model.

            If the equal-path option is given the result will
            contain multiple cost-equal paths if possible.
            """

   def help_anycast(self):
      print """
            Usage: anycast (add|remove|clear) (<node1 node2 ...>)

            Add or remove nodes as anycast nodes.
            When no argument is given list the current anycast
            nodes and display a plot of anycast members.            
            """
            
   def help_minimize(self):
      print """
            Usage: minimize

            Try to reduce as many of the metrics as possible,
            making them as small as possible whilst preserving
            every shortest path in the simulated model.
            """

   def help_load(self):
      print """
            Usage: load <filename>

            Load the topology and metrics from
            the given file, replacing base model.

            Useful if you work with several models,
            you don't have to restart the program
            to switch between them.
            """
      
   def help_reload(self):
      print """
            Usage: reload

            Reload the topology and metrics from
            file, replacing base model.
            """
      
   def help_plot(self):
      print """
            Usage: plot (with-load)

            Display metrics and topology graphically.

            If given the 'with-load' option, instead
            display current link utilizations.
            """

   def help_asymmetric(self):
      print """
            Usage: asymmetric

            List links with asymmetric metrics.
            If a simulation is active this shows
            data from the current simulated topology.
            """

   def help_quit(self):
      print """
            Usage: quit

            End program.
            """

   def help_areaplot(self):
      print """
            Usage: areaplot

            Display metrics and topology graphically.
            Show the different IS-IS areas with different
            colors
            """      

   def help_stop(self):
      print """
            Usage: stop

            End current simulation.
            """

   def help_list(self):
      print """
            Usage: list

            List the name of all nodes.
            """

   def help_linkinfo(self):
      print """
            Usage: linkinfo <source> <destination>

            Show various information about the link
            between the source and destination nodes.
            If available, also shows the capacity and
            current utilization i Mbit/s.
            """

   def help_listequal(self):
      print """
            Usage: listequal

            List all equal-cost (source, destinations) pairs.
            If a simulation is active, the simulated model
            is used in the computations.

            If number of paths is greater than two, the
            number of paths is printed after each destination.
            """
      
   def help_png(self):
      print """
            Usage: png (filename)

            Save current topology to a PNG file.
            If a simulation is active, saves the
            simulated topology.
            
            """
      
   def help_simplot(self):
      print """
            Usage: simplot (with-load)

            Display metrics and topology graphically
            given the current simulated changes.

            If any anycast nodes exists, plot with
            anycast groups displayed (equivalent to
            running 'anycast' without arguments)

            If given the 'with-load' option, instead
            display current simulated link utilizations.
            """

   def help_stats(self):
      print """
            Usage: stats

            Display some statistics of current
            topology. If a simulation is active
            this shows statistics for the current
            simulated topology.
            """

   def help_linkloads(self):
      print """
            Usage: linkloads

            Fetch updated link utilizations for the
            last hour (average load). At the moment this
            is a UNINETT specific command.
            """

   def help_info(self):
      print """
            Usage: info <node>

            Display some information and stats
            about the given node. If a simulation
            is active this uses the data from
            the simulation.
            """
   
   def help_changes(self):
      print """
            Usage: changes (as-commands|no-effects) (<source>)

            Display current simulated changes
            and their effects. For more detail
            provide an optional source node.

            To show current changes as commands (i.e.
            to input on another machine/simulation)
            use the 'as-commands' option.

            To list only the changes, and not the effects
            use the 'no-effects' option. These options can
            both be given at the same time.
            """

   def help_sim(self):
      self.help_simulation()
   
   def help_simulation(self):
      print """
            Usage: simulation (start|stop)

            Enter or leave simulation mode.  In order to simulate
            metric changes and failures you have to be in simulation
            mode.

            In simulation mode the prompt changes to reflect the
            changes to the model. The numbers and letters mean:

              * Nc     - #changes simulated
              * N:N:Nn - #nodes with changed paths to >= 1, 5
                         and 20 nodes.
              * N:N:Nu - #links with >= 50, 75 and 95% utilization
                         respectively. (If linkloads are active)
              * NL     - change in average shortest path length
              * Nr     - change in radius, ie. minimum all-pairs
                         shortest path distance.
              * Nd     - change in diameter, ie. maximum all-pairs
                         shortest path distance.
              * N:Np   - #partitions and #nodes not in the largest
                         partition, when parts of the network are cut
                         off from each other.

           If anycast simulation is active, the following numbers
           are displayed:

              * Na     - #anycast nodes/groups
              * N:N:Nm - #members belonging to each anycast group


           For more detail of changes, the 'stats', 'changes', 'linkinfo'
           and 'diffpath'-commands might be useful.
            """

   def help_metric(self):
      print """
            Usage:

            metric <src> <dst> <value> (one-way|two-way)

            Set metric for link between source and
            destination to value. If metric is symmetric
            this will set metric between destination
            and source as well.

            If the optional (one-way|two-way) argument is
            given the metric is applied accordingly.
            """

   def help_utilizations(self):
      print """
            Usage: utilizations

            Display any links with utilization >= 75%, as well
            as the top 10 utilized links.
            """

   def help_linkfail(self):
      print """
            Usage: linkfail <source> <destination>

            Simulate link failure between source and
            destination
            """

   def help_routerfail(self):
      print """
            Usage: routerfail <node>

            Simulate router failure by removing
            the node.
            """
      
   def help_undo(self):
      print """
            Usage: undo <change #>

            Undo the change with given number.
            Type 'changes' to get a list of the changes.
            """

   #
   # Quit-methods
   #
   def do_EOF(self, arg):
      if self.simulation.is_active():
         print
         self.do_simulation("stop")
         print
         return
      self.do_quit(arg)

   def do_quit(self, arg):
      print "Bye!"
      try:
         readline.write_history_file(self.histfile)
      except IOError:
         pass
      sys.exit(0)


   # Private methods
   def _colormode(self, on):
      self.bt = lambda x: colored(x, attrs=['bold'])
      self.pbt = lambda x: colored2(x, attrs=['bold'])      

      if on:
         self.termcolor = True
         self.prt = lambda x: colored2(x, 'red')
         self.blt = lambda x: colored(x, 'magenta')
         self.pblt = lambda x: colored2(x, 'magenta')
         self.pbblt = lambda x: colored3(x, 'magenta')
         self.pgrt = lambda x: colored2(x, 'cyan')
         self.pbgrt = lambda x: colored3(x, 'cyan')         
      else:
         self.termcolor = False
         self.blt = lambda x: x
         self.prt = lambda x: x
         self.pblt = lambda x: x
         self.pbblt = self.pbt
         self.pgrt = lambda x: x
         self.pbgrt = self.pbt         
   
