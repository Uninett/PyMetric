import networkx as nx
from pajek import read_pajek
import utils

class Model:

   def __init__(self, graph, config, debug=False):
      self.graph = graph
      self.config = config
      self.debug = debug
      self.G = self._make_weighted_copy()
      self._refresh_betweenness()
      self.linkloads = {}
      self.all_paths = {}
      self.paths_using_edge = {}
      self.linkload_parts = {}
      self._refresh_all_paths()

   def refresh_linkloads(self):
      if not self.config.get('use_linkloads'): return False
      self.linkloads = utils.read_linkloads(self.graph,
                                            self.config.get('linkloads_host'),
                                            self.config.get('linkloads_url'))
      if not self.linkloads: return False
      self.linkload_parts = {}
      return True

   def has_linkloads(self):
      return len(self.linkloads.keys()) > 0
   
   def get_in_link_load(self, u,v):
      if not (u,v) in self.linkloads:
         return False
      return int(self.linkloads[(v,u)])

   def get_out_link_load(self, u,v):
      if not (u,v) in self.linkloads:
         return False
      return int(self.linkloads[(u,v)])
   
   def get_betweenness(self, top=None):
      if not top:
         return self.betweenness
      bc = self.betweenness
      toplist = sorted(bc, lambda x,y: cmp(bc[x], bc[y]))
      toplist.reverse()
      return toplist[:top]      

   def get_edge_betweenness(self, top=None):
      if not top:
         return self.edge_betweenness
      ebc = self.edge_betweenness
      toplist = sorted(ebc,
                       lambda (x1,y1), (x2, y2): cmp(ebc[(x1, y1)], ebc[(x2, y2)]))
      toplist.reverse()
      return toplist[:top]

   def uneven_metrics(self):
      G = self.G
      return filter(lambda x: G[x[0]][x[1]] != G[x[1]][x[0]],
                    G.edges())

   def get_total_in_load(self, node, G=None, loads=None):
      sum = 0
      if not loads: loads = self.linkloads
      if not G: G = self.graph
      for neighbor in G[node]:
         sum += loads[neighbor, node]
      return sum

   def get_total_out_load(self, node, G=None, loads=None):
      sum = 0
      if not loads: loads = self.linkloads
      if not G: G = self.graph
      for neighbor in G[node]:
         sum += loads[node, neighbor]
      return sum

   def get_transit_links(self, u, v):
      paths = self.nodes_and_paths_using_edge(u,v,self.G, True)[1]
      return paths.keys()

   def nodes_and_paths_using_edge(self, u, v, G=None, transit_only=False):
      import time
      stime = time.time()
      if not G:
         G = self.G

      if not transit_only and (G == self.G or G == self.graph) and (u,v) in self.paths_using_edge:
         return self.paths_using_edge[(u,v)]
      
      candidates = set()
      retpaths = {}
      #print "      Finding candidates (%s secs)" % (time.time() - stime)
      for node in G:
         if node == v: continue
         paths = self.path(node, v, G)
         if not paths: continue
         for path in paths[1]:
            if path[-2] == u:
               candidates.add(node)
      #print "      Done. (%s secs)" % (time.time() - stime)

      for node in candidates:
         for dest in (set(G.nodes()) - candidates):
            paths = self.path(node, dest, G)
            if not paths: continue
            paths = paths[1]
            for path in paths:
               edges = zip(path, path[1:])
               if (u,v) in edges:
                  if (node,dest) not in retpaths:
                     if transit_only:
                        if node not in (u,v) and dest not in (u,v):
                           retpaths[(node,dest)] = [path]
                     else:
                        retpaths[(node,dest)] = [path]
                  else:
                     if transit_only:
                        if node not in (u,v) and dest not in (u,v):
                           retpaths[(node,dest)].append(path)
                     else:
                        retpaths[(node,dest)].append(path)
      #print "      Returning (%s secs)" % (time.time() - stime)
      if not transit_only:
         self.paths_using_edge[(u,v)] = (candidates, retpaths)
      return candidates, retpaths

   def get_link_load_part(self, u, v, loads=None, G=None):
      import time
      stime = time.time()
      use_cache = False
      if not G:
         G = self.G
      if not loads:
         loads = self.linkloads

      if loads == self.linkloads:
         use_cache = True
         #print "    Cache is possible, keys:"
         #print "    %s" % self.linkload_parts.keys()

      if use_cache and (u,v) in self.linkload_parts:
         #print "  Returning from cache (%s secs)" % (time.time() - stime)
         return self.linkload_parts[(u,v)]
      
      #print "  Finding nodes_and_paths (%s, %s) (%s secs)" % (u,v,time.time()-stime)
      nodes, pathlist = self.nodes_and_paths_using_edge(u, v, G)
      #print "  Nodes: %s    --    Pathlist: %s" % (nodes, pathlist)
      #print "  Done. (%s secs)" % (time.time()-stime)
      partloads = {}
      counts = {}
      
      for paths in pathlist.values():
         numpaths = len(paths)
         pathloads = {}
         for path in paths:
            #print "  Finding path_loads (%s, %s) (%s secs)" % (u,v,time.time()-stime)            
            edges = self.get_path_loads(u, v, path, numpaths, loads, G)
            for (s,t) in edges:
               if (s,t) not in pathloads:
                  pathloads[(s,t)] = edges[(s,t)]
               else:
                  pathloads[(s,t)] += edges[(s,t)]
         partloads.update(pathloads)
                  

      for (s,t) in partloads:
         try:
            assert float(partloads[(s,t)]) -1 <= float(loads[(s,t)])
         except:
            print "Assertion failed for (%s,%s) %s > %s" \
                % (s,t, partloads[(s,t)], loads[(s,t)])
      #print "  Returning (%s secs)" % (time.time()-stime)
      if use_cache:
         self.linkload_parts[(u,v)] = partloads
      return partloads
                  
         
         
   def get_path_loads(self, u, v, path,
                      numpaths=1,
                      loads=None,
                      G=None):

      if not loads:
         loads = self.linkloads
      if not G:
         G = self.G
      edges = zip(path, path[1:])
      if path[0] == u:
         pass
      elif path[-1] == v:         
         edges.reverse()
      elif (u,v) not in edges:
         print "Invalid call:"
         print "get_path_loads: (%s -> %s) [%s]" % (u,v,path)         
         return False
      else:
         path1, path2 = [], []
         i = 0
         while path[i] != v:
            path1.append(path[i])
            i += 1
         path1.append(v)
         path2.append(u)
         path2.append(v)
         
         i += 1
         for node in path[i:]:
            path2.append(node)

         #print "Splitting call in two: %s, %s" % (path1, path2)

         res1 = self.get_path_loads(u,v,path1,numpaths,loads,G)
         res2 = self.get_path_loads(u,v,path2,numpaths,loads,G)

         res1.update(res2)

         return res1


      #print "get_path_loads: (%s -> %s) [%s]" % (u,v,path)
      cr = utils.calc_ratio
      ndio = utils.node_diff_in_out

      loadmatrix = {}

      loadmatrix[u] = {'out': loads[(u,v)] / float(numpaths)}
      loadmatrix[v] = { 'in': loads[(u,v)] / float(numpaths)}

      for i in range(len(edges)):
         (s,t) = edges[i]

         #print "Looking at [%s] (%s,%s)" % (i,s,t)
         if s in loadmatrix:
            if not 'out' in loadmatrix[s]:
               loadmatrix[s]['out'] = loadmatrix[s]['in'] * cr(G, loads,
                                                               s, t, True, False)
         
               #print "Load(in) :", loadmatrix[s]['in']
               #print "Load(out):", loadmatrix[s]['out']
            loadmatrix[t] = {'in': loadmatrix[s]['out']}
         elif t in loadmatrix:
            if not 'in' in loadmatrix[t]:
               newpath = path[:]
               #print "Newpath before slice: %s" % newpath
               newpath = newpath[-(i+2):]
               #print "Doing self(newpath: %s)" % newpath
               pathcalc = self.get_path_loads(newpath[0], newpath[1],
                                              newpath, numpaths, loads, G)
               loadmatrix[t]['in'] = pathcalc[(newpath[-2], newpath[-1])]
            loadmatrix[s] = {'out': loadmatrix[t]['in']}
         else:
            print "Can't find loaddata for (%s,%s)" % (s,t)

      edges = zip(path, path[1:])

      retloads = {}
      for (s,t) in edges:
         retloads[(s,t)] = loadmatrix[s]['out']

      return retloads

   def get_link_info(self, u, v):
      G = self.G
      if not G.has_edge(u,v): return {}
      bc = self.edge_betweenness
      retinfo = {}
      edgedata = self.graph.get_edge(u,v)
      name = ""
      capacity = 0
      if 'c' in edgedata:
         capacity = edgedata['c']
      if 'l' in edgedata:
         name = edgedata['l']

      utilization = "NA"
      if capacity != 0 and (u,v) in self.linkloads:
         utilization = "%.2f%%" % (self.get_link_utilization(u,v)*100)
      load = "NA"
      if (u,v) in self.linkloads:
         load = "%.2f Mbit/s" % (self.get_out_link_load(u,v)/float(1024))
      retinfo['name'] = name
      retinfo['betweenness'] = "%.3f (%.2f%% of max, %.2f%% of avg)" \
                             % (bc[(u,v)], (bc[(u,v)]/max(bc.values()))*100,
                                (bc[(u,v)]/(sum(bc.values())/len(bc)))*100)      
      retinfo['capacity'] = utils.cap2str(capacity)
      retinfo['load'] = load
      retinfo['utilization'] = utilization

      return retinfo   

   def get_node_info(self, node):
      G = self.graph
      if node not in G.nodes(): return {}
      bc = self.betweenness
      retinfo = {}
      retinfo['name'] = node
      retinfo['degree'] = G.out_degree(node)
      retinfo['links'] = map(lambda x: x[2]['l'] + \
                                " (" + str(int(x[2]['value'])) + ")",
                             G.edges(node, data=True))
      retinfo['neighbors'] = G.neighbors(node)
      retinfo['longest paths'] = self.get_max_cost_paths(nodes=[node])      
      retinfo['eccentricity'] = nx.eccentricity(G, node)
      retinfo['betweenness'] = "%.3f (%.2f%% of max, %.2f%% of avg)" \
                             % (bc[node], (bc[node]/max(bc.values()))*100,
                                (bc[node]/(sum(bc.values())/len(bc)))*100)

      return retinfo

   def get_max_cost_paths(self, top=8, nodes=None):
      sources = self.G.nodes()
      if nodes:
         sources = nodes
      pathcosts = {}
      retval = []
      for source in sources:
         costs = nx.dijkstra_predecessor_and_distance(self.G, source)[1]

         for dest in costs:
            pathcosts[(source, dest)] = costs[dest]

      spathcosts = sorted(pathcosts,
                          cmp=lambda x,y: cmp(pathcosts[x], pathcosts[y]))

      spathcosts.reverse()

      sp = spathcosts
      pc = pathcosts
      seen = {}

      for (u,v) in sp[:top]:
         if (u,v) in seen and pc[(u,v)] == pc[(v,u)]: continue
         retval.append("%s (%s)" % (" <-> ".join([u,v]), pc[(u,v)]))
         seen[(u,v)] = True
         if (v,u) in sp and pc[(u,v)] == pc[(v,u)]:
            seen[(v,u)] = True         

      return retval
   
   def get_node_groups(self, threshold=0.095, n=10, nodes=None, path=None):
      groups = {}
      bc = self.betweenness
      top = self.get_betweenness(top=n)

      for node in self.G.nodes():
         if nodes != None and node not in nodes:
            continue
         if bc[node] > threshold or node in top:
            if path and node == path[0]:
               if not 'mainstart' in groups:
                  groups['mainstart'] = [node]
               else:
                  groups['mainstart'].append(node)
            elif path and node == path[-1]:
               if not 'mainstop' in groups:
                  groups['mainstop'] = [node]
               else:
                  groups['mainstop'].append(node)
            elif path and node in path:
               if not 'mainpath' in groups:
                  groups['mainpath'] = [node]
               else:
                  groups['mainpath'].append(node)
            else:
               if not 'main' in groups:
                  groups['main'] = [node]
               else:
                  groups['main'].append(node)
         else:
            if path and node == path[0]:
               if not 'normalstart' in groups:
                  groups['normalstart'] = [node]
               else:
                  groups['normalstart'].append(node)
            elif path and node == path[-1]:
               if not 'normalstop' in groups:
                  groups['normalstop'] = [node]
               else:
                  groups['normalstop'].append(node)
            elif path and node in path:
               if not 'normalpath' in groups:
                  groups['normalpath'] = [node]
               else:
                  groups['normalpath'].append(node)
            else:
               if not 'normal' in groups:
                  groups['normal'] = [node]
               else:
                  groups['normal'].append(node)

      return [(groups[k], k) for k in groups]

   def get_path_capacity(self, path, as_string=False, slowest_only=False):

      path_links = zip(path, path[1:])
      slowest = None
      if slowest_only:
         slowest = min([self.get_link_capacity(u,v)
                        for (u,v) in path_links])
         if as_string:
            return utils.cap2str(slowest)
         return slowest
         
      return [self.get_link_capacity(u,v,as_string) for (u,v) in path_links]      
   
   def get_link_capacity(self, u, v, as_string=False):
                  
      if not self.graph.has_edge(u,v):
         return False

      linkinfo = self.graph.get_edge(u,v)

      if not 'c' in linkinfo:
         if as_string:
            return "Unknown"
         return False

      if as_string:
         return utils.cap2str(int(linkinfo['c']))
      return int(linkinfo['c'])

   def get_link_utilization(self, u, v):
      return self.get_out_link_load(u,v)/float(self.get_link_capacity(u,v))

   def get_link_utilizations(self):
      utils = {}
      for (u,v) in self.G.edges():
         utils[(u,v)] = self.get_link_utilization(u,v)
         
      return utils

   def has_capacity_info(self):
      for (u,v) in self.graph.edges():
         if 'c' in self.graph[u][v]:
            return True
      return False
   
   def get_edge_groups(self, threshold=0.01, n=20, edges=None, path=None):
      groups, mpath_edges, rpath_edges = {}, [], []
      multi = False
      mpath = path
      if path != None:
         if type(path[0]) == type([]):
            if len(path) > 1: multi = True
            mpath = path[0]
            
            for p in path[1:]:
               rpath_edges += zip(p, p[1:])
               rpath_edges += zip(p[1:], p)

         mpath_edges = zip(mpath, mpath[1:])
         mpath_edges += zip(mpath[1:], mpath)
      
      ebc = self.edge_betweenness
      top = self.get_edge_betweenness(top=n)

      for (u, v, d) in self.G.edges(data=True):
         if edges != None and (u, v) not in edges:
            continue
         if (ebc[(u,v)] > threshold and ebc[(v,u)] > threshold) \
                or (u,v) in top:
            #print "Path: %s, multi: %s, (%s,%s), %s" % (path, multi, u,v,mpath_edges)
            if (path != None) and (not multi) and ((u,v) in mpath_edges):
               if 'mainpath' not in groups:
                  groups['mainpath'] = [(u,v,d)]
               else:
                  groups['mainpath'].append((u,v,d))
            elif multi and mpath_edges and (u,v) in mpath_edges \
                   and (u,v) not in rpath_edges:
               if 'mainaltpath' not in groups:
                  groups['mainaltpath'] = [(u,v,d)]
               else:
                  groups['mainaltpath'].append((u,v,d))
            elif mpath_edges and (u,v) in mpath_edges:
               if 'mainpath' not in groups:
                  groups['mainpath'] = [(u,v,d)]
               else:
                  groups['mainpath'].append((u,v,d))                  
            elif rpath_edges and (u,v) in rpath_edges:
               if 'mainaltpath' not in groups:
                  groups['mainaltpath'] = [(u,v,d)]
               else:
                  groups['mainaltpath'].append((u,v,d))
            else:
               if 'main' not in groups:
                  groups['main'] = [(u,v,d)]
               else:
                  groups['main'].append((u,v,d))
         else:
            if path != None and not multi and (u,v) in mpath_edges:
               if 'normalpath' not in groups:
                  groups['normalpath'] = [(u,v,d)]
               else:
                  groups['normalpath'].append((u,v,d))
            elif multi and mpath_edges and (u,v) in mpath_edges \
                   and (u,v) not in rpath_edges:
               if 'normalaltpath' not in groups:
                  groups['normalaltpath'] = [(u,v,d)]
               else:
                  groups['normalaltpath'].append((u,v,d))
            elif mpath_edges and (u,v) in mpath_edges:
               if 'normalpath' not in groups:
                  groups['normalpath'] = [(u,v,d)]
               else:
                  groups['normalpath'].append((u,v,d))
            elif rpath_edges and (u,v) in rpath_edges:
               if 'normalaltpath' not in groups:
                  groups['normalaltpath'] = [(u,v,d)]
               else:
                  groups['normalaltpath'].append((u,v,d))
            else:
               if 'normal' not in groups:
                  groups['normal'] = [(u,v,d)]
               else:
                  groups['normal'].append((u,v,d))

      return [(groups[k], k) for k in groups]
   
   def get_nodes(self):
      return self.G.nodes()

   def get_areas(self, nodes):
      na = self.graph.node_attr
      areas = {}
      for n in nodes:
         if 'area' in na[n]:
            areas[n] = na[n]['area']
         else:
            areas[n] = None
            
      return areas

   def get_positions(self, nodes):
      na = self.graph.node_attr
      pos = {}
      for n in nodes:
         pos[n] = (float(na[n]['x']), float(na[n]['y']))
      return pos
   
   def get_stats(self):
      top = self.get_betweenness(top=20)
      stats = {}
      stats["nodes"] = nx.number_of_nodes(self.graph)
      stats["edges"] = nx.number_of_edges(self.graph)
      stats["radius"] = nx.radius(self.graph)
      stats["diameter"] = nx.diameter(self.graph)
      stats["center"] = nx.center(self.graph)
      stats["periphery"] = nx.periphery(self.graph)
      stats["density"] = nx.density(self.graph)
      stats["reciprocity"] = utils.reciprocity(self.graph)
      stats["mean length"] = utils.mean_shortest_path_length(self.graph)
      stats["longest paths"] = self.get_max_cost_paths()      
      stats["top 20 transit"] = top

      return stats
   
   def path(self, source, dest, G=None):

      if not G:
         G = self.G
      if G == self.G:
         preds, costs = self.all_paths[source]
      else:
         preds, costs = nx.dijkstra_predecessor_and_distance(G, source)

      if not dest in costs:
         return False, None

      def _get_paths(preds, path, paths, dest):
         if dest in path:
            return
         path.append(dest)
         if len(preds[dest]) == 0:
            paths.append(path)            
            return
         for newdest in preds[dest]:
            _get_paths(preds, path[:], paths, newdest)
         return paths

      paths = _get_paths(preds, [], [], dest)
      for path in paths:
         path.reverse()

      return costs[dest], paths

   def refresh_from_file(self, filename):
      self.graph = read_pajek(filename)
      self.G = self._make_weighted_copy()
      self.linkloads = {}
      self._refresh_betweenness()
      self._refresh_all_paths()

   def _make_weighted_copy(self):
      G = self.graph.copy()
      for (u,v,d) in G.edges(data=True):
         G.delete_edge(u,v)
         G.add_edge(u,v,d['value'])
      return G

   def _refresh_betweenness(self):
      self.betweenness = nx.load_centrality(self.G, weighted_edges=True)
      self.edge_betweenness = nx.edge_betweenness(self.G, True, True)

   def _refresh_all_paths(self):
      for node in self.G:
         self.all_paths[node] = nx.dijkstra_predecessor_and_distance(self.G, node)
      for edge in self.G.edges():
         self.paths_using_edge[edge[0], edge[1]] = \
                          self.nodes_and_paths_using_edge(edge[0], edge[1], self.G)

   def _routeselection(self, paths):
      p_attr = {}

      pathnodes = reduce(lambda x,y: x+y, paths)
      areas = self.get_areas(pathnodes)

      if not areas:
         return paths

      for i in range(len(paths)):
         areahops = map(lambda x: areas[x[0]] == areas[x[1]],
                        zip(paths[i], paths[i][1:]))
         p_attr[i] = {'areahops': areahops, 'candidate': True}

      for hop in range(1, max([2] + [len(p) for p in paths]) - 2):
         diff = False
         last_hop = None
         last_areahop = None
         for i, path in enumerate(paths):
            if hop+1 > len(path) - 1: continue
            if p_attr[i]['candidate'] == False: continue

            pathhop = (path[hop], path[hop+1])
            pathah  = p_attr[i]['areahops'][hop]

            print "Comparing %s to %s and %s to %s (hop %s)" \
                  % (pathhop, last_hop, pathah, last_areahop, hop)
            if last_hop == None:
               last_hop = pathhop
               last_areahop = pathah
            elif pathhop != last_hop:
               if pathah != last_areahop:
                  diff = True
                  print "breaking at hop %s" % hop
                  break

         if diff:
            for i in range(len(paths)):
               if hop > len(paths[i]) - 1: continue
               print "Looking at path %s with areahops %s, index %s" \
                  % (paths[i], p_attr[i]['areahops'], hop)
               if p_attr[i]['areahops'][hop] != True:
                  p_attr[i]['candidate'] = False
            diff = False

      return [paths[i] for i in range(len(paths)) if p_attr[i]['candidate']]

            
class Simulation:

   SC_METRIC      = 1
   SC_LINKFAIL    = 2
   SC_ROUTERFAIL  = 4
   
   def __init__(self, model, debug=False):
      self.model = model
      self.graph = model.G.copy()
      self.active = False
      self.changes = []
      self._refresh_betweenness()
      self.debug = debug
      self.acnodes = set()
      self.acgroups = {}
      self.all_paths = {}
      self._refresh_all_paths()
      self.linkloads = self.model.linkloads
      
   def get_stats(self):
      bc = self.betweenness
      top = sorted(bc, lambda x,y: cmp(bc[x], bc[y]))
      top.reverse()

      stats = {}
      stats["nodes"] = nx.number_of_nodes(self.graph)
      stats["edges"] = nx.number_of_edges(self.graph)
      stats["radius"] = nx.radius(self.graph)
      stats["diameter"] = nx.diameter(self.graph)
      stats["center"] = nx.center(self.graph)
      stats["periphery"] = nx.periphery(self.graph)
      stats["density"] = nx.density(self.graph)
      stats["reciprocity"] = utils.reciprocity(self.graph)
      stats["mean length"] = utils.mean_shortest_path_length(self.graph)
      stats["longest paths"] = self.get_max_cost_paths()
      stats["top 20 transit"] = top[0:20]

      return stats

   def get_link_utilization(self, u, v):
      return self.get_out_link_load(u,v)/float(self.model.get_link_capacity(u,v))

   def get_link_utilizations(self):
      utils = {}
      for (u,v) in self.graph.edges():
         utils[(u,v)] = self.get_link_utilization(u,v)
         
      return utils
   def get_in_link_load(self, u,v):
      if not (u,v) in self.linkloads:
         return False
      return int(self.linkloads[(v,u)])

   def get_out_link_load(self, u,v):
      if not (u,v) in self.linkloads:
         return False
      return int(self.linkloads[(u,v)])
   
   def get_node_info(self, node):
      G = self.graph
      if node not in G.nodes(): return {}
      bc = self.betweenness
      retinfo = {}
      retinfo['name'] = node
      retinfo['degree'] = G.out_degree(node)
      retinfo['links'] = map(lambda x: self.model.graph.get_edge(x[0], x[1])['l']\
                                + " (" + str(int(x[2])) + ")",
                             G.edges(node, data=True))
      retinfo['neighbors'] = G.neighbors(node)
      retinfo['longest paths'] = self.get_max_cost_paths(nodes=[node])
      retinfo['eccentricity'] = nx.eccentricity(G, node)
      retinfo['betweenness'] = "%.3f (%.2f%% of max, %.2f%% of avg)" \
                             % (bc[node], (bc[node]/max(bc.values()))*100,
                                (bc[node]/(sum(bc.values())/len(bc)))*100)      
      if self.acnodes:
         acstr = " and ".join(self.acgroups[node])
         retinfo['anycast group'] = acstr
         if node in self.acnodes:
            retinfo['anycast group'] += '*'

      return retinfo

   def get_link_info(self, u, v):
      G = self.graph
      if not G.has_edge(u,v): return {}
      bc = self.edge_betweenness
      retinfo = {}
      edgedata = self.model.graph.get_edge(u,v)
      name = ""
      capacity = 0
      if 'c' in edgedata:
         capacity = edgedata['c']
      if 'l' in edgedata:
         name = edgedata['l']

      utilization = "NA"
      if capacity != 0 and (u,v) in self.linkloads:
         utilization = "%.2f%%" % (self.get_link_utilization(u,v)*100)
      load = "NA"
      if (u,v) in self.linkloads:
         load = "%.2f Mbit/s" % (self.get_out_link_load(u,v)/float(1024))
      retinfo['name'] = name
      retinfo['betweenness'] = "%.3f (%.2f%% of max, %.2f%% of avg)" \
                             % (bc[(u,v)], (bc[(u,v)]/max(bc.values()))*100,
                                (bc[(u,v)]/(sum(bc.values())/len(bc)))*100)      
      retinfo['capacity'] = utils.cap2str(capacity)
      retinfo['load'] = load
      retinfo['utilization'] = utilization

      return retinfo

   def get_transit_links(self, u, v):
      paths = self.model.nodes_and_paths_using_edge(u,v,self.G, True)[1]
      return paths.keys()

   def get_max_cost_paths(self, top=8, nodes=None):
      sources = self.graph.nodes()
      if nodes:
         sources = nodes
      pathcosts = {}
      retval = []
      for source in sources:
         costs = nx.dijkstra_predecessor_and_distance(self.graph, source)[1]

         for dest in costs:
            pathcosts[(source, dest)] = costs[dest]

      spathcosts = sorted(pathcosts,
                          cmp=lambda x,y: cmp(pathcosts[x], pathcosts[y]))

      spathcosts.reverse()

      sp = spathcosts
      pc = pathcosts
      seen = {}
      for (u,v) in sp[:top]:
         if (u,v) in seen and pc[(u,v)] == pc[(v,u)]: continue
         retval.append("%s (%s)" % (" <-> ".join([u,v]), pc[(u,v)]))
         seen[(u,v)] = True
         if (v,u) in sp and pc[(u,v)] == pc[(v,u)]:
            seen[(v,u)] = True         

      return retval
   
   def start(self):
      self.active = True
      self.graph = self.model.G.copy()
      self._refresh_betweenness()
      self.changes = []
      self.effects = []
      self.linkloads = self.model.linkloads

   def stop(self):
      self.acnodes = set()
      self.acgroups = {}
      self.active = False

   def get_changes(self):
      return self.changes

   def get_changes_strings(self, commands=False):
      strings = []
      for change in self.changes:         
         if change['type'] == Simulation.SC_METRIC:
            connector = "->"
            bidirstr = " one-way"
            if change['bidir']:
               connector = "<->"
               bidirstr = ""
            if commands:
               strings.append("metric %s %s %s%s"\
                     % (change['pair'][0], change['pair'][1],
                        change['metrics'][1], bidirstr))
               continue
            strings.append("Metric for %s%s%s [%s->%s]"\
                  % (change['pair'][0], connector, change['pair'][1],
                     change['metrics'][0], change['metrics'][1]))
         elif change['type'] == Simulation.SC_LINKFAIL:
            if commands:
               strings.append("linkfail %s %s"\
                     % (change['pair'][0], change['pair'][1]))
               continue
            strings.append("Link failure between %s and %s" \
                  % (change['pair'][0], change['pair'][1]))
         elif change['type'] == Simulation.SC_ROUTERFAIL:
            if commands:
               strings.append("routerfail %s"\
                     % (change['node']))
               continue                                 
            strings.append("Router failure of %s" \
                  % (change['node']))
      return strings

   def uneven_metrics(self):
      G = self.graph
      return filter(lambda x: G[x[0]][x[1]] != G[x[1]][x[0]],
                    G.edges())
   
   def has_changes(self):
      return len(self.changes) > 0

   def no_changes(self):
      return len(self.changes)

   def get_effects(self):
      return self.effects

   def get_effects_node(self, node):
      if not node in self.effects: return {}
      return self.effects[node]

   def get_effects_summary(self):

      dstsummary, srcsummary = {}, {}
      
      for source in self.effects:
         no_changes = 0

         for dest in self.effects[source].keys():
            ddiffs = self.effects[source][dest]
            no_changes += len(ddiffs)            
            if dest in dstsummary:
               dstsummary[dest].append(source)
            else:
               dstsummary[dest] = [source]
            if source in srcsummary:
               srcsummary[source].append(dest)
            else:
               srcsummary[source] = [dest]

      return srcsummary, dstsummary
      
   def get_nodes(self):
      return self.graph.nodes()

   def get_betweenness(self, top=None):
      if not top:
         return self.betweenness
      bc = self.betweenness
      toplist = sorted(bc, lambda x,y: cmp(bc[x], bc[y]))
      toplist.reverse()
      return toplist[:top]      

   def get_edge_betweenness(self, top=None):
      if not top:
         return self.edge_betweenness
      ebc = self.edge_betweenness
      toplist = sorted(ebc, lambda (x1,y1), (x2, y2): cmp(ebc[(x1, y1)], ebc[(x2, y2)]))
      toplist.reverse()
      return toplist[:top]

   def get_anycast_groups_by_source(self):
      return self.acgroups
   
   def get_anycast_group(self, node):
      if node not in self.acnodes:
         return None
      return filter(lambda x: node in self.acgroups[x], self.acgroups.keys())
   
   def get_anycast_nodes(self):
      return list(self.acnodes)

   def add_anycast_nodes(self, nodes):
      self.acnodes.update(nodes)
      self._refresh_anycast()

   def remove_anycast_nodes(self, nodes):
      for n in nodes:
         self.acnodes.discard(n)
      self._refresh_anycast()
   
   def get_node_groups(self, threshold=0.095, n=10, path=None):
      groups = {}
      bc = self.betweenness
      top = self.get_betweenness(top=n)
      for node in self.graph.nodes():
         if bc[node] > threshold or node in top:
            if path and node == path[0]:
               if not 'mainstart' in groups:
                  groups['mainstart'] = [node]
               else:
                  groups['mainstart'].append(node)
            elif path and node == path[-1]:
               if not 'mainstop' in groups:
                  groups['mainstop'] = [node]
               else:
                  groups['mainstop'].append(node)            
            elif path and node in path:
               if not 'mainpath' in groups:
                  groups['mainpath'] = [node]
               else:
                  groups['mainpath'].append(node)
            else:
               if not 'main' in groups:
                  groups['main'] = [node]
               else:
                  groups['main'].append(node)
         else:
            if path and node == path[0]:
               if not 'normalstart' in groups:
                  groups['normalstart'] = [node]
               else:
                  groups['normalstart'].append(node)
            elif path and node == path[-1]:
               if not 'normalstop' in groups:
                  groups['normalstop'] = [node]
               else:
                  groups['normalstop'].append(node)            
            elif path and node in path:
               if not 'normalpath' in groups:
                  groups['normalpath'] = [node]
               else:
                  groups['normalpath'].append(node)
            else:
               if not 'normal' in groups:
                  groups['normal'] = [node]
               else:
                  groups['normal'].append(node)

      return [(groups[k], k) for k in groups]

   def get_diff_edge_groups(self, path, spath, threshold=0.01, n=20):      
      groups = {}

      #print "get_diff_edge_groups called (%s, %s)" % (path, spath)
      smpath_edges, srpath_edges = [], []
      mpath_edges, rpath_edges = [], []
      
      smpath = spath
      mpath = path
      
      if type(spath[0]) == type([]):
         if len(spath) > 1: smulti = True
         smpath = spath[0]
            
         for p in spath[1:]:
            srpath_edges += zip(p, p[1:])
            srpath_edges += zip(p[1:], p)

      if type(path[0]) == type([]):
         if len(path) > 1: multi = True
         mpath = path[0]
            
         for p in path[1:]:
            rpath_edges += zip(p, p[1:])
            rpath_edges += zip(p[1:], p)
      
      mpath_edges = zip(mpath, mpath[1:])
      mpath_edges += zip(mpath[1:], mpath)
      smpath_edges = zip(smpath, smpath[1:])
      smpath_edges += zip(smpath[1:], smpath)      

      mopath_edges = list(set(mpath_edges) - set(smpath_edges))
      mupath_edges = list(set(mpath_edges).intersection(set(smpath_edges)))

      ropath_edges = list(set(rpath_edges) - set(srpath_edges))
      #rupath_edges = list(set(rpath_edges).intersection(set(srpath_edges)))


      #print "mpath:  %s" % mpath_edges
      #print "rpath:  %s" % rpath_edges
      #print "smpath: %s" % smpath_edges
      #print "srpath: %s" % srpath_edges

      a = set(srpath_edges) ^ set(smpath_edges)
      b = set(rpath_edges) ^ set(mpath_edges)

      if not srpath_edges and not rpath_edges:
         a = set()
         b = set()

      c = set(srpath_edges).intersection((a & b))
      d = set(smpath_edges).intersection((a & b))
      rupath_edges = list(c|d)

      #rupath_edges = list(set(srpath_edges).intersection((a & b)))

      #print "mupath: %s" % mupath_edges
      #print "rupath: %s" % rupath_edges
      ebc = self.edge_betweenness
      top = self.get_edge_betweenness(top=n)


      redges = list(set(self.model.G.edges()) \
                  - set(self.graph.edges()))

      for (u, v, d) in self.graph.edges(data=True):
         debug = False
         #if u == 'oslo-gw' or v == 'oslo-gw': debug = True
         if debug: print "Looking at (%s, %s, %s)" % (u, v, d)
         if (u,v) in redges:
            if debug: print "In redges...ignoring"
            continue
         if (ebc[(u,v)] > threshold and ebc[(v,u)] > threshold) \
                or (u,v) in top:            
            if debug: print "Is main edge"
            
            if (u,v) in mupath_edges and (u,v) not in rupath_edges:
               if debug: print "Is mupath_edge"
               if 'mainupath' not in groups:
                  groups['mainupath'] = [(u,v,d)]
               else:
                  groups['mainupath'].append((u,v,d))
            elif (u,v) in rupath_edges:
               if debug: print "Is rupath_edge"                              
               if 'mainualtpath' not in groups:
                  groups['mainualtpath'] = [(u,v,d)]
               else:
                  groups['mainualtpath'].append((u,v,d))
            elif (u,v) in smpath_edges and srpath_edges and (u,v) not in srpath_edges:
               if debug: print "Is smpath_edge (not sr)"               
               if 'mainaltpath' not in groups:
                  groups['mainaltpath'] = [(u,v,d)]
               else:
                  groups['mainaltpath'].append((u,v,d))
            elif (u,v) in smpath_edges:
               if debug: print "Is smpath_edge"               
               if 'mainpath' not in groups:
                  groups['mainpath'] = [(u,v,d)]
               else:
                  groups['mainpath'].append((u,v,d))
            elif (u,v) in srpath_edges:
               if debug: print "Is srpath_edge"               
               if 'mainaltpath' not in groups:
                  groups['mainaltpath'] = [(u,v,d)]
               else:
                  groups['mainaltpath'].append((u,v,d))
            elif (u,v) in mopath_edges:
               if debug: print "Is mopath_edge"               
               if 'mainopath' not in groups:
                  groups['mainopath'] = [(u,v,d)]
               else:
                  groups['mainopath'].append((u,v,d))
            elif (u,v) in ropath_edges:
               if debug: print "Is ropath_edge"               
               if 'mainoaltpath' not in groups:
                  groups['mainoaltpath'] = [(u,v,d)]
               else:
                  groups['mainoaltpath'].append((u,v,d))                  
            else:
               if debug: print "Is notpath_edge"                              
               if 'main' not in groups:
                  groups['main'] = [(u,v,d)]
               else:
                  groups['main'].append((u,v,d))
         else:
            if debug: print "Is normal edge"            
            if (u,v) in mupath_edges and (u,v) not in rupath_edges:
               if debug: print "Is mupath_edge"               
               if 'normalupath' not in groups:
                  groups['normalupath'] = [(u,v,d)]
               else:
                  groups['normalupath'].append((u,v,d))
            elif (u,v) in rupath_edges:
               if debug: print "Is rupath_edge"                              
               if 'normalualtpath' not in groups:
                  groups['normalualtpath'] = [(u,v,d)]
               else:
                  groups['normalualtpath'].append((u,v,d))                  
            elif (u,v) in smpath_edges and srpath_edges and (u,v) not in srpath_edges:
               if debug: print "Is smpath_edge (not sr)"               
               if 'normalaltpath' not in groups:
                  groups['normalaltpath'] = [(u,v,d)]
               else:
                  groups['normalaltpath'].append((u,v,d))                  
            elif (u,v) in rupath_edges:
               if debug: print "Is rupath_edge"               
               if 'normalualtpath' not in groups:
                  groups['normalualtpath'] = [(u,v,d)]
               else:
                  groups['normalualtpath'].append((u,v,d))                  
            elif (u,v) in smpath_edges:
               if debug: print "Is smpath_edge"               
               if 'normalpath' not in groups:
                  groups['normalpath'] = [(u,v,d)]
               else:
                  groups['normalpath'].append((u,v,d))
            elif (u,v) in srpath_edges:
               if debug: print "Is srpath_edge"               
               if 'normalaltpath' not in groups:
                  groups['normalaltpath'] = [(u,v,d)]
               else:
                  groups['normalaltpath'].append((u,v,d))
            elif (u,v) in mopath_edges:
               if debug: print "Is mopath_edge"               
               if 'normalopath' not in groups:
                  groups['normalopath'] = [(u,v,d)]
               else:
                  groups['normalopath'].append((u,v,d))
            elif (u,v) in ropath_edges:
               if debug: print "Is ropath_edge"               
               if 'normaloaltpath' not in groups:
                  groups['normaloaltpath'] = [(u,v,d)]
               else:
                  groups['normaloaltpath'].append((u,v,d))
            else:
               if debug: print "Is notpath_edge"               
               if 'normal' not in groups:
                  groups['normal'] = [(u,v,d)]
               else:
                  groups['normal'].append((u,v,d))

      redge_data = self.model.get_edge_groups(edges=redges, path=path)

      for (edges, etype) in redge_data:
         if etype == 'mainpath':
            if 'mainopath' in groups:
               groups['mainopath'] += edges
            else:
               groups['mainopath'] = edges
         elif etype == 'mainaltpath':
            if 'mainoaltpath' in groups:
               groups['mainoaltpath'] += edges
            else:
               groups['mainoaltpath'] = edges
         elif etype == 'normalpath':
            if 'normalopath' in groups:
               groups['normalopath'] += edges
            else:
               groups['normalopath'] = edges
         elif etype == 'normalaltpath':
            if 'normaloaltpath' in groups:
               groups['normaloaltpath'] += edges
            else:
               groups['normaloaltpath'] = edges

      return [(groups[k], k) for k in groups]

   def get_diff_node_groups(self, path, spath, threshold=0.095, n=10):
      groups = {}
      bc = self.betweenness
      top = self.get_betweenness(top=n)

      opath = list(set(path) - set(spath))
      upath = list(set(path).intersection(set(spath)))
      
      rnodes = list(set(self.model.G.nodes()) - set(self.graph.nodes()))

      for node in self.graph.nodes():
         if node in rnodes: continue
         if bc[node] > threshold or node in top:
            if node == path[0]:
               if not 'mainstart' in groups:
                  groups['mainstart'] = [node]
               else:
                  groups['mainstart'].append(node)
            elif node == path[-1]:
               if not 'mainstop' in groups:
                  groups['mainstop'] = [node]
               else:
                  groups['mainstop'].append(node)
            elif node in upath:
               if not 'mainupath' in groups:
                  groups['mainupath'] = [node]
               else:
                  groups['mainupath'].append(node)
            elif node in spath:
               if not 'mainpath' in groups:
                  groups['mainpath'] = [node]
               else:
                  groups['mainpath'].append(node)               
            elif node in opath:
               if not 'mainopath' in groups:
                  groups['mainopath'] = [node]
               else:
                  groups['mainopath'].append(node)
            else:
               if not 'main' in groups:
                  groups['main'] = [node]
               else:
                  groups['main'].append(node)
         else:
            if node == path[0]:
               if not 'normalstart' in groups:
                  groups['normalstart'] = [node]
               else:
                  groups['normalstart'].append(node)
            elif node == path[-1]:
               if not 'normalstop' in groups:
                  groups['normalstop'] = [node]
               else:
                  groups['normalstop'].append(node)            
            elif node in upath:
               if not 'normalupath' in groups:
                  groups['normalupath'] = [node]
               else:
                  groups['normalupath'].append(node)
            elif node in spath:
               if not 'normalpath' in groups:
                  groups['normalpath'] = [node]
               else:
                  groups['normalpath'].append(node)               
            elif node in opath:
               if not 'normalopath' in groups:
                  groups['normalopath'] = [node]
               else:
                  groups['normalopath'].append(node)
            else:
               if not 'normal' in groups:
                  groups['normal'] = [node]
               else:
                  groups['normal'].append(node)

      rnode_data = self.model.get_node_groups(nodes=rnodes, path=path)

      for (nodes, ntype) in rnode_data:
         if ntype == 'mainpath':
            if 'mainopath' in groups:
               groups['mainopath'] += nodes
            else:
               groups['mainopath'] = nodes
         elif ntype == 'normalpath':
            if 'normalopath' in groups:
               groups['normalopath'] += nodes
            else:
               groups['normalopath'] = nodes
      
      return [(groups[k], k) for k in groups]

   def get_edge_groups(self, threshold=0.01, n=20, path=None):
      groups, mpath_edges, rpath_edges = {}, [], []
      multi = False
      mpath = path
      if path != None:
         if type(path[0]) == type([]):
            if len(path) > 1: multi = True
            mpath = path[0]
            
            for p in path[1:]:
               rpath_edges += zip(p, p[1:])
               rpath_edges += zip(p[1:], p)

         mpath_edges = zip(mpath, mpath[1:])
         mpath_edges += zip(mpath[1:], mpath)
      
      ebc = self.edge_betweenness
      top = self.get_edge_betweenness(top=n)

      for (u, v, d) in self.graph.edges(data=True):
         if (ebc[(u,v)] > threshold and ebc[(v,u)] > threshold) \
                or (u,v) in top:
            if (path  != None) and (not multi) and ((u,v) in mpath_edges):
               if 'mainpath' not in groups:
                  groups['mainpath'] = [(u,v,d)]
               else:
                  groups['mainpath'].append((u,v,d))
            elif multi and mpath_edges and (u,v) in mpath_edges \
                   and (u,v) not in rpath_edges:
               if 'mainaltpath' not in groups:
                  groups['mainaltpath'] = [(u,v,d)]
               else:
                  groups['mainaltpath'].append((u,v,d))
            elif mpath_edges and (u,v) in mpath_edges:
               if 'mainpath' not in groups:
                  groups['mainpath'] = [(u,v,d)]
               else:
                  groups['mainpath'].append((u,v,d))                  
            elif rpath_edges and (u,v) in rpath_edges:
               if 'mainaltpath' not in groups:
                  groups['mainaltpath'] = [(u,v,d)]
               else:
                  groups['mainaltpath'].append((u,v,d))                  
            else:
               if 'main' not in groups:
                  groups['main'] = [(u,v,d)]
               else:
                  groups['main'].append((u,v,d))
         else:
            if (path != None) and (not multi) and ((u,v) in mpath_edges):
               if 'normalpath' not in groups:
                  groups['normalpath'] = [(u,v,d)]
               else:
                  groups['normalpath'].append((u,v,d))
            elif multi and mpath_edges and (u,v) in mpath_edges \
                   and (u,v) not in rpath_edges:
               if 'normalaltpath' not in groups:
                  groups['normalaltpath'] = [(u,v,d)]
               else:
                  groups['normalaltpath'].append((u,v,d))
            elif mpath_edges and (u,v) in mpath_edges:
               if 'normalpath' not in groups:
                  groups['normalpath'] = [(u,v,d)]
               else:
                  groups['normalpath'].append((u,v,d))
            elif rpath_edges and (u,v) in rpath_edges:
               if 'normalaltpath' not in groups:
                  groups['normalaltpath'] = [(u,v,d)]
               else:
                  groups['normalaltpath'].append((u,v,d))
            else:
               if 'normal' not in groups:
                  groups['normal'] = [(u,v,d)]
               else:
                  groups['normal'].append((u,v,d))

      return [(groups[k], k) for k in groups]
   
   def has_effects(self):
      return len(self.effects) > 0

   def linkfail(self, n1, n2, record=True):
      if not self.graph.has_edge(n1,n2):
         return False

      metrics = (self.graph.get_edge(n1, n2), self.graph.get_edge(n2,n1))

      if record:
         self.changes.append({'type': Simulation.SC_LINKFAIL, 'pair': (n1,n2),
                                      'metrics': metrics})
      
      self.graph.delete_edge(n1, n2)
      self.graph.delete_edge(n2, n1)
      
      self._refresh_betweenness()
      self._refresh_anycast()
      self.effects = self._refresh_effects()
      self._refresh_linkload()      

      return True

   def routerfail(self, n1, record=True):
      if not self.graph.has_node(n1):
         return False

      removed_edges = []
      for node in self.graph.neighbors(n1):
         removed_edges.append((n1, node, self.graph.get_edge(n1, node)))
         self.graph.delete_edge(n1, node)
         removed_edges.append((node, n1, self.graph.get_edge(node, n1)))
         self.graph.delete_edge(node, n1)
         
      self.graph.delete_node(n1)

      if record:
         self.changes.append({'type': Simulation.SC_ROUTERFAIL, 'node': n1,
                                      'edges': removed_edges})

      self._refresh_betweenness()
      self._refresh_anycast()
      self.effects = self._refresh_effects()
      self._refresh_linkload()      

      return True

   def change_metric(self, n1, n2, metric, record=True, bidir=None):
      bidirectional = False
      metric = float(metric)

      if not self.graph.has_edge(n1, n2):
         return False

      old_metric = self.graph.get_edge(n1, n2)

      if old_metric == self.graph.get_edge(n2,n1):
         bidirectional = True

      if bidir == False:
         bidirectional = False
         
      if not record:
         bidirectional = False
      self.graph.delete_edge(n1, n2)
      self.graph.add_edge(n1,n2, metric)

      if bidirectional or bidir:
         self.graph.delete_edge(n2, n1)
         self.graph.add_edge(n2,n1, metric)         

      if record:
         self.changes.append({'type': Simulation.SC_METRIC, 'pair': (n1, n2),
                                      'metrics': (int(old_metric), int(metric)),
                                      'bidir': bidirectional or bidir})

      self._refresh_betweenness()
      self._refresh_anycast()
      self.effects = self._refresh_effects()
      self._refresh_linkload()
      
      return True

   def path(self, source, dest, G=None):
      if not G:
         G = self.graph
      if G == self.graph:
         preds, costs = self.all_paths[source]
      else:
         preds, costs = nx.dijkstra_predecessor_and_distance(G, source)

      if not dest in costs:
         return False, None

      def _get_paths(preds, path, paths, dest):
         if dest in path:
            return
         path.append(dest)
         if len(preds[dest]) == 0:
            paths.append(path)            
            return
         for newdest in preds[dest]:
            _get_paths(preds, path[:], paths, newdest)
         return paths

      paths = _get_paths(preds, [], [], dest)
      for path in paths:
         path.reverse()
      return costs[dest], paths
      
   def undo(self, change_no):
      if change_no > len(self.changes) or change_no < 1:
         return False

      idx = change_no - 1
      change = self.changes[idx]

      if change['type'] == Simulation.SC_METRIC:
         (u, v) = change['pair']
         w = change['metrics'][0]
         self.change_metric(u, v, w, record=False)
         if change['bidir']:
            self.change_metric(v, u, w, record=False)

      elif change['type'] == Simulation.SC_LINKFAIL:
         (u, v)   = change['pair']
         (m1, m2) = change['metrics']

         self.graph.add_edge(u, v, m1)
         self.graph.add_edge(v, u, m2)

      elif change['type'] == Simulation.SC_ROUTERFAIL:
         router = change['node']
         edges  = change['edges']

         self.graph.add_node(router)
         for (u, v, w) in edges:
            self.graph.add_edge(u, v, w)

      del self.changes[idx]

      self._refresh_betweenness()
      self._refresh_anycast()
      self.effects = self._refresh_effects()
      self._refresh_linkload()      
      return True
      
   def is_active(self):
      return self.active

   def reroute(self, start, stop, via, equal=False, timeout=2*60):
      import time
      debug = False

      max_metric = self.model.config.get('max_metric')
      
      G = self.graph
      H = self.graph.copy()
      I = self.graph.copy()
      K = self.graph.copy()
      success = False
      results = {}

      target_no_paths = [1]
      if equal:
         target_no_paths = range(2,10) + [1]
      
      ocost, opaths = self.path(start, stop)
      cost1, paths1 = self.path(start, via)

      if not start in G.nodes() \
      or not stop in G.nodes() \
      or not via in G.nodes():
         print "Invalid nodename"
         return []

      for path in paths1:
         for node in path:
            if node == stop:
               print "Path to via-node is through stop-node."
               print "Exiting"
               return (success, results)
            if node == via:
               continue
            H.delete_node(node)

      J = H.copy()
      cost2, paths2 = self.path(via, stop, H)

      S = set(reduce(lambda x,y: x+y, opaths))
      U = set(reduce(lambda x,y: x+y, paths1))
      V = set(reduce(lambda x,y: x+y, paths2))

      A = V.copy()
      for node in V:
         A.update(G.neighbors(node))

      allowed_nodes = list(U.union(A))

      if debug: print "Parameters:"
      if debug: print "S: %s" % (S)
      if debug: print "U: %s" % (U)
      if debug: print "V: %s" % (V)

      if debug: print "Allowed nodes: %s" % (allowed_nodes)
      
      finished = False
      neighbor_inc = 1
      start_t = time.time()
      
      while not finished:

         if time.time() - start_t >= timeout:
            finished = True
            success = False
            print "Timed out!"
            return (success, results)
         
         ocost, opaths = self.path(start, stop, K)
         W = set(reduce(lambda x,y: x+y, opaths))
         cost1, paths1 = self.path(start, via, I)
         cost2, paths2 = self.path(via, stop, J)

         if debug: print "Opath now: %s" % opaths

         if debug: print "Comparing %s+%s to %s" % (cost2, cost1, ocost)
         if via in W and len(opaths) in target_no_paths:
            if debug: print "Success!"
            finished = True
            success = True
            continue
         
         nochange = True

         if debug: print "Negative adjustment loop"
         for path1 in paths1:
            for (u,v) in zip(path1, path1[1:]):
               w = I.get_edge(u,v)
               if debug: print "Considering (%s,%s,%s) (-1)" % (u,v,w)

               if u == start or u == stop or v == start or v == stop:
                  if debug: print "A: Bad effect = False"
                  bad_effect = False
                  minmax = False
                  if debug: print "Inner negative adjustment loop"
                  while (not bad_effect) and (not minmax):
                     w = w - 1

                     if w < 1:
                        if debug: print "Reached minimum metric..."
                        minmax = True
                        break

                     I.add_edge(u,v,w)
                     K.add_edge(u,v,w)               

                     effects = self._refresh_effects(G, I)

                     if debug: print "B: Bad effect = False"
                     bad_effect = False
                     for src in effects:
                        if src not in allowed_nodes:
                           for dest in effects[src]:
                              #print dest
                              if dest not in allowed_nodes:
                                 bad_effect = True

                     if not bad_effect:
                        ocost, opaths = self.path(start, stop, K)
                        W = set(reduce(lambda x,y: x+y, opaths))                  
                        cost1, paths1 = self.path(start, via, I)
                        if debug: print "Opath now: %s" % opaths
                        if debug: print "Comparing %s+%s to %s" % (cost2, cost1, ocost)                             
                        if via in W and len(opaths) in target_no_paths:
                           if debug: print "Success!"
                           finished = True
                           success = True
                           break
                  if minmax:
                     if debug: print "A2: Bad effect = 2"
                     bad_effect = 2

               else:
                  w = w - 1

                  if w < 1:
                     if debug: print "Reached minimum metric..."
                     continue

                  I.add_edge(u,v,w)
                  K.add_edge(u,v,w)               

                  effects = self._refresh_effects(G, I)

                  if debug: print "C: Bad effect = False"
                  bad_effect = False
                  for src in effects:
                     if src not in allowed_nodes:
                        for dest in effects[src]:
                           #print dest
                           if dest not in allowed_nodes:
                              bad_effect = True
                              
               if bad_effect == True:
                  I.add_edge(u,v,w+1)
                  K.add_edge(u,v,w+1)                  
                  continue
               elif bad_effect == 2:
                  continue
               else:
                  if debug: print "A: nochange = False"
                  nochange = False

         ocost, opaths = self.path(start, stop, K)
         W = set(reduce(lambda x,y: x+y, opaths))                  
         cost1, paths1 = self.path(start, via, I)
         if debug: print "Opath now: %s" % opaths
         if debug: print "Comparing %s+%s to %s" % (cost2, cost1, ocost)                             
         if via in W and len(opaths) in target_no_paths:
            if debug: print "Success!"
            finished = True
            success = True
            continue
         
         if debug: print "Positive adjustment loop"
         for opath in opaths:
            for (u,v) in zip(opath, opath[1:]):
               if u in V and v in V: continue
               w = I.get_edge(u,v)

               if debug: print "Considering (%s,%s,%s) (+1)" % (u,v,w)

               if u == start or u == stop or v == start or v == stop:
                  if debug: print "D: Bad effect = False"
                  bad_effect = False
                  minmax = False
                  if debug: print "Inner positive adjustment loop"
                  while (not bad_effect) and (not minmax):
                     w = w + 1

                     if w > max_metric:
                        if debug: print "Reached maximum metric..."
                        minmax = True
                        continue

                     I.add_edge(u,v,w)
                     K.add_edge(u,v,w)               

                     effects = self._refresh_effects(G, I)

                     if debug: print "E: Bad effect = False"
                     bad_effect = False
                     for src in effects:
                        if src not in allowed_nodes:
                           for dest in effects[src]:
                              #print dest
                              if dest not in allowed_nodes:
                                 bad_effect = True

                     if not bad_effect:
                        ocost, opaths = self.path(start, stop, K)
                        W = set(reduce(lambda x,y: x+y, opaths))                  
                        cost1, paths1 = self.path(start, via, I)
                        if debug: print "Opath now: %s" % opaths
                        if debug: print "Comparing %s+%s to %s" % (cost2, cost1, ocost)                             
                        if via in W and len(opaths) in target_no_paths:
                           if debug: print "Success!"
                           finished = True
                           success = True
                           break
                  if minmax:
                     if debug: print "D2: Bad effect = 2"
                     bad_effect = 2

               else:
                  w = w + 1
                  if w > max_metric:
                     if debug: print "Reached maximum metric..."
                     continue
                  I.add_edge(u,v,w)
                  K.add_edge(u,v,w)

                  effects = self._refresh_effects(G, I)

                  if debug: print "F: Bad effect = False"
                  bad_effect = False
                  for src in effects:
                     if src not in allowed_nodes:
                        for dest in effects[src]:
                           #print dest
                           if dest not in allowed_nodes:
                              bad_effect = True

               if bad_effect == True:
                  I.add_edge(u,v,w-1)
                  K.add_edge(u,v,w-1)
                  continue
               elif bad_effect == 2:
                  continue
               else:
                  if debug: print "B: nochange = False"
                  nochange = False
         ocost, opaths = self.path(start, stop, K)
         W = set(reduce(lambda x,y: x+y, opaths))                  
         cost1, paths1 = self.path(start, via, I)
         if debug: print "Opath now: %s" % opaths

         if debug: print "Comparing %s+%s to %s" % (cost2, cost1, ocost)         
         if via in W and len(opaths) in target_no_paths:
            if debug: print "Success!"
            finished = True
            success = True
            continue

         if debug: print "2nd negative adjustment loop"
         for path2 in paths2:
            for (u,v) in zip(path2, path2[1:]):
               w = J.get_edge(u,v)
               if debug: print "Considering (%s,%s,%s) (-1)" % (u,v,w)
               w = w - 1

               if w < 1:
                  if debug: print "Reached minimum metric..."
                  continue                  
               J.add_edge(u,v,w)
               K.add_edge(u,v,w)               

               effects = self._refresh_effects(H, J)

               if debug: print "G: Bad effect = False"
               bad_effect = False
               for src in effects:
                  if src not in allowed_nodes:
                     for dest in effects[src]:
                        #print dest
                        if dest not in allowed_nodes:
                           bad_effect = True

               if bad_effect:
                  J.add_edge(u,v,w+1)
                  K.add_edge(u,v,w+1)                  
                  continue
               else:
                  if debug: print "C: nochange = False"
                  nochange = False

         if debug: print "Considering increasing allowed nodes"
         if nochange:
            if neighbor_inc > 2:
               if debug: print "No solution found"
               finished = True
               success = False
               continue
            append_nodes = []
            for node in allowed_nodes:
               append_nodes += G.neighbors(node)
            if debug: print "Increasing set of nodes"
            allowed_nodes += append_nodes
            neighbor_inc += 1
         else:
            if debug: print "nochange was False, so going on"


      for (u,v,w) in K.edges(data=True):
         if (u,v) in results: continue
         old_w = G.get_edge(u,v)
         if old_w != w:
            results[(u,v)] = w
            results[(v,u)] = w

      #for (u,v,w) in J.edges():
      #   if (u,v) in results: continue
      #   old_w = H.get_edge(u,v)
      #   if old_w != w:
      #      results[(u,v)] = w
      #      results[(v,u)] = w

      return (success, results)
         
         

            
   def minimal_link_costs(self):
      debug = False

      ebc = self.edge_betweenness
      G = self.graph
      H = self.graph.copy()

      edges = sorted(H.edges(data=True), cmp=lambda x,y: cmp(y[2], x[2]) \
                                    or cmp(ebc[(x[0],x[1])],
                                           ebc[(y[0],y[1])]))
      
      finished = False

      while not finished:

         adjustment_found = False

         for (u,v,w) in edges:
            if not w == H.get_edge(v,u):
               continue

            old_w = G.get_edge(u,v)
            if debug: print "Considering (%s,%s)" % (u,v)

            count = 1
            while count:
               w = w - 1
               count = 0
               if w < 1: continue

               if debug: print "Trying metrics..",
               H.add_edge(u,v,w)
               H.add_edge(v,u,w)
               effects = self._refresh_effects(G, H)
               if effects:
                  if abs(old_w - w) < 2:
                     H.add_edge(u,v,old_w)
                     H.add_edge(v,u,old_w)
                  else:
                     H.add_edge(u,v,w+1)
                     H.add_edge(v,u,w+1)
                  if debug: print "failed! (%s->%s)" % (old_w, w+1)
               else:
                  count = 1
                  adjustment_found = True
                  if debug: print "ok"

         if not adjustment_found:
            finished = True
            
      return H

   def _refresh_betweenness(self):
      self.betweenness = nx.load_centrality(self.graph, weighted_edges=True)
      self.edge_betweenness = nx.edge_betweenness(self.graph, True, True)

   def _refresh_effects(self, OG=None, NG=None):

      self._refresh_all_paths()
      
      if not OG:
         OG = self.model.G
      if not NG:
         NG = self.graph
         
      diff_paths = {}
      sources = OG.nodes()
      
      for source in sources:
         diff_by_dst = {}
         if not source in NG: continue
         opreds = self.model.all_paths[source][0]
         if OG != self.model.G:
            opreds = nx.dijkstra_predecessor_and_distance(OG, source)[0]
         npreds = self.all_paths[source][0]
         if NG != self.graph:
            npreds = nx.dijkstra_predecessor_and_distance(NG, source)[0]

         for dest in opreds:
            if not dest in npreds:
               diff_by_dst[dest] = [{'old': opreds[dest], 'new': []}]
               continue
               
            diff_res = self._path_cmp(opreds[dest], npreds[dest])
            if diff_res:
               if dest in diff_by_dst:
                  diff_by_dst[dest].append(diff_res)
               else:
                  diff_by_dst[dest] = [diff_res]

         if diff_by_dst.keys():
            diff_paths[source] = diff_by_dst

      #print diff_paths
      return diff_paths

   def _path_cmp(self, oldpaths, newpaths):
      if cmp(oldpaths, newpaths) != 0:
         return {'old': oldpaths, 'new': newpaths}
      return None

   def _refresh_anycast(self):

      accosts = {}
      acgroups = {}
      
      for source in self.graph.nodes():

         if source in self.acnodes:
            acgroups[source] = [source]
            continue

         lengths = nx.single_source_dijkstra_path_length(self.graph, source)

         for dest in self.acnodes:
            if dest not in lengths:
               continue
            else:
               cost = lengths[dest]
               if not source in accosts:
                  accosts[source] = cost
                  acgroups[source] = [dest]
               elif cost == accosts[source]:
                  acgroups[source] += [dest]
               elif cost < accosts[source]:
                  accosts[source] = cost
                  acgroups[source] = [dest]

      self.acgroups = acgroups

   def _apply_load_changes(self,effects):

      import time
      stime = time.time()
      via_edges_seen = {}
      adjustments = {}
      after_adjustments = {}
      traverse_edges = []
      old_path_parts = {}
      old_paths = {}
      
      for node in effects:
         for dest in effects[node]:

            no_old = len(effects[node][dest][0]['old'])
            no_new = len(effects[node][dest][0]['new'])
            old_vias = effects[node][dest][0]['old']
            new_vias = effects[node][dest][0]['new']

            for vianode in old_vias:
               if (vianode, dest) in via_edges_seen:
                  continue
               #print "  Considering viapath (%s, %s)" % (vianode, dest)
               traverse_edges.append((vianode,dest))
               via_edges_seen[(vianode,dest)] = True

      #print "Viapaths found (%s secs)" % (time.time() - stime)
      for (vianode, dest) in traverse_edges:
         old_paths = self.model.nodes_and_paths_using_edge(vianode,dest)[1]            
         # reduce_old(node, dest)
         loads = self.model.linkloads
         G = self.model.graph
         #print "Finding load parts for (%s, %s) (%s secs)" % (vianode, dest,
         #                                                     time.time() - stime)
         old_path_load_parts = self.model.get_link_load_part(vianode, dest)
         old_path_parts[(vianode, dest)] = old_path_load_parts.copy()
         for (u,v) in old_path_load_parts:
            change = -old_path_load_parts[(u,v)]
            if (u,v) in adjustments:
               if change < adjustments[(u,v)]:
                  #if u == 'porsgrunn-gw' or v == 'porsgrunn-gw':
                  #print "    Setting (%s, %s) to %s (<%s)" % (u,v, change,
                  #                                        adjustments[(u,v)])
                  adjustments[(u,v)] = change
                  if u in effects:
                     if dest in effects[u] \
                        and vianode in effects[u][dest][0]['old']:
                        new_paths = self.path(vianode,dest)[1]
                        if new_paths == None:
                           new_paths = [[]]
                        no_new_paths = len(effects[u][dest][0]['new'])
                        no_old_paths = len(effects[u][dest][0]['old'])
                        deduct = 0
                        
                        for npath in new_paths:
                           edges = zip(npath, npath[1:])
                           if (v,u) in edges:
                              deduct = old_path_parts[(vianode, dest)][(u,v)]
                              deduct *= float(no_old_paths/no_new_paths)
                              if (v,u) in after_adjustments:
                                 if -deduct < after_adjustments[(v,u)]:
                                    after_adjustments[(v,u)] = -deduct
                                    #print "Deducting %s from (%s,%s)" % (deduct, v,u)
                              else:
                                 after_adjustments[(v,u)] = -deduct
                                 #print "Deducting %s from (%s,%s)" % (deduct, v,u)
            else:
               #if u == 'porsgrunn-gw' or v == 'porsgrunn-gw':
               #print "    Setting (%s, %s) to %s" % (u,v, change)
               adjustments[(u,v)] = change
               if u in effects:
                  if dest in effects[u] \
                     and vianode in effects[u][dest][0]['old']:
                     new_paths = self.path(vianode,dest)[1]
                     if new_paths == None:
                        new_paths = [[]]
                     no_new_paths = len(effects[u][dest][0]['new'])
                     no_old_paths = len(effects[u][dest][0]['old'])
                     deduct = 0
                     for npath in new_paths:
                        edges = zip(npath, npath[1:])
                        if (v,u) in edges:
                           deduct = old_path_parts[(vianode, dest)][(u,v)]
                           deduct *= float(no_old_paths/no_new_paths)
                           if (v,u) in after_adjustments:
                              if -deduct < after_adjustments[(v,u)]:
                                 after_adjustments[(v,u)] = -deduct
                                 #print "Deducting %s from (%s,%s)" % (deduct, v,u)
                           else:
                              after_adjustments[(v,u)] = -deduct
                              #print "Deducting %s from (%s,%s)" % (deduct, v, u)

      #print "Negative adjustments complete (%s secs)" % (time.time() - stime)
      pos_changes = {}
      for (vianode, dest) in traverse_edges:
         old_paths = self.model.nodes_and_paths_using_edge(vianode,dest)[1]            
         for (n1, n2) in old_paths:
            if not (n1 in self.graph and n2 in self.graph): continue
            if not (n2 == dest or n1 == vianode): continue

            new_paths = self.path(n1,n2)[1]
            if new_paths == None:
               new_paths = [[]]
            opaths = old_paths[(n1,n2)]
            ofirst_edges = []
            for opath in opaths:
               if n2 == dest: 
                  ofirst_edges.append((opath[0], opath[1]))
               else:
                  ofirst_edges.append((opath[-2], opath[-1]))
            old = 0
            for oedge in ofirst_edges:
               if oedge in old_path_parts[(vianode, dest)]:
                  old += old_path_parts[(vianode, dest)][oedge]

            if old == 0: continue
            #print "Applying old load %s to new path (%s,%s)" \
            #    % (old, n1, n2)

            for path in new_paths:
               edges = zip(path, path[1:])                     
               for (u,v) in edges:
                  if (u,v) not in pos_changes:
                     pos_changes[(u,v)] = old
                  else:
                     if old > pos_changes[(u,v)]:
                        pos_changes[(u,v)] = old

      #print "Positive adjustments complete (%s secs)" % (time.time() - stime)
      for (u,v) in pos_changes:
         #if (u,v) == ('trd-gw1', 'hovedbygget-gw1'):
         #   print "    Adjusting (%s, %s) += %s" % (u, v, pos_changes[(u,v)])
         if (u,v) not in adjustments:
            if (u,v) not in after_adjustments:
               adjustments[(u,v)] = pos_changes[(u,v)]
            else:
               adjustments[(u,v)] = pos_changes[(u,v)] \
                                  + after_adjustments[(u,v)]
         else:
            if (u,v) not in after_adjustments:
               adjustments[(u,v)] += pos_changes[(u,v)]
            else:
               adjustments[(u,v)] += pos_changes[(u,v)] \
                                   + after_adjustments[(u,v)]

      #print "Returning adjustments (%s secs)" % (time.time() - stime)
      return adjustments
      
         
   def _refresh_linkload(self):

      self.linkloads = self.model.linkloads.copy()
      newloads = self.model.linkloads.copy()
      if not self.linkloads: return

      effects = self.effects
      adjustments = self._apply_load_changes(effects)

      for (u,v) in adjustments:
         if adjustments[(u,v)] == 0: continue
         if self.graph.has_edge(u,v):
            #print "Final adjustment for (%s, %s) += %s" % (u,v, adjustments[(u,v)])
            newloads[(u,v)] += adjustments[(u,v)]

      for (u,v) in sorted(newloads):
         if newloads[(u,v)] < 0:
            print "Assertion failed for load on (%s,%s): %s" \
                % (u,v, newloads[(u,v)])

      self.linkloads = newloads


   def _refresh_all_paths(self):
      for node in self.graph:
         self.all_paths[node] = nx.dijkstra_predecessor_and_distance(self.graph, node)
