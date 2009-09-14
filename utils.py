import networkx as nx
import httplib

def mean_shortest_path_length(G):
    """Calculate mean shortest path lenght of graph."""
    apspl = nx.path.all_pairs_shortest_path_length(G)
    sum_of_paths = sum([sum(apspl[n].values()) for n in apspl])
    num_of_paths = sum(map(len, apspl.values()))
    return float(sum_of_paths) / num_of_paths

def reciprocity(G):
    """Calculate reciprocity of graph, i.e. the ratio of the edges in
    one direction that have and edge going in the other direction."""
    return sum([G.has_edge(e[1], e[0]) 
    for e in G.edges_iter()]) / float(G.number_of_edges())

def short_names(names):
   labels = {}
   for n in names:
      mainlabel = n[0:3]
      if n[2] == '-':
         mainlabel = n[0:2]
      if n[-2:].startswith('w'):
         labels[n] = (mainlabel + n[-1])
      else:
         labels[n] = mainlabel
   return labels

def edge_labels(edges, edgegroups):
   labels = []

   i = 0

   edgedone = {}
   
   for (u,v,w) in edges:      
      for el, type in edgegroups:
         for (s,t,x) in el:
            if (u,v) == (s,t):
               if not (u,v) in edgedone:
                  metric = int(x)
                  m = str(metric)
                  if metric == 10:
                      m = ""
                  if type.endswith("opath") and x != w:
                     m = "%s (%s)" % (str(int(x)), str(int(w)))
                  l = (i, m, type.startswith('main'))
                  labels.append((i, m, type.startswith('main')))
                  edgedone[(u,v)] = l
                  i = i + 1
      if (u,v) not in edgedone:
         labels.append((i, "", False))
         i = i + 1

   try:
      assert len(edges) == len(labels)
   except:
      print "Assertion fail: %s != %s" % (len(edges), len(labels))

   return labels


def cap2str(capacity):

    mapping = {  1984    : '2Mbit/s',
                34000    : '34Mbit/s',
                34010    : '34Mbit/s',
                100000   : '100Mbit/s',
                155000   : '155Mbit/s',
                1000000  : '1Gbit/s',
                2488000  : '2.5Gbit/s',
                10000000 : '10Gbit/s'
              }

    if type(capacity) != type(int):
        capacity = int(capacity)
    if not capacity in mapping: return "Unkown"
    return mapping[capacity]
      
def read_linkloads(G, host, url):

    conn = httplib.HTTPConnection(host)
    conn.request("GET", url)
    r1 = conn.getresponse()
    if r1.status != 200:
        conn.close()        
        return {}
    data1 = r1.read()
    if not data1:
        conn.close()        
        return {}
    conn.close()

    loads_by_descr = {}

    retloads = {}

    for line in data1.split('\n'):
        if not line: continue
        tokens = line.split()
        descr = tokens[0].strip()
        avg_in = int(tokens[3].strip())
        avg_out = int(tokens[4].strip())

        loads_by_descr[descr] = (avg_out, avg_in)

    for (u,v,edgedata) in G.edges(data=True):
        if not 'l' in edgedata: continue
        label = edgedata['l']
        if label in loads_by_descr:
            retloads[(u,v)] = loads_by_descr[label]

    sanitized_loads = {}
    for (u,v) in retloads:
        if (v,u) in retloads:
            if retloads[(v,u)][1] > retloads[(u,v)][0]:
                sanitized_loads[(u,v)] = retloads[(v,u)][1]
            else:
                sanitized_loads[(u,v)] = retloads[(u,v)][0]
        else:
            sanitized_loads[(u,v)] = retloads[(u,v)][0]
            sanitized_loads[(v,u)] = retloads[(u,v)][1]

    return sanitized_loads


def calc_ratio(G, loads, u, v, discard_inverse=False, no_diff=False, exclude_edges=[]):
  sum = 0
  totload = loads[(u,v)]
  for (u1,v1,d) in G.in_edges(u):
      if (u1,v1) in exclude_edges:
          #totload -= loads[(u1,v1)] * calc_ratio(G, loads, u,v)
          continue
      if discard_inverse and (u1,v1) == (v,u): continue
      sum += float(loads[(u1,v1)])
  ee = []
  #if discard_inverse: ee += [(v,u)]
  ndiff = node_diff_in_out(G, loads, u, False, ee)
  if not no_diff:
      if ndiff < 0:
          sum += -ndiff
  if sum == 0:
     return 0
  ratio = totload / float(sum)
  if ratio < 0:
      print "Assertion failed for ratio (%s, %s): %s" \
          % (u, v, ratio)
  if ratio > 1:
      ratio = 1
  return ratio

def calc_contrib_ratio2(G, loads, u, v, discard_inverse=False, exclude_edges=[]):
  sum = 0
  for (u1,v1,d) in G.in_edges(v):
      if (u1,v1) in exclude_edges: continue
      if discard_inverse and (u1,v1) == (u,v): continue
      sum += float(loads[(u1,v1)])
  if sum == 0:
     return 0
  ratio = loads[(u,v)] / float(sum)
  #if ratio < 0 or ratio > 1:
  #    print "Assertion failed for ratio (%s, %s): %s" \
  #        % (u, v, ratio)
  return ratio

def calc_contrib_ratio(G, loads, u, v, discard_diff=False, exclude_edges=[]):
   sum = 0
   for (u1,v1,d) in G.in_edges(v):
      if (u1,v1) in exclude_edges: continue
      sum += float(loads[(u1,v1)])
   print "Sum before diff:", sum
   ee = []
   #if discard_inverse: ee += [(u,v)]
   ndiff = node_diff_in_out(G, loads, v, False, ee)
   if not discard_diff:
       if ndiff > 0:
           sum += ndiff
   print "Sum after diff:", sum
   if sum == 0:
      return 0

   print "Initial load:", loads[(u,v)]
   ratio = loads[(u,v)] / float(sum)
   return ratio

def node_diff_in_out(G, loads, node, inverse=False, exclude_edges=[]):
   sum_out = 0
   sum_in = 0

   for (u1,v1,d) in G.out_edges(node):
      if (u1,v1) in exclude_edges: continue
      sum_out += loads[(u1,v1)]
   for (u1,v1,d) in G.in_edges(node):
      if (u1,v1) in exclude_edges: continue       
      sum_in += loads[(u1,v1)]

   if inverse:
      return sum_in - sum_out
       
   return sum_out - sum_in

