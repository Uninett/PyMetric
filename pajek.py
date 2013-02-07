"""
Read graphs in Pajek format.

See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
for format information.

This implementation handles only directed and undirected graphs including
those with self loops and parallel edges.  

Adapted by Morten Knutsen (morten.knutsen@uninett.no).
"""
__author__ = """Aric Hagberg (hagberg@lanl.gov)"""
#    Copyright (C) 2008 by 
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    Distributed under the terms of the GNU Lesser General Public License
#    http://www.gnu.org/copyleft/lesser.html
import networkx
from networkx.utils import is_string_like

def read_pajek(path):
    """Read graph in pajek format from path. Returns an XGraph or XDiGraph.
    """
    fh=open(path,mode='r')
    lines = fh.readlines()
    G=parse_pajek(lines)
    return G

def parse_pajek(lines):
    """Parse pajek format graph from string or iterable.."""
    import shlex
    if is_string_like(lines): lines=iter(lines.split('\n'))
    lines = iter([line.rstrip('\n') for line in lines])
    G=networkx.DiGraph() # are multiedges allowed in Pajek?
    G.node_attr={} # dictionary to hold node attributes
    directed=True # assume this is a directed network for now
    while lines:
        try:
            l=lines.next()
            l=l.lower()
        except: #EOF
            break
        if l.startswith("#"):
            continue
        if l.startswith("*network"):
            label,name=l.split()
            G.name=name
        if l.startswith("*vertices"):
            nodelabels={}
            l,nnodes=l.split()
            while not l.startswith("*arcs"):
                if l.startswith('#'):
                    l = lines.next()
                    l = l.lower()
                    continue
                if l.startswith('*'):
                    l = lines.next()
                    l = l.lower()
                    continue
                splitline=shlex.split(l)
                #print splitline
                id, label = splitline[0:2]
                G.add_node(label)
                nodelabels[id]=label
                G.node_attr[label]={'id':id}                
                if len(splitline) > 2:
                    id,label,x,y=splitline[0:4]                
                    G.node_attr[label]={'id':id,'x':x,'y':y}
                extra_attr=zip(splitline[4::2],splitline[5::2])
                #print extra_attr
                G.node_attr[label].update(extra_attr)
                l = lines.next()
                l = l.lower()
        if l.startswith("*arcs"):
            for l in lines:
                if not l: break
                if l.startswith('#'): continue
                splitline=shlex.split(l)
                ui,vi,w=splitline[0:3]
                u=nodelabels.get(ui,ui)
                v=nodelabels.get(vi,vi)
                edge_data={'value':float(w)}
                extra_attr=zip(splitline[3::2],splitline[4::2])
                edge_data.update(extra_attr)
                if G.has_edge(u,v):
                    if G[u][v]['value'] > float(w):
                        G.add_edge(u,v,edge_data)
                else:
                    G.add_edge(u,v,edge_data)
    if not G.name:
        raise Exception("No graph definition found")
    if len(G.nodes()) == 0:
        raise Exception("No graph definition found")
    return G

