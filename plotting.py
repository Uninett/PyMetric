import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import networkx as nx
import numpy as np
import utils
import random
import time
import readline, sys

class PlotUI:

   nodesizes  = {'main'        : 700,
                 'mainpath'    : 800,
                 'mainstart'   : 800,
                 'mainstop'    : 800,
                 'mainopath'   : 700,
                 'mainupath'   : 800,
                 'normal'      : 350,
                 'normalstart' : 425,
                 'normalstop'  : 425,
                 'normalpath'  : 425,
                 'normalopath' : 350,
                 'normalupath' : 425}

   nodecolors  = {'main'       : '#cccccc',
                 'mainpath'    : '#77aaff',
                 'mainstart'   : '#44dd44',
                 'mainstop'    : '#dd7777',
                 'mainopath'   : '#cccc99',
                 'mainupath'   : '#4477cc',
                 'normal'      : '#dddddd',
                 'normalstart' : '#44dd44',
                 'normalstop'  : '#dd7777',
                 'normalpath'  : '#77aaff',
                 'normalopath' : '#cccc99',
                 'normalupath' : '#4477cc'}   

   edgewidths = {'main'          : 4,
                 'mainpath'      : 5,
                 'mainaltpath'   : 5,
                 'mainopath'     : 4,
                 'mainupath'     : 5,
                 'mainoaltpath'  : 4,
                 'mainualtpath'  : 5,                 
                 'normal'        : 1,
                 'normalpath'    : 2.5,
                 'normalaltpath' : 2.5,
                 'normalopath'   : 2,
                 'normalupath'   : 2.5,
                 'normaloaltpath': 2,
                 'normalualtpath': 2.5,
                 10000000        : 10,
                  2488000        : 6,
                  1000000        : 3,
                   155000        : 1.75,
                   100000        : 1.5,
                    34010        : 1.0,
                    34000        : 1.9,
                     1984        : 0.75}   

   edgecolors = {'main'           : '#bbbbbb',
                 'mainpath'       : '#77aaff',
                 'mainaltpath'    : '#77aaff',                 
                 'mainopath'      : '#aaaa77',
                 'mainupath'      : '#4477cc',
                 'mainoaltpath'   : '#aaaa77',
                 'mainualtpath'   : '#2255aa',                 
                 'normal'         : '#dddddd',
                 'normalpath'     : '#77aaff',
                 'normalaltpath'  : '#77aaff',                 
                 'normalopath'    : '#aaaa77',
                 'normalupath'    : '#4477cc',
                 'normaloaltpath' : '#bbbb88',
                 'normalualtpath' : '#2255aa'}   

   edgestyles = {'main'           : 'solid',
                 'mainpath'       : 'solid',
                 'mainaltpath'    : 'dotted',
                 'mainopath'      : 'solid',
                 'mainupath'      : 'solid',
                 'mainoaltpath'   : 'dotted',
                 'mainualtpath'   : 'dotted',                 
                 'normal'         : 'solid',
                 'normalpath'     : 'solid',
                 'normalaltpath'  : 'dotted',
                 'normalopath'    : 'solid',
                 'normalupath'    : 'solid',
                 'normaloaltpath' : 'dotted',
                 'normalualtpath' : 'dotted'}      


   areacolors = [(0.65,0.5,0,1), (0.5,0.75,0.35,1), (0.28,0.8,0.68,1),
                 (0.7,0.5,0.95,1), (0.35,0.5,0.7,1), (0.7,0.9,0.35,1),
                 (0.6,0.8,0.2,1), (0.9,0.7,0.6,1), (0.4,0.5,0.4,1),
                 (0.8,0.5,0.8,1), (0.45,0.65,0.55,1), (0.75,0.75,0.9,1),
                 (0.3,0.4,0.8,1), (0.1,0.9,0.85,1), (0.35,0.83,0.68,1),
                 (0.3,0.83,0.69,1), (0.95,0.95,0.8,1), (0.14,0.78,0.87,1),
                 (0.74,0.23,0.95,1)]
   
   def __init__(self, command):

      self.has_plotted = False
      self.command = command
      self.plottednodes = {}

   def picktest(self, event):
      #print "Picker fired, details:"

      #print " Index: %s"      % (event.ind)
      #print " (x,y): (%s,%s)" % (event.mouseevent.xdata,
      #                           event.mouseevent.ydata)
      #print "button: %s"      % (event.mouseevent.button)

      if not self.command: return

      if not event.artist in self.plottednodes: return
      
      nodes = self.plottednodes[event.artist]

      for i in event.ind:
         self.command.fromui(nodes[i])

   def savefig(self, fname, load=False):

      f = plt.gcf()
      f.set_figwidth(8)
      f.set_figheight(13)
      plt.subplot(111)
      plt.axis("off")
      if load:
         plt.savefig(fname, facecolor='#000000', dpi=72)
      else:
         plt.savefig(fname, dpi=72)

      return

   def clear(self):
      plt.clf()

   def plot(self, graph, data, opacity=1, areagroups=False, anycast=False,
            edge_cmap=False, edge_capa=False):
      ax = plt.gca()

      interactive = matplotlib.is_interactive()
      ci = 0

      for (nodes, type) in data['nodegroups']:

         nodecolors = PlotUI.nodecolors[type]
         nodealpha = self._get_alpha(type, opacity)

         if edge_cmap:
            nodecolors = '#444488'
            nodealpha = 0.9
         
         if areagroups:

            areas = [data['areagroups'][nodes[i]] for i in range(len(nodes))]

            for a in areas:
               if not a in PlotUI.nodecolors:
                  if ci < len(PlotUI.areacolors):
                     PlotUI.nodecolors[a] = PlotUI.areacolors[ci]
                     ci += 1
                  else:
                     PlotUI.nodecolors[a] = random.choice(PlotUI.areacolors)
                  
            nodecolors = [PlotUI.nodecolors[data['areagroups'][nodes[i]]]
                          for i in range(len(nodes))]

         elif anycast:

            for group in data['acnodes'] + ['multi']:

               if not group in PlotUI.nodecolors:
                  if ci < len(PlotUI.areacolors):
                     PlotUI.nodecolors[group] = PlotUI.areacolors[ci]
                     ci += 1
                  else:
                     PlotUI.nodecolors[group] = random.choice(PlotUI.areacolors)

            nodecolors = []
            for i in range(len(nodes)):
               current_node = nodes[i]
               current_group = data['acgroups'][current_node]
               if len(current_group) > 1:
                  color = PlotUI.nodecolors['multi']
                  color = (color[0], color[1], color[2], 0.3)
                  nodecolors.append(color)
               else:
                  color = PlotUI.nodecolors[current_group[0]]
                  color = (color[0], color[1], color[2], 0.3)
                  if current_node in data['acnodes']:
                     color = (color[0], color[1], color[2], 1)
                  nodecolors.append(color)

         nodec = nx.draw_networkx_nodes(graph, pos=data['pos'],
                                nodelist=nodes,
                                node_size=PlotUI.nodesizes[type],
                                node_color=nodecolors,
                                alpha = nodealpha,
                                hold=True)


         self.plottednodes[nodec] = nodes
         nodec.set_picker(True)
         #self.nodec.set_urls(nodes)


      for (edges, type) in data['edgegroups']:

         edgecolors = PlotUI.edgecolors[type]
         
         #print "colors:", edgecolors

         if nx.__version__ > "0.36":
            edgecolors = [PlotUI.edgecolors[type]] * len(edges)
            
         ealpha = self._get_alpha(type, opacity)
         if anycast:
            ealpha = 0.5

         if edge_cmap != False:
            assert edge_capa != False
            ccm = {
               'red'  :  ((0., 0., 0.), (0.005, 0.0, 0.0),
                          (0.01, 0.2, 0.2), (0.1, 0., 0.),
                          (0.6, 1, 1),
                          (0.7, 0.75, 0.75), (0.8, 0.9, 0.9),
                          (1., 1., 1.)),
               'green':  ((0., 0., 0.), (0.01, 0., 0.),
                          (0.05, 0.3, 0.3), (0.1, 0.75, 0.75),
                          (0.3, 1., 1.),
                          (0.6, 0.75, 0.75), (1., 0., 0.)),
               'blue' :  ((0., 0.45, 0.45), (0.005, 0.55, 0.55),
                          (0.05, 0.8, 0.8), (0.10, 1., 1.),
                          (0.25, 0., 0.), (1., 0., 0.))
               }
            my_cmap = mc.LinearSegmentedColormap('my_colormap', ccm, 1024)
            ealpha = 0.7
            f = plt.gcf()
            f.set_facecolor('#000000')
            edgecolors = [max(edge_cmap[(edges[i][0], edges[i][1])],
                              edge_cmap[(edges[i][1], edges[i][0])])
                          for i in range(len(edges))]
            edgewidths = [PlotUI.edgewidths[edge_capa[(edges[i][0], edges[i][1])]]
                          for i in range(len(edges))]            
            self.edgec = nx.draw_networkx_edges(graph, pos=data['pos'],
                                edgelist=edges,
                                width=edgewidths,
                                edge_color=edgecolors,
                                edge_cmap=my_cmap,
                                edge_vmin=0.005,
                                edge_vmax=1.0,
                                style=PlotUI.edgestyles[type],
                                alpha=ealpha,
                                arrows=False,
                                hold=True)
         else:
            self.edgec = nx.draw_networkx_edges(graph, pos=data['pos'],
                                edgelist=edges,
                                width=PlotUI.edgewidths[type],
                                edge_color=edgecolors,
                                style=PlotUI.edgestyles[type],
                                alpha=ealpha,
                                arrows=False,
                                hold=True)            

      if 'labels' in data:
         color = 'k'
         fsize = 7.5
         if edge_cmap:
            color = '#bbbbbb'
            fsize = 8.5
         
         self.nlabelc = nx.draw_networkx_labels(graph, pos=data['pos'],
                                 labels=data['labels'],
                                 font_size=fsize,
                                 font_color=color,
                                 alpha=1,
                                 hold=True)
      
      if 'edgelabels' in data and not edge_cmap:
         self.elabelc = self._plot_edge_labels(graph,
                                               data['edgelabels'],
                                               pos=data['pos'])
      plt.axis("off")      
      plt.subplots_adjust(0,0,1,1)
      ax.set_xbound(-10, 210)
      ax.set_ybound(0, 200)
      plt.xlim(-10, 210)
      plt.ylim(0,200)

      
      f = plt.gcf()
      if not edge_cmap:
         f.set_facecolor('w')
      else:
         cb = plt.colorbar(self.edgec, shrink=0.15,
                           fraction=0.10, pad=0)
         for t in cb.ax.get_yticklabels():
            t.set_color('#bbbbbb')
         
         cb.set_label("Link utilization", color='#bbbbbb')         
      f.set_lod(True)
      fm = plt.get_current_fig_manager()      

      if (not self.has_plotted) and interactive:
         f.set_figheight(13)         
         f.set_size_inches(8,13)
         if matplotlib.get_backend() == 'TkAgg':
            maxh = fm.window.winfo_screenheight() - 60
            fm.window.minsize(600,maxh)         
      if interactive:
         fm.set_window_title("PyMetric - " + data['title'])
         f.canvas.mpl_connect('pick_event', self.picktest)
      plt.draw()         
      self.has_plotted = True

   def _get_alpha(self, type, opacity):
      if not (type.endswith('path') \
                 or type.endswith('start') \
                 or type.endswith('stop')):
         return opacity
      else:
         return 1
         
   def _plot_edge_labels(self, G, edgelabels, pos):
      from math import sqrt

      lpos = []
      for (u,v,w) in G.edges(data=True):

         (x1, y1) = pos[u]
         (x2, y2) = pos[v]
         
         x_diff = (x2 - x1)
         y_diff = (y2 - y1)
         
         d = sqrt( x_diff**2 + y_diff**2 )

         x = (x1+x2) / 2
         y = (y1+y2) / 2

         if d < 70 and G[v][u]['weight'] == w['weight']:
            pass

         else:
            wy = (9.0/d)*y_diff
            wx = (9.0/d)*x_diff

            if abs(wy) > 7:
               if wy < 0:
                  wy = -7
               else:
                  wy = 7
            if abs(wx) > 10:
               if wx < 0:
                  wx = -10
               else:
                  wx = 10
                  
            if x_diff == 0:
               y = y1 + wy
            elif y_diff == 0:
               x = x1 + wx
            else:
               x = x1 + wx
               y = y1 + wy

            


         lpos.append((x,y))
         
      ax = plt.gca()
      text_items = {}
         
      for (i, label, bold) in edgelabels:
         label = str(label)
         (x,y) = lpos[i]
         fw = 'normal'
         fs = 8
         if bold:
            fs = 10.5
            fw = 'bold'

         color = '#662222'

         t = ax.text(x, y,
                     label,
                     size=fs,
                     color=color,
                     family="sans-serif",
                     weight=fw,
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform = ax.transData)
         text_items[i] = t

      return text_items

