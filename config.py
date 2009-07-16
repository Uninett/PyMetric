import os.path

class Config():

   def __init__(self):

      self.options = {
                 'max_metric':        63,
                 'use_areas':          1,
                 'use_linkloads':      1,
                 'linkloads_host':    'drift.uninett.no',
                 'linkloads_url':     '/nett/ip-nett/load-now'
                }
      
      localconf = os.path.join(os.environ["HOME"], ".pymetricrc")
      if os.path.isfile(localconf):
         self._read_localconf(localconf)

   def get(self, key):
      if key not in self.options: return False
      return self.options[key]

   def _read_localconf(self, filename):
      fh = open(filename, "r")

      for line in fh.readlines():
         if line.startswith('#'): continue
         if not line: continue
         line = line.strip()
         tokens = line.split()

         if len(tokens) > 1:
            key, value = tokens
            key = key.strip()
            value = value.strip()
            try:
               self.options[key] = int(value)
            except:
               self.options[key] = value
               



      
         
