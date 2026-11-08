import sys
src=open(sys.argv[1]).read()
add = """        F, sd = f.mean, f.sd
        if p['oracle_path']:
            if not hasattr(self, '_osp'):
                a = np.load(p['oracle_path'])
                self._ot, self._os = a[:, 0].copy(), a[:, 1].copy()
            i = int(np.searchsorted(self._ot, self.now_t, side='right')) - 1
            if p['oracle_log']:
                if not hasattr(self, '_olog'):
                    self._olog = []; self._onext = 0.0
                if self.now_t >= self._onext:
                    self._onext = self.now_t + 0.05
                    self._olog.append((self.now_t, F, sd, f.pv, f.exc, float(self._os[i]), float(f.drought), float(f.hot), self.pos))
                    if len(self._olog) % 2000 == 0 or self.now_t > 5999.0:
                        np.save(p['oracle_log'], np.array(self._olog))
            elif p['oracle_reg'] == 1 and f.pv >= 0.5:
                pass
            elif p['oracle_reg'] == 2 and f.pv < 0.5:
                pass
            else:
                F = float(self._os[i])
                if p['oracle_sd'] >= 0:
                    sd = p['oracle_sd']
            self._osp = 1
"""
assert src.count("        F, sd = f.mean, f.sd\n")==1
src=src.replace("        F, sd = f.mean, f.sd\n", add)
src=src.replace("    'ss_on': 1,\n","    'ss_on': 1,\n    'oracle_path': '', 'oracle_sd': 0.05, 'oracle_log': '', 'oracle_reg': 0,\n",1)
open(sys.argv[2],'w').write(src)
