# usage: mkinst_set.py SRC OUT LOG k=v ...  (instrumented copy with PARAMS overrides)
import sys, subprocess, ast
src, out, log = sys.argv[1:4]
subprocess.check_call([sys.executable, 'mkinst.py', src, out, log])
over = {}
for kv in sys.argv[4:]:
    k, v = kv.split('=', 1); over[k] = ast.literal_eval(v)
if over:
    open(out, 'a').write('\nPARAMS.update(%r)\n' % over)
