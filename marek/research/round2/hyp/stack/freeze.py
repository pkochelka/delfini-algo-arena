import re,sys
src,dst=sys.argv[1],sys.argv[2]; s=open(src).read()
for kv in sys.argv[3:]:
    k,v=kv.split('=')
    pat=re.compile(r"(\n    '%s': )([^,]+)(,)"%re.escape(k))
    assert len(pat.findall(s))==1,k
    s=pat.sub(lambda m:m.group(1)+v+m.group(3),s)
open(dst,'w').write(s)
