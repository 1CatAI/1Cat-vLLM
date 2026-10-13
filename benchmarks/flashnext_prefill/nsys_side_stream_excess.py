# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Excess main-stream kernel time while the cold-expert transcode side stream runs.

Usage: python nsys_side_stream_excess.py trace.sqlite [deviceId]
Compares each (kernel, grid) family's duration with vs without overlap.
"""
import sqlite3,collections,bisect,statistics as S,sys
c=sqlite3.connect(sys.argv[1]); dev=int(sys.argv[2]) if len(sys.argv)>2 else 0
names=dict(c.execute("select id,value from StringIds").fetchall())
ks=c.execute("select start,end,streamId,demangledName,gridX,gridY,gridZ,blockX from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start",(dev,)).fetchall()
main=collections.Counter(r[2] for r in ks).most_common(1)[0][0]
mer=[]
for s,e,sid,n,*_ in ks:
    if sid!=main and 'transcode' in names[n] and _[0]<=8:
        if mer and s<=mer[-1][1]: mer[-1][1]=max(mer[-1][1],e)
        else: mer.append([s,e])
starts=[m[0] for m in mer]
def ov(s,e):
    i=bisect.bisect_right(starts,e); t=0
    for a,b in mer[max(0,i-3):i]: t+=max(0,min(b,e)-max(a,s))
    return t/max(1,e-s)
fam=collections.defaultdict(lambda:[[],[]])
for s,e,sid,n,gx,gy,gz,bx in ks:
    if sid!=main: continue
    k=(names[n][:70]+" g=%d,%d,%d b=%d"%(gx,gy,gz,bx),gx,gy,gz,bx); o=ov(s,e)
    (fam[k][0] if o>0.5 else fam[k][1]).append(e-s)
tot=0; rows=[]
for k,(a,b) in fam.items():
    if len(b)>=3 and a:
        base=S.median(b); ex=sum(max(0,x-base) for x in a); tot+=ex; rows.append((ex/1e6,len(a),base/1e3,S.median(a)/1e3,k[0][-40:]))
print('excess on main stream while cold transcode runs: %.0f ms'%(tot/1e6))
for r in sorted(rows,reverse=True)[:12]: print('%6.0f ms n=%4d base %6.0f us -> %6.0f us  %s'%r)
