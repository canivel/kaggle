import subprocess, sys, json, concurrent.futures as cf
users=[u.strip() for u in open('users_140.txt') if u.strip()]
def q(u):
    try:
        p=subprocess.run(['kaggle','kernels','list','--user',u,'--competition','arc-prize-2026-arc-agi-3','--page-size','50','--csv'],capture_output=True,text=True,timeout=180)
        return u,p.stdout.strip()
    except Exception as e:
        return u,'ERR '+str(e)
out={}
with cf.ThreadPoolExecutor(8) as ex:
    for u,s in ex.map(q,users):
        out[u]=s
json.dump(out,open('user_kernels.json','w'),indent=1)
tot=0
for u,s in out.items():
    lines=[l for l in s.splitlines() if l and not l.startswith('ref,') and 'No kernels found' not in l and not l.startswith('Warning')]
    if lines:
        print('###',u)
        for l in lines: print('  ',l[:160]); tot+=1
print('TOTAL rows',tot)
