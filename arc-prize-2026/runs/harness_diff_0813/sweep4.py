import subprocess, json, time, os, sys
env=dict(os.environ); env['PYTHONIOENCODING']='utf-8'
users=[u.strip() for u in open('users_140.txt') if u.strip()]
out={}
for i,u in enumerate(users):
    s=''
    for t in range(3):
        p=subprocess.run(['kaggle','kernels','list','--user',u,'--page-size','100','--csv'],capture_output=True,text=True,timeout=180,env=env,encoding='utf-8',errors='replace')
        s=(p.stdout or '').strip()
        if s.startswith('ref,') or 'Not found' in s or 'No kernels' in s: break
        time.sleep(2)
    out[u]=s
json.dump(out,open('user_kernels2.json','w'),indent=1)
tot=0
for u,s in sorted(out.items()):
    lines=[l for l in s.splitlines() if l and not l.startswith('ref,') and 'Not found' not in l]
    if lines:
        print('###',u)
        for l in lines: print('   ',l[:170]); tot+=1
print('TOTAL',tot)
