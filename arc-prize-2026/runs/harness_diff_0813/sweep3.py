import subprocess, json, time, sys
users=[u.strip() for u in open('users_140.txt') if u.strip()]
out={}
for i,u in enumerate(users):
    got=None
    for t in range(3):
        p=subprocess.run(['kaggle','kernels','list','--user',u,'--page-size','100','--csv'],capture_output=True,text=True,timeout=180)
        s=p.stdout.strip()
        if s.startswith('ref,') or 'Not found' in s or 'No kernels' in s:
            got=s; break
        time.sleep(2)
    out[u]=got if got is not None else 'FAIL'
    print(i,u,len(out[u].splitlines()),flush=True)
json.dump(out,open('user_kernels2.json','w'),indent=1)
