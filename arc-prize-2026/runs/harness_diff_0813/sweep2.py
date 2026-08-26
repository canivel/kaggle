import subprocess, json, time, sys
rows={}
def q(sort,page,tries=3):
    for t in range(tries):
        p=subprocess.run(['kaggle','kernels','list','--competition','arc-prize-2026-arc-agi-3','--sort-by',sort,'-p',str(page),'--page-size','100','--csv'],capture_output=True,text=True,timeout=240)
        out=p.stdout
        n=len([l for l in out.splitlines() if l and not l.startswith('ref,') and not l.startswith('Warning')])
        if n: return out,n
        time.sleep(3)
    return '',0
for sort in ['dateRun','hotness','voteCount','scoreDescending','relevance']:
    for page in range(1,9):
        out,n=q(sort,page)
        for l in out.splitlines():
            if not l or l.startswith('ref,') or l.startswith('Warning') or 'Not found' in l: continue
            rows[l.split(',')[0]]=l
        print(sort,page,n,'cum',len(rows),flush=True)
        if n==0: break
json.dump(rows,open('all_comp_kernels.json','w'),indent=0)
users=set(u.strip() for u in open('users_140.txt') if u.strip())
hits=sorted([l for r,l in rows.items() if r.split('/')[0] in users])
print('TOTAL',len(rows),'| by 1.40+ members:',len(hits))
for l in hits: print('  ',l[:170])
