import subprocess, sys, json, concurrent.futures as cf, itertools
rows={}
def q(args):
    sort,page=args
    try:
        p=subprocess.run(['kaggle','kernels','list','--competition','arc-prize-2026-arc-agi-3','--sort-by',sort,'-p',str(page),'--page-size','100','--csv'],capture_output=True,text=True,timeout=240)
        return p.stdout
    except Exception as e:
        return ''
tasks=[(s,p) for s in ['dateRun','hotness','voteCount','scoreDescending'] for p in range(1,8)]
with cf.ThreadPoolExecutor(6) as ex:
    for out in ex.map(q,tasks):
        for l in out.splitlines():
            if not l or l.startswith('ref,') or l.startswith('Warning') or 'Not found' in l: continue
            ref=l.split(',')[0]
            rows[ref]=l
print('unique kernels',len(rows))
json.dump(rows,open('all_comp_kernels.json','w'),indent=0)
users=set(u.strip() for u in open('users_140.txt') if u.strip())
hits=[(r,l) for r,l in rows.items() if r.split('/')[0] in users]
print('kernels owned by 1.40+ members:',len(hits))
for r,l in sorted(hits): print('  ',l[:170])
