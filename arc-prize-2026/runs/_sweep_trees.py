import json,re,time,sys,os
import kaggle as kg
from kagglesdk.discussions.types.discussions_api_service import ApiGetTopicRequest
BASE=r'F:\kaggle\arc-prize-2026\runs\topics_0831'
ids=list(json.load(open(os.path.join(BASE,'_all.json'),encoding='utf-8')).keys())
outp=os.path.join(BASE,'_trees.json')
out=json.load(open(outp,encoding='utf-8')) if os.path.exists(outp) else {}
fails=[]
with kg.api.build_kaggle_client() as c:
    for i,tid in enumerate(ids):
        if tid in out: continue
        for attempt in range(5):
            try:
                req=ApiGetTopicRequest(); req.id=int(tid)
                r=c.discussions.discussion_api_client.get_topic(req); t=r.topic
                out[tid]={'title':t.title,'author':t.author_name,'post_date':str(t.post_date),
                          'content':t.content or '','votes':t.votes,
                          'comments':[{'a':x.author_name,'d':str(x.post_date),'v':x.votes,'c':x.content or ''} for x in (r.comments or [])]}
                break
            except Exception as e:
                if attempt==4: fails.append((tid,str(e)[:60]))
                else: time.sleep(3+4*attempt)
        if i%20==0:
            json.dump(out,open(outp,'w',encoding='utf-8'),ensure_ascii=False)
            print(f"{i}/{len(ids)} ok={len(out)} fail={len(fails)}",flush=True)
json.dump(out,open(outp,'w',encoding='utf-8'),ensure_ascii=False)
print(f"DONE topics={len(out)} comments={sum(len(v['comments']) for v in out.values())} empty_body={sum(1 for v in out.values() if not v['content'])} FAILED={len(fails)} {fails[:15]}",flush=True)
