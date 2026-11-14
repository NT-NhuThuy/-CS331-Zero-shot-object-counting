#!/usr/bin/env python3
"""Post-hoc DEV evaluator for localization and Safe-IIC clusters."""
from __future__ import annotations
import argparse,json
from collections import defaultdict,Counter
from pathlib import Path

def iou(a,b):
    x0,y0=max(a[0],b[0]),max(a[1],b[1]);x1,y1=min(a[2],b[2]),min(a[3],b[3]);z=max(0,x1-x0)*max(0,y1-y0);u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z;return z/u if u else 0.

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();h=a.experiment
    anns={int(x['query_id']):x for x in map(json.loads,a.annotations.read_text().splitlines()) if x};c={x['candidate_id']:x for x in map(json.loads,(h/'consensus_candidates.jsonl').read_text().splitlines()) if x};u={x['candidate_id']:x for x in map(json.loads,(h/'identity_unary_decisions.jsonl').read_text().splitlines()) if x};pairs=[json.loads(z) for z in (h/'identity_pair_decisions.jsonl').read_text().splitlines() if z.strip()]
    import sys;sys.path.insert(0,str(Path(__file__).parent));from safe_iic import safe_cluster
    candidates=[]
    for cid,x in c.items():
        d=u[cid];candidates.append({'candidate_id':cid,'query_id':int(x['query_id']),'view_provenance':x.get('support_views',[]),'semantic':d['semantic'],'whole_instance':d['whole'],'box_xyxy':x['box_xyxy']})
    clusters=safe_cluster(candidates,pairs)['components'];tp=fp=fn=0;ious=[];gt_seen=set();cluster_purity=[];false_merge=0;clusters_by_gt=defaultdict(set);stage={k:{'tp':0,'fp':0,'gt_hit':set()} for k in ('semantic','whole','combined')}
    for q,ann in anns.items():
        gboxes={x['instance_id']:x['box_xyxy'] for x in ann['target_instances']};assign={}
        for name,ok in [('semantic',lambda x:x['semantic']=='YES'),('whole',lambda x:x['whole_instance']=='YES'),('combined',lambda x:x['semantic']=='YES' and x['whole_instance']=='YES')]:
            for x in [z for z in candidates if z['query_id']==q and ok(z)]:
                score,gid=max(((iou(x['box_xyxy'],b),gid) for gid,b in gboxes.items()),default=(0,None));stage[name]['tp']+=score>=.5;stage[name]['fp']+=score<.5
                if score>=.5:stage[name]['gt_hit'].add(gid)
        valid=[x for x in candidates if x['query_id']==q and x['semantic']=='YES' and x['whole_instance']=='YES']
        for x in valid:
            best=max(((iou(x['box_xyxy'],b),gid) for gid,b in gboxes.items()),default=(0,None));ious.append(best[0]);tp+=best[0]>=.5;fp+=best[0]<.5
            if best[0]>=.5:gt_seen.add((q,best[1]))
            assign[x['candidate_id']]=best[1]
        fn+=max(0,len(gboxes)-sum(1 for qq,gid in gt_seen if qq==q))
        for cl in [z for z in clusters if z['query_id']==q]:
            ids=[assign[cid] for cid in cl['members'] if cid in assign and assign[cid] is not None];gids=set(ids)
            if ids:cluster_purity.append(max(Counter(ids).values())/len(ids))
            false_merge+=len(gids)>1
            for gid in gids:clusters_by_gt[(q,gid)].add(cl['representative'])
    false_split=sum(len(v)>1 for v in clusters_by_gt.values())
    for z in stage.values():z['gt_hit']=len(z['gt_hit']);z['precision_iou50']=z['tp']/(z['tp']+z['fp']) if z['tp']+z['fp'] else 0;z['candidate_count']=z['tp']+z['fp']
    out={'status':'FQIE_INSTANCE_EVIDENCE_DEV_METRICS','uses_gt_for_inference':False,'queries':len(anns),'eligible_candidates':sum(x['semantic']=='YES' and x['whole_instance']=='YES' for x in candidates),'stage_metrics':stage,'box_precision_iou50':tp/(tp+fp) if tp+fp else 0,'box_recall_iou50':len(gt_seen)/(len(gt_seen)+fn) if gt_seen or fn else 0,'mean_best_iou':sum(ious)/len(ious) if ious else 0,'cluster_purity_mean':sum(cluster_purity)/len(cluster_purity) if cluster_purity else 0,'false_merge_clusters':false_merge,'false_split_gt_instances':false_split,'clusters':len(clusters)}
    a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
