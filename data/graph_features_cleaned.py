"""Graph features for the supplied Person 1 cleaned schema.
AI assistance: adapted and tested with OpenAI Codex, 2026-10-24.
Dependencies tested: Python 3.12, pandas 2.2.3, numpy 2.3.5, pyarrow.
Topology uses customer ACCOUNT -> JSON Destination once per transaction.
All windows are [t-window, t); same-time transactions are excluded.
Partial history is retained; supported observed counts can be zero.
Unsupported measurements remain NaN. Labels are never exported.
No verified service registry was supplied: endpoints remain unknown services;
features are descriptive, not suspiciousness flags or accusations.
FIFO excludes conversions, preserves dust, ignores transaction fees, and
is an inferred allocation of observed receipts, not fund provenance.
Prior-history length is based on this sample's first supported event and
must not be interpreted as verified complete ledger coverage.
Train/test must be processed separately, never concatenated as history.
No Person 4 behavioural features are changed.
"""
import json
from collections import defaultdict, deque, Counter
import numpy as np
import pandas as pd

def make_graph_features(data, dust_xrp=1.0):
    d=data.copy().reset_index(drop=True)
    if d.TRANSACTION_HASH.isna().any() or not d.TRANSACTION_HASH.is_unique:
        raise ValueError('Transaction hashes must be present and unique.')
    def obj(v):
        if isinstance(v,dict): return v
        try:
            z=json.loads(v)
            return z if isinstance(z,dict) else {}
        except (ValueError,TypeError): return {}
    extra=d.EXTRA_TRANSACTION_FIELDS.map(obj)
    d['_src']=d.ACCOUNT
    d['_dst']=extra.map(lambda z:z.get('Destination'))
    d['_t']=pd.to_numeric(d.TIMESTAMP,errors='coerce')
    def drops(v):
        if isinstance(v,str):
            try: v=json.loads(v)
            except (ValueError,TypeError): pass
        if isinstance(v,bool): return None
        if isinstance(v,(int,np.integer)): return int(v) if v>0 else None
        if isinstance(v,str) and v.isdigit(): return int(v) if int(v)>0 else None
        return None
    d['_drops']=pd.Series([drops(v) for v in d.DELIVERED_AMOUNT],dtype=object)
    addr=lambda v:isinstance(v,str) and bool(v.strip())
    d['_top']=d._src.map(addr)&d._dst.map(addr)&np.isfinite(d._t)&d.TRANSACTION_TYPE.eq('Payment')&d.TRANSACTION_RESULT.eq('tesSUCCESS')
    d['_fifo']=d._top&d._src.ne(d._dst)&d._drops.notna()&d.transfer_count.eq(1)&d.has_sendmax.eq(0)&d.has_paths.eq(0)&extra.map(lambda z:'SendMax' not in z and not z.get('Paths'))
    f=pd.DataFrame(index=d.index)
    f['TRANSACTION_HASH']=d.TRANSACTION_HASH
    f['graph_topology_supported']=d._top.astype(int)
    f['graph_fifo_supported']=d._fifo.astype(int)
    f['graph_native_dust']=np.nan
    f.loc[d._fifo,'graph_native_dust']=d.loc[d._fifo,'_drops'].map(lambda v:float(v<dust_xrp*1e6))
    windows={'1h':3600,'1d':86400,'7d':604800}
    start=d.loc[d._top,'_t'].min()
    f['graph_observed_prior_hours']=((d._t-start)/3600).clip(lower=0).where(d._top)
    for label,seconds in windows.items():
        f[f'graph_partial_history_{label}']=((d._t-start)<seconds).astype(float).where(d._top)
        for side in ['sender','receiver']:
            for name in ['sent_n','received_n','fan_out','fan_in']:
                f[f'graph_{side}_{name}_{label}']=np.nan
        queues=defaultdict(deque); counts=defaultdict(Counter)
        def stat(a,direction,t):
            key=(a,direction); q=queues[key]; c=counts[key]
            while q and q[0][0]<t-seconds:
                _,peer=q.popleft(); c[peer]-=1
                if not c[peer]: del c[peer]
            return len(q),len(c)
        for t,batch in d.loc[d._top].groupby('_t',sort=True):
            for i,r in batch.iterrows():
                for side,a in [('sender',r['_src']),('receiver',r['_dst'])]:
                    n,k=stat(a,'out',t); m,j=stat(a,'in',t)
                    
                    for name,val in zip(['sent_n','received_n','fan_out','fan_in'],[n,m,k,j]): f.at[i,f'graph_{side}_{name}_{label}']=val
            for _,r in batch.iterrows():
                for a,direction,peer in [(r['_src'],'out',r['_dst']),(r['_dst'],'in',r['_src'])]:
                    queues[(a,direction)].append((t,peer)); counts[(a,direction)][peer]+=1
    for label in windows: f[f'graph_sender_fifo_fraction_{label}']=np.nan
    f['graph_sender_fifo_observed_fraction']=np.nan
    f['graph_sender_fifo_age_minutes']=np.nan
    inventory=defaultdict(deque); evidence=[]
    for t,batch in d.loc[d._top].groupby('_t',sort=True):
        # Unknown debits may consume recorded XRP inventory. Clear it before
        # processing any same-time supported sends; no arbitrary tie ordering.
        for _,r in batch.loc[~batch._fifo].iterrows(): inventory[r['_src']].clear()
        # Multiple sends from an account at one timestamp have unknown order.
        ambiguous=set(batch.loc[batch._fifo,'_src'].loc[lambda s:s.duplicated(False)])
        for a in ambiguous: inventory[a].clear()
        for i,r in batch.loc[batch._fifo].iterrows():
            if r['_src'] in ambiguous: continue
            remaining=int(r['_drops']); allocations=[]; q=inventory[r['_src']]
            while remaining and q:
                receipt=q[0]; amount=min(remaining,receipt[2]); delay=t-receipt[1]
                allocations.append((amount,delay)); evidence.append((receipt[0],r.TRANSACTION_HASH,amount/1e6,delay/60))
                remaining-=amount; receipt[2]-=amount
                if not receipt[2]: q.popleft()
            total=int(r['_drops'])
            f.at[i,'graph_sender_fifo_observed_fraction']=sum(a for a,_ in allocations)/total
            for label,seconds in windows.items(): f.at[i,f'graph_sender_fifo_fraction_{label}']=sum(a for a,dt in allocations if dt<=seconds)/total
            if allocations: f.at[i,'graph_sender_fifo_age_minutes']=sum(a*dt for a,dt in allocations)/sum(a for a,_ in allocations)/60
        # Receipts become available only after all sends at this timestamp.
        for _,r in batch.loc[batch._fifo].iterrows(): inventory[r['_dst']].append([r.TRANSACTION_HASH,t,int(r['_drops'])])
    evidence=pd.DataFrame(evidence,columns=['receipt_hash','send_hash','matched_xrp','delay_minutes'])
    coverage={'transactions':len(d),'topology_supported':int(d._top.sum()),'native_fifo_eligible':int(d._fifo.sum()),'fifo_computed':int(f.graph_sender_fifo_observed_fraction.notna().sum()),'native_dust':int(f.graph_native_dust.eq(1).sum()),'fully_observed_7d':int(f.graph_partial_history_7d.eq(0).sum()),'limitations':['History is observed sample coverage, not complete account history.','No service registry supplied: service status is unknown.','FIFO ignores fees and assumes observed eligible receipts fund later eligible sends.','Unsupported debits and simultaneous sends clear inventory conservatively.','FIFO fractions of zero mean no allocation in retained observed inventory, not no real funding.']}
    assert len(f)==len(data) and f.TRANSACTION_HASH.is_unique
    assert all(c=='TRANSACTION_HASH' or c.startswith('graph_') for c in f)
    return f,evidence,coverage


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--dust-xrp", type=float, default=1.0)
    args = parser.parse_args()
    if not np.isfinite(args.dust_xrp) or args.dust_xrp < 0:
        parser.error("--dust-xrp must be finite and nonnegative")
    features, matches, coverage = make_graph_features(
        pd.read_parquet(args.input), dust_xrp=args.dust_xrp
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output_dir / f"{args.split}_graph_features.parquet", index=False)
    matches.to_parquet(args.output_dir / f"{args.split}_fifo_evidence.parquet", index=False)
    coverage["feature_nonmissing_counts"] = features.notna().sum().to_dict()
    coverage["input_path"] = str(args.input.resolve())
    coverage["dust_threshold_xrp"] = args.dust_xrp
    (args.output_dir / f"{args.split}_graph_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n"
    )
    print(json.dumps(coverage, indent=2))
