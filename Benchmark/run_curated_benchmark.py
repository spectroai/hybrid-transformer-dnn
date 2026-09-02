"""Reproducible MR-FMN benchmark reanalysis for revised Figure 3.

Outputs exact-mass tolerance sensitivity, plain/modified cosine and
Transformer-DNN retrieval metrics, paired bootstrap contrasts, reaction-class
performance, and a mass-rule decoy analysis. The curated benchmark is used;
this script does not claim to assess chromatographic in-source fragmentation.
"""
from pathlib import Path
import argparse
import ast, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors

SEED = 42
N_BOOT = 2000
N_DECOY = 1000
PPM_LEVELS = [5, 10, 20]
ABS_LEVEL = 0.05
FRAG_TOL = 0.02

COLORS = {"MW only":"#9AA3AD", "Cosine":"#3D8DC4", "Modified cosine":"#1C607F", "Transformer-DNN":"#8C62BD"}

def exact_mass(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    return Descriptors.ExactMolWt(m) if m is not None else np.nan

def arr(x):
    if isinstance(x, str):
        try: return np.asarray(ast.literal_eval(x), dtype=float)
        except Exception: return np.array([], dtype=float)
    try: return np.asarray(x, dtype=float)
    except Exception: return np.array([], dtype=float)

def prep(mz, inten):
    mz, inten = arr(mz), arr(inten); n=min(len(mz),len(inten)); mz, inten=mz[:n],inten[:n]
    ok=np.isfinite(mz)&np.isfinite(inten)&(inten>0); mz,inten=mz[ok],inten[ok]
    if not len(mz): return mz,inten
    inten=inten/inten.max(); ok=inten>=0.01; mz,inten=mz[ok],inten[ok]
    if len(mz)>80:
        k=np.argpartition(inten,-80)[-80:];mz,inten=mz[k],inten[k]
    k=np.argsort(mz); return mz[k],inten[k]

def greedy_similarity(a_mz,a_i,b_mz,b_i,shift=None,tol=FRAG_TOL):
    if not len(a_mz) or not len(b_mz): return np.nan
    cand=[]
    for i,x in enumerate(a_mz):
        lo=np.searchsorted(b_mz,x-tol); hi=np.searchsorted(b_mz,x+tol)
        for j in range(lo,hi): cand.append((a_i[i]*b_i[j],i,j))
        if shift is not None and abs(shift)>tol:
            y=x-shift; lo=np.searchsorted(b_mz,y-tol); hi=np.searchsorted(b_mz,y+tol)
            for j in range(lo,hi): cand.append((a_i[i]*b_i[j],i,j))
    useda=set();usedb=set();dot=0.0
    for val,i,j in sorted(cand,reverse=True):
        if i not in useda and j not in usedb: dot+=val;useda.add(i);usedb.add(j)
    den=np.linalg.norm(a_i)*np.linalg.norm(b_i)
    return dot/den if den else np.nan

def fp_bits(x):
    v=arr(x)
    return (v>=0.4).astype(np.uint8) if len(v) else None

def tanimoto(a,b):
    if a is None or b is None or len(a)!=len(b): return np.nan
    inter=np.bitwise_and(a,b).sum(); union=np.bitwise_or(a,b).sum()
    return float(inter/union) if union else 0.0

def ap_hit(group, score_col):
    y=group.is_true.to_numpy(int); s=group[score_col].to_numpy(float)
    ok=np.isfinite(s);y,s=y[ok],s[ok]
    if not len(y) or y.sum()==0:return np.nan,np.nan
    order=np.argsort(-s,kind='mergesort'); yy=y[order]
    ranks=np.flatnonzero(yy)+1; ap=np.mean(np.arange(1,len(ranks)+1)/ranks)
    top=s==np.nanmax(s); hit=y[top].mean() # expected Hit@1 under random tie-breaking
    return float(ap),float(hit)

def bootstrap(metric_df, methods, n=N_BOOT):
    rng=np.random.default_rng(SEED); ids=metric_df.product_index.unique(); rows=[]
    piv_ap=metric_df.pivot(index='product_index',columns='method',values='AP')
    piv_h=metric_df.pivot(index='product_index',columns='method',values='Hit@1')
    for metric,piv in [('mAP',piv_ap),('Hit@1',piv_h)]:
        for m in methods:
            vals=[]
            for _ in range(n): vals.append(np.nanmean(piv.loc[rng.choice(ids,len(ids),replace=True),m]))
            rows.append([metric,m,np.nanmean(piv[m]),*np.quantile(vals,[.025,.975])])
    return pd.DataFrame(rows,columns=['metric','method','estimate','ci_low','ci_high'])

def paired_boot(metric_df, ref='Transformer-DNN', n=N_BOOT):
    rng=np.random.default_rng(SEED+1); ids=metric_df.product_index.unique(); rows=[]
    for metric,col in [('mAP','AP'),('Hit@1','Hit@1')]:
        piv=metric_df.pivot(index='product_index',columns='method',values=col)
        for other in ['Cosine','Modified cosine']:
            d=(piv[ref]-piv[other]).dropna().to_numpy(); vals=[]
            for _ in range(n): vals.append(np.mean(rng.choice(d,len(d),replace=True)))
            rows.append([metric,f'{ref} - {other}',d.mean(),*np.quantile(vals,[.025,.975])])
    return pd.DataFrame(rows,columns=['metric','contrast','difference','ci_low','ci_high'])

def main(data_dir: Path, output_dir: Path):
    DATA = data_dir.resolve()
    OUT = output_dir.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    gt=pd.read_csv(DATA/'reactions_verified_true.csv')
    p0=pd.read_csv(DATA/'products_enriched.csv').drop_duplicates('product_index').copy()
    r0=pd.read_csv(DATA/'reactants_enriched.csv').drop_duplicates('reactant_index').copy()
    p0['exact_mass']=p0.product_smiles.map(exact_mass);r0['exact_mass']=r0.reactant_smiles.map(exact_mass)
    p0['spec']=[prep(x,y) for x,y in zip(p0.mz_values,p0.intensities)];r0['spec']=[prep(x,y) for x,y in zip(r0.mz_values,r0.intensities)]
    p0['fp']=p0.Transformer_embedding.map(fp_bits);r0['fp']=r0.Transformer_embedding.map(fp_bits)
    P=p0.set_index('product_index');R=r0.set_index('reactant_index')
    gt=gt.copy();gt['p_exact']=gt.product_smiles.map(exact_mass);gt['r_exact']=gt.reactant_smiles.map(exact_mass);gt['exact_delta']=gt.p_exact-gt.r_exact
    rules=gt.groupby('reaction_type').exact_delta.median().to_dict(); rules.pop('MonomerToDimer',None)
    pids=P.index.to_numpy();rids=R.index.to_numpy();pm=P.exact_mass.to_numpy();rm=R.exact_mass.to_numpy();diff=pm[:,None]-rm[None,:]
    true_set=set(zip(gt.product_index.astype(int),gt.reactant_index.astype(int)))

    # Sensitivity of mass gating.
    sens=[]; pools={}
    settings=[('5 ppm',5),('10 ppm',10),('20 ppm',20),('±0.05 Da',None)]
    for label,ppm in settings:
        mask=np.zeros(diff.shape,bool)
        for delta in rules.values():
            tol=ABS_LEVEL if ppm is None else np.maximum(pm[:,None]*ppm*1e-6,1e-6)
            mask |= np.abs(diff-delta)<=tol
        ii,jj=np.where(mask); pairs=set(zip(pids[ii].astype(int),rids[jj].astype(int)))
        retained=sum(x in pairs for x in true_set)/len(true_set)
        sens.append([label,len(pairs),len(pairs)/len(pids),retained])
        pools[label]=(ii,jj)
    sensitivity=pd.DataFrame(sens,columns=['window','candidate_pairs','candidates_per_product','ground_truth_retention'])
    sensitivity.to_csv(OUT/'mass_window_sensitivity.csv',index=False)

    # Use 10 ppm as the primary high-resolution setting.
    ii,jj=pools['10 ppm']; records=[]
    gt_type=gt.set_index(['product_index','reactant_index']).reaction_type.to_dict()
    for i,j in zip(ii,jj):
        pid,rid=int(pids[i]),int(rids[j]); ps,rs=P.loc[pid,'spec'],R.loc[rid,'spec']
        delta=pm[i]-rm[j]
        records.append([pid,rid,(pid,rid) in true_set,gt_type.get((pid,rid),'Candidate'),
                        greedy_similarity(*ps,*rs),greedy_similarity(*ps,*rs,shift=delta),tanimoto(P.loc[pid,'fp'],R.loc[rid,'fp'])])
    cand=pd.DataFrame(records,columns=['product_index','reactant_index','is_true','reaction_type','cosine','modified_cosine','transformer'])
    cand['mw_only']=0.0;cand.to_csv(OUT/'candidate_scores_10ppm.csv',index=False)
    methods={'MW only':'mw_only','Cosine':'cosine','Modified cosine':'modified_cosine','Transformer-DNN':'transformer'}
    rows=[]
    for pid,g in cand.groupby('product_index'):
        rt=gt.loc[gt.product_index==pid,'reaction_type'].mode().iat[0]
        for m,c in methods.items():
            ap,h=ap_hit(g,c);rows.append([pid,rt,m,ap,h,len(g)])
    md=pd.DataFrame(rows,columns=['product_index','reaction_type','method','AP','Hit@1','candidate_count'])
    md.to_csv(OUT/'metrics_per_product.csv',index=False)
    summary=bootstrap(md,list(methods));summary.to_csv(OUT/'metrics_bootstrap_summary.csv',index=False)
    contrasts=paired_boot(md);contrasts.to_csv(OUT/'paired_bootstrap_contrasts.csv',index=False)
    cls=md.groupby(['reaction_type','method']).agg(mAP=('AP','mean'),Hit1=('Hit@1','mean'),n=('product_index','nunique')).reset_index()
    cls.to_csv(OUT/'metrics_by_reaction_class.csv',index=False)

    # Decoy rules: shift every real delta by a random chemically implausible offset.
    # Transformer all-pair similarities allow the same score threshold to be applied.
    pf=list(P.fp);rf=list(R.fp); tsim=np.full(diff.shape,np.nan)
    for i,a in enumerate(pf):
        if a is None: continue
        for j,b in enumerate(rf):
            if b is not None: tsim[i,j]=tanimoto(a,b)
    rng=np.random.default_rng(SEED+2); real_mask=np.zeros(diff.shape,bool)
    for delta in rules.values(): real_mask|=np.abs(diff-delta)<=np.maximum(pm[:,None]*10e-6,1e-6)
    real=[real_mask.sum(),np.sum(real_mask&(tsim>=0.85))]
    dec=[]; deltas=np.array(list(rules.values()))
    for _ in range(N_DECOY):
        shifts=rng.choice([-1,1],len(deltas))*rng.uniform(25,75,len(deltas))
        dm=np.zeros(diff.shape,bool)
        for delta in deltas+shifts: dm|=np.abs(diff-delta)<=np.maximum(pm[:,None]*10e-6,1e-6)
        dec.append([dm.sum(),np.sum(dm&(tsim>=0.85))])
    dec=np.asarray(dec); decdf=pd.DataFrame(dec,columns=['mass_candidates','transformer_ge_0.85']);decdf.to_csv(OUT/'decoy_rule_null_1000.csv',index=False)
    decsum=pd.DataFrame({'metric':['Mass-matched pairs','Transformer ≥ 0.85'],'observed':real,
        'decoy_mean':dec.mean(0),'decoy_low':np.quantile(dec,.025,axis=0),'decoy_high':np.quantile(dec,.975,axis=0),
        'empirical_p':[(1+np.sum(dec[:,k]>=real[k]))/(N_DECOY+1) for k in range(2)]})
    decsum.to_csv(OUT/'decoy_rule_summary.csv',index=False)

    # Combined figure.
    sns.set_theme(style='whitegrid',context='paper');fig=plt.figure(figsize=(14.8,8.8),dpi=200)
    gs=fig.add_gridspec(2,3,width_ratios=[1.08,1.12,1.15],hspace=.45,wspace=.52)
    ax=fig.add_subplot(gs[0,0]); x=np.arange(len(sensitivity));
    ax.plot(x,sensitivity.ground_truth_retention*100,'o-',color='#D55345',lw=2,label='Ground-truth retained')
    ax.set_xticks(x,sensitivity.window,rotation=20,ha='right');ax.set_ylabel('Ground-truth retained (%)');ax.set_ylim(96,100.4)
    ax2=ax.twinx();ax2.plot(x,sensitivity.candidates_per_product,'s--',color='#4A83B7',lw=1.8,label='Candidates per product');ax2.set_ylabel('Candidates per product')
    ax.set_title('a  Exact-mass window sensitivity',loc='left',fontweight='bold')
    lines=ax.lines+ax2.lines;ax.legend(lines,[l.get_label() for l in lines],frameon=False,fontsize=8,loc='lower right')

    ax=fig.add_subplot(gs[0,1]); sub=summary.copy(); xx=np.arange(4);w=.34
    for k,metric in enumerate(['mAP','Hit@1']):
        q=sub[sub.metric==metric].set_index('method').loc[list(methods)];pos=xx+(k-.5)*w
        ax.bar(pos,q.estimate,w,color=[COLORS[m] for m in methods],alpha=.55 if k==1 else 1,hatch='//' if k==1 else None)
        ax.errorbar(pos,q.estimate,yerr=[q.estimate-q.ci_low,q.ci_high-q.estimate],fmt='none',ecolor='#263442',capsize=2,lw=1)
    ax.set_xticks(xx,['MW only','Cosine','Modified\ncosine','Transformer-\nDNN'],rotation=18,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Score');ax.set_title('b  Reaction-pair retrieval',loc='left',fontweight='bold')
    ax.text(.02,.96,'solid: mAP   hatched: Hit@1',transform=ax.transAxes,va='top',fontsize=8)

    ax=fig.add_subplot(gs[0,2]); con=contrasts.copy(); yy=np.arange(len(con));
    ax.axvline(0,color='#8A939C',lw=1);ax.errorbar(con.difference,yy,xerr=[con.difference-con.ci_low,con.ci_high-con.difference],fmt='o',color='#7A4FA3',capsize=3)
    short=[]
    for a,b in zip(con.contrast,con.metric): short.append(('vs cosine' if a.endswith('Cosine') else 'vs modified cosine')+f' ({b})')
    ax.set_yticks(yy,short);ax.tick_params(axis='y',labelsize=8);ax.set_xlabel('Transformer-DNN minus comparator');ax.set_title('c  Paired bootstrap contrasts',loc='left',fontweight='bold');ax.invert_yaxis()

    ax=fig.add_subplot(gs[1,:2]); keep=cls[cls.n>=10];mat=keep.pivot(index='reaction_type',columns='method',values='mAP').reindex(columns=list(methods))
    sns.heatmap(mat,annot=True,fmt='.2f',cmap='Blues',vmin=0,vmax=1,cbar_kws={'label':'mAP'},ax=ax,linewidths=.5)
    ax.set_xlabel('');ax.set_ylabel('');ax.set_title('d  Performance by reaction class (classes with ≥10 products)',loc='left',fontweight='bold')

    ax=fig.add_subplot(gs[1,2]); ypos=np.arange(2)
    dlo=np.maximum(decsum.decoy_low.to_numpy(float),0.5)
    ax.errorbar(decsum.decoy_mean,ypos,xerr=[decsum.decoy_mean-dlo,decsum.decoy_high-decsum.decoy_mean],fmt='o',color='#8C62BD',capsize=3,label='Decoy rules')
    ax.scatter(decsum.observed,ypos,marker='D',s=48,color='#D55345',label='Observed rules',zorder=3)
    ax.set_yticks(ypos,decsum.metric);ax.set_xscale('log');ax.set_xlim(.5,1000);ax.set_xlabel('Number of candidate pairs (log scale)');ax.set_title('e  Reaction-rule shift null model',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8,loc='center right');ax.invert_yaxis()
    fig.suptitle('MR-FMN reaction retrieval: mass specificity, ranking performance and null-model validation',fontsize=14,fontweight='bold',y=.99)
    fig.savefig(OUT/'Figure3_MRFMN_combined.png',dpi=400,bbox_inches='tight');fig.savefig(OUT/'Figure3_MRFMN_combined.pdf',bbox_inches='tight');fig.savefig(OUT/'Figure3_MRFMN_combined.svg',bbox_inches='tight')
    (OUT/'analysis_summary.json').write_text(json.dumps({'primary_window':'10 ppm','n_pairs':len(gt),'n_products':gt.product_index.nunique(),'n_reactants':gt.reactant_index.nunique(),'summary':summary.to_dict('records'),'contrasts':contrasts.to_dict('records'),'decoy':decsum.to_dict('records')},indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print('\n',contrasts.to_string(index=False));print('\n',sensitivity.to_string(index=False));print('\n',decsum.to_string(index=False));print('\nOUTPUT',OUT)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Run the curated 453-pair MR-FMN benchmark.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the three benchmark CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
