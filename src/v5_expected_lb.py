"""
V5 — RMSE oficial esperado de la candidata V4b si sus probabilidades están calibradas, y dispersión por azar.

    python src/v5_expected_lb.py      # desde la raíz del proyecto; requiere models/experiments/v4b/final/ y candidate_v4b

Para cada fila del ranking se toma la mezcla de V4b (clase normal con varianza 200², clase 1 = MVT − SCHED, clase 2 =
86 400 + b; probabilidades de la compuerta o de la celda de fecha desplazada) y se calcula el error cuadrático esperado y su
varianza. No usa ningún target del ranking ni el leaderboard.
"""
import sys, os, numpy as np, polars as pl
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage"); sys.path.insert(0,'src')
from v3_protocol import shifted_no_aobt3_expr
from v2_protocol import DAY_SEC
U='models/experiments/v4b/final'
rk = pl.read_parquet('models/experiments/v4b/features_rank.parquet')
s = rk['mvt_minus_sched'].cast(pl.Float64).to_numpy(); n = s.size
cell = rk.select(shifted_no_aobt3_expr().fill_null(False))[:,0].to_numpy()
apt = rk['ADEP_mvt'].to_numpy(); A = rk['has_aobt3'].to_numpy()==0
L = lambda c, sd: np.load(f'{U}/{c}__s{sd}__rank.npy').astype(float)
PR=[]
for sd in (42,7,2025):
    b=L('c2_n',sd); p1=L('c1_clf',sd); P0=L('c0_clf',sd)
    PR.append(np.maximum(np.where(cell, P0[:,0]*b + P0[:,1]*s + P0[:,2]*(DAY_SEC+b), b + p1*(s-b)), 0))
pred=np.mean(PR,0)
b=np.mean([L('c2_n',sd) for sd in (42,7,2025)],0); p1=np.mean([L('c1_clf',sd) for sd in (42,7,2025)],0); P0=np.mean([L('c0_clf',sd) for sd in (42,7,2025)],0)
sig2=200.0**2
q0=np.where(cell,P0[:,0],1-p1); q1=np.where(cell,P0[:,1],p1); q2=np.where(cell,P0[:,2],0.0)
e0=(b-pred)**2+sig2; e1=(s-pred)**2; e2=(DAY_SEC+b-pred)**2+sig2
ESE=q0*e0+q1*e1+q2*e2
VSE=np.maximum(q0*e0**2+q1*e1**2+q2*e2**2-ESE**2,0)
tE=ESE.sum(); sd_=np.sqrt(VSE.sum())
print(f'RMSE esperado V4b (si está calibrado): {np.sqrt(tE/n):.1f} s; rango ±1σ por azar en filas de cola: [{np.sqrt((tE-sd_)/n):.1f}, {np.sqrt((tE+sd_)/n):.1f}]; ±2σ: [{np.sqrt(max(tE-2*sd_,0)/n):.1f}, {np.sqrt((tE+2*sd_)/n):.1f}]')
D=np.abs(s-b)
segs={'celda (fecha desplazada, sin NM)':cell,'LIRF sin NM (fuera de celda)':(~cell)&A&(apt=='LIRF'),'resto con |s−b|>1h':(~cell)&~(A&(apt=='LIRF'))&(D>3600),'resto (grueso)':(~cell)&~(A&(apt=='LIRF'))&(D<=3600)}
for k,mk in segs.items():
    print(f'  {k:34s} filas {mk.sum():7d}  MSE esperado {ESE[mk].sum()/tE*100:5.1f}%  varianza (azar) {VSE[mk].sum()/VSE.sum()*100:5.1f}%')
o=np.argsort(-ESE)
for k in (10,50,100,500):
    print(f'  top-{k} filas: {ESE[o[:k]].sum()/tE*100:.1f}% del MSE esperado, {VSE[o[:k]].sum()/VSE.sum()*100:.1f}% de la varianza')
sub=pl.read_parquet('submissions/candidate_v4b.parquet')
j=sub.join(pl.DataFrame({'MVT_ID_mvt':rk['MVT_ID_mvt'],'p':pred}),on='MVT_ID_mvt')
print('comprobación parquet vs recomputada: max |dif| =', float((j['TAXITIME_SEC_mvt']-j['p']).abs().max()))
