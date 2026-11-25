"""Run with Python 3 and NumPy. All vector operations use float32."""
import numpy as np

rng = np.random.default_rng(7900)
np.seterr(invalid="ignore", divide="ignore")
def dot(a,b):
    return np.sum(a*b,axis=-1,keepdims=True)
def unit(a):
    return a/np.sqrt(dot(a,a))
def project(a,n):
    return a-n*dot(a,n)
def choose(mask,a,b):
    return np.where(mask,a,b)
def frame(method,d,n,t,b):
    axis = choose(abs(n[:,1:2]) < (.9 if method=="fix" else .99),
                  np.float32([0,1,0]),np.float32([1,0,0]))
    if method=="fix":
        p=np.cross(n,d)
        alt=np.cross(n,b)
        alt=choose(dot(alt,alt)<=1e-8,np.cross(n,axis),alt)
        mesh=np.cross(n,t)
        alt=choose(dot(mesh,mesh)<=1e-8,alt,mesh)
        p=choose(dot(p,p)<=np.maximum(1e-20,1e-8*dot(d,d)),alt,p)
        p=np.cross(p,n)
    else:
        p=project(unit(d) if method in ("old","O") else d,n)
        world=np.cross(axis,n)
        if method=="O":
            p=choose(dot(p,p)<1e-8,world,p)
        if method in ("N","candidate"):
            valid=dot(p,p)>=1e-8 if method=="N" else (dot(d,d)>1e-20)&(dot(p,p)>1e-8*dot(d,d))
            p=choose(valid,p,project(t,n))
            if method=="candidate":
                p=choose(dot(p,p)>1e-8,p,project(b,n))
            p=choose(dot(p,p)>=1e-8,p,world)
    t=unit(p)
    return t,np.cross(n,t)

def random_vec():
    return rng.normal(size=(100000,3)).astype("f")
n=unit(random_vec())
d=unit(random_vec())
t=unit(np.cross(n,random_vec()))
b=np.cross(n,t)
# Deterministic adversarial rows preceding the random sample.
angles=np.float32([0,0,1e-6,1e-6,0,0,0,0,0,.999e-4,1.001e-4,0])
k=len(angles)
nn=np.tile(np.float32([0,0,1]),(k,1))
tt=np.tile(np.float32([0,1,0]),(k,1))
bb=np.cross(nn,tt)
dd=nn.copy()
dd[:,0]=angles
dd[[1,3],2]=-1
dd[4]=0
dd[5]*=1e-18
tt[6:8]=nn[6:8]
bb[7]=0
nn[8]*=1.000001
bb[11]*=-1
tt[11]=nn[11]
d,n,t,b=[np.concatenate((x,y)) for x,y in zip((dd,nn,tt,bb),(d,n,t,b))]
rot=np.float32([[.8,.6,0],[-.36,.48,.8],[.48,-.64,.6]])
old=frame("old",d,n,t,b)[0]
for method in ("old","N","O","candidate","fix"):
    # Earlier proposals read fd.T/B, which are zero at this call site.
    mt,mb=(t,b) if method=="fix" else (0*t,0*b)
    T,B=frame(method,d,n,mt,mb)
    rotated=frame(method,*[x@rot for x in (d,n,mt,mb)])[0]
    err=lambda x: float(np.nanmax(abs(x)))
    length=lambda x: np.sqrt(dot(x,x))
    print(method,"invalid",np.sum(~np.isfinite(T+B).all(1)),
          "length errors",err(length(T)-1),err(length(B)-1),
          "dot NT NB TB",*[err(dot(x,y)) for x,y in ((n,T),(n,B),(T,B))],
          "random delta",err(length(T[k:]-old[k:])),
          "rotation error (excluding world fallback)",err(np.delete(length(rotated-T@rot),7)),
          "threshold jump",length(T[9]-T[10]).item())
    if method=="fix":
        assert np.isfinite(T+B).all() and err(length(T)-1)<2e-6 and err(length(B)-1)<2e-6
        assert max(err(dot(x,y)) for x,y in ((n,T),(n,B),(T,B)))<1e-6
        np.testing.assert_allclose(T[[0,1,2,3,4,5,8]],t[[0,1,2,3,4,5,8]])
        np.testing.assert_allclose(T[[6,11]],b[[6,11]])
        assert err(length(T[k:]-old[k:]))<1e-4 and err(np.delete(length(rotated-T@rot),7))<1e-4
