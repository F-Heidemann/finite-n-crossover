"""Deterministic checks for the 16 October review. Run from the repo root."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import rrr
import a3_rrr_order as A
import t1_mais_o78 as T

def scalar_logz(n, delta, d):
    """Independent integral: scalar a, Gaussian vector b integrated analytically."""
    def integrand(u):
        a=np.exp(u); q=1+n*a*a
        return np.exp(u-a*a/2-d*np.log(q)/2-n*delta*delta/(2*q))
    z=quad(integrand,-50,4,epsabs=1e-15,epsrel=1e-10,points=[-np.log(n)/2],limit=300)[0]
    return np.log(np.sqrt(2/np.pi)*z)

worst=0.
for n in (1e2,1e4,1e6):
    for s in (.2,2.,10.):
        delta=np.sqrt(2*s/n)
        lam,z=rrr.lam_n(n,3,1,1,('rank1',s),return_logZ=True)
        z0=rrr.lam_n(n,3,1,1,('rank1',0),return_logZ=True)[1]
        err=abs((z-z0)-(scalar_logz(n,delta,3)-scalar_logz(n,0,3)))
        h=1e-4
        fd=-(scalar_logz(n*np.exp(h),delta,3)-scalar_logz(n*np.exp(-h),delta,3))/(2*h)
        worst=max(worst,err,abs(lam-fd))
assert worst<1e-6,worst
print('Independent scalar Gaussian integral and fixed-truth derivative: max error',worst)

for s in (.1,1.,10.,100.):
    value=quad(lambda t: quad(lambda u: np.exp(-s*np.sin(t)**2*np.sin(u)**2),0,np.pi/2,epsabs=1e-11)[0],0,np.pi/2,epsabs=1e-11)[0]*4/np.pi**2
    expected=float(rrr.Phi_rank1(s,2,2,2,1))
    assert abs(value-expected)<1e-9,(s,value,expected)
print('Interior rank-one profile: beta-product quadrature agrees within 1e-9')

Sigma=np.array([[2.,.8],[.8,1.]])
L=np.linalg.cholesky(Sigma)
old=np.linalg.solve(L,np.eye(2)).T
new=np.linalg.solve(L.T,np.eye(2)).T
print('Old noise covariance max error:',np.max(np.abs(old.T@old-np.linalg.inv(Sigma))))
assert np.allclose(new.T@new,np.linalg.inv(Sigma),rtol=1e-13,atol=1e-13)
class FixedRNG:
    def chisquare(self,df): return {10:20.,9:6.8}[df]
    def standard_normal(self,shape):
        if shape==0: return np.empty(0)
        if shape==1: return np.array([.8*np.sqrt(5)])
        return np.eye(2)
hat,sig=T.dataset(10,0.,FixedRNG())
assert np.allclose(sig,Sigma)
assert np.allclose(10*hat.T@hat,np.linalg.inv(sig))
print('Corrected dataset covariance: verified to floating-point precision')

for n in (1e4,1e8,1e32):
    vals=[]
    for factor in (1,2):
        A.NU,A.NTH=1100*factor,192*factor
        def fn(s):
            d=np.sqrt(2*s/n); C=np.zeros((2,2)); C[0,0]=d
            return A.post(n,2,C,d)[2]-.5
        vals.append(brentq(fn,.1,5,xtol=1e-9))
    assert abs(vals[0]-vals[1])<1e-5,vals
    print('Tie window n=',n,'coarse/refined=',vals)
A.NU,A.NTH=1100,192
rrr.selftest()
print('All asserted review checks passed.')
