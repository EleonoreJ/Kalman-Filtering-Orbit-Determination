import numpy as np
from scipy.linalg import sqrtm
import scipy.sparse as sp
import scipy.sparse.linalg as spln
from measurement import w, g


def EKF(X, Y, mu0, Sigma0, Q, R, fDynamics, JDynamics, propagator, dt, fMeas=g, JMeas=None, X_gps=None):
    """
    Extended Kalman Filter.
    """
    T, n = X.shape

    mu = np.zeros((T, n))
    Sigma = np.zeros((T, n, n))
    A = np.zeros((T, n, n))

    mu[0,:] = mu0
    Sigma[0,:,:] = Sigma0
    A[0,:,:] = JDynamics(mu[0,:])

    for t in range(1,T):
        #--- Predict ---#
        mu[t,:] = propagator(mu[t-1,:], dt, fDynamics)
        Sigma[t,:,:] = np.dot(A[t-1,:,:], np.dot(Sigma[t-1,:,:], A[t-1,:,:].T))
        Sigma[t,:,:] += Q

        A[t,:,:] = JDynamics(mu[t,:])

        #--- Update ---#
        if X_gps is None or JMeas is None:
            C = np.eye(6)
            y = fMeas(C, mu[t,:])
        else:
            C = JMeas(X_gps[:,:,t], mu[t,:])
            y = fMeas(X_gps[:,:,t], mu[t,:])
        K = np.dot(Sigma[t,:,:], np.dot(C.T, np.linalg.inv(np.dot(C,Sigma[t,:,:]).dot(C.T) + R)))
        mu[t,:] += np.dot(K, Y[t,:] - y)
        Sigma[t,:,:] -= np.dot(K, np.dot(C, Sigma[t,:,:]))
        
    return mu, Sigma


def UT(mu, Sigma, lda=2):
    """
    Unscented Transform.
    """
    n = len(mu)
    xtab = np.zeros((2*n+1, n))
    wtab = np.ones((2*n+1, 1))/2/(lda+n)
    xtab[0] = mu
    wtab[0] = lda/(lda+n)
    S = sqrtm((lda+n)*Sigma)
    for i in range(1,n+1):
        xtab[i] = mu + S[:,i-1]
        xtab[i+n] = mu - S[:,i-1]  
    assert np.sum(wtab) == 1
    return xtab, wtab

def UTi(xtab, wtab, lda=2):
    """
    Inverse Unscented Transform.
    """
    mu = np.sum(xtab*wtab, axis=0)
    xh = xtab - mu
    Sigma = np.dot(xh.T, xh*wtab)
    return mu, Sigma

def UKF(X, Y, mu0, Sigma0, Q, R, fDynamics, propagator, dt, fMeas=g, X_gps=None):
    """
    Unscented Kalman Filter.
    """
    T, n = X.shape

    mu = np.zeros((T, n))
    Sigma = np.zeros((T, n, n))

    mu[0,:] = mu0
    Sigma[0,:,:] = Sigma0
    
    for t in range(1,T):
        #--- Predict ---#
        # Compute sigma-points and weights
        xtab, wtab = UT(mu[t-1,:], Sigma[t-1,:,:])
        for i in range(0,len(xtab)):
            xtab[i] = propagator(xtab[i], dt, fDynamics) + w(n,Q)
        
        # Predict mean and covariance
        mu[t,:], Sigma[t,:,:] = UTi(xtab, wtab)
        Sigma[t,:,:] += Q
        
        #--- Update ---#
        # Recompute sigma-points with predictions
        xtab, wtab = UT(mu[t,:], Sigma[t,:,:])
        
        # Sigma-point measurements
        if X_gps is None:
            ytab = np.zeros((2*n+1, 6))
            for i in range(0,len(xtab)):
                ytab[i] = fMeas(np.eye(6), xtab[i])
        else:
            ytab = np.zeros((2*n+1, 16))
            for i in range(0,len(xtab)):
                ytab[i] = fMeas(X_gps[:,:,t], xtab[i])
        
        # Expected measurement
        yh = np.sum(ytab*wtab,axis=0)
        
        # Empirical covariances
        Sigma_y = np.dot((ytab-yh).T, wtab*(ytab-yh))
        Sigma_y += R
        Sigma_xy = np.dot((xtab-mu[t,:]).T, wtab*(ytab-yh))
        
        # Update
        K = np.dot(Sigma_xy, np.linalg.inv(Sigma_y))
        mu[t,:] += np.dot(K, (Y[t,:] - yh))
        Sigma[t,:,:] -= np.dot(K, Sigma_xy.T)

    return mu, Sigma


def low_variance_resampling(xtab,wtab,N): 
    """
    Low variance resampling implementation.
    """
    s=np.random.rand()/N
    j=0
    xlist=[]
    c=wtab[0]
    wlist=[]
    for i in range(N):
        u=s+i/N
        while u>c:
            j+=1
            c+=wtab[j]
        xlist.append(xtab[j])
        wlist.append(wtab[j])
    return np.array(xlist), np.array(wlist)/sum(wlist)

def create_particles(mu,Sigma,N):
    """
    Creates a randomly distributed population with uniform initial weights.
    """
    xtab=np.random.multivariate_normal(mu,Sigma,N)
    wtab=np.ones(N)/N
    return xtab, wtab

def lognorm_pdf(x,mu,Sigma):
    """
    Computes the PDF of a multivariate log-normal distribution.
    """
    nx = len(Sigma)
    norm_coeff = nx*np.log(2*np.pi)+np.linalg.slogdet(Sigma)[1]
    err = x-mu
    if (sp.issparse(Sigma)):
        numerator = spln.spsolve(Sigma, err).T.dot(err)
    else:
        numerator = np.linalg.solve(Sigma, err).T.dot(err)
    return -0.5*(norm_coeff+numerator)

def exp_normalize(x):
    """
    Exp-normalization implementation to avoid numerical overflow.
    """
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def PF(X, Y, mu0, Sigma0, Q, R, N, fDynamics, propagator, dt, fMeas=g, X_gps=None):
    """
    Particle Filter.
    """
    T, n = X.shape

    mu = np.zeros((T, n))
    mu[0,:] = mu0

    # Initialize population using initial guess for mu0 and sigma0
    xtab,wtab=create_particles(mu0,Sigma0,N)
    
    for t in range(1,T):
        #--- Predict ---#
        for j in range(0,len(xtab)):
            xtab[j] = propagator(xtab[j], dt, fDynamics) + w(n,Q)
        
        if X_gps is None:
            m = 6
            ytab = np.zeros((N, m))
            for j in range(0,len(xtab)):
                ytab[j] = fMeas(np.eye(m), xtab[j])
        else:
            m = 16
            ytab = np.zeros((N, m))
            for j in range(0,len(xtab)):
                ytab[j] = fMeas(X_gps[:,:,t], xtab[j])
        
        # Update weights
        wtab=np.zeros(N)
        for j in range(N):
            wtab[j] = lognorm_pdf((Y[t,:]-ytab[j]), np.zeros(m), R)
        wtab = exp_normalize(wtab)
        
        # Resample
        xtab, wtab = low_variance_resampling(xtab,wtab,N)

        #--- Update ---#
        mu[t,:] = np.dot(xtab.T,wtab)
       
    return mu


def low_variance_resampling_UPF(xtab,Sigmatab,wtab,N): 
    """
    Low variance resampling implementation for the UPF.
    """
    s=np.random.rand()/N
    j=0
    xlist=[]
    Sigmalist=[]
    c=wtab[0]
    wlist=[]
    for i in range(N):
        u=s+i/N
        while u>c:
            j+=1
            c+=wtab[j]
        xlist.append(xtab[j])
        Sigmalist.append(Sigmatab[j])
        wlist.append(wtab[j])
    return np.array(xlist), np.array(Sigmalist), np.array(wlist)/sum(wlist)

def UPF(X, Y, mu0, Sigma0, Q, R, N, fDynamics, propagator, dt, fMeas=g, X_gps=None):
    """
    Unscented Particle Filter.
    """
    T, n = X.shape
    
    mu = np.zeros((T, n))
    Sigma = np.zeros((T, n, n))
    mu[0,:] = mu0
    Sigma[0,:,:] = Sigma0
    
    # Initialize population
    mutab = np.zeros((N,T,n))
    Sigmatab = np.zeros((N,T,n,n))
    for j in range(N):
        mutab[j,0,:] = mu0
        Sigmatab[j,0,:,:] = Sigma0 
    Wtab = np.ones(N)/N
    
    for t in range(1,T):

        #--- Update the particles with the UKF ---#
        for j in range(N):
            #--- Predict ---#
            # Compute sigma-points and weights
            xtab, wtab = UT(mutab[j,t-1,:], Sigmatab[j,t-1,:,:])
            for i in range(0,len(xtab)):
                xtab[i] = propagator(xtab[i], dt, fDynamics) + w(n,Q)
            
            # Predict mean and covariance
            mutab[j,t,:], Sigmatab[j,t,:,:] = UTi(xtab, wtab)
            Sigmatab[j,t,:,:] += Q

            #--- Update ---#
            # Recompute sigma-points with predictions
            xtab, wtab = UT(mutab[j,t,:], Sigmatab[j,t,:,:])
            
            # Sigma-point measurements
            if X_gps is None:
                ytab = np.zeros((2*n+1, 6))
                for i in range(0,len(xtab)):
                    ytab[i] = fMeas(np.eye(6), xtab[i])
            else:
                ytab = np.zeros((2*n+1, 16))
                for i in range(0,len(xtab)):
                    ytab[i] = fMeas(X_gps[:,:,t], xtab[i])
            
            # Expected measurement
            yh = np.sum(ytab*wtab,axis=0)
            
            # Empirical covariances
            Sigma_y = np.dot((ytab-yh).T, wtab*(ytab-yh))
            Sigma_y += R
            Sigma_xy = np.dot((xtab-mu[t,:]).T, wtab*(ytab-yh))
            
            # Update
            K = np.dot(Sigma_xy, np.linalg.inv(Sigma_y))
            mutab[j,t,:] += np.dot(K, (Y[t,:] - yh))
            Sigmatab[j,t,:,:] -= np.dot(K, Sigma_xy.T)
            
        # Measurements of the particles states
        if X_gps is None:
            m = 6
            Ytab = np.zeros((N, m))
            for j in range(N):
                Ytab[j] = fMeas(np.eye(m), mutab[j,t,:])
        else:
            m = 16
            ytab = np.zeros((N, m))
            for j in range(N):
                ytab[j] = fMeas(X_gps[:,:,t], mutab[j,t,:])
        
        # Update weights
        for j in range(N):
            Wtab[j] = lognorm_pdf((Y[t,:]-Ytab[j]), np.zeros(m), R)
        Wtab = exp_normalize(Wtab)
        
        # Resample
        mutab_, Sigmatab_, Wtab = low_variance_resampling_UPF(mutab[:,t,:],Sigmatab[:,t,:,:],Wtab,N)
        mutab[:,t,:] = mutab_
        Sigmatab[:,t,:,:] = Sigmatab_

        # Update
        mu[t,:] = np.dot(mutab_.T,Wtab)
        Sigma[t,:,:] = np.dot(Sigmatab_.T,Wtab)
        
    return mu
