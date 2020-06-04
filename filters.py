import numpy as np
from scipy.linalg import sqrtm
import scipy.sparse as sp
import scipy.sparse.linalg as spln


def g(C_t, X_t):
    return np.dot(C_t, X_t)

def w(dim, Cov):
    return np.random.multivariate_normal(np.zeros(dim), Cov)

def generateMeasurements(C, X, R, noise=True):
    T, m = X.shape[0], C.shape[1]
    Y = np.zeros((T, m))
    for t in range(T):
        Y[t,:] = g(C[t,:,:], X[t,:])
        if noise:
            Y[t,:] += w(m, R)
    
    return Y


def EKF(X, Y, C, mu0, Sigma0, Q, R, fDynamics, fMeas, JDynamics, propagator, dt):
    """
    Extended Kalman Filter.
    """
    T, n = X.shape
    m = C.shape[1]

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
        K = np.dot(Sigma[t,:,:], np.dot(C[t,:,].T, np.linalg.inv(np.dot(C[t,:,:],Sigma[t,:,:]).dot(C[t,:,:].T) + R)))
        mu[t,:] += np.dot(K, Y[t,:] - fMeas(C[t,:,:], mu[t,:]))
        Sigma[t,:,:] -= np.dot(K, np.dot(C[t,:,:], Sigma[t,:,:]))
        
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

def UKF(X, Y, C, mu0, Sigma0, Q, R, fDynamics, fMeas, propagator, dt):
    """
    Unscented Kalman Filter.
    """
    T, n = X.shape
    m = C.shape[1]

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
        ytab = np.zeros((2*n+1, m))
        for i in range(0,len(xtab)):
            ytab[i] = fMeas(C[t,:,:], xtab[i])
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

def PF(X, Y, C, mu0, Sigma0, Q, R, N, fDynamics, fMeas, propagator, dt):
    """
    Particle Filter.
    """
    T, n = X.shape
    m = C.shape[1]

    mu = np.zeros((T, n))
    mu[0,:] = mu0

    # Initialize population using initial guess for mu0 and sigma0
    xtab,wtab=create_particles(mu0,Sigma0,N)
    
    for t in range(1,T):
        #--- Predict ---#
        for j in range(0,len(xtab)):
            xtab[j] = propagator(xtab[j], dt, fDynamics) + w(n,Q)
        # Measurements of the particles states
        ytab = np.zeros((N, m))
        for j in range(0,len(xtab)):
            ytab[j] = fMeas(C[t,:,:], xtab[j])
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

def UPF(X, Y, C, mu0, Sigma0, Q, R, N, fDynamics, fMeas, propagator, dt):
    """
    Unscented Particle Filter.
    """
    T, n = X.shape
    m = C.shape[1]
    
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
            ytab = np.zeros((2*n+1, m))
            for i in range(0,len(xtab)):
                ytab[i] = fMeas(C[t,:,:], xtab[i])
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
        Ytab = np.zeros((N, m))
        for j in range(N):
            Ytab[j] = fMeas(C[t,:,:], mutab[j,t,:])
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
