import numpy as np


def w(dim, Cov):
    return np.random.multivariate_normal(np.zeros(dim), Cov)

## Naive measurement model

def g(C_t, X_t):
    return np.dot(C_t, X_t)

def generateMeasurementsNaive(C, X, R, noise=True):
    T, m = X.shape[0], C.shape[1]
    Y = np.zeros((T, m))
    for t in range(T):
        Y[t,:] = g(C[t,:,:], X[t,:])
        if noise:
            Y[t,:] += w(m, R)
    return Y

## GPS measurement model

def rangeMeas(X_gps_t, X_t):
    return np.linalg.norm(X_gps_t[:, :3] - X_t[:3].T, axis = 1)

def dopplerMeas(X_gps_t, X_t):
    norm = np.linalg.norm(X_gps_t[:, :3] - X_t[:3].T, axis = 1, keepdims = True)
    y = (X_gps_t[:, :3] - X_t[:3].T)*(X_gps_t[:, :3] - X_t[:3].T)/norm
    y = np.reshape(y, -1)
    return y

def measurements(X_gps_t, X_t):
    r = rangeMeas(X_gps_t, X_t)
    doppler = dopplerMeas(X_gps_t, X_t)
    meas = np.concatenate((r, doppler), axis = 0)
    return meas

def generateRangeMeasurements(X_gps, X, R, noise=True):
    m, T = X_gps.shape[0], X.shape[0]
    Y = np.zeros((T,m))
    for t in range(T):
        Y[t,:] = rangeMeas(X_gps[:,:,t], X[t,:])  
        if noise:
            Y[t,:] += w(m, R)
    return Y

def generateDopplerMeasurements(X_gps, X, R, noise=True):
    m, T = X_gps.shape[0], X.shape[0]
    Y = np.zeros((T,3*m))
    for t in range(T):
        Y[t, :] = dopplerMeas(X_gps[:,:,t], X[t,:])
        if noise:
            Y[t,:] += w(3*m, R)
    return Y

def generateMeasurements(X_gps, X, R, noise=True):
    r = generateRangeMeasurements(X_gps, X, R[:4,:4])
    doppler = generateDopplerMeasurements(X_gps, X, R[4:,4:])
    measures = np.concatenate((r,doppler), axis = 1)
    return measures

def rangeMeasJacobian(X_gps_t, X_t):
    num_sat = X_gps_t.shape[0]
    norm = np.linalg.norm(X_gps_t[:, :3] - X_t[:3].T, axis = 1, keepdims = True)
    C = (X_t[:3].T - X_gps_t[:, :3])/norm
    C = np.concatenate((C, np.zeros((num_sat, 3))), axis = 1)
    return C

def dopplerMeasJacobian(X_gps_t, X_t):

    num_sat = X_gps_t.shape[0]
    C = np.zeros((num_sat*3, 6))
    norm = np.linalg.norm(X_gps_t[:, :3] - X_t[:3].T, axis = 1, keepdims = True)
    
    for i in range(num_sat):
        C[np.arange(3) + 3*i,np.arange(3)+3] = (X_t[:3].T - X_gps_t[i, :3])/norm[i]
        
        C[3*i, 0] = -(X_gps_t[i, 3] - X_t[3])*(1/norm[i] - np.square(X_gps_t[i, 0] - X_t[0])/norm[i]**3)
        C[3*i, 1] = (X_gps_t[i, 4] - X_t[4])*(X_gps_t[i, 0] - X_t[0])*(X_gps_t[i, 1] - X_t[1])/norm[i]**3
        C[3*i, 2] = (X_gps_t[i, 5] - X_t[5])*(X_gps_t[i, 0] - X_t[0])*(X_gps_t[i, 2] - X_t[2])/norm[i]**3
    
        C[3*i + 1, 1] = -(X_gps_t[i, 4] - X_t[4])*(1/norm[i] - np.square(X_gps_t[i, 1] - X_t[1])/norm[i]**3)
        C[3*i + 1, 0] = (X_gps_t[i, 3] - X_t[3])*(X_gps_t[i, 1] - X_t[1])*(X_gps_t[i, 0] - X_t[0])/norm[i]**3 
        C[3*i + 1, 2] = (X_gps_t[i, 5] - X_t[5])*(X_gps_t[i, 1] - X_t[1])*(X_gps_t[i, 2] - X_t[2])/norm[i]**3
    
        C[3*i + 2, 2] = -(X_gps_t[i, 5] - X_t[5])*(1/norm[i] - np.square(X_gps_t[i, 2] - X_t[2])/norm[i]**3) 
        C[3*i + 1, 0] = (X_gps_t[i, 3] - X_t[3])*(X_gps_t[i, 2] - X_t[2])*(X_gps_t[i, 0] - X_t[0])/norm[i]**3 
        C[3*i + 1, 1] = (X_gps_t[i, 4] - X_t[4])*(X_gps_t[i, 2] - X_t[2])*(X_gps_t[i, 1] - X_t[1])/norm[i]**3  
    
    return C

def measJacobian(X_gps_t, X_t):
    Jr = rangeMeasJacobian(X_gps_t, X_t)
    Jdoppler = dopplerMeasJacobian(X_gps_t, X_t)
    J = np.concatenate((Jr, Jdoppler), axis = 0)
    return J

