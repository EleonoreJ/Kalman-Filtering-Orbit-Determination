import numpy as np
from ostk.physics.environment.objects.celestial_bodies import Earth
from ostk.physics import Environment


def keplerianDynamics(X):
    """
    Computes the acceleration due to the Earth gravitational force (Keplerian dynamics).
    @param: X The spacecraft state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
    """
    mu = Earth.gravitational_parameter
    mu = float(mu.in_unit(mu.get_unit()))
  
    r = float(np.linalg.norm(X[0:3]))
    
    return -np.array(X[0:3])*mu/r**3


def keplerianJacobian(X):
    """
    Computes the Jacobian for Keplerian dynamics.
    @param: X The spacecraft state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
    """
    mu = Earth.gravitational_parameter
    mu = float(mu.in_unit(mu.get_unit()))
    
    r = float(np.linalg.norm(X[0:3]))
    ri = float(X[0])
    rj = float(X[1])
    rk = float(X[2])
    
    J = np.zeros((6, 6))
    J[0,3] = 1
    J[1,4] = 1
    J[2,5] = 1
    J[3,0] = -mu/r**3 + 3*mu*ri**2/r**5
    J[3,1] = 3*mu*ri*rj/r**5
    J[3,2] = 3*mu*ri*rk/r**5
    J[4,0] = J[3,1]
    J[4,1] = -mu/r**3 + 3*mu*rj**2/r**5
    J[4,2] = 3*mu*rj*rk/r**5
    J[5,0] = J[3,2]
    J[5,1] = J[4,2]
    J[5,2] = -mu/r**3 + 3*mu*rk**2/r**5
        
    return J


def j2Dynamics(X):
    """
    Computes the acceleration due to the Earth gravitational force 
    and Earth oblateness (J2).
    @param: X The spacecraft state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
    """
    mu = Earth.gravitational_parameter
    Re = Earth.equatorial_radius
    J2 = Earth.J2
    mu = float(mu.in_unit(mu.get_unit()))
    Re = float(Re.in_unit(Re.get_unit()))
    J2 = float(J2)
    
    r = float(np.linalg.norm(X[0:3]))
    rk = float(X[2])
    K = np.array([0,0,1])
    
    fJ2 = (-1/2)*mu*J2*Re**2*((6*rk/r**5)*K + (3/r**4 - 15*(rk**2/r**6))*np.array(X[0:3])/r)
    
    return keplerianDynamics(X) + fJ2


def j2Jacobian(X):
    """
    Computes the Jacobian for Keplerian + J2 dynamics.
    @param: X The spacecraft state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
    """

    mu = Earth.gravitational_parameter
    Re = Earth.equatorial_radius
    J2 = Earth.J2
    mu = float(mu.in_unit(mu.get_unit()))
    Re = float(Re.in_unit(Re.get_unit()))
    J2 = float(J2)
    
    r = float(np.linalg.norm(X[0:3]))
    ri = float(X[0])
    rj = float(X[1])
    rk = float(X[2])
    
    J = keplerianJacobian(X)
   
    a = -1/2*mu*J2*Re**2
    J[3,0] += a*((3*r**-5 - 15*rk**2*r**-7) + (-15*r**-7 + 105*rk**2*r**-9)*ri**2)
    J[3,1] += a*(-15*r**-7 + 105*rk**2*r**-9)*ri*rj
    J[3,2] += a*((-30*rk*ri*r**-7) + (-15*r**-7 + 105*rk**2*r**-9)*ri*rk)
    
    J[4,0] += a*(-15*r**-7 + 105*rk**2*r**-9)*ri*rj
    J[4,1] += a*((3*r**-5 - 15*rk**2*r**-7) + (-15*r**-7 + 105*rk**2*r**-9)*rj**2)
    J[4,2] += a*((-30*rk*rj*r**-7) + (-15*r**-7 + 105*rk**2*r**-9)*rj*rk)
    
    J[5,0] += a*(-45*rk*r**-7 + 105*rk**3*r**-9)*ri
    J[5,1] += a*(-45*rk*r**-7 + 105*rk**3*r**-9)*rj
    J[5,2] += a*((-45*rk*r**-7 + 105*rk**3*r**-9)*rk + (9*r**-5 - 45*rk**2*r**-7))
	
    return J


def dragDynamics(X, Cd=2.3, A=20, m=1500):
    """
    Computes the acceleration due to the Earth gravitational force,
    Earth oblateness (J2) and atmospheric drag.
    @param: X  The spacecraft state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
    @param: Cd The drag coefficient (default 2.3).
    @param: A  The satellite area subject to drag [m^2] (default 20).
    @param: m  The satellite mass [kg] (default 1500).
    """
    h0 = 0             # Sea level [m]
    rho0 = 1.2250      # Air density at sea level [kg/m^3]
    H = 10000          # Characteristic height [m]
    
    Re = Earth.equatorial_radius
    Re = float(Re.in_unit(Re.get_unit()))
    
    # Earth angular velocity vector in ECI
    omega = np.array([0, 0, 2*np.pi/86164.09])
    
    # Satellite velocity relative to Earth
    vRel = X[3:6] - np.cross(omega, X[0:3])
    
    # Atmospheric density
    h = np.linalg.norm(X[0:3]) - Re
    rho = rho0*np.exp(-(h-h0)/H)
    
    # Acceleration vector due to drag
    B = Cd*A/m
    fDrag = (-1/2)*B*rho*np.linalg.norm(vRel)*vRel
    
    return j2Dynamics(X) + fDrag


def dragJacobian(X, Cd=2.3, A=20, m=1500):
    """
    Computes the Jacobian for Keplerian + J2 + atmospheric drag dynamics.
    @param: X The spacecraft state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
    @param: Cd The drag coefficient (default 2.3).
    @param: A  The satellite area subject to drag [m^2] (default 20).
    @param: m  The satellite mass [kg] (default 1500).
    """
    h0 = 0             # Sea level [m]
    rho0 = 1.2250      # Air density at sea level [kg/m^3]
    H = 10000          # Characteristic height [m]
    
    Re = Earth.equatorial_radius
    Re = float(Re.in_unit(Re.get_unit()))
    
    # Earth angular velocity vector in ECI
    omega = np.array([0, 0, 2*np.pi/86164.09])
    om = omega[2]
    
    # Satellite velocity relative to Earth
    vRel = X[3:6] - np.cross(omega, X[0:3])
    v = np.linalg.norm(vRel)
    
    # Atmospheric density
    h = np.linalg.norm(X[0:3]) - Re
    rho = rho0*np.exp(-(h-h0)/H)

    B = Cd*A/m
    
    r = float(np.linalg.norm(X[0:3]))
    ri,rj,rk,vi,vj,vk = X
    
    J = j2Jacobian(X)
    J[3,0] += -1/2*B*rho/H * -ri*(vi + om*rj)*v/r
    J[3,1] += -1/2*B*rho/H * (om*np.linalg.norm(vRel) - rj*(vi + om*rj)*v/r)
    J[3,2] += -1/2*B*rho/H * -rk*(vi + om*rj)*v/r
    
    J[4,0] += -1/2*B*rho/H * (om*v- ri*(vj + om*ri)*v/r)
    J[4,1] += -1/2*B*rho/H * -rj*(vj + om*ri)*v/r
    J[4,2] += -1/2*B*rho/H * -rk*(vj + om*ri)*v/r
    
    J[5,0] += -1/2*B*rho/H * -(vk*ri)*v/r
    J[5,1] += -1/2*B*rho/H * -(vk*rj)*v/r
    J[5,2] += -1/2*B*rho/H * -(vk*rk)*v/r
    
    J[3,3] += -1/2*B*rho *(v + vi*(vi + om*rj)/v)
    J[3,4] += -1/2*B*rho *(vj*(vi + om*rj)/v)
    J[3,5] += -1/2*B*rho *(vk*(vi + om*rj)/v)
    
    J[4,3] += -1/2*B*rho *(vi*(vj + om*ri)/v)
    J[4,4] += -1/2*B*rho *(v + vj*(vj + om*ri)/v)
    J[4,5] += -1/2*B*rho *(vk*(vj + om*ri)/v)
    
    J[5,3] += -1/2*B*rho *vi*vk/v
    J[5,4] += -1/2*B*rho *vj*vk/v
    J[5,3] += -1/2*B*rho *(v + vk**2/v)
    
    return J


# def moonDynamics(X, X_moon=None):
#     """
#     Computes the acceleration due to the Earth gravitational force,
#     Earth oblateness (J2), atmospheric drag and Moon gravitational force.
#     @param: X      The spacecraft state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
#     @param: X_moon The moon state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
#     """
#     moon = Environment.default().access_celestial_object_with_name("Moon")
#     mu = moon.get_gravitational_parameter()
#     mu = float(mu.in_unit(mu.get_unit()))
    
#     if not X_moon:
#         # by default, assume some fixed position and velocity of the Moon for the time of simulation
#         X_moon = np.array([3.901856345891698e+08, -7.652260694831179e+08, -7.072466144052049e+08,
#                           0.248727753078019e+03, 0.872460710727378e+03, 0.340065120865137e+03])
#     d = X_moon[0:3] - X[0:3]
#     rm = float(np.linalg.norm(X_moon[0:3]))
    
#     fMoon = mu*(d/np.linalg.norm(d)**3 - X_moon[0:3]/rm**3)
#     return dragDynamics(X) + fMoon
   
            
# def moonJacobian(X, X_moon=None):
#     """
#     Computes the acceleration due to the Earth gravitational force,
#     Earth oblateness (J2), atmospheric drag and Moon gravitational force.
#     @param: X      The spacecraft state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
#     @param: X_moon The moon state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
#     """
#     moon = Environment.default().access_celestial_object_with_name("Moon")
#     mu = moon.get_gravitational_parameter()
#     mu = float(mu.in_unit(mu.get_unit()))
    
#     if not X_moon:
#         # by default, assume some fixed position and velocity of the Moon for the time of simulation
#         X_moon = np.array([3.901856345891698e+08, -7.652260694831179e+08, -7.072466144052049e+08,
#                           0.248727753078019e+03, 0.872460710727378e+03, 0.340065120865137e+03])
#     d = X_moon[0:3] - X[0:3]  
#     r = float(np.linalg.norm(X[0:3]))
#     ri,rj,rk,vi,vj,vk = X
#     rm = float(np.linalg.norm(X_moon[0:3]))
#     rim,rjm,rkm = X_moon
            
#     J = dragJacobian(X)
#     J[3,0] += 3*(rim-ri)**2/np.linalg.norm(d)**5 - 1/np.linalg.norm(d)**2
#     J[3,1] += 3*(rim-ri)*(rjm-rj)/np.linalg.norm(d)**5
#     J[3,2] += 3*(rim-ri)*(rkm-rk)/np.linalg.norm(d)**5
                
#     J[4,0] += 3*(rim-ri)*(rjm-rj)/np.linalg.norm(d)**5
#     J[4,1] += 3*(rjm-rj)**2/np.linalg.norm(d)**5 - 1/np.linalg.norm(d)**2
#     J[4,2] += 3*(rjm-rj)*(rkm-rk)/np.linalg.norm(d)**5
                
#     J[5,0] += 3*(rim-ri)*(rkm-rk)/np.linalg.norm(d)**5
#     J[5,1] += 3*(rjm-rj)*(rkm-rk)/np.linalg.norm(d)**5
#     J[5,2] += 3*(rkm-rk)**2/np.linalg.norm(d)**5 - 1/np.linalg.norm(d)**2
            
#     return J
            
            
def srpDynamics(X, X_sun=None, Csr=0.7, psr=4.57e-6, A=20, m=1500):
    """
    Computes the acceleration due to the Earth gravitational force, Earth oblateness (J2), 
    atmospheric drag, Moon gravitational force and solar radiation pressure.
    @param: X      The spacecraft state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
    @param: X_sun  The sun state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
    @param: Csr    The drag coefficient (default 0.7).
    @param: psr    The drag coefficient (default 4.57e-6).
    @param: A      The satellite area exposed to Sun [m^2] (default 20).
    @param: m      The satellite mass [kg] (default 1500).
    """
    if not X_sun:
        # by default, assume some fixed position and velocity of the Sun for the time of simulation
        X_sun = np.array([2.488497222696136e+10, -1.330174881730062e+11, -5.766341113228035e+10,
                          29.848920446520168e+03, 4.736679395943737e+03, 2.052798795667299e+03])
    d = X_sun[0:3] - X[0:3]
                
    rs = float(np.linalg.norm(X_sun[0:3]))
    
    fSun = -psr*Csr*A/m/np.linalg.norm(d) * d
    return dragDynamics(X) + fSun
                

def srpJacobian(X, X_sun=None, Csr=0.7, psr=4.57e-6, A=20, m=1500):
    """
    Computes the acceleration due to the Earth gravitational force, Earth oblateness (J2), 
    atmospheric drag, Moon gravitational force and solar radiation pressure.
    @param: X      The spacecraft state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
    @param: X_sun  The sun state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
    @param: Csr    The drag coefficient (default 0.7).
    @param: psr    The drag coefficient (default 4.57e-6).
    @param: A      The satellite area exposed to Sun [m^2] (default 20).
    @param: m      The satellite mass [kg] (default 1500).
    """
    if not X_sun:
        # by default, assume some fixed position and velocity of the Sun for the time of simulation
        X_sun = np.array([2.488497222696136e+10, -1.330174881730062e+11, -5.766341113228035e+10,
                          29.848920446520168e+03, 4.736679395943737e+03, 2.052798795667299e+03])
    d = X_sun[0:3] - X[0:3]
    r = float(np.linalg.norm(X[0:3]))
    ri,rj,rk,vi,vj,vk = X
    rs = float(np.linalg.norm(X_sun[0:3]))
    ris,rjs,rks = X_sun[0:3]
                
    J = dragJacobian(X)
    a = -psr*Csr*A/m
    J[3,0] += a*((ris-ri)**2/np.linalg.norm(d)**3 - 1/np.linalg.norm(d))
    J[3,1] += a*((ris-ri)*(rjs-rj)/np.linalg.norm(d)**3)
    J[3,2] += a*((ris-ri)*(rks-rk)/np.linalg.norm(d)**3)
                
    J[4,0] += a*((ris-ri)*(rjs-rj)/np.linalg.norm(d)**3)
    J[4,1] += a*((rjs-rj)**2/np.linalg.norm(d)**3 - 1/np.linalg.norm(d))
    J[4,2] += a*((rks-rk)*(rjs-rj)/np.linalg.norm(d)**3)
                
    J[5,0] += a*((ris-ri)*(rks-rk)/np.linalg.norm(d)**3)
    J[5,1] += a*((rks-rk)*(rjs-rj)/np.linalg.norm(d)**3)
    J[5,2] += a*((rks-rk)**2/np.linalg.norm(d)**3 - 1/np.linalg.norm(d))
            
    return J
       