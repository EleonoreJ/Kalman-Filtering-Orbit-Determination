import numpy as np
from ostk.physics.environment.objects.celestial_bodies import Earth


def keplerianDynamics(X):
    """
    Computes the acceleration due to the Earth gravitational force (Keplerian dynamics).
    @param: X The spacecraft state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
    """
    mu = Earth.gravitational_parameter
    mu = float(mu.in_unit(mu.get_unit()))
    
    r = float(np.linalg.norm(X[0:3]))
    
    return - np.array(X[0:3])*mu/r**3


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
    
    f = keplerianDynamics(X)
    f += (-1/2)*mu*J2*Re**2*((6*rk/r**5)*K + (3/r**4 - 15*(rk**2/r**6))*np.array(X[0:3])/r)
    
    return f


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
   
    A = -1/2*mu*J2*Re**2
    J[3,0] += A*((3*r**-5 - 15*rk**2*r**-7) + (-15*r**-7 + 105*rk**2*r**-9)*ri**2)
    J[3,1] += A*(-15*r**-7 + 105*rk**2*r**-9)*ri*rj
    J[3,2] += A*((-30*rk*ri*r**-7) + (-15*r**-7 + 105*rk**2*r**-9)*ri*rk)
    
    J[4,0] += A*(-15*r**-7 + 105*rk**2*r**-9)*ri*rj
    J[4,1] += A*((3*r**-5 - 15*rk**2*r**-7) + (-15*r**-7 + 105*rk**2*r**-9)*rj**2)
    J[4,2] += A*((-30*rk*rj*r**-7) + (-15*r**-7 + 105*rk**2*r**-9)*rj*rk)
    
    J[5,0] += A*(-45*rk*r**-7 + 105*rk**3*r**-9)*ri
    J[5,1] += A*(-45*rk*r**-7 + 105*rk**3*r**-9)*rj
    J[5,2] += A*((-45*rk*r**-7 + 105*rk**3*r**-9)*rk + (9*r**-5 - 45*rk**2*r**-7))
	
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
    
    f = j2Dynamics(X) + fDrag
    return f



def dragJacobian(X, Cd=2.3, A=20, m=1500):
    """
    Computes the Jacobian for Keplerian + J2 + atmospheric drag dynamics.
    @param: X The spacecraft state [ri rj rk vi vj vk] expressed in ECI frame ([m], [m/s]).
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
    
    # Acceleration vector due to drag
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

