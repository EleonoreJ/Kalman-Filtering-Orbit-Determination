import numpy as np

from ostk.physics.time import Scale
from ostk.physics.coordinate.spherical import LLA
from ostk.physics.coordinate import Frame
from ostk.physics.environment.objects.celestial_bodies import Earth


def propagator(X, dt, dynamics):
    f = dynamics(X)
    Xp = np.zeros((6,))
    Xp[3:6] = X[3:6] + dt*f
    Xp[0:3] = X[0:3] + (1/2)*dt*(X[3:6] + Xp[3:6])
    return Xp

def convertState(state):
    return [
                *state.get_position().get_coordinates().transpose()[0],
                *state.get_velocity().get_coordinates().transpose()[0]
            ]

def convertStateLLA(instant, state):
    lla = LLA.cartesian(state.get_position().in_frame(Frame.ITRF(), state.get_instant()).get_coordinates(), Earth.equatorial_radius, Earth.flattening)
    
    return [
                repr(instant),
                float(instant.get_modified_julian_date(Scale.UTC)),
                *state.get_position().get_coordinates().transpose()[0].tolist(),
                *state.get_velocity().get_coordinates().transpose()[0].tolist(),
                float(lla.get_latitude().in_degrees()),
                float(lla.get_longitude().in_degrees()),
                float(lla.get_altitude().in_meters())
            ]
