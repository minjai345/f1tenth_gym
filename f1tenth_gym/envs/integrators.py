from enum import IntEnum

class IntegratorType(IntEnum):
    EULER = 1
    RK4 = 2

def rk4_integration(f, x, u, dt, *args):
    k1 = f(x, u, *args)

    k2_state = x + dt * (k1 / 2)
    k2 = f(k2_state, u, *args)

    k3_state = x + dt * (k2 / 2)
    k3 = f(k3_state, u, *args)

    k4_state = x + dt * k3
    k4 = f(k4_state, u, *args)

    # dynamics integration
    x = x + dt * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x

def euler_integration(f, x, u, dt, *args):
    x = x + dt * f(x, u, *args)
    return x

def integrator_from_type(integrator_type: 'IntegratorType'):
    if integrator_type == IntegratorType.EULER:
        return euler_integration
    elif integrator_type == IntegratorType.RK4:
        return rk4_integration
    else:
        raise ValueError(f"Unknown integrator type: {integrator_type}")