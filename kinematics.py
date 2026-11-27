import numpy as np
import pandas as pd


def smooth(values, window=5):
    #moyenne
    return (pd.Series(values)
            .rolling(window, center=True, min_periods=1)
            .mean()
            .values)


def compute_kinematics(traj, px_per_m=None, window=7):
    
    #Renvoie {'speed', 'accel', 'direction'}
    traj = np.asarray(traj, dtype=float)
    t = traj[:, 0]
    x = smooth(traj[:, 1], window)
    y = smooth(traj[:, 2], window)

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    speed = np.hypot(vx, vy)
    if px_per_m:
        speed = speed / px_per_m

    accel = np.gradient(smooth(speed, window), t)
    direction = np.degrees(np.arctan2(vy, vx))
    return {"speed": speed, "accel": accel, "direction": direction}
