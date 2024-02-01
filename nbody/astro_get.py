#import sys
#import requests
#import os
import numpy as np
from astroquery.jplhorizons import Horizons

id_list = [10, 199, 299, 399, 499, 599, 699, 799, 899]
id_list += [503, 606, 504, 501, 301,
            502, 801, 703,704, 605, 608, 901,
            702, 701, 604, 603,602, 705,808,601,
            609, 607, 807, 806, 505, 506, 610, 805, 611, 514,
            804, 616, 617, 803, 903, 902, 516, 904, 401]

positions = []
velocities = []
for id in id_list:
    print('processing ', id)
    #barycenter
    #body = Horizons(id=id, location='@ssb', epochs = {'start':'1980-01-1', 'stop':'2015-06-21','step':'360m'})
    #sun center
    body = Horizons(id=id, location='@10', epochs = {'start':'1980-01-1', 'stop':'2015-06-21','step':'720m'})
    body = body.vectors()
    x = body['x'].data.data
    y = body['y'].data.data
    z = body['z'].data.data

    vx = body['vx'].data.data
    vy = body['vy'].data.data
    vz = body['vz'].data.data

    X = np.stack([x,y,z],axis=1)
    V = np.stack([vx,vy,vz],axis=1)

    positions.append(X)
    velocities.append(V)
    print(X.shape)

positions = np.stack(positions, axis=0)
velocities = np.stack(velocities, axis=0)

print('positions ', positions.shape)

np.save('astro_pos_720_sun.npy', positions)
np.save('astro_vel_720_sun.npy', velocities)
