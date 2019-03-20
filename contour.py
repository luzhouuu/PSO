import numpy as np
import multiprocessing as mp
from itertools import repeat
import heapq
import time
import pandas as pd

import PSOTestFuncs as tf
from PSOInit import pso_init
from PSOInit import qpso_init
from PSOUpdate import veloc_update
from PSOUpdate import point_update
from PSOUpdate import qpoint_update
from PSOUpdate import dist



import matplotlib.pyplot as plt
plt.style.use('ggplot')

#import colour map
import brewer2mpl
bmap = brewer2mpl.get_map('Set2', 'qualitative', 8)
colors = bmap.mpl_colors




def pso_algo(f, s, bounds, params, maxrounds, p=0.9,optimum_dis=0.01):
    n = len(bounds)
    pcurr, vcurr, pbest, fbest, pgbest, fgbest = pso_init(f, s, bounds)
    t = 0
    while t < maxrounds:
        for i in range(s):
            for d in range(n):
                vcurr[i][d] = veloc_update(pcurr[i][d], vcurr[i][d], pbest[i][d], pgbest[d], params)
            newp = pcurr[i] + vcurr[i]
            for d in range(n):
                if newp[d] > bounds[d][0] and newp[d] < bounds[d][1]:
                    pcurr[i][d] = newp[d] + 0
            fcurr = f(pcurr[i])
            if fcurr < fbest[i]:
                fbest[i] = fcurr + 0
                pbest[i] = pcurr[i] + 0
                if fcurr < fgbest:
                    fgbest = fcurr + 0
                    pgbest = pcurr[i] + 0
        t += 1
        # pcurr_best = pcurr[fbest > min(heapq.nlargest(int(fbest.size*p), fbest))]
        # if dist(pcurr_best, pgbest) <= optimum_dis:
        #     break
    return [pcurr, pgbest]
s = 500
params = [0.715, 1.7, 1.7]
maxrounds = 5000
sims = 50

funcnamelist = ["X-Squared", "Booth", "Beale", "ThreeHumpCamel", "GoldsteinPrice", "Levi_n13", "Sphere", "Rosebrock", "StyblinskiTang", "Ackley", "Schaffer_n2", "Eggholder", "McCormick", "Rastrigin", "Schaffer_n4", "Easom", "Bukin_n6", "Matyas"]
functionlist = [tf.xsq, tf.booth, tf.beale, tf.threehumpcamel, tf.goldsteinprice, tf.levi_n13, tf.sphere, tf.rosenbrock, tf.Styblinski_Tang, tf.ackley, tf.schaffer_n2, tf.eggholder, tf.mccormick, tf.rastrigin, tf.schaffer_n4, tf.easom, tf.bukin_n6, tf.matyas]
pminlist = [[0], [1,3], [3,0.5], [0,0], [0, -1],[1,1], [0,0,0,0], [1,1,1,1], [-2.903534,-2.903534,-2.903534,-2.903534,-2.903534,-2.903534], [0,0], [0,0], [512, 404.2319], [-0.54719, -1.54719], [0,0,0,0,0,0,0,0], [0,1.25313], [np.pi, np.pi], [-10,1], [0,0]]
boundlist = [[[-200, 200]], [[-10, 10], [-10, 10]], [[-4.5, 4.5], [-4.5, 4.5]], [[-5, 5], [-5, 5]], [[-2, 2], [-2, 2]], [[-10, 10], [-10, 10]], [[-100, 100], [-100, 100], [-100, 100], [-100, 100]], [[-100, 100], [-100, 100], [-100, 100], [-100, 100]], [[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5]], [[-5, 5], [-5, 5]], [[-100, 100], [-100, 100]], [[-512, 512], [-512, 512]], [[-1.5, 4], [-3, 4]], [[-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12]], [[-100, 100], [-100, 100]], [[-100, 100], [-100, 100]], [[-15, -5], [-3, 3]], [[-10.00, 10.00], [-10.00, 9.00]]]

i = 10
f = functionlist[i]
bounds = boundlist[i]
#pcurr, vcurr, pbest, fbest, pgbest, fgbest = pso_init(f, s, bounds)
#position_0 = pcurr
#plt.figure(figsize=(15,6))
#plt.scatter(position_0[:,0], position_0[:,1], c= colors[1])
#plt.ylabel("Y")
#plt.xlabel("X")
#plt.title("Countour Plot for" + " " + funcnamelist[i] + " " + "when iteration = 0")

for maxrounds in [0,10,20,100]:
    res = pso_algo(f, s, bounds, params, maxrounds)
    position = res[0]
    pgbest = res[1]     
    plt.figure(figsize=(15,6))
    plt.subplot(121)
    plt.scatter(position[:,0],position[:,1], c = colors[1])
    plt.scatter(pgbest[0], pgbest[1], c = "k")
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.title("Countour Plot for" + " " + funcnamelist[i]+ " " + "when iteration =" +" " +str(maxrounds))


