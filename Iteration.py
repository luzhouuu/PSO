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
    return pgbest, fgbest, t

def qpso_algo(f, s, bounds, maxrounds, p=0.9,optimum_dis=0.01):
    n = len(bounds)
    pcurr, pbest, fbest, pgbest, fgbest = qpso_init(f, s, bounds)
    x = np.copy(pcurr, order="k")
    t = 0
    while t < maxrounds:
        mbest = np.mean(pbest, axis=0)
        beta = 0.5*(maxrounds-t)/maxrounds + 0.5

        for i in range(s):
            for d in range(n):
                phi = np.random.uniform()
                u = np. random.uniform()
                coinToss = np.random.uniform() < 0.5
                pcurr[i,d] = phi*pbest[i,d] + (1- phi)*pgbest[d]
                changeParam = beta * abs(mbest[d] - x[i, d]) * (-1) * np.log(u)
                newx_id = pcurr[i, d] + changeParam if coinToss else pcurr[i, d] - changeParam
                if newx_id > bounds[d][0] and newx_id < bounds[d][1]:
                    x[i,d] = newx_id + 0
            fcurr = f(x[i])
            if fcurr < fbest[i]:
                fbest[i] = fcurr + 0
                pbest[i] = x[i] + 0
                if fcurr < fgbest:
                    fgbest = fcurr + 0
                    pgbest = x[i] + 0
        t += 1
        # pcurr_best = pcurr[fbest > min(heapq.nlargest(int(fbest.size*p), fbest))]
        # if dist(pcurr_best, pgbest) <= optimum_dis:
        #     break
    return pgbest, fgbest, t








if __name__ == '__main__':

    s = 50
    params = [0.715, 1.7, 1.7]
    maxrounds = 1000
    sims = 50

    funcnamelist = ["X-Squared", "Booth", "Beale", "ThreeHumpCamel", "GoldsteinPrice", "Levi_n13", "Sphere", "Rosebrock", "StyblinskiTang", "Ackley", "Schaffer_n2", "Eggholder", "McCormick", "Rastrigin", "Schaffer_n4", "Easom", "Bukin_n6", "Matyas"]
    functionlist = [tf.xsq, tf.booth, tf.beale, tf.threehumpcamel, tf.goldsteinprice, tf.levi_n13, tf.sphere, tf.rosenbrock, tf.Styblinski_Tang, tf.ackley, tf.schaffer_n2, tf.eggholder, tf.mccormick, tf.rastrigin, tf.schaffer_n4, tf.easom, tf.bukin_n6, tf.matyas]
    pminlist = [[0], [1,3], [3,0.5], [0,0], [0, -1],[1,1], [0,0,0,0], [1,1,1,1], [-2.903534,-2.903534,-2.903534,-2.903534,-2.903534,-2.903534], [0,0], [0,0], [512, 404.2319], [-0.54719, -1.54719], [0,0,0,0,0,0,0,0], [0,1.25313], [np.pi, np.pi], [-10,1], [0,0]]
    boundlist = [[[-200, 200]], [[-10, 10], [-10, 10]], [[-4.5, 4.5], [-4.5, 4.5]], [[-5, 5], [-5, 5]], [[-2, 2], [-2, 2]], [[-10, 10], [-10, 10]], [[-100, 100], [-100, 100], [-100, 100], [-100, 100]], [[-100, 100], [-100, 100], [-100, 100], [-100, 100]], [[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5]], [[-5, 5], [-5, 5]], [[-100, 100], [-100, 100]], [[-512, 512], [-512, 512]], [[-1.5, 4], [-3, 4]], [[-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12]], [[-100, 100], [-100, 100]], [[-100, 100], [-100, 100]], [[-15, -5], [-3, 3]], [[-10.00, 10.00], [-10.00, 9.00]]]

    outdata = pd.DataFrame()

    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)

    for j in range(len(functionlist)):
        for maxrounds in range(200):

            f = functionlist[j]
            bounds = boundlist[j]
            trueval = f(pminlist[j])
    
            pmin, fmin, nrounds = pso_algo(f, s, bounds, params, maxrounds)
            outdata = outdata.append([[maxrounds, funcnamelist[j], "PSO", nrounds, pmin, pminlist[j], fmin, trueval]])
    
               # start = time.time()
               # pmin, fmin, nrounds = pso_algo_par(f, s, bounds, params, maxrounds)
               # end = time.time()
               # outdata = outdata.append([[k, funcnamelist[j], "PSO_Par", end-start, nrounds, pmin, pminlist[j], fmin, trueval]])
    
            pmin, fmin, nrounds = qpso_algo(f, s, bounds, maxrounds)
            outdata = outdata.append([[k, funcnamelist[j], "QPSO", nrounds, pmin, pminlist[j], fmin, trueval]])
    
            #start = time.time()
            #pmin, fmin, nrounds = qpso_algo_par(f, s, bounds, maxrounds)
            #end = time.time()
            #outdata = outdata.append([[k, funcnamelist[j], "QPSO_Par", end-start, nrounds, pmin, pminlist[j], fmin, trueval]])

    outdata.columns = ["times", "Function", "Method", "rounds", "FoundMinLoc", "TrueMinLoc", "FoundMinVal", "TrueMinVal"]
    outdata.sort_values(["Function", "Method"], inplace = True)
    outdata = outdata.reset_index(drop = True)
    outdata.to_csv("Output.csv")



