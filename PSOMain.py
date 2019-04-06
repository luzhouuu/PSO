import numpy as np
import multiprocessing as mp
from itertools import repeat
import time
import pandas as pd


# Custom functions from other files we wrote
import PSOTestFuncs as tf
from PSOInit import pso_init
from PSOInit import qpso_init
from PSOUpdate import veloc_update
from PSOUpdate import point_update
from PSOUpdate import qpoint_update










############################ Contents ############################


# Defines the 4 algorithms that will be used:
#    - PSO - pso_algo()
#    - QPSO - qpso_algo()
#    - Parallelized PSO - pso_algo_par()
#    - Parallelized QPSO - qpso_algo_par()

# Runs 50 simulations of each algorithm on each test function within PSOTestFuncs.py
# Saves the output to a CSV

















############################ Primary Variable Descriptions ############################

# D is the number of dimensions (int)
# N is the number of particles (int)
# bounds of the search area - of the form [[x1min, x1max], ... , [xDmin, xDmax]]
# f is the function to be optimized
# params are the necessary parameters (omega, c1, c2) for PSO
    # - omega is the inertia weight which balances global and local exploitation
    # - c1 is the cognitive parameter - how much weight is given to the particle's best (pbest)
    # - c2 is the social parameter - how much weight is given to the global best (gbest)
# t is the current iteration of the algorithm
# sims is number of simulations to run on each function

# maxrounds is the maximum number of iterations allowed
# tol is the amount of change in fgbest to be considered an improvement
# nochange is the number or iterations without a sufficient improvement in fgbest before stopping
# fgbest_compare is the old fgbest value to  compare
# samebest is a counter for how many rounds in a row with improvement in fgbest of less than tol

# pcurr is the current position of each particles
# vcurr is the current velocity of each particles (PSO and PSO_par only)
# pbest is the best position of each particle
# fbest is the minimum value found for each particle
# fgbest is the overall minimum value found of all particles
# pgbest is the overall best position of each particle
# mbest is mean of each particle's best position by dimension (QPSO and QPSO_par only)
# x is the proposed movement, which is made if it is an improvement from the particles last location 
    # (QPSO and QPSO_par only)


# r1, r2 randomly variables in algorithm ~U(0,1) (PSO and PSO_par only)
# u, phi randomly variables in algorithm ~U(0,1) (QPSO and QPSO_par only) 
# beta is the contraction-expansion coefficient
# coinToss - 50% chance of being True / False, used in QPSO to decide to add or subtract changeParam



# funcnamelist is a list with strings of all the functions
# functionlist is a list of the callable names of each coded function
# pminlist is a list of the location of the true minimum point for each function, each entry is [x1,...,xn]
# boundlist is a list of the bounds of each function, each entry is the form [[x1min, x1max], ... , [xDmin, xDmax]]











############################ Algorithm Functions ############################



# Takes in f, N, bounds, params, maxrounds, tol, nochange
# Runs PSO on f over the search area bounds using N particles and parameters params,
#    and stopping criteria specified by maxrounds, tol, nochange
# Returns pgbest, fgbest, and t

def pso_algo(f, N, bounds, params, maxrounds, tol, nochange):

    # Initialize all necessary variables
    D = len(bounds)
    pcurr, vcurr, pbest, fbest, pgbest, fgbest = pso_init(f, N, bounds)
    t = 0
    samebest = 0

    # Start algorithm
    while t < maxrounds:
        fgbest_compare = fgbest
        for i in range(N):

            # Updates velocity of each dimension of the particle 
            for d in range(D):
                vcurr[i][d] = veloc_update(pcurr[i][d], vcurr[i][d], pbest[i][d], pgbest[d], params)

            # Only updates the dimension of pcurr if it would be within the bounds
            newp = pcurr[i] + vcurr[i]

            # Only updates the dimension of pcurr if it would be within the bounds
            for d in range(D):
                if newp[d] > bounds[d][0] and newp[d] < bounds[d][1]:
                    #Adding 0 creates a new object in memory instead of variable that references same object
                    pcurr[i][d] = newp[d] + 0 

            # Updates the particle's best and global best
            fcurr = f(pcurr[i])
            if fcurr < fbest[i]:
                fbest[i] = fcurr + 0
                pbest[i] = pcurr[i] + 0
                if fcurr < fgbest:
                    fgbest = fcurr + 0
                    pgbest = pcurr[i] + 0
        t += 1


        # Stopping criteria
        if abs(fgbest_compare - fgbest) > tol :
            samebest = 0
        else :
            samebest += 1
        if samebest >= nochange :
            break

    return pgbest, fgbest, t

















# Takes in f, N, bounds, maxrounds, tol, nochange
# Runs QPSO on f over the search area bounds using N particles,
#     and stopping criteria specified by maxrounds, tol, nochange
# Returns pgbest, fgbest, and t

def qpso_algo(f, N, bounds, maxrounds, tol, nochange):

    # Initialize all necessary variables
    D = len(bounds)
    pcurr, pbest, fbest, pgbest, fgbest = qpso_init(f, N, bounds)
    x = np.copy(pcurr, order="k")
    t = 0
    samebest = 0

    # Start algorithm
    while t < maxrounds:
        fgbest_compare = fgbest

        # Calculates mbest and beta
        mbest = np.mean(pbest, axis=0)
        beta = 0.5*(maxrounds-t)/maxrounds + 0.5

        # Updates each particle by dimension according to QPSO alogirthm
        for i in range(N):
            for d in range(D):
                phi = np.random.uniform()
                u = np. random.uniform()
                coinToss = np.random.uniform() < 0.5
                pcurr[i,d] = phi*pbest[i,d] + (1- phi)*pgbest[d]
                changeParam = beta * abs(mbest[d] - x[i, d]) * (-1) * np.log(u)
                newx_id = pcurr[i, d] + changeParam if coinToss else pcurr[i, d] - changeParam

                # Only updates the dimension of x if it would be within the bounds
                if newx_id > bounds[d][0] and newx_id < bounds[d][1]:
                    #Adding 0 creates a new object in memory instead of variable that references same object
                    x[i,d] = newx_id + 0
            fcurr = f(x[i])

            # Updates the particle's best and global best
            if fcurr < fbest[i]:
                fbest[i] = fcurr + 0
                pbest[i] = x[i] + 0
                if fcurr < fgbest:
                    fgbest = fcurr + 0
                    pgbest = x[i] + 0
        t += 1


        # Stopping criteria
        if abs(fgbest_compare - fgbest) > tol:
            samebest = 0
        else:
            samebest += 1
        if samebest >= nochange:
            break

    return pgbest, fgbest, t














# Takes in f, N, bounds, params, maxrounds, tol, nochange
# Runs parallelized PSO on f over the search area bounds using N particles and parameters params,
#	and stopping criteria specified by maxrounds, tol, nochange
# We update all the points in an iteration at once, so no communication within an iteration
# Returns pgbest, fgbest, and t

def pso_algo_par(f, N, bounds, params, maxrounds, tol, nochange):

    # Initialize all necessary variables
    pcurr, vcurr, pbest, fbest, pgbest, fgbest = pso_init(f, N, bounds)
    t = 0
    samebest = 0

    # Start algorithm
    while t < maxrounds:
        fgbest_compare = fgbest

        # Puts inputs in format so each list is the info for one particle, where sublist is [pcurr_i, vcurr_i, etc]
        inputs = zip(pcurr, vcurr, pbest, fbest, repeat(pgbest), repeat(params), repeat(bounds), repeat(f))

    # Calls starmap using the pool object to run point_update on each particle using inputs as formatted above
        # results_0 is a list where each sublist is the output for each particle
        results_0 = pool.starmap(point_update, inputs)

        # Reformat results so each sublist corresponds to variable rather than particle
        results = list(map(list, zip(*results_0)))

        # Assign the sublists of results to the variables themselves
        pcurr = np.array(list(results)[0])
        vcurr = np.array(list(results)[1])
        pbest = np.array(list(results)[2])
        fbest = np.array(list(results)[3])


        # Updates global best
        if min(fbest) < fgbest:
            #Adding 0 creates a new object in memory instead of variable that references same object
            fgbest = min(fbest) + 0
            pgbest = np.copy(pbest[fbest == fgbest], order="k")[0]

        t += 1


        # Stopping criteria
        if abs(fgbest_compare - fgbest) > tol:
            samebest = 0
        else:
            samebest += 1
        if samebest >= nochange:
            break

    return pgbest, fgbest, t




















# Takes in f, N, bounds, maxrounds, tol, nochange
# Runs parallelized QPSO on f over the search area bounds using N particles and parameters params,
#	and stopping criteria specified by maxrounds, tol, nochange
# We update all the points in an iteration at once, so no communication within an iteration
# Returns pgbest, fgbest, and t

def qpso_algo_par(f, N, bounds, maxrounds, tol, nochange):

    # Initialize all necessary variables
    pcurr, pbest, fbest, pgbest, fgbest = qpso_init(f, N, bounds)
    x = np.copy(pcurr, order="k")
    t = 0
    samebest = 0

    # Start algorithm
    while t < maxrounds:
        fgbest_compare = fgbest

        # Calculates mbest and beta
        mbest = np.mean(pbest, axis=0)
        beta = 0.5*(maxrounds-t)/maxrounds + 0.5

        # Puts inputs in format so each list is the info for one particle, where sublist is [x_ i, pcurr_i, etc]
        inputs = zip(x, pcurr, pbest, fbest, repeat(mbest), repeat(pgbest), repeat(beta), repeat(bounds), repeat(f))

        # Calls starmap using the pool object to run point_update on each particle using inputs as formatted above
        # results_0 is a list where each sublist is the output for each particle
        results_0 = pool.starmap(qpoint_update, inputs)

        # Reformat results so each sublist corresponds to variable rather than particle
        results = list(map(list, zip(*results_0)))

        # Assign the sublists of results to the variables themselves
        x = np.array(list(results)[0])
        pcurr = np.array(list(results)[1])
        pbest = np.array(list(results)[2])
        fbest = np.array(list(results)[3])


        # Updates global best
        if min(fbest) < fgbest:
            #Adding 0 creates a new object in memory instead of variable that references same object
            fgbest = min(fbest) + 0
            pgbest = np.copy(pbest[fbest == fgbest], order="k")[0]

        t += 1


        # Stopping criteria
        if abs(fgbest_compare - fgbest) > tol:
            samebest = 0
        else:
            samebest += 1
        if samebest >= nochange:
            break


    return pgbest, fgbest, t




























############################ Simulations and Testing ############################



if __name__ == '__main__':


    # Specifies the necessary parameters to be used by the algorithms, and # of simulations

    N = 50
    params = [0.715, 1.7, 1.7]
    maxrounds = 1000
    tol = 10**(-9)
    nochange = 20
    sims = 50




    # Stores the information for each function including names of function as a string, 
    # how to call it, where the true minimum occurs, and what the bounds are

    funcnamelist = ["X-Squared", "Booth", "Beale", "ThreeHumpCamel", "GoldsteinPrice", "Levi_n13", "Sphere", "Rosebrock", "StyblinskiTang", "Ackley", "Schaffer_n2", "Eggholder", "McCormick", "Rastrigin", "Schaffer_n4", "Easom", "Bukin_n6", "Matyas"]
    functionlist = [tf.xsq, tf.booth, tf.beale, tf.threehumpcamel, tf.goldsteinprice, tf.levi_n13, tf.sphere, tf.rosenbrock, tf.Styblinski_Tang, tf.ackley, tf.schaffer_n2, tf.eggholder, tf.mccormick, tf.rastrigin, tf.schaffer_n4, tf.easom, tf.bukin_n6, tf.matyas]
    pminlist = [[0], [1,3], [3,0.5], [0,0], [0, -1],[1,1], [0,0,0,0], [1,1,1,1], [-2.903534,-2.903534,-2.903534,-2.903534,-2.903534,-2.903534], [0,0], [0,0], [512, 404.2319], [-0.54719, -1.54719], [0,0,0,0,0,0,0,0], [0,1.25313], [np.pi, np.pi], [-10,1], [0,0]]
    boundlist = [[[-200, 200]], [[-10, 10], [-10, 10]], [[-4.5, 4.5], [-4.5, 4.5]], [[-5, 5], [-5, 5]], [[-2, 2], [-2, 2]], [[-10, 10], [-10, 10]], [[-100, 100], [-100, 100], [-100, 100], [-100, 100]], [[-100, 100], [-100, 100], [-100, 100], [-100, 100]], [[-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5], [-5, 5]], [[-5, 5], [-5, 5]], [[-100, 100], [-100, 100]], [[-512, 512], [-512, 512]], [[-1.5, 4], [-3, 4]], [[-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12]], [[-100, 100], [-100, 100]], [[-100, 100], [-100, 100]], [[-15, -5], [-3, 3]], [[-10.00, 10.00], [-10.00, 10.00]]]




    # Sets up a dataframe to store the data 

    outdata = pd.DataFrame()




    # Sets up for parallel computing

    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)









    # Forloop for each function containing for-loop for each simulation which runs all 4 algorithms and times each
    # Stores the results of each simulation and true function values in outdata, and saves this as a CSV

    for j in range(len(functionlist)):
        for k in range(sims):
            fname = funcnamelist[j]
            f = functionlist[j]
            bounds = boundlist[j]
            truemin = pminlist[j]
            trueval = f(truemin)

            start = time.time()
            pmin, fmin, nrounds = pso_algo(f, N, bounds, params, maxrounds, tol, nochange)
            end = time.time()
            outdata = outdata.append([[k, fname, "PSO", end-start, nrounds, pmin, truemin, fmin, trueval]])

            start = time.time()
            pmin, fmin, nrounds = pso_algo_par(f, N, bounds, params, maxrounds, tol, nochange)
            end = time.time()
            outdata = outdata.append([[k, fname, "PSO_Par", end-start, nrounds, pmin, truemin, fmin, trueval]])

            start = time.time()
            pmin, fmin, nrounds = qpso_algo(f, N, bounds, maxrounds, tol, nochange)
            end = time.time()
            outdata = outdata.append([[k, fname, "QPSO", end-start, nrounds, pmin, truemin, fmin, trueval]])

            start = time.time()
            pmin, fmin, nrounds = qpso_algo_par(f, N, bounds, maxrounds, tol, nochange)
            end = time.time()
            outdata = outdata.append([[k, fname, "QPSO_Par", end-start, nrounds, pmin, truemin, fmin, trueval]])

    pool.close()




    #Cleans up dataframe and saves it to a CSV file

    outdata.columns = ["Simulation#", "Function", "Method", "time", "rounds", "FoundMinLoc", "TrueMinLoc", "FoundMinVal", "TrueMinVal"]
    outdata.sort_values(["Function", "Method"], inplace = True)
    outdata = outdata.reset_index(drop = True)
    outdata.to_csv("OutputData.csv")
