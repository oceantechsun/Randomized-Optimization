import sklearn as sk
import mlrose as ml
import numpy as np
import matplotlib.pyplot as plt
import random
import time


#fitness = ml.FourPeaks(t_pct=0.15)
#state = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
#fitness.evaluate(state)

# Create list of city coordinates

"""
coords_list = [(random.random()*100, random.random()*100) for _ in range(50)]

print(coords_list)

coords_list = [(77.40596349267919, 52.936386234356135), (59.58802687721355, 7.744677328731598), (57.57305494618471, 80.91313755494713), (3.018737419255779, 56.5647841653613), (99.98568848156795, 96.66242320898988), (64.33569306753807, 63.9299767842776), (17.625567994180415, 49.55244325774327), (36.81056314605032, 37.95069104471168), (55.29981120908227, 59.529227438799495), (7.7733781972427085, 43.70279470683194), (1.1659180753084253, 35.49723247546971), (75.66206087786465, 28.379494189932263), (84.63423385227622, 25.820787053716245), (41.6278112522615, 79.68001731287666), (47.90854119610534, 40.958596372026804), (31.200220340943407, 77.94617352573259), (37.88865048246252, 7.256346323403506), (52.67829630680916, 24.16944673883591), (18.16576952728094, 89.34225014743761), (9.007355266603167, 96.51934665936507), (5.946017451757857, 27.22667560678298), (0.570000867272924, 48.95217530994878), (6.188575054526046, 36.692965850038426), (76.13042790908646, 58.992979940399145), (32.34225729597063, 3.0106172897140127), (68.98898698869684, 81.5865331997617), (99.42559141685311, 5.3174517877926375), (79.0252132274689, 35.01762936929046), (15.167301211710749, 18.54809221893764), (10.704755201808602, 60.52470028775154), (21.447233723470006, 12.521383516454742), (84.43878564931485, 25.2946165154034), (49.398146555598544, 60.96584286629118), (90.67769392623599, 48.41972578974221), (69.3891436872516, 54.86300122260954), (6.648312754594354, 55.47062318072871), (10.897212854825778, 50.10161778349794), (55.53357445220867, 85.70138117480334), (32.2709233018365, 21.628885429308298), (28.338677386649124, 62.03359714749619), (42.76518817585665, 5.66798998442144), (40.69238678877179, 89.99970119200661), (84.06872639938594, 14.418898363842436), (77.51237325158993, 77.24741229336085), (58.51696437571151, 57.853751726492995), (62.766776542010035, 50.08154510946372), (46.11102391501021, 51.25346534772087), (45.48357231565821, 81.66366884634029), (41.02651759157758, 25.601766687555518), (25.126365902513758, 64.13245711534248)]

xcoords = []
ycoords = []

for x in range(len(coords_list)):
    xcoords.append(coords_list[x][0])
    ycoords.append(coords_list[x][1])


plt.scatter(xcoords,ycoords)
plt.grid()
plt.savefig("coordinates.png")

# Initialize fitness function object using coords_list
fitness_coords = ml.TravellingSales(coords = coords_list)
# Define optimization problem object
problem_fit = ml.TSPOpt(length = 50, fitness_fn = fitness_coords, maximize=False)



# Solve problem using the genetic algorithm
#best_state, best_fitness = ml.genetic_alg(problem_fit, random_state = 2)

#best_state, best_fitness = ml.random_hill_climb(problem_fit, random_state = 2)

#best_state, best_fitness = ml.simulated_annealing(problem_fit, random_state = 2)

best_state, best_fitness = ml.mimic(problem_fit, random_state = 2)


print(best_state)


print(best_fitness)

"""
fitness = ml.FourPeaks(t_pct=0.15)
main_best_fit = []
main_execution_times = []
main_total_iterations = []
main_opt_iterations= []
sample_sizes = []

for y in range(10):
    best_fit = []
    execution_times = []
    total_iterations = []
    opt_iterations= []
    for x in range(5):
        start = time.time()
        opt = ml.DiscreteOpt((y*10)+10, fitness)
        #best_state, best_fitness, curve = ml.random_hill_climb(opt, max_iters=5, restarts = 0, curve=True, random_state = 2)
        best_state, best_fitness, curve = ml.genetic_alg(opt, pop_size=(y*10)+10, max_attempts=100, curve=True, random_state = x)
        
        end = time.time()
        best_fit.append(best_fitness)
        total_iterations.append(len(curve))
        opt_iterations.append(np.argmax(curve))
        
        if(len(sample_sizes) < 10 and x==0):
            sample_sizes.append((y*10)+10)
        execution_times.append(end-start)
    main_best_fit.append(np.mean(best_fit))
    main_execution_times.append(np.mean(execution_times))
    main_total_iterations.append(np.mean(total_iterations))
    main_opt_iterations.append(np.mean(opt_iterations))

plt.plot(sample_sizes, main_execution_times)
plt.savefig("times.png")
plt.clf()
plt.ylim([0,100])
plt.plot(sample_sizes, main_best_fit)
plt.savefig("accuracy.png")
plt.clf()
plt.plot(sample_sizes, main_opt_iterations)
plt.savefig("opt_iters.png")
plt.clf()
plt.plot(sample_sizes, main_total_iterations)
plt.savefig("total_iters.png")
plt.clf()
plt.plot(main_opt_iterations, main_execution_times)
plt.savefig("time_iters.png")
plt.clf()
print(main_best_fit)
print(main_execution_times)
print(main_opt_iterations)
print(main_total_iterations)
print(sample_sizes)
#state = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, \
# 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, \
# 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0])
#print(fitness.evaluate(state))

#best_state, best_fitness = ml.genetic_alg(opt, random_state = 2)

best_state, best_fitness, curve = ml.random_hill_climb(opt, max_iters=5, restarts = 0, curve=True, random_state = 2)

#best_state, best_fitness = ml.simulated_annealing(opt, max_iters=150, random_state = 2)

#best_state, best_fitness, curve = ml.mimic(opt, curve=True, random_state = 2)

print(curve)

plt.plot(curve)
plt.savefig("4pks.png")
print(best_state)


print(best_fitness)

weights = [10.0, 5.0, 2.0, 8.0, 15.0, 3.0, 11.0]
values = [1, 12, 3, 14, 5, 10, 12]
max_weight_pct = 0.35
fitness_two = ml.Knapsack(weights=weights, values=values, max_weight_pct=max_weight_pct)

opt_two = ml.DiscreteOpt(length=7, fitness_fn=fitness_two)
#state = np.array([0, 1, 1, 1, 0])
#sprint(fitness_two.evaluate(state))

best_state_two, best_fitness_two = ml.genetic_alg(opt_two, random_state = 2)

#best_state_two, best_fitness_two = ml.random_hill_climb(opt_two, random_state = 2)

best_state_two, best_fitness_two = ml.simulated_annealing(opt_two, max_iters=1, random_state = 2)

#best_state_two, best_fitness_two = ml.mimic(opt_two, max_iters=1, random_state = 2)

print(best_state_two)


print(best_fitness_two)

#plt.scatter([0,1,0,1,2,3],[1,2,2,3,3,4])
#plt.savefig("test.png")
edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]

fitness_three = ml.MaxKColor(edges)
opt_three = ml.DiscreteOpt(length=5, fitness_fn=fitness_three, maximize=False)
#best_state_three, best_fitness_three = ml.genetic_alg(opt_three, random_state = 2)
#best_state_three, best_fitness_three = ml.random_hill_climb(opt_three, random_state = 2)
best_state_three, best_fitness_three = ml.simulated_annealing(opt_three, random_state = 2)
best_state_three, best_fitness_three = ml.mimic(opt_three, max_iters=1, random_state = 2)

print(fitness_three.evaluate([1, 0, 0, 1, 0]))

print(best_state_three)

print(best_fitness_three)