import sklearn as sk
import mlrose as ml
import numpy as np
import matplotlib.pyplot as plt
import random
import time

sample_sizes = []
fitness = ml.OneMax()


main_best_fit_gene = []
main_execution_times_gene = []
main_total_iterations_gene = []
main_opt_iterations_gene = []


for y in range(10):
    best_fit_gene = []
    execution_times_gene = []
    total_iterations_gene = []
    opt_iterations_gene = []
    for x in range(5):
        start = time.time()
        opt = ml.DiscreteOpt((y*6)+6, fitness)
        #best_state, best_fitness, curve = ml.random_hill_climb(opt, max_iters=5, restarts = 0, curve=True, random_state = 2)
        best_state_gene, best_fitness_gene, curve_gene = ml.genetic_alg(opt, pop_size=(y*6)+6, max_attempts=100, curve=True, random_state = x)
        
        end = time.time()
        best_fit_gene.append(best_fitness_gene)
        total_iterations_gene.append(len(curve_gene))
        opt_iterations_gene.append(np.argmax(curve_gene))
        
        if(len(sample_sizes) < 10 and x==0):
            sample_sizes.append((y*6)+6)
        execution_times_gene.append(end-start)
    main_best_fit_gene.append(np.mean(best_fit_gene))
    main_execution_times_gene.append(np.mean(execution_times_gene))
    main_total_iterations_gene.append(np.mean(total_iterations_gene))
    main_opt_iterations_gene.append(np.mean(opt_iterations_gene))

print("genetic alg data")
print(main_best_fit_gene)
print(main_execution_times_gene)
print(main_opt_iterations_gene)
print(main_total_iterations_gene)
print(sample_sizes)


main_best_fit_hill = []
main_execution_times_hill = []
main_total_iterations_hill = []
main_opt_iterations_hill = []


for y in range(10):
    best_fit_hill = []
    execution_times_hill = []
    total_iterations_hill = []
    opt_iterations_hill = []
    for x in range(5):
        start = time.time()
        opt = ml.DiscreteOpt((y*6)+6, fitness)
        #best_state, best_fitness, curve = ml.random_hill_climb(opt, pop_size=(y*10)+10, max_iters=5, restarts = 0, curve=True, random_state = 2)
        best_state_hill, best_fitness_hill, curve_hill = ml.random_hill_climb(opt, max_iters=5, restarts = 0, curve=True, random_state = 2)
        
        end = time.time()
        best_fit_hill.append(best_fitness_hill)
        total_iterations_hill.append(len(curve_hill))
        opt_iterations_hill.append(np.argmax(curve_hill))
        
        if(len(sample_sizes) < 10 and x==0):
            sample_sizes.append((y*6)+6)
        execution_times_hill.append(end-start)
    main_best_fit_hill.append(np.mean(best_fit_hill))
    main_execution_times_hill.append(np.mean(execution_times_hill))
    main_total_iterations_hill.append(np.mean(total_iterations_hill))
    main_opt_iterations_hill.append(np.mean(opt_iterations_hill))

print("hill data")
print(main_best_fit_hill)
print(main_execution_times_hill)
print(main_opt_iterations_hill)
print(main_total_iterations_hill)
print(sample_sizes)



main_best_fit_anneal = []
main_execution_times_anneal = []
main_total_iterations_anneal = []
main_opt_iterations_anneal = []
print("anneal fitness scores")
for y in range(10):
    #print("y = "+y)
    best_fit_anneal = []
    execution_times_anneal = []
    total_iterations_anneal = []
    opt_iterations_anneal = []
    for x in range(5):
        start = time.time()
        opt = ml.DiscreteOpt((y*6)+6, fitness)
        #best_state, best_fitness, curve = ml.random_anneal_climb(opt, pop_size=(y*10)+10, max_iters=5, restarts = 0, curve=True, random_state = 2)
        best_state_anneal, best_fitness_anneal, curve_anneal = ml.simulated_annealing(opt, max_attempts=100, curve=True, random_state = 2)
        print(best_fitness_anneal)
        end = time.time()
        best_fit_anneal.append(best_fitness_anneal)
        total_iterations_anneal.append(len(curve_anneal))
        opt_iterations_anneal.append(np.argmax(curve_anneal))
        
        if(len(sample_sizes) < 10 and x==0):
            sample_sizes.append((y*6)+6)
        execution_times_anneal.append(end-start)
    main_best_fit_anneal.append(np.mean(best_fit_anneal))
    main_execution_times_anneal.append(np.mean(execution_times_anneal))
    main_total_iterations_anneal.append(np.mean(total_iterations_anneal))
    main_opt_iterations_anneal.append(np.mean(opt_iterations_anneal))

print("anneal data")
print(main_best_fit_anneal)
print(main_execution_times_anneal)
print(main_opt_iterations_anneal)
print(main_total_iterations_anneal)
print(sample_sizes)




main_best_fit_mimic = []
main_execution_times_mimic = []
main_total_iterations_mimic = []
main_opt_iterations_mimic = []

for y in range(10):
    best_fit_mimic = []
    execution_times_mimic = []
    total_iterations_mimic = []
    opt_iterations_mimic = []
    for x in range(2):
        start = time.time()
        opt = ml.DiscreteOpt((y*6)+6, fitness)
        #best_state, best_fitness, curve = ml.random_mimic_climb(opt, pop_size=(y*10)+10, max_iters=5, restarts = 0, curve=True, random_state = 2)
        best_state_mimic, best_fitness_mimic, curve_mimic = ml.mimic(opt, pop_size=(y*6)+6, max_attempts=10,max_iters=900, curve=True, random_state = 2)
        
        end = time.time()
        best_fit_mimic.append(best_fitness_mimic)
        total_iterations_mimic.append(len(curve_mimic))
        opt_iterations_mimic.append(np.argmax(curve_mimic))
        
        if(len(sample_sizes) < 10 and x==0):
            sample_sizes.append((y*6)+6)
        execution_times_mimic.append(end-start)
        #print("x = " + x)
        #print("y = " + y)
    main_best_fit_mimic.append(np.mean(best_fit_mimic))
    main_execution_times_mimic.append(np.mean(execution_times_mimic))
    main_total_iterations_mimic.append(np.mean(total_iterations_mimic))
    main_opt_iterations_mimic.append(np.mean(opt_iterations_mimic))

print("mimic data")
print(main_best_fit_mimic)
print(main_execution_times_mimic)
print(main_opt_iterations_mimic)
print(main_total_iterations_mimic)
print(sample_sizes)





plt.title("Execution times vs Population Size")
plt.xlabel("Population Size")
plt.ylabel("Mean Execution Time") 
plt.grid()
plt.plot(sample_sizes, main_execution_times_gene,color="g",linestyle='dashed', marker='o',label="Genetic Algorithm")
plt.plot(sample_sizes, main_execution_times_anneal,color="r",linestyle='dashed', marker='o',label="Simulated Annealing")
plt.plot(sample_sizes, main_execution_times_hill, color="b",linestyle='dashed', marker='o',label="Random Hill Climbing")
plt.plot(sample_sizes, main_execution_times_mimic, color="m",linestyle='dashed', marker='o',label="MIMIC")
plt.legend(loc="best")
plt.savefig("onemax_times.png")
plt.clf()
#plt.ylim([0,100])
plt.title("Best Fitness Score vs Population Size")
plt.xlabel("Population Size")
plt.ylabel("Mean Best Fitness Score")
plt.grid()
plt.plot(sample_sizes, main_best_fit_gene,color="g",linestyle='dashed', marker='o',label="Genetic Algorithm")
plt.plot(sample_sizes, main_best_fit_anneal,color="r",linestyle='dashed', marker='o',label="Simulated Annealing")
plt.plot(sample_sizes, main_best_fit_hill, color="b",linestyle='dashed', marker='o',label="Random Hill Climbing")
plt.plot(sample_sizes, main_best_fit_mimic, color="m",linestyle='dashed', marker='o',label="MIMIC")
plt.legend(loc="best")
plt.savefig("onemax_accuracy.png")
plt.clf()

plt.title("No. Iterations to Optimization vs Population Size")
plt.xlabel("Population Size")
plt.ylabel("Mean Iterations Needed for Optimization")
plt.grid()
plt.plot(sample_sizes, main_opt_iterations_gene,color="g",linestyle='dashed', marker='o',label="Genetic Algorithm")
plt.plot(sample_sizes, main_opt_iterations_anneal,color="r",linestyle='dashed', marker='o',label="Simulated Annealing")
plt.plot(sample_sizes, main_opt_iterations_hill, color="b",linestyle='dashed', marker='o',label="Random Hill Climbing")
plt.plot(sample_sizes, main_opt_iterations_mimic, color="m",linestyle='dashed', marker='o',label="MIMIC")
plt.legend(loc="best")
plt.savefig("onemax_opt_iters.png")
plt.clf()

plt.title("Total Iterations vs Population Size")
plt.xlabel("Population Size")
plt.ylabel("Mean Total Iterations")
plt.grid()
plt.plot(sample_sizes, main_total_iterations_gene,color="g",linestyle='dashed', marker='o',label="Genetic Algorithm")
plt.plot(sample_sizes, main_total_iterations_anneal,color="r",linestyle='dashed', marker='o',label="Simulated Annealing")
plt.plot(sample_sizes, main_total_iterations_hill, color="b",linestyle='dashed', marker='o',label="Random Hill Climbing")
plt.plot(sample_sizes, main_total_iterations_mimic, color="m",linestyle='dashed', marker='o',label="MIMIC")
plt.legend(loc="best")
plt.savefig("onemax_total_iters.png")
plt.clf()

plt.title("Execution Time vs Iterations for Optimization")
plt.xlabel("Mean Iterations Needed for Optimization")
plt.ylabel("Mean Execution Time")
plt.grid()
plt.plot(main_opt_iterations_gene, main_execution_times_gene,color="g",linestyle='dashed', marker='o',label="Genetic Algorithm")
plt.plot(main_opt_iterations_anneal, main_execution_times_anneal,color="r",linestyle='dashed', marker='o',label="Simulated Annealing")
plt.plot(main_opt_iterations_hill, main_execution_times_hill, color="b",linestyle='dashed', marker='o',label="Random Hill Climbing")
plt.plot(main_opt_iterations_mimic, main_execution_times_mimic, color="m",linestyle='dashed', marker='o',label="MIMIC")
plt.legend(loc="best")
plt.savefig("onemax_time_iters.png")
plt.clf()