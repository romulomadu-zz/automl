from sklearn.svm import SVC

import math
import os
import numpy
from numpy import mean, sum
from sklearn.model_selection import cross_val_score
from sklearn import datasets


def create_metabase(data, algorithm, params):
    metabase = []
    best_solution = ([], None)

    count = 0

    #We can define the interval (In this case, 1.0)
    c_params = numpy.arange(params[0][0], params[0][1] +1, 1)
    gamma_params = numpy.arange(params[1][0], params[1][1] +1, 1)
    epsilon_params = numpy.arange(params[2][0], params[2][1] +1, 1)

    tam_c = len(c_params)
    tam_gamma = len(gamma_params)
    tam_epsilon = len(epsilon_params)

    total_parameters = len(c_params) * len(gamma_params) * len(epsilon_params)

    for p1 in c_params:
        algorithm.C = 2**int(p1);
        for p2 in gamma_params:
            algorithm.gamma = 2**int(p2)

            for p3 in epsilon_params:
                algorithm.epsilon = 2**int(p3)

                count += 1

                rate = cross_validation(algorithm, data.data, data.target, 10);
                
                #print(rate)

                metabase.append(((math.log(algorithm.C, 2), math.log(algorithm.gamma,2), math.log(algorithm.epsilon,2)), rate))

                if best_solution[0] == []:
                    best_solution = ((math.log(algorithm.C, 2), math.log(algorithm.gamma,2), math.log(algorithm.epsilon,2)), rate)
                    
                    #print(best_solution)
                else:
                    '''
                    When the problem is a classification one, uses greater than (success rate). In regression problems, 
                    use less than (mean error).
                    '''
                    if rate > best_solution[1]:
                        best_solution = ((math.log(algorithm.C, 2), math.log(algorithm.gamma,2), math.log(algorithm.epsilon,2)), rate)

        #Report
        print(str((float(count)*100)/total_parameters) + "%")

    return metabase, best_solution


def cross_validation(algorithm, examples, target, k_attr):
        predictions = cross_val_score(algorithm, examples, target, cv=k_attr, scoring='r2')
        
        mean_error = predictions.mean()

        return mean_error
    
if __name__ == "__main__":
    algorithm = SVC()
             #C      #gamma   #epsilon
    params = [[-5, 15], [-15, 3], [-8,-1]]
    
    data = datasets.load_iris()
    
    metabase, best = create_metabase(data, algorithm, params)
    
    print(best)
