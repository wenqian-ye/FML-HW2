#!/usr/bin/env python
from libsvm.svmutil import *
from libsvm.svm import *
import random
import numpy as np
import matplotlib.pyplot as plt 

y, x = svm_read_problem('./abalone/abalone_train_binary.txt.scale')
training_set = list(zip(y, x))
random.shuffle(training_set)
y, x = zip(*training_set)
y_test, x_test = svm_read_problem('./abalone/abalone_test_binary.txt.scale')

def split_fold(m, n):
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        out_list = [quotient] * (n - remainder) + [quotient + 1] * remainder
    elif remainder < 0:
        out_list = [quotient - 1] * -remainder + [quotient] * (n + remainder)
    else:
        out_list = [quotient] * n
    for i in range(1, len(out_list)):
        out_list[i] = out_list[i-1] + out_list[i]
    return [0] + out_list

folder_5 = split_fold(3133 , 5)
model_dict = {}

for d in [1, 2, 3, 4, 5]:
    for logc in range(-8, 9):
        model_dict[(logc, d)] = []
        for split in range(0, 5):
            val_start = folder_5[split]
            val_end = folder_5[split+1]
            
            y_train = y[0:val_start] + y[val_end:len(y)]
            x_train = x[0:val_start] + x[val_end:len(x)]
            y_val = y[val_start:val_end]
            x_val = x[val_start:val_end]
            
            c = 3 ** logc
            m = svm_train(y_train, x_train, '-h 0 -t 1 -d ' + str(d) + ' -c ' + str(c))
            p_label, p_acc, p_val = svm_predict(y_val, x_val, m)
            model_dict[(logc, d)] += [1 - p_acc[0] / 100]   # Training error

for d in [1, 2, 3, 4, 5]:
    logc_plot = range(-8, 9)
    error_mean_plot = []
    error_up_plot = []
    error_low_plot = []
    
    for logc in range(-8, 9):
        error_mean = np.mean(model_dict[(logc, d)])
        error_std = np.std(model_dict[(logc, d)])
        error_mean_plot.append(error_mean)
        error_up_plot.append(error_mean + error_std)
        error_low_plot.append(error_mean - error_std)
        print("C = 3^" + str(logc) + " ; d = " + str(d) + " ; Training Error = " + str(error_mean))
    
    plt.figure(figsize=(10,10))
    plt.title("d = " + str(d))
    plt.plot(logc_plot, error_up_plot, color="black", linestyle="--")
    plt.plot(logc_plot, error_mean_plot, color="black")
    plt.plot(logc_plot, error_low_plot, color="black", linestyle="--")
    plt.grid()
    plt.xlabel("log_3(C)")
    plt.ylabel("Cross Validation Error")
    plt.savefig("figures/Q3_Error" + "_d_" + str(d) + ".jpg")
