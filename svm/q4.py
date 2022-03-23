#!/usr/bin/env python
from libsvm.svmutil import *
from libsvm.svm import *
import random
import numpy as np
import matplotlib.pyplot as plt

# Best (C, d) is (3^6, 3)

y,x = svm_read_problem('./abalone/abalone_train_binary.txt.scale')
y_test,x_test = svm_read_problem('./abalone/abalone_test_binary.txt.scale')

training_set = list(zip(y, x))
random.shuffle(training_set)
y, x = zip(*training_set)

def split_integer(m, n):
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

folder_5 = split_integer(3133 , 5)
val_errors = {}
test_errors = {}

# Select C = 3**6
c = 3 ** 6

for d in [1, 2, 3, 4, 5]:
    val_errors[d] = []
    test_errors[d] = []
    for split in range(0, 5):
        val_start = folder_5[split]
        val_end = folder_5[split+1]
        
        y_train = y[0:val_start] + y[val_end:len(y)]
        x_train = x[0:val_start] + x[val_end:len(x)]
        y_val = y[val_start:val_end]
        x_val = x[val_start:val_end]
        
        m = svm_train(y_train, x_train, '-h 0 -t 1 -d ' + str(d) + ' -c ' + str(c))
        p_label, p_acc, p_val = svm_predict(y_val, x_val, m)
        val_errors[d] += [1 - p_acc[0] / 100]
        p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
        test_errors[d] += [1 - p_acc[0] / 100]

d_plot = [1, 2, 3, 4, 5]
val_error_mean_plot = []
val_error_up_plot = []
val_error_low_plot = []
test_error_mean_plot = []
test_error_up_plot = []
test_error_low_plot = []

for d in d_plot:
    val_error_mean = np.mean(val_errors[d])
    val_error_std = np.std(val_errors[d])
    test_error_mean = np.mean(test_errors[d])
    test_error_std = np.std(test_errors[d])
    
    val_error_mean_plot.append(val_error_mean)
    val_error_up_plot.append(val_error_mean + val_error_std)
    val_error_low_plot.append(val_error_mean - val_error_std)
    
    test_error_mean_plot.append(test_error_mean)
    test_error_up_plot.append(test_error_mean + test_error_std)
    test_error_low_plot.append(test_error_mean - test_error_std)
    print("d = " + str(d) + " ; Val Error = " + str(val_error_mean) + " ; Test Error = " + str(test_error_mean))

plt.figure(figsize=(10,10))
plt.title("5-fold Cross-validation Error")
plt.plot(d_plot, val_error_up_plot, color="black", linestyle="--")
plt.plot(d_plot, val_error_mean_plot, color="black")
plt.plot(d_plot, val_error_low_plot, color="black", linestyle="--")
plt.grid()
plt.xlabel("d")
plt.ylabel("Error")
plt.savefig("figures/q4_Val_Error.jpg")

plt.figure(figsize=(10,10))
plt.title("Test Error")
plt.plot(d_plot, test_error_up_plot, color="black", linestyle="--")
plt.plot(d_plot, test_error_mean_plot, color="black")
plt.plot(d_plot, test_error_low_plot, color="black", linestyle="--")
plt.grid()
plt.xlabel("d")
plt.ylabel("Error")
plt.savefig("figures/q4_Test_Error.jpg")

nsv = [
        [1193, 1247, 1211, 1217, 1229],
        [1146, 1195, 1166, 1176, 1193],
        [1142, 1186, 1163, 1174, 1181],
        [1174, 1210, 1189, 1193, 1206],
        [1218, 1253, 1231, 1240, 1244]
    ]

nbsv = [
        [1184, 1238, 1201, 1207, 1219],
        [1121, 1171, 1142, 1151, 1170],
        [1105, 1152, 1131, 1135, 1144],
        [1131, 1171, 1153, 1151, 1167],
        [1187, 1213, 1195, 1196, 1207]
    ]

nsv_mean = []
nbsv_mean = []

for i in nsv:
    nsv_mean.append(np.mean(i))

for i in nbsv:
    nbsv_mean.append(np.mean(i))

d_plot = [1, 2, 3, 4, 5]

plt.figure(figsize=(10,10))
plt.title("Number of Support Vectors")
plt.plot(d_plot, nsv_mean, color="black")
plt.grid()
plt.xlabel("d")
plt.ylabel("nSV")
plt.savefig("figures/q4_nSV.jpg")

for i in range(0, 5):
    print(nsv_mean[i] - nbsv_mean[i])