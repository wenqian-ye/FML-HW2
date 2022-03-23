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
    return out_list

trainset_size = split_fold(3133, 10)
train_errors = []
test_errors = []

c = 3 ** 6
d = 3

for t_size in trainset_size:
    y_train = y[0:t_size]
    x_train = x[0:t_size]
        
    m = svm_train(y_train, x_train, '-h 0 -t 1 -d ' + str(d) + ' -c ' + str(c))
    p_label, p_acc, p_val = svm_predict(y_train, x_train, m)
    train_errors.append(1 - p_acc[0] / 100)
    p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
    test_errors.append(1 - p_acc[0] / 100)

plt.figure(figsize=(10,10))
plt.title("Training sample Training Error")
plt.plot(trainset_size, train_errors, color="black")
plt.grid()
plt.xlabel("Training Set Size")
plt.ylabel("Training Error")
plt.savefig("figures/q5_training_error.jpg")

plt.figure(figsize=(10,10))
plt.title("Training sample Test Error")
plt.plot(trainset_size, test_errors, color="black")
plt.grid()
plt.xlabel("Training Set Size")
plt.ylabel("Test Error")
plt.savefig("figures/q5_test_error.jpg")

