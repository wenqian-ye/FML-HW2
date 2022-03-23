# Foundations of Machine Learning HW2

## SVM Hands-on

### 3.2 

Split the dataset into train and test:

```powershell
python svm/subset.py -s 2 abalone/abalone.txt 3133 abalone/abalone_train.txt abalone/abalone_test.txt
```

Generate labels and scale:

```powershell
python svm/generate_binary.py abalone/abalone_train.txt abalone/abalone_train_binary.txt
python svm/generate_binary.py abalone/abalone_test.txt abalone/abalone_test_binary.txt
svm-scale -r abalone/abalone_train_binary.txt.range abalone/abalone_test_binary.txt > abalone/abalone_test_binary.txt.scale
```

### 3.3

```powershell
python svm/q3.py > results/q3_output.txt
```

### 3.4

```powershell
python svm/q4.py > results/q4_output.txt
```

### 3.5

```powershell
python svm/q5.py > results/q5_output.txt
```