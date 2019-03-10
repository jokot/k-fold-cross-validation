from csv import reader
from math import exp
from random import shuffle
import matplotlib.pyplot as plt


##Import data from file---------------------------------------------------------------------------------------------------------------------
def import_data(file_name):
    dataset = list()
    with open(file_name,'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)

    return dataset


##convert input string to float---------------------------------------------------------------------------------------------------------------------
def str_to_float(dataset,column):
    for row in dataset:
        row[column] = float(row[column].strip())

##convert string type to int type ---------------------------------------------------------------------------------------------------------------------
def str_to_int(dataset,column):
    for row in dataset:
        if row[column] == "Iris-setosa":
            row[column] = 0
        else:
            row[column] = 1

##function fold ---------------------------------------------------------------------------------------------------------------------
def fold(dataset,
         k_fold, l_rate,
         n_epoch, error_train, error_test,
         accuracy_train,accuracy_test):
    
    weights = [0.5 for i in range(len(dataset[0]))]
    shuffle(dataset)
    
    dataset_test = list()

    #split dataset to data train and data test
    for k in range(k_fold):
        dataset_train = list(dataset)
        dataset_test = list()
        
        a = int(k*len(dataset)/k_fold)
        b = int(a+(len(dataset)/k_fold))
        
        for m in range(a,b):
            dataset_test.append(dataset_train[m])
        del dataset_train[a:b]

        weights = train(dataset_train, l_rate,
                        n_epoch, error_train,accuracy_train)
        
        test(dataset_test,n_epoch,
             weights,error_test,accuracy_test)
        
    averrage(error_train, error_test,
             accuracy_train,accuracy_test,
             n_epoch,k_fold)

## train the data---------------------------------------------------------------------------------------------------------------------       
def train(dataset_train, l_rate,
          n_epoch, error_train,accuracy_train):
    weight = [0. for i in range(len(dataset_train))]
    sum_error = list()
    accuracy = list()
    
    for epoch in range(n_epoch):
        sumAccuracy = 0
        sumError = 0

        for row in dataset_train:
            activ = activation(row, weight)
            prediction = 1.0 if activ >= 0.5 else 0.0
            if prediction == row[-1]:
                sumAccuracy+=1
            error = (row[-1] - activ)**2
            sumError+=error
            
            dweight = d_weight(row,activ)
            weight[-1] = weight[0] + l_rate * dweight[-1]
            for i in range(len(row)-1):
                weight[i] = weight[i] + l_rate * dweight[i]*row[i]
                
        accuracy_train.append(sumAccuracy/len(dataset_train))
        error_train.append(sumError)

    return weight


##find dwaight / update weight ---------------------------------------------------------------------------------------------------------------------
def d_weight(row,activation):
    weight = [0.0 for i in range(len(row))]
    
    for i in range(len(row)):
        weight[i] = 2.0*(row[-1]-activation)*(1.0-activation)*activation
            
    return weight
    
##find activation---------------------------------------------------------------------------------------------------------------------
def activation(row, weights):
    activation = weights[-1]
    for i in range(len(row)-1):
	    activation += weights[i] * row[i]
    
    return 1/(1+exp(-activation))
    
##test the data after training---------------------------------------------------------------------------------------------------------------------
def test(dataset_test, n_epoch, weights, error_test,accuracy_test):
    for epoch in range(n_epoch):
        sumAccuracy = 0
        sumError = 0        
        for row in dataset_test:
            activ = activation(row, weights)
            prediction = 1.0 if activ >= 0.5 else 0.0
            
            if prediction == row[-1]:
                sumAccuracy+=1
            
            error = (row[-1] - activ)**2
            sumError += error
            
        error_test.append(sumError)
        accuracy_test.append(sumAccuracy/len(dataset_test))
        

##calculate the avverage error and acuracy from k fold ---------------------------------------------------------------------------------------------------------------------
def averrage(error_train,error_test,
             accuracy_train,accuracy_test,
             n_epoch,k_fold):
    
    errorTrain = [0.0 for i in range(n_epoch)]
    errorTest = [0.0 for i in range(n_epoch)]
    accTrain = [0.0 for i in range(n_epoch)]
    accTest = [0.0 for i in range(n_epoch)]
    
    for k in range(k_fold):
        err_train = list()
        err_test = list()
        acc_train = list()
        acc_test = list()
        
        a = int(k*len(error_train)/k_fold)
        b = int(a+(len(error_train)/k_fold))

        for m in range(a,b):
            err_test.append(error_test[m])
            err_train.append(error_train[m])
            acc_test.append(accuracy_test[m])
            acc_train.append(accuracy_train[m])
            
        for m in range(n_epoch):
            errorTrain = [x + y for x, y in zip(err_train, errorTrain)]
            errorTest= [x + y for x, y in zip(err_test, errorTest)]
            accTrain = [x + y for x, y in zip(acc_train, accTrain)]
            accTest = [x + y for x, y in zip(acc_test, accTest)]

    errorTrain = [x / n_epoch*k_fold for x in errorTrain]
    accTrain = [x / n_epoch*k_fold for x in accTrain]
    accTest = [x / n_epoch*k_fold for x in accTest]
    errorTest = [x / n_epoch*k_fold for x in errorTest]

    draw_grafik(errorTrain,errorTest,
                'Averrage Error',
                'Error',
                'Grafik Error k-fold')
    draw_grafik(accTrain,accTest,
                'Averrage Accuracy',
                'Accuracy',
                'Grafik Accurasi k-fold')


##draw grafik ---------------------------------------------------------------------------------------------------------------------
def draw_grafik(data_train,data_test,label,ylabel,title):
    
    plt.plot(data_train,label = label+' (Train)')
    plt.plot(data_test, label = label+' (Test)')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    

##Main program ---------------------------------------------------------------------------------------------------------------------
dataset = import_data("iris.csv")
for i in range(len(dataset[0])-1):
    str_to_float(dataset,i)
str_to_int(dataset,len(dataset[0])-1)

k_fold = 5
l_rate = 0.2
#l_rate = 0.8
epoch = 300

error_train = list()
error_test = list()
accuracy_train = list()
accuracy_test =list()

fold(dataset, k_fold, l_rate, epoch,
     error_train, error_test,accuracy_train,accuracy_test)





