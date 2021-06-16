import numpy as np
import math
import matplotlib.pyplot as plt


def data():
    #open the file
    f = open("raw_iris_data_original.txt", "r+", encoding ="utf-8")
    raw = f.read().split()
    f.close()
    data = np.array(list(raw))
    #some lists that i will use to process data
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    a5 = []
    na1 = []
    na2 = []
    na3 = []
    na4 = []
    dataset = []
    setosa = []
    versicolor = []
    virginica = []
    train = []
    test = []
# since the input data is in text, we have to change the type of each attribute to float (excluding the type attribute)
# so we can normalize the data in a easier way
    for i in range (len(data)):
        k = data[i].split(",")
        a1.append(float(k[0]))
        a2.append(float(k[1]))
        a3.append(float(k[2]))
        a4.append(float(k[3]))
        a5.append(k[4])
#normalize the data
    for i in range (len(data)):
        na1.append(round((a1[i]-min(a1))/(max(a1) - min(a1)),3))
        na2.append(round((a2[i]-min(a2))/(max(a2) - min(a2)),3))
        na3.append(round((a3[i]-min(a3))/(max(a3) - min(a3)),3))
        na4.append(round((a4[i]-min(a4))/(max(a4) - min(a4)),3))

#append each attribute list to a big datasetet, each plant is a dependent list
    for i in range (len(data)):
        d =[]
        d.append(na1[i])
        d.append(na2[i])
        d.append(na3[i])
        d.append(na4[i])
        d.append(a5[i])
        dataset.append(d)
#the original dataset is sorted in order setosa-versicolor-virginica with 50 plants each type
#so we slice the list into three part to get the list of plants in each type
    setosa = dataset[:50]
    versicolor = dataset[50:100]
    virginica = dataset[100:150]
#random so we can get different set each time
    np.random.shuffle(setosa)
    np.random.shuffle(versicolor)
    np.random.shuffle(virginica)
#with the ratio 3:2 for trainSet and testSet, we can create train set by selecting the first 30 plants of each type
#test set by slecting the later 20 plants of  each type. This will ensure to maintain the ratio throughout the testing
#process
    for i in range (0,30):
        train.append(setosa[i])
        train.append(versicolor[i])
        train.append(virginica[i])
        
    for i in range (30,50):
        test.append(setosa[i])
        test.append(versicolor[i])
        test.append(virginica[i])  
        
    return train, test

#calculate the distance of points and put it in a dictionary, the k closet points will be selected (sorted ascending)
#the distance is calculated by the Euclidean formula
def KKN(train, point, k):
    distances = []
    for item in train:
        distance = 0
        for i in range(4):
            distance += (item[i] - point[i]) ** 2
        distance =  math.sqrt(distance)
        distances.append({
            "label": item[-1],
            "value": distance
        })
    distances.sort(key=lambda x: x["value"])
    labels = [item["label"] for item in distances]
    return labels[:k]


#the algorithm checks all k closet points and which attribute has the most 
# repeated points will be the attribute for the choosen point
def repeated(kpoints):
    attributes = set(kpoints) 
    predictedattr = ""
    mostRep = 0
    for attr in attributes:
        n = kpoints.count(attr)
        if n > mostRep:
            mostRep = n
            predictedattr = attr
    return predictedattr

#run the algorithm

def main():
    results = []
    trialno = list(range(1,61))
    #run the algorithm 60 times and get the results
    for i in range (60):    
        train, test = data()
        correct = 0
        for item in test:
            closepoint = KKN(train, item, 5)
            predictedattr = repeated(closepoint)
            if item[-1] == predictedattr:
                correct += 1
        results.append(correct/len(test))
        #accuracy calculator
        print("Accuracy of trial number {}: {}%".format(i+1,round(correct/len(test)*100,2)))
    print('Algorithms performance in correctly predicting the category for the testing points: {} %'.format(round((sum(results) / len(results))*100,3)))
    plt.scatter(trialno, results)
    plt.xlabel("Trial Number")
    plt.ylabel("Percentages (%)")
    
main()