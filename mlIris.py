"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Introduction Into Machine Learning 
Iris Dataset
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import mode
import numpy as np
import copy


# import some data to play with
iris = datasets.load_iris()
data = iris.data
targets = iris['target']

samples = len(data)

"""
K-Nearest Neighbor
**User-input**
"""""

alpha = [1,1,1,1]

""""""

#### GRAPHS ###
    


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Based on the correlation data presented in the Iris Data Set, the main feature 
that will be looked at are the Pedals, and from that creating a 2D look at the
grouping of the data.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

data = iris.data

petal_lengths = data[:,2]
petal_widths = data[:,3]

setpl = petal_lengths[0:50]
setpw = petal_widths[0:50]

verpl = petal_lengths[50:100]
verpw = petal_widths[50:100]

virpl = petal_lengths[100:150]
virpw = petal_widths[100:150]


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plotting the Iris Data Set for Pedal width vs Pedal length
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

plt.figure(1)
plt.plot(setpl,setpw,'c*',
         verpl,verpw,'bo',
         virpl,virpw,'mv')
plt.ylabel('Pedal Widths')
plt.xlabel('Pedal Lengths')



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Just for experimenting puposes, we will also look at the sepal length and
width, and from that creating a 2D look at the grouping of the data
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

sepal_lengths = data[:,0]
sepal_widths = data[:,1]

setsl = sepal_lengths[0:50]
setsw = sepal_widths[0:50]

versl = sepal_lengths[50:100]
versw = sepal_widths[50:100]

virsl = sepal_lengths[100:150]
virsw = sepal_widths[100:150]


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plotting the Iris Data Set for Pedal width and length
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

plt.figure(2)
plt.plot(setsl,setsw,'gv',
         versl,versw,'b*',
         virsl,virsw,'c<')
plt.ylabel('Sepal Widths')
plt.xlabel('Sepal Lengths')

###############################################################################

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Helper functions to calculate distance between two data points and find
the k nearest neighbors to a specific data point sample/test
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Using a distance formula, calulates the distance between two data points
# MODEL TO CLASSIFY NEW DATA POINTS GIVEN THE IRIS DATASET
def distance(sarr,darr):
    sum1 = 0
    n = len(sarr)
    for i in range(n):
        sum1 += alpha[i] * ((sarr[i]-darr[i])**2)
    sum1 = sum1**0.5
    return sum1


# Function that takes an an array for a sample/test and gives a classification
# based on the k nearest neighbors
def classify(sarr,kn):
    #Create the arrays needed to store values (nearest neighbors, indices, etc.)
    distances = [] #distances between sample and all points
    species = [] # species for the k nearest neighbors
    for i in range(samples):
        dist = distance(sarr,data[i])
        distances.append(dist)
    for i in range(kn):
        minimum = min(distances)
        maximum = max(distances)
        indx = distances.index(minimum)
        species.append(targets[indx])
        distances[indx] = maximum
    return mode(species)[0][0]
    

            
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Testing the above model against the actual data set to see how accurately
it can predict a flower's species
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    

classtest = []

for i in range(samples):
    classtest.append(classify(data[i],10))
    
    
correct = 0

for i in range(samples):
    if classtest[i] == targets[i]:
        correct += 1.0
        
accuracy = correct/samples      





"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Testing the above model against the actual data set to see how accurately
it can predict a flower's species while varying k
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""  


def findingK(u):
    accur = []
    ks = range(1,u)
    for i in range(1,u):
        k = i
        correct = 0
        results = []
        for j in range(samples):
            results.append(classify(data[j],k))
            if results[j] == targets[j]:
                correct += 1.0
        accur.append(correct/samples)
    return ks, accur

ks, acc = findingK(30)
        


plt.figure(3)
plt.plot(ks,acc,'-o')
plt.ylabel('Score Against Data Set')
plt.xlabel('K')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Testing the above model against the actual data set to see how accurately
it can predict a flower's species while varying all the alphas
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""  


def findingAlphas(step):
    als = np.linspace(0,1,step)
    knn = 14
    opdata = []
    temparr = [1,1,1,1,1]
    for i in range(1,step):
        alpha[0] = als[i]
        temparr[0] = als[i]
        for j in range(1,step):
            alpha[1] = als[j]
            temparr[1] = als[j]
            for m in range(1,step):
                alpha[2] = als[m]
                temparr[2] = als[m]
                for p in range(1,step):
                    alpha[3] = als[p]
                    temparr[3] = als[p]
                    results = []
                    correct = 0
                    for po in range(samples):
                        results.append(classify(data[po],knn))
                        if results[po] == targets[po]:
                            correct += 1.0
                    accur = (correct/samples)
                    temparr[4] = accur
                    darr = copy.deepcopy(temparr)
                    opdata.append(darr)
    return opdata
        

def optimalAlphas():
    ldata = np.array(findingAlphas(11))
    accuracies = ldata[:,4]
    ma = max(accuracies)
    indx = np.where(accuracies==ma)
    alphas = ldata[indx[0][0]][0:4]
    print("Alpha1 is equal to ",round(alphas[0],2))
    print("Alpha2 is equal to ",round(alphas[1],2))
    print("Alpha3 is equal to ",round(alphas[2],2))
    print("Alpha4 is equal to ",round(alphas[3],2))
    print("Which give a model with an accuracy of ",round(ma,3)*100,'%')
    
    
    
    
    
    
    
    




