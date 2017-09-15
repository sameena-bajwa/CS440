"""
NeuralNet.py

CS440 - PA1

Sameena Bajwa 

"""
import numpy as np 
import matplotlib.pyplot as plt 
#import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score
import pandas as pd


class NeuralNet:
    """
    This class implements a 3 layer neural network.
    """
    
    def __init__(self, input_dim, output_dim, hidden, epsilon):
        """
        Initializes the parameters of the logistic regression classifer to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
        """
        self.epsilon = epsilon
        self.hidden = hidden
        #weights and biases for hidden layer
        self.thetaA = np.random.randn(input_dim, hidden) / np.sqrt(input_dim)
        self.biasA = np.zeros((1, hidden))
        
        #weights and biases for output layer
        self.thetaB = np.random.randn(hidden, output_dim) / np.sqrt(hidden)
        self.biasB = np.zeros((1, output_dim))
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total cost on the dataset.
        
        args:
            X: Data array
            y: Labels corresponding to input data
        
        returns:
            cost: average cost per data sample
        """
        num_samples = len(X)
        # Forward propogation
        zA = np.dot(X, self.thetaA) + self.biasA
        # Use tanh activation function for hidden layers 
        aA = np.tanh(zA) 
        # Result of activation function used to calculate output layers weight
        zB = np.dot(aA, self.thetaB) + self.biasB
        exp_z = np.exp(zB)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        # Calculate cross-entropy loss
        cross_ent_err = -np.log(softmax_scores[range(num_samples), y.astype(int)])
        data_loss = np.sum(cross_ent_err)
        return 1./num_samples * data_loss
        
    
    #--------------------------------------------------------------------------
 
    def predict(self,X):
        """
        Makes a prediction based on current model parameters.
        
        args:
            X: Data array
            
        returns:
            predictions: array of predicted labels
        """
        
        # See comments for compute_cost
        zA = np.dot(X, self.thetaA) + self.biasA
        aA = np.tanh(zA)        
        zB = np.dot(aA, self.thetaB) + self.biasB
        exp_z = np.exp(zB)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis = 1)

        return predictions
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y):
        
        """
        Learns model parameters to fit the data.
        """  

        for i in range(0,1000):
            
            # Holds the change in cost with respect to the input's weight
            deltaThetaA = 0
            deltaBiasA = 0
            deltaThetaB = 0
            deltaBiasB = 0
            num_samples = len(X)
            
            for j in range(0,num_samples):
                
                # Changed code based off of compute_cost and predict                
                len_input = len(X[j])
                
                zA = np.dot(X[j], self.thetaA) + self.biasA
                aA = np.tanh(zA)
                zB = np.dot(aA, self.thetaB) + self.biasB
                exp_z = np.exp(zB)
               
                
                
                soft_output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                
                #### Uncomment to calculate ground truth for problem 7 #######
#                ground_truth = np.zeros(10, dtype = int)
#                index = y[j].astype(int) 
#                ground_truth[(np.array(index))] = 1
                ##################       end problem 7     ###################
                
                ###  Uncomment to calculate ground truth for problems 3-5  ###
#                if y[j] == 0:
#                   ground_truth = np.array([1,0])
#                else:
#                    ground_truth = np.array([0,1])
                ##########             end problems 3-5              #########
                    
                # Holds difference between target and calculated ouputs                          
                beta = soft_output - ground_truth

                ################## uncomment for problem 7 ###################
#                deltaThetaB += np.dot(aA.T, beta)
#                deltaBiasB += np.dot(beta.reshape(1,10), np.ones((10,10)))
#                
#                beta2 = np.dot(beta, self.thetaB.T) * (1-np.power(aA,2)) 
#                
#                deltaThetaA += np.dot(X[j].reshape(64, 1), beta2)
#                deltaBiasA += np.dot(beta.reshape(1,10), np.ones((10,self.hidden)))
#                
                ##################       end problem 7      ###################

#                
                #############      uncomment for problems 3-5     #############
#                deltaThetaB += np.dot(aA.T, beta)
#                deltaBiasB += np.dot(np.ones((len_input)), beta.reshape(2, 1))
#                beta2 = np.dot(beta, self.thetaB.T) * (1-np.power(aA, 2))
#                deltaThetaA += np.dot(X[j].reshape(2,1), beta2)
#                deltaBiasA += np.dot(np.ones(len_input), beta.reshape(2,1))
                ##########             end problems 3-5             #########

                
            # Gradient descent - update model weight according to the partial 
            # derivative of the cost function with respect to weight             
            self.thetaA = self.thetaA - epsilon * deltaThetaA/num_samples
            self.biasA = self.biasA - epsilon * deltaBiasA/num_samples
            self.thetaB = self.thetaB - epsilon * deltaThetaB/num_samples
            self.biasB = self.biasB - epsilon * deltaBiasB/num_samples
##--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def plot_decision_boundary(model, X, y):
    """
    Function to print the decision boundary given by model.
    
    args:
        model: model, whose parameters are used to plot the decision boundary.
        X: input data
        y: input labels
    """

    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]

    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)

    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    plt.show()


################################################################################    
###########   uncomment entire section when running problem 3    ############

## Uncomment for linear data
##X = np.genfromtxt('DATA/Linear/X.csv', delimiter=',')
##y = np.genfromtxt('DATA/Linear/y.csv', delimiter=',')         
##y.astype(int)
##  
# 
##  Uncomment for nonlinear data
#X = np.genfromtxt('DATA/NonLinear/X.csv', delimiter=',')
#y = np.genfromtxt('DATA/NonLinear/y.csv', delimiter=',')         
#y.astype(int)
# 
##plot data
#plt.scatter(X[:,0], X[:,1], s = 40, c=y, cmap=plt.cm.bwr)
#plt.show()
#
##gradient descent parameter - tells us how much we should change weight and
##bias in order lower the cost 
#epsilon = 0.3
#
#input_dim = 2
#output_dim = 2
##change to get different amount of hidden dimensions
#hidden = 5
#
#NN = NeuralNet(input_dim, output_dim, hidden, epsilon)
#NN.fit(X, y)
#plot_decision_boundary(NN, X, y)
 
#############################################################################
 
###########   uncomment entire section when running problem 7    ############

 
# Uncomment for X_training data
#X_train = np.genfromtxt('DATA/Digits/X_train.csv', delimiter = ',')
#y_train = np.genfromtxt('DATA/Digits/y_train.csv', delimiter = ',')
## Uncomment for X_test data
#X_test = np.genfromtxt('DATA/Digits/X_test.csv', delimiter = ',')
#y_test = np.genfromtxt('DATA/Digits/y_test.csv', delimiter = ',')
#
#input_dim = 64
#output_dim = 10
#hidden = 10
#epsilon = 0.3
#NN = NeuralNet (input_dim, output_dim, hidden, epsilon)
#NN.fit(X_train, y_train)
#
## Computes and prints confusion matrix 
#y_actual = pd.Series(y_test.astype(int), name='Actual')
#y_predicted = pd.Series(NN.predict(X_test), name = 'Predicted')
#df_confusion = pd.crosstab(y_actual, y_predicted, rownames=['Actual'], colnames=['Predicted'], margins=True)
#print df_confusion
#
## Computes and prints accuracy score
#score = accuracy_score(y_test.astype(int), NN.predict(X_test))
#print score

#############################################################################

########  uncomment entire section when running problems 4 and 5    ##########


##  Uncomment for nonlinear data - used for problems 4 and 5
#X = np.genfromtxt('DATA/NonLinear/X.csv', delimiter=',')
#y = np.genfromtxt('DATA/NonLinear/y.csv', delimiter=',')         
#y.astype(int)


## Problem 4 - Gives plot with difference in costs when learning rate is changed
#def learningRates():
#    
#    rates = [.00001, .0001, .001, .01, .1]
#    costs = []
#    count = 10
#    
#    for i in rates:
#
#        average = []
#        #find cost 10 times for each learning rate, use average
#        for j in range(count):
#            NN = NeuralNet(2, 2, 5, i)
#            tempcost = NN.compute_cost(X, y)
#            average += [tempcost]
#            
#        #print np.mean(average)
#        costs.append(np.mean(average))
#    
#    
#    plt.xlim(.000001, .01)
#
#    plt.ylabel("Cost")
#    plt.xlabel("Learning Rate Parameter")
#    plt.plot(rates, costs, 'b-')
#    plt.show()
#  
#learningRates()
#  
## Problem 5 - Shows effect of changing number of hidden layers
#def hiddenLayers():
#    epislon = .3
#    
#    layers = [3, 5, 7, 10]
#    for i in layers:
#        NN = NeuralNet(2, 2, i, epsilon)
#        NN.fit(X,y)
#        print "Number of hidden layers: ", i
#        plot_decision_boundary(NN, X, y)
#
#hiddenLayers()        


##############################################################################

