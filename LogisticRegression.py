"""
LogisticRegression.py

CS440 - PA1

Sameena Bajwa 

"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score

class LogisticRegression:
    """
    This class implements a Logistic Regression Classifier.
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initializes the parameters of the logistic regression classifer to 
        random values.
        
        args:
            input_dim: Number of dimensions of the input data
            output_dim: Number of classes
        """
        
        self.theta = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.bias = np.zeros((1, output_dim))
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
        z = X.dot(self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        # Calculate cross-entropy loss
        cross_ent_err = -np.log(softmax_scores[range(num_samples), y])
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
        z = np.dot(X,self.theta) + self.bias
        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis = 1)
        return predictions
        
    #--------------------------------------------------------------------------
    
    def fit(self,X,y):
        
        """
        Learns model parameters to fit the data.
        """  
        
        num_samples = len(X)
        for i in range(0,5000):
            
            # Holds the change in cost with respect to the input's weight
            deltaTheta = 0
            deltaBias = 0
            
            for j in range(0,num_samples):
                
                len_input = len(X[j])
                z = np.dot(X[j], self.theta) + self.bias
                exp_z = np.exp(z)
                soft_output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                
                #### Uncomment to calculate ground truth for problem 6 #######
#                ground_truth = np.zeros(10, dtype = int)
#                index = y[j].astype(int) 
#                ground_truth[(np.array(index))] = 1
    
                ##################       end problem 6     ###################

    
                ### Uncomment to calculate ground truth for problems 1 and 2 ###
                #if y[j] == 0:
                #   ground_truth = np.array([1,0])
                #else:
                #    ground_truth = np.array([0,1])
                
                ##########           end problems 1 and 2             #########
                
                
                # Holds difference between target and calculated ouputs 
                beta = soft_output - ground_truth

                # Reweights every node in the layer according to the beta value
                # Accumulated weight changes held in deltaTheta

                ################## uncomment for problem 6 ###################
#                deltaTheta +=  np.dot(X[j].reshape(64, 1), beta)
#                deltaBias += beta
                ##################       end problem 6      ###################

                #############    uncomment for problems 1 and 2   #############
                #deltaTheta +=  np.dot(X[j].reshape(2,1), beta)
                #deltaBias += np.dot(np.ones((len_input)), beta.reshape(2,1))
                ##########           end problems 1 and 2             #########

            # Gradient descent - update model weight according to the partial 
            # derivative of the cost function with respect to weight (deltaTheta)          
            self.theta = self.theta - epsilon * deltaTheta / num_samples
            self.bias = self.bias - epsilon * deltaBias / num_samples


#--------------------------------------------------------------------------
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

########### uncomment entire section when running problems 1 and 2 ############

##Problem 1 - uncomment for linear data
#X = np.genfromtxt('DATA/Linear/X.csv', delimiter=',')
#y = np.genfromtxt('DATA/Linear/y.csv', delimiter=',')         
#y.astype(int)
  
 
##Problem 2 - uncomment for nonlinear data
#X = np.genfromtxt('DATA/NonLinear/X.csv', delimiter=',')
#y = np.genfromtxt('DATA/NonLinear/y.csv', delimiter=',')         
#y.astype(int)
 
##plot data
#plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.bwr)
#plt.show()

#gradient descent parameter - tells us how much we should change weight and
#bias in order lower the cost 
#epsilon = 0.01

#input_dim = 2
#output_dim = 2
#LR = LogisticRegression(input_dim, output_dim)

#LR.fit(X, y)
#plot_decision_boundary(LR, X, y)


#############################################################################

################ uncomment entire section for problem 6 ###################

## X_training data
#X_train = np.genfromtxt('DATA/Digits/X_train.csv', delimiter = ',')
#y_train = np.genfromtxt('DATA/Digits/y_train.csv', delimiter = ',')
## X_test data
#X_test = np.genfromtxt('DATA/Digits/X_test.csv', delimiter = ',')
#y_test = np.genfromtxt('DATA/Digits/y_test.csv', delimiter = ',')
#
#epsilon = .3
#input_dim = 64
#output_dim = 10
#LR = LogisticRegression(input_dim, output_dim)
#LR.fit(X_train, y_train)
#
#
## Computes and prints confusion matrix 
#y_actual = pd.Series(y_test.astype(int), name='Actual')
#y_predicted = pd.Series(LR.predict(X_test), name = 'Predicted')
#df_confusion = pd.crosstab(y_actual, y_predicted, rownames=['Actual'], colnames=['Predicted'], margins=True)
#print df_confusion
#
## Computes and prints accuracy score
#score = accuracy_score(y_test, LR.predict(X_test))
#print score

#############################################################################