# import os, re, sys, pickle, datetime
# import itertools
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
# from matplotlib.colors import ListedColormap,LinearSegmentedColormap
# import pandas as pd
# from scipy import stats
# import seaborn as sns
# import copy

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA,NMF
# from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
# from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_regression
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.linear_model import LogisticRegression,Lasso,LinearRegression,Ridge,ElasticNetCV,ElasticNet,Lars,LassoCV,RidgeCV,LarsCV,LassoLarsCV,LassoLarsIC,OrthogonalMatchingPursuitCV,OrthogonalMatchingPursuit
# from sklearn.manifold import TSNE,MDS
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix,f1_score
# from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedKFold,LeaveOneOut,cross_val_score,cross_validate
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier,MLPRegressor
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler,MinMaxScaler,PolynomialFeatures
# from sklearn.svm import LinearSVC,SVR
# from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
# from sklearn.model_selection import StratifiedShuffleSplit

### Really just a data container to keep a bunch of stuff in one place
class Threshold:
    def __init__(self, dt, index, cut_value, operator, accuracy, f1, correlations):
        self.dt = dt
        self.index = index
        self.cut_value = cut_value
        self.operator = operator
        self.accuracy = accuracy
        self.f1 = f1
        self.correlations = correlations
        
        self.feature_label = f'x{index+1}'
        self.feature_name = X_names[index]
        self.correlated_features = np.where(correlations==1)[0]
        self.added_accuracy = 0
        
    def __str__(self):
        return f'{self.feature_label} {self.feature_name} {self.operator} {self.cut_value:.3f} with Added Accuracy {self.added_accuracy:.3f}'    

### This is where most of the actual computations happen
class Hotspot:
    def __init__(self, thresholds, X, y, y_cut, ts=[], vs=[], low_is_good=False):
        self.thresholds = []
        self.X = X
        self.y = y
        self.y_cut = y_cut
        
        if(low_is_good):
            self.y_class = np.array([0 if i > y_cut else 1 for i in y])
        else:
            self.y_class = np.array([0 if i < y_cut else 1 for i in y])
        
        if(ts == []):
            ts = [range(len(y))]
        else:
            self.ts = ts
        self.vs = vs
        self.X_train = X[ts]
        self.X_test = X[vs]
        self.y_train = y[ts]
        self.y_test = y[vs]
        self.y_class_train = self.y_class[ts]
        self.y_class_test = self.y_class[vs]

        self.accuracy = sum(self.y_class)/len(self.y_class)
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.add_threshold(thresholds[0])
    
        self.threshold_indexes = self.__set_threshold_indexes()
        self.correlated_features = self.__set_correlated_features()
            
    ### What comes out when you call the object as a string
    def __str__(self):
        output = 'Thresholds: \n'
        output = output + f'\tAccuracy with no thresholds: {sum(self.y_class)/len(self.y_class):.3f}\n'
        for thresh in self.thresholds:
            output = output + '\t' + str(thresh) + '\n'
        output = output + f'Total accuracy with {len(self.thresholds)} thresholds: {self.accuracy:.3f}\n'
        if(len(self.y_test)>0):
            output = output + f'Training set accuracy with {len(self.thresholds)} thresholds: {self.train_accuracy:.3f}\n'
            output = output + f'Test set accuracy with {len(self.thresholds)} thresholds: {self.test_accuracy:.3f}\n'
        return output
    
    ### Add a threshold to the hotspot and update accuracy
    def add_threshold(self, threshold):
        temp_accuracy = self.accuracy
        
        self.thresholds.append(threshold)
        self.accuracy = self.__get_accuracy()
        self.train_accuracy = self.__get_train_accuracy()
        self.test_accuracy = self.__get_test_accuracy()
        
        added_accuracy = self.accuracy-temp_accuracy
        self.thresholds[-1].added_accuracy = added_accuracy
        
        self.threshold_indexes = self.__set_threshold_indexes()
        self.correlated_features = self.__set_correlated_features()
       
    ### Bookeeping to update threshold_indexes
    def __set_threshold_indexes(self):
        indexes = []
        for thresh in self.thresholds:
            indexes.append(thresh.index)
        return indexes
    
    ### Bookeeping to update correlated_features
    def __set_correlated_features(self):
        features = []
        for thresh in self.thresholds:
            features.append(thresh.correlated_features)
        return features
    
    ### Returns the truth value of {value < or > cutoff}
    def __evaluate_threshold(self, value, op, cutoff):
        output = False
        if (op == '<'):
            if(value < cutoff):
                output = True
        elif(op == '>'):
            if(value > cutoff):
                output = True
        return output
        
    ### Evaluates if the point at a given index is inside or outside the hotspot
    ### Returns a bool of whether or not the point is inside the hotspot
    def __is_inside(self, index, X_space, verbose=False):
        bool_list = []
        for thresh in self.thresholds:
            value = X_space[index, thresh.index]
            if(verbose):
                print(f'Thresh Index:{thresh.index}')
                print(f'Value: {value}')
            bool_list.append(self.__evaluate_threshold(value, thresh.operator, thresh.cut_value))
        return all(bool_list)
    
    ### Gives total accuracy as a number
    def __get_accuracy(self):
        correct = 0
        for i, hit in enumerate(self.y_class):
            if (hit == 1 and self.__is_inside(i, self.X)):
                correct = correct + 1
            elif(hit == 0 and not self.__is_inside(i, self.X)):
                correct = correct + 1
                
        accuracy = correct/len(self.y)
        #print(f'__get_accuracy test \n\tCorrect: {correct}, Total: {len(self.y)}\n')
        return accuracy
    
    ### Gives accuracy of the training set as a number
    def __get_train_accuracy(self):
        if(len(self.y_train)==len(self.y)):
            return self.accuracy
        
        correct = 0
        for i, hit in enumerate(self.y_class_train):
            if (hit == 1 and self.__is_inside(self.ts[i], self.X)):
                correct = correct + 1
            elif(hit == 0 and not self.__is_inside(self.ts[i], self.X)):
                correct = correct + 1
                
        accuracy = correct/len(self.y_train)
        return accuracy   
    
    ### Gives accuracy of the test set as a number
    def __get_test_accuracy(self):
        if(len(self.y_test)==0):
            return 0
        
        correct = 0
        for i, hit in enumerate(self.y_class_test):
            if (hit == 1 and self.__is_inside(self.vs[i], self.X)):
                correct = correct + 1
            elif(hit == 0 and not self.__is_inside(self.vs[i], self.X)):
                correct = correct + 1
        
        accuracy = correct/len(self.y_test)
        return accuracy
      
    ### For internal use to get the chemical space after applying a threshold
    def __apply_threshold(self, thresh, X_current, y_current):
        if(thresh.operator == '<'):
            mask_prop = X_current[:,thresh.index] < thresh.cut_value
        elif(thresh.operator == '>'):
            mask_prop = X_current[:,thresh.index] > thresh.cut_value
        else:
            print('THERE IS A PROBLEM HERE')
        X_sel,y_sel = X_current[mask_prop],y_current[mask_prop]
        return X_sel,y_sel
    
    ### Returns a cut down X, y based on the hotspot
    def get_hotspot_space(self):
        X_out = self.X
        y_out = self.y
        for thresh in self.thresholds:
            X_out, y_out = self.__apply_threshold(thresh, X_out, y_out)
        return X_out, y_out
        
    ### Returns a hotspots with the same thresholds but different X and y 
    ### to see if the hotspot applies to another data set
    def test_transfer(self, test_X, test_y, test_y_cut):
        test_hs = Hotspot(self.thresholds, test_X, test_y, test_y_cut)
        return test_hs      
    
    ### Returns a list of all ligands in the hotspot
    ### {style} should be 'index' or 'id' depending on what format you want out
    def get_hotspot_ligands(self, style='index'):
        output = []
        ### Iterates through each ligand by index and if it's within the hotspot appends the ligand ID to output
        if(style=='index'):
            for i in range(len(self.y)):
                if(self.__is_inside(i,self.X, False)):
                    output.append(i)
        elif(style=='id'):
            for i in range(len(self.y)):
                #print(f'Ligand{y_labels_dict[i]}')
                if(self.__is_inside(i,self.X, False)):
                    output.append(y_labels_dict[i])
        return output
    
    ### Print the hotspot in 3d space
    def plot_3d(self, subset='all', coloring='scaled'):
        #%matplotlib notebook
        #print('AT LEAST THIS IS WORKING')
        x_col,y_col,z_col = self.thresholds[0].index, self.thresholds[1].index, self.thresholds[2].index
        
        ### This section auto-scales the plot
        x_min = min(self.X[:,x_col])
        x_max = max(self.X[:,x_col])
        y_min = min(self.X[:,y_col])
        y_max = max(self.X[:,y_col])
        z_min = min(self.X[:,z_col]) 
        z_max = max(self.X[:,z_col])

        dx = abs(x_min-x_max)
        dy = abs(y_min-y_max)
        dz = abs(z_min-z_max)

        x_min = x_min - abs(dx*0.05)
        x_max = x_max + abs(dx*0.05)
        y_min = y_min - abs(dy*0.05)
        y_max = y_max + abs(dy*0.05)
        z_min = z_min - abs(dz*0.05)
        z_max = z_max + abs(dz*0.05)

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection = '3d')

        ### Sets the points to plot based on {subset}
        if(subset=='all'):
            X_space = self.X
            y_scaled = self.y
            y_binary = self.y_class
        elif(subset=='train'):
            X_space = self.X_train
            y_scaled = self.y_train
            y_binary = self.y_class_train
        elif(subset=='test'):
            X_space = self.X_test
            y_scaled = self.y_test
            y_binary = self.y_class_test
        else:
            print('---Bad subset input---\n')
            
        
        ### Changes how the points are colored, controlled by the {coloring} variable
        if(coloring=='scaled'):
            mapping_cl = y_scaled
            cMap_cl = 'coolwarm'
        elif(coloring=='binary'):
            mapping_cl = y_binary
            cMap_cl = 'Oranges'

        ax.scatter(X_space[:,x_col], X_space[:,y_col], X_space[:,z_col],c=mapping_cl,cmap=cMap_cl,alpha=0.95,marker="s",s=20, edgecolors='k') 
            
        ### Plots the z-axis threshold
        temp_x = np.linspace(x_min, x_max, num=10)
        temp_y = np.linspace(y_min, y_max, num=10)
        temp_x, temp_y = np.meshgrid(temp_x, temp_y)
        temp_z = self.thresholds[2].cut_value + 0*temp_x + 0*temp_y
        ax.plot_surface(temp_x, temp_y, temp_z, alpha=0.15, color='gray')
        
        ### Plots the x-axis threshold
        temp_y = np.linspace(y_min, y_max, num=10)
        temp_z = np.linspace(z_min, z_max, num=10)
        temp_z, temp_y = np.meshgrid(temp_z, temp_y)
        temp_x = self.thresholds[0].cut_value + 0*temp_z + 0*temp_y
        ax.plot_surface(temp_x, temp_y, temp_z, alpha=0.15, color='gray') 
        
        ### Plot the y-axis threshold
        temp_x = np.linspace(x_min, x_max, num=10)
        temp_z = np.linspace(z_min, z_max, num=10)
        temp_x, temp_z = np.meshgrid(temp_x, temp_z)
        temp_y = self.thresholds[1].cut_value + 0*temp_x + 0*temp_z
        ax.plot_surface(temp_x, temp_y, temp_z, alpha=0.15, color='gray')
        
        plt.xticks(fontsize=10) 
        plt.yticks(fontsize=10)

        
        ax.set_xlabel(f'{self.thresholds[0].feature_label} {self.thresholds[0].feature_name}',fontsize=12.5)
        ax.set_ylabel(f'{self.thresholds[1].feature_label} {self.thresholds[1].feature_name}',fontsize=12.5)
        ax.set_zlabel(f'{self.thresholds[2].feature_label} {self.thresholds[2].feature_name}',fontsize=12.5)
        plt.locator_params(axis='y', nbins=8)

        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        ax.set_zlim(z_min, z_max)

        plt.tight_layout()
        plt.show()  
        
    ### Print the hotspot in 2d space
    def plot_2d(self, subset='all', coloring='scaled', output_label='Yield %'):
        #%matplotlib inline
        
        x_col,y_col = self.thresholds[0].index, self.thresholds[1].index

        ### This section auto-scales the plot
        x_min = min(self.X[:,x_col])
        x_max = max(self.X[:,x_col])
        y_min = min(self.X[:,y_col])
        y_max = max(self.X[:,y_col])

        dx = abs(x_min-x_max)
        dy = abs(y_min-y_max)

        x_min = x_min - abs(dx*0.05)
        x_max = x_max + abs(dx*0.05)
        y_min = y_min - abs(dy*0.05)
        y_max = y_max + abs(dy*0.05)
        
        ### Sets the points to plot based on {subset}
        if(subset=='all'):
            X_space = self.X
            y_scaled = self.y
            y_binary = self.y_class
        elif(subset=='train'):
            X_space = self.X_train
            y_scaled = self.y_train
            y_binary = self.y_class_train
        elif(subset=='test'):
            X_space = self.X_test
            y_scaled = self.y_test
            y_binary = self.y_class_test
        else:
            print('---Bad subset input---\n')
            
        
        ### Changes how the points are colored, controlled by the {coloring} variable
        if(coloring=='scaled'):
            mapping_cl = y_scaled
            cMap_cl = 'coolwarm'
        elif(coloring=='binary'):
            mapping_cl = y_binary
            cMap_cl = 'Oranges'
        
        ### Axis setup
        plt.figure(figsize=(12, 8))
        plt.xlabel(f'{self.thresholds[0].feature_label} {self.thresholds[0].feature_name}',fontsize=15)
        plt.ylabel(f'{self.thresholds[1].feature_label} {self.thresholds[1].feature_name}',fontsize=15)
        plt.xticks(fontsize=18)
        plt.xlim(x_min, x_max)
        plt.locator_params(axis='x', nbins=5)
        plt.yticks(fontsize=18)
        plt.ylim(y_min, y_max)
        plt.locator_params(axis='y', nbins=4)

        ### This is where you choose which descriptors to plot
        plt.scatter(X_space[:,x_col],X_space[:,y_col], c=mapping_cl,cmap=cMap_cl, edgecolor='black', s=100)  
        
        ### Set the gradient bar on the side
        if(coloring == 'scaled'):
            cbar = plt.colorbar()
            cbar.set_label(output_label, rotation=90,size=18)
            if('yield' in output_label or 'Yield' in output_label):
                plt.clim(vmin=0, vmax=100)
            else:
                plt.clim(vmin=0, vmax=max(y_scaled)*1.05)

        plt.axhline(y=self.thresholds[1].cut_value, color='r', linestyle='--')
        plt.axvline(x=self.thresholds[0].cut_value, color='r', linestyle='--')

        plt.show() 
        
    ### Print the hotspot in 1d space
    def plot_1d(self, subset='all', coloring='scaled', output_label='Yield %'):
        #%matplotlib inline
        
        x_col = self.thresholds[0].index

        ### This section auto-scales the plot
        x_min = min(self.X[:,x_col])
        x_max = max(self.X[:,x_col])
        y_min = min(self.y)
        y_max = max(self.y)

        dx = abs(x_min-x_max)
        dy = abs(y_min-y_max)

        x_min = x_min - abs(dx*0.05)
        x_max = x_max + abs(dx*0.05)
        y_min = y_min - abs(dy*0.05)
        y_max = y_max + abs(dy*0.05)
        
        ### Sets the points to plot based on {subset}
        if(subset=='all'):
            X_space = self.X
            y_scaled = self.y
            y_binary = self.y_class
        elif(subset=='train'):
            X_space = self.X_train
            y_scaled = self.y_train
            y_binary = self.y_class_train
        elif(subset=='test'):
            X_space = self.X_test
            y_scaled = self.y_test
            y_binary = self.y_class_test
        else:
            print('---Bad subset input---\n')
            
        
        ### Changes how the points are colored, controlled by the {coloring} variable
        if(coloring=='scaled'):
            mapping_cl = y_scaled
            cMap_cl = 'Oranges'
        elif(coloring=='binary'):
            mapping_cl = y_binary
            cMap_cl = 'Oranges'
        
        ### Axis setup
        plt.figure(figsize=(12, 8))
        plt.xlabel(f'{self.thresholds[0].feature_label} {self.thresholds[0].feature_name}',fontsize=15)
        plt.ylabel(output_label,fontsize=15)
        plt.xticks(fontsize=18)
        plt.xlim(x_min, x_max)
        plt.locator_params(axis='x', nbins=5)
        plt.yticks(fontsize=18)
        plt.ylim(y_min, y_max)
        plt.locator_params(axis='y', nbins=4)

        ### This is where you choose which descriptors to plot
        plt.scatter(X_space[:,x_col],y_scaled, c=mapping_cl,cmap=cMap_cl, edgecolor='black', s=100)  
        
        ### Set the gradient bar on the side
        if(coloring == 'scaled'):
            cbar = plt.colorbar()
            cbar.set_label(output_label, rotation=90,size=18)
            if('yield' in output_label or 'Yield' in output_label):
                plt.clim(vmin=0, vmax=100)
            else:
                plt.clim(vmin=0, vmax=max(y_scaled)*1.05)          
            
        plt.axvline(x=self.thresholds[0].cut_value, color='r', linestyle='--')

        plt.show() 
       
    ### plots the hotspot in 2d or 3d depending on how many thresholds are present
    ### subset should be 'all', 'test', or 'train'
    ### coloring should be 'scaled' or 'binary' --- only works on 3D at the moment
    def plot(self, subset='all', coloring='scaled', output_label='Yield %'):
        if(len(self.thresholds)==1):
            self.plot_1d(subset, coloring, output_label)
        elif(len(self.thresholds)==2):
            self.plot_2d(subset, coloring, output_label)
        elif(len(self.thresholds)==3):
            self.plot_3d(subset, coloring)
      
    ### Gives a bool column of which ligands are inside the hotspot
    ### {X_all} is the space you're expanding to, likely the X_all variable from reading in data
    ### Returns a list of bools
    def expand(self, X_all):
        bool_list = [self.__is_inside(i, X_all) for i in range(len(X_all))]
        return bool_list
    
    ### returns a list of all hotspots that can be made from colinear parameters
    def colinear_hotspots(self):
        hotspot_list = []
        ### Iterates through each of the correlated sets
        if(len(self.correlated_features) == 3):
            for colinear_1 in self.correlated_features[0]:
                for colinear_2 in self.correlated_features[1]:
                    for colinear_3 in self.correlated_features[2]:
                        hotspot_list.append([colinear_1, colinear_2, colinear_3])
        else:
            for colinear_1 in self.correlated_features[0]:
                for colinear_2 in self.correlated_features[1]:
                    hotspot_list.append([colinear_1, colinear_2])

        hotspot_list.remove(self.threshold_indexes)
        return hotspot_list

### Accepts {X_space} a matrix of parameters, {y_space} a list of outputs, {y_cut} a cutoff value for classifying success, and class_weight
### Returns a list of threshold objects, one for each parameter in X
def threshold_generation(X_space, y_space, y_cut, class_weight, feature=-1, invert_y=False):
    if(invert_y):
        y_class = np.array([0 if i > y_cut else 1 for i in y_space])
    else:
        y_class = np.array([0 if i < y_cut else 1 for i in y_space])
    
    if(feature==-1):
        features = range(X_space.shape[1])
    else:
        features = feature
    
    all_thresholds = []
    for f_ind in features:
        dt = DecisionTreeClassifier(max_depth=1,class_weight=class_weight).fit(X_space[:,f_ind].reshape(-1, 1), y_class)

        #Turns the dt into a Threshold object
        accuracy = dt.score(X_space[:,f_ind].reshape(-1, 1), y_class)
        
        if(len(dt.tree_.children_left)>1):            
            ### If the amount of hits in the left subtree is greater than hits in the right subtree
            #if(X_space[best_performer_index,f_ind] < dt.tree_.threshold[0]):
            if(dt.tree_.value[1][0][1] > dt.tree_.value[2][0][1]):
                 operator = '<'
            else:
                operator = '>'
        else:
            operator = '>'
            
        temp_threshold = Threshold(
            dt, 
            f_ind, 
            dt.tree_.threshold[0], 
            operator, 
            accuracy, 
            metrics.f1_score(y_class,dt.predict(X_space[:,f_ind].reshape(-1, 1))),
            correlations=binary_corrmap[f_ind,:]
        )

        all_thresholds.append(temp_threshold)
    
    return all_thresholds

### returns the more accurate {percentage} of the input hotspots
def prune_hotspots(hotspots, percentage):
    accuracy_list=[]
    for hs in hotspots:
        accuracy_list.append(hs.accuracy)

    cut = np.percentile(accuracy_list, 100-percentage)
    
    hs_out=[]
    for hs in hotspots:
        if(hs.accuracy>cut):
            hs_out.append(hs)
    
    return hs_out

### Returns a list of all possible hotspots with an additional threshold added on from {hs}
### Doesn't repeat hotspots with different ordering, so it will give (x1, x2) but not (x2, x1)
def hs_next_extensive_thresholds(hs, class_weight):
    ### Makes all possible hotspots by adding one threshold
    all_hotspots = []
    for f_ind in range(hs.thresholds[-1].index+1, hs.X.shape[1]):
        dt = DecisionTreeClassifier(max_depth=1,class_weight=class_weight).fit(hs.X[:,f_ind].reshape(-1, 1), hs.y_class)

        #Turns the dt into a Threshold object
        accuracy = dt.score(hs.X[:,f_ind].reshape(-1, 1), hs.y_class)
        if(accuracy > .25):

            if(len(dt.tree_.children_left)>1):            
                ### If the amount of hits in the left subtree is greater than hits in the right subtree
                if(dt.tree_.value[1][0][1] > dt.tree_.value[2][0][1]):
                     operator = '<'
                else:
                    operator = '>'
            else:
                operator = '>'


            temp_threshold = Threshold(
                dt, 
                f_ind, 
                dt.tree_.threshold[0], 
                operator, 
                accuracy, 
                metrics.f1_score(hs.y_class,dt.predict(hs.X[:,f_ind].reshape(-1, 1))),
                correlations=binary_corrmap[f_ind,:]
            )

            temp_hs = copy.deepcopy(hs)
            temp_hs.add_threshold(temp_threshold)
            all_hotspots.append(temp_hs)


    return all_hotspots

### Accepts {hs_list} a list of hotspots and {n} the number of best hotspots desired
### Returns a list of the {n} most accurate hotspots from the list
def find_best_hotspots(hs_list, n, repetition_cut=5):
    best_hotspots = []
    repetitions = np.zeros(num_par)

    ### Repeat until you have n hotspots
    for i in range(n):        
        best_accuracy = 0
        best_hs = 0
        
        ### Check each hotspot in the list to see if it's the next best
        for hs in hs_list:
            unique=True
            for i in hs.threshold_indexes:
                if(repetitions[i]>repetition_cut):
                    unique=False
            if((hs.accuracy > best_accuracy) and unique):
                best_accuracy = hs.accuracy
                best_hs = hs

        if(best_hs not in hs_list):
            continue
        else:
            best_hotspots.append(best_hs)
            hs_list.remove(best_hs)
        
        for thresh in best_hs.thresholds:
            repetitions[thresh.index] = repetitions[thresh.index]+1
                      
    return best_hotspots

### Accepts {hs} a hotspot, {n} number of new hotspots, and class_weight
### Returns a list of {n} hotspots each with an additional threshold
def hs_next_thresholds(hs, class_weight):
    
    ### Makes all possible hotspots by adding one threshold
    all_hotspots = []
    for f_ind in range(hs.X.shape[1]):
        dt = DecisionTreeClassifier(max_depth=1,class_weight=class_weight).fit(hs.X[:,f_ind].reshape(-1, 1), hs.y_class)

        #Turns the dt into a Threshold object
        accuracy = dt.score(hs.X[:,f_ind].reshape(-1, 1), hs.y_class)

        if(len(dt.tree_.children_left)>1):            
            ### If the amount of hits in the left subtree is greater than hits in the right subtree
            if(dt.tree_.value[1][0][1] > dt.tree_.value[2][0][1]):
                 operator = '<'
            else:
                operator = '>'
        else:
            operator = '>'
            
            
        temp_threshold = Threshold(
            dt, 
            f_ind, 
            dt.tree_.threshold[0], 
            operator, 
            accuracy, 
            metrics.f1_score(hs.y_class,dt.predict(hs.X[:,f_ind].reshape(-1, 1))),
            correlations=binary_corrmap[f_ind,:]
        )

        temp_hs = copy.deepcopy(hs)
        temp_hs.add_threshold(temp_threshold)
        all_hotspots.append(temp_hs)

    return all_hotspots

### Train/Test split function to consolidate
def train_test_splits(X_sel,y_sel,labels_sel,split, test_ratio, randomstate=0):
    if split == "random":
        X_train_, X_test_, y_train_, y_test_ = train_test_split(
            X_sel, y_sel, random_state=randomstate+4, test_size=test_ratio)    
        TS = [np.argwhere(np.all(X==i,axis=1))[0,0] for i in X_train_]
        VS = [np.argwhere(np.all(X==i,axis=1))[0,0] for i in X_test_]

    elif split == "ks":
        import kennardstonealgorithm as ks
        TS,VS = ks.kennardstonealgorithm(X_sel,int((1-test_ratio)*np.shape(X_sel)[0]))
        X_train_, y_train_,X_test_, y_test_ = X_sel[TS], y_sel[TS],X_sel[VS], y_sel[VS]

        TS = [np.argwhere(np.all(X==i,axis=1))[0,0] for i in X_train_]
        VS = [np.argwhere(np.all(X==i,axis=1))[0,0] for i in X_test_]   

    elif split == "y_equidist":
        no_extrapolation = True

        import kennardstonealgorithm as ks
        if no_extrapolation:
            minmax = [np.argmin(y_sel),np.argmax(y_sel)]
            y_ks = np.array(([i for i in y_sel if i not in [np.min(y_sel),np.max(y_sel)]]))
            y_ks_indices = [i for i in range(len(y_sel)) if i not in minmax]

            # indices relative to y_ks:
            VS_ks,TS_ks = ks.kennardstonealgorithm(y_ks.reshape(np.shape(y_ks)[0],1),int((test_ratio)*(2+np.shape(y_ks)[0])))
            # indices relative to y_sel:
            TS_ = sorted([y_ks_indices[i] for i in list(TS_ks)]+minmax)
            VS_ = sorted([y_ks_indices[i] for i in VS_ks])

        else:
            VS_,TS_ = ks.kennardstonealgorithm(y_sel.reshape(np.shape(y_sel)[0],1),int((test_ratio)*np.shape(y_sel)[0]))

        X_train_, y_train_,X_test_, y_test_ = X_sel[TS_], y_sel[TS_],X_sel[VS_], y_sel[VS_]

        # indices relative to y
        TS = [np.argwhere(np.all(X==i,axis=1))[0,0] for i in X_train_]
        VS = [np.argwhere(np.all(X==i,axis=1))[0,0] for i in X_test_]

    elif split == "none":
        TS, VS = [i for i in range(X.shape[0]) if i not in exclude],[]
        X_train_, y_train_,X_test_, y_test_ = X[TS],y[TS],X[VS],y[VS]

    else: 
        raise ValueError("split option not recognized")

    return X_train_,y_train_,X_test_,y_test_,TS,VS

