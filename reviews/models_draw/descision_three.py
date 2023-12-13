import pandas as pd
import numpy as np
class Node():
    def __init_(self,feature_index,treshold,left,right,info_gain,value):
        # for descision node
        self.feature_index = feature_index
        self.treshold = treshold
        self.left = left    
        self.right = right
        self.info_gain = info_gain
        #for leaf node
        self.value = value

class DescisiontreeClassifier():
    #recursive function to build the tree
    def __init__(self,root,min_samples_split=2,max_deph=5):
        #initialize the root of the three
        self.root = root
        #stopping conditions
        self.min_samples_split = min_samples_split
        self.maxdeph = max_deph
    def build_three(self,dataset,curr_deph=0):
        # revisar com lucas 1
        x,y= dataset[:,:-1]
        numsamples,num_features = np.shape(x)
        # revisar com lucas 1

        # split until stopping conditions are met
        if numsamples>=self.min_samples_split and curr_deph<= self.maxdeph:
            #find best split
            best_split = self.get_best_split(dataset,numsamples,num_features)
            if best_split['info_gain']>0:
                # recur left
                left_subtree = self.build_three(best_split['data_left'],curr_deph+1)
                # recur right
                right_subtree = self.build_three(best_split['data_right'],curr_deph+1)
                return Node(best_split['feature_index'],best_split['treshold'],left_subtree,right_subtree,best_split['info_gain'],None)
        
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)
    def get_best_split(self,dataset,numsamples,num_features):

        best_split = {}
        max_info_gain = -float('inf')

        for feature_index in range(num_features):

            feature_values = dataset[:,feature_index]
            possible_tresholds = np.unique(feature_values)

            for treshold in possible_tresholds:

                data_left,data_right = self.split(dataset,feature_index,treshold)

                if len(data_left)>0 and len(data_right)>0:

                    y,y_left,y_right = dataset[:,-1],data_left[:,-1],data_right[:,-1]
                    curr_info_gain = self.information_gain(y,y_left,y_right,'gini')

                    if curr_info_gain>max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split['treshold'] = treshold
                        best_split['data_left'] = data_left
                        best_split['data_right'] = data_right
                        best_split['info_gain'] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split
    
    def split(self,dataset,feature_index,treshold):
        data_left = np.array([row for row in dataset if row[feature_index]<=treshold])

    