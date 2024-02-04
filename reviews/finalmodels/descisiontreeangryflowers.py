import numpy as np
class Node():
    def __init__(self,right,left,treshold,information_gain,value,feature_index):
        self.right = right
        self.left = left
        self.treshold = treshold
        self.information_gain = information_gain
        self.value = value
        self.feature_index = feature_index

class DecisionTreeClassifier():
    
    def __init__(self,root,min_samples_split=2,max_deph=5):
        self.root = root
        self.min_samples_split = min_samples_split
        self.max_deph = max_deph
        self.tree = None
        self.X = None
        self.y = None
        self.classes = None
        self.features = None
    
    def fit(self,X,y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.features = np.arange(X.shape[1])
        self.tree = self.__create_tree(X,y,0)
   
    def __create_tree(self,X,y,deph):
        if deph >= self.max_deph or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return Node(None,None,None,None,self.__most_common(y),None)
        best_feature,best_treshold,best_information_gain = self.__best_criteria(X,y)
        left_indices,right_indices = self.__split(X[:,best_feature],best_treshold)
        left = self.__create_tree(X[left_indices],y[left_indices],deph+1)
        right = self.__create_tree(X[right_indices],y[right_indices],deph+1)
        return Node(right,left,best_treshold,best_information_gain,self.__most_common(y),best_feature)
    
    def __best_criteria(self,X,y):
        best_gain = -1
        best_treshold = None
        best_feature = None
        for feature_index in self.features:
            feature_values = X[:,feature_index]
            tresholds = np.unique(feature_values)
            for treshold in tresholds:
                gain = self.__information_gain(y,feature_values,treshold)
                if gain > best_gain:
                    best_gain = gain
                    best_treshold = treshold
                    best_feature = feature_index
        return best_feature,best_treshold,best_gain

        