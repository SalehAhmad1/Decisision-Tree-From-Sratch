import numpy as np
import pandas as pd
from copy import deepcopy
import os


def powerset(iterable):
    from itertools import chain, combinations
    xs = list(iterable)
    return chain.from_iterable( combinations(xs,n) for n in range(len(xs)+1) )

def get_powerset(iterable, length):
    subsets=set(powerset(iterable))
    ss=set([frozenset(s) for s in subsets if len(s)>=1 and len(s)<=length])
    return ss

def getSplits(categories):
    categories = set(categories)
    tsplits = get_powerset(categories,len(categories)-1)
    flist=[]
    for s in tsplits:
        if not s in flist:
            r = categories.difference(s)
            flist.append(s)
            flist.append(r)

    olist=[]
    for s in flist[::-1]:
        ilist=[]
        for k in s:
            ilist.append(k)
        olist.append(tuple(ilist))    
    return olist

class Node:
    def __init__(self,purity,variance,mean=-1,fidx=-1, split=-1):
        self.lchild,self.rchild = None,None     
        self.split = split
        self.variance = variance
        self.mean = mean #Leaf Nodes will have mean
        self.fidx = fidx
        self.purity = purity
        self.ftype = 'categorical' if type(self.split) in [tuple, str, np.string_] else 'continuous'

    def set_childs(self,lchild,rchild):
        self.lchild = lchild
        self.rchild = rchild

    def isleaf(self):
        if self.mean != None:
            return True
        else:
            return False

    def isless_than_eq(self, X):
        if self.ftype == 'categorical':
            if X[self.fidx] in self.split:
                return True
            else:
                return False
        else:
            if X[self.fidx] <= self.split:
                return True
            else:
                return False

    def get_str(self):        
        if self.isleaf():
            return 'L(Mean={},Purity={})'.format(self.mean,self.purity)
        else:
            return 'I(Fidx={},Purity={},Split={})'.format(self.fidx,self.purity,self.split)
    
class DecisionTreeRegressor:
    '''Implements the Decision Tree For Regression'''
    def __init__(self,Path = None):
        self.tree = None   
        self.purity = 0.0
        self.exthreshold = 0
        self.maxdepth = 0
        self.path = Path

    def __init__(self, purityp=0.05, exthreshold=5, maxdepth=10, Path = None):        
        self.purity = purityp
        self.exthreshold = exthreshold
        self.maxdepth = maxdepth
        self.path = Path
        self.tree = []

    def fit(self, X, Y):
        self.tree = self.build_tree(X,Y,self.maxdepth)
        self.Save_Model(self.path)

    def Save_Model(self,Path = None):
        if Path == None:
            Path = os.getcwd()
        print("Model Saved at: ",Path)
        np.save(Path,self.tree)

    def build_tree(self, X, Y, depth):
        nexamples, nfeatures = X.shape
        
        VarianceOfY = np.var(Y)
        Purity = VarianceOfY

        if ((nexamples < self.exthreshold) or (Purity < self.purity) or (depth < 0)):
            node = Node(Purity,-1,np.mean(Y))
        else:
            BEST_SP,BEST_VAR,BEST_LD,BEST_RD = None,None,None,None
            SP,VR,LD,RD= None,None,None,None
            Threshold = float("inf")
            Best_Feature = -1
            
            for FeatureIndex in range(nfeatures):
                if np.dtype(X[0,FeatureIndex]) in [tuple, str, np.string_]:
                    if len(np.unique(X[:,FeatureIndex])) == 1:
                        continue
                    else:
                        SP,VR,LD,RD = self.evaluate_categorical_attribute(deepcopy(X[:,FeatureIndex]), deepcopy(Y))
                else:
                    SP,VR,LD,RD = self.evaluate_numerical_attribute(deepcopy(X[:, FeatureIndex]), deepcopy(Y))
                
                if  VR != None and VR < Threshold:
                    Best_Feature = FeatureIndex
                    BEST_SP,BEST_VAR,BEST_LD,BEST_RD = SP,VR,LD,RD
                    Threshold = BEST_VAR
                    
            node = None
            if BEST_VAR == None: #Leaf Node
                node = Node(Purity,-1,np.mean(Y)) #Purity, Variance, Mean
            else: #Internal Node
                node = Node(Purity,BEST_VAR,None,Best_Feature,BEST_SP) #Purity, Variance, NO Mean cauz leaf node will have mean only, FeatureIndex, Split
                node.set_childs(self.build_tree(deepcopy(X[BEST_LD]), deepcopy(Y[BEST_LD]), depth - 1), self.build_tree(deepcopy(X[BEST_RD]), deepcopy(Y[BEST_RD]), depth - 1))
        return node
        
    def Test(self, X):
        if (self.path == None) and (self.tree == None or self.tree == []):
            print("No Model Found")
            return
        else:
            self.tree = np.load(self.path,allow_pickle=True)
            Pred = self.predict(X)
            return np.array(Pred)

    def evaluate_categorical_attribute(self, feat, Y):
            categories = set(feat)
            splits = getSplits(categories)
            Vaiance_Matrix = []
            for idx,LeftSplit in range(0,len(splits),2):
                RightSplit = LeftSplit + 1

                LeftY = Y[np.isin(feat, splits[LeftSplit])]
                RightY = Y[np.isin(feat, splits[RightSplit])]

                LeftVariance = np.var(LeftY)
                RightVariance = np.var(RightY)

                Vaiance_Matrix.append(LeftVariance+RightVariance)

            LeftSplits = splits[0::2]
            RightSplits = splits[1::2]
            BestSplitIndex = np.argmin(Vaiance_Matrix)
            BestSplit = LeftSplits[BestSplitIndex]
            LeftDataIndices = np.isin(feat, LeftSplits[BestSplitIndex])
            RightDataIndices = np.isin(feat, RightSplits[BestSplitIndex])
            
            return BestSplit, Vaiance_Matrix[BestSplitIndex], LeftDataIndices, RightDataIndices
        
    def evaluate_numerical_attribute(self, feat, Y):
        UniquesInF = np.unique(sorted(feat))
        SplitPoints = []
        for i in range(len(UniquesInF)-1):
            SplitPoints.append((UniquesInF[i] + UniquesInF[i+1])/2)
        
        Vaiance_Matrix = []
        for SplitPoint in SplitPoints:
            LeftY = Y[feat <= SplitPoint]
            RightY = Y[feat > SplitPoint]

            Left_Variance = np.var(LeftY)
            Right_Variance = np.var(RightY)

            Vaiance_Matrix.append(Left_Variance+Right_Variance)
        
        BestSplitIndex = np.argmin(Vaiance_Matrix)
        BestSplit = SplitPoints[BestSplitIndex]
        LeftDataIndices = np.where(feat <= BestSplit)[0]
        RightDataIndices = np.where(feat > BestSplit)[0]
        return BestSplit, Vaiance_Matrix[BestSplitIndex], LeftDataIndices, RightDataIndices

    def predict(self, X):
        z = []
        for idx in range(X.shape[0]):
            z.append(self._predict(self.tree, X[idx, :]))
        return z 
    
    def _predict(self, node, X):
        TreeRootTemp = node
        while TreeRootTemp.isleaf() == False:
            if TreeRootTemp.isless_than_eq(X) == True:
                TreeRootTemp = TreeRootTemp.lchild
            else:
                TreeRootTemp = TreeRootTemp.rchild
        return TreeRootTemp.mean #The leaf node at which the new test example fits best will have the mean of the target variable as the prediction

    def __str__(self):
        str = '---------------------------------------------------'
        str += '\n A Decision Tree With Depth={}'.format(self.find_depth())
        str += self.__print(self.tree)
        str += '\n---------------------------------------------------'
        return self.__print(self.tree)        
   
    def find_depth(self):
        return self._find_depth(self.tree)

    def _find_depth(self, node):
        if not node:
            return
        if node.isleaf():
            return 1
        else:
            return max(self._find_depth(node.lchild), self._find_depth(node.rchild)) + 1

    def __print(self, node, depth=0):
        ret = ""
        if node.rchild:
            ret += self.__print(node.rchild, depth + 1)
        ret += "\n" + ("    "*depth) + node.get_str()
        if node.lchild:
            ret += self.__print(node.lchild, depth + 1)
        return ret

