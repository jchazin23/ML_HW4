# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:08:38 2016

@author: jchazin
"""

from sklearn import tree

features = [[140,1],[130,1],[150,0],[170,0]]
labels = [0,0,1,1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)

print clf.predict([[150,0]])