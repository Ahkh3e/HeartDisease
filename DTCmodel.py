import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree,model_selection

def DTC(train):


    target = train["DEATH_EVENT"].values
    features_names = ["age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction","high_blood_pressure","platelets","serum_creatinine","serum_sodium","sex","smoking","time"]
    features = train[features_names].values

    generalized_tree = tree.DecisionTreeClassifier(
        random_state = 1,
        max_depth = 7,
        min_samples_split = 2
    )

    generalized_tree_ = generalized_tree.fit(features, target)

    print(generalized_tree_.score(features,target))

    scores = model_selection.cross_val_score(generalized_tree, features,target, scoring = 'accuracy' , cv =50)
    print (scores)
    print (scores.mean())

    #tree.export_graphviz(generalized_tree_ , feature_names = features_names, out_file = 'tree.dot')
