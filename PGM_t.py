# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:05:02 2018

@author: aumale
"""
# Import libraries
import numpy as np
import pandas as pd
import json as js

from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BdeuScore
from pgmpy.inference import VariableElimination

class PGM_t :    
    
    def __init__(self, data_rel_path = './Dataset/'):
        self.data_rel_path = data_rel_path
        with open(self.data_rel_path+'states.json') as fp:
            self.states_all = js.load(fp)

        with open(self.data_rel_path+'states_9.json') as fp:
            self.states_9 = js.load(fp)

    def match_dict(self, dict1, dict2,weighted=False):
        incorrect_match = 0
        for key in (set(dict1) & set(dict2)):
            if dict1[key] != dict2[key]:
                incorrect_match +=1  
        if weighted :
            return (len(set(dict1) & set(dict2)) - incorrect_match)/len(set(dict1) & set(dict2))
        else :
            if incorrect_match > 0 : return 0
            else : return 1

    def chk_accuracy(self, data,est_obj, variables=[], evidence=[],sample_size = 100):
        test = data.sample(n=sample_size, random_state=1)
    
        correct = 0
        for index,observation in test.iterrows():
#            print(observation[evidence].to_dict())
            prob_same = est_obj.map_query(variables, observation[evidence].to_dict())
#            print(prob_same, observation[variables].to_dict())
            correct += self.match_dict(prob_same, observation[variables].to_dict(),weighted = True)
        accuracy = correct/sample_size
        return (accuracy)


    def load_features(self, filepath = 'AND_Features.csv', start = 1, end = 11,seperated = False):
        filepath = self.data_rel_path+'AND_Features.csv'
        # Load And Features from csv file and preprocess to generate training and testing sets
        self.img_features = pd.read_csv(filepath, header = 0, usecols = range(start,end))
        self.img_features.sort_values(by=['ImageId'])
        self.img_features['owner'] = self.img_features.ImageId.apply(lambda x: int(x[:-1]))
        self.img_features['key'] = 1
        self.img_features.columns = list(pd.Series(self.img_features.columns.values[0:-1]).apply(lambda x :  x + '_a' )) + list(self.img_features.columns.values[-1:])
        img_features_copy = self.img_features.copy()
        img_features_copy.columns = list(pd.Series(img_features_copy.columns.values[0:-1]).apply(lambda x :  x[0:-2] + '_b' )) + list(img_features_copy.columns.values[-1:])
        
        # Cartesian product of feature with itself to give dataset of 18 variables
        new_set = pd.merge(self.img_features, img_features_copy, on = 'key')[list(self.img_features.columns[:-1]) + list(img_features_copy.columns[:-1])]
        # Add variable 'same' to identify if writer is same, 0 is for same writer
        new_set['same'] = np.where(new_set['owner_a'] == new_set['owner_b'], 0 , 1)
        
        final_set = new_set.drop(['owner_a','owner_b', 'ImageId_a','ImageId_b'], axis = 1)
        #final_set.to_csv(path_or_buf='./Full_set.csv', sep=',', index = False)
        
        # Get the training data set, used to learn the structure of network
        train_set = final_set.sample(frac=.8,random_state=1)
        train_set_same = train_set[train_set['same']==0]
        train_set_not_same = train_set[train_set['same']==1]
        
        # Get the testing data set, used to measure accuracy of variable inference
        test_set = final_set[~final_set.index.isin(train_set.index)]
        test_set_same = test_set[test_set['same']==0]
        test_set_not_same = test_set[test_set['same']==1]
    
        if seperated:
            return train_set_same, train_set_not_same, test_set_same, test_set_not_same
        else:
            return train_set, test_set
    

def main():
    
    andPGM = PGM_t()
    print('loading features..')
    train_set, test_set = andPGM.load_features()
    print('loading features.. Done')
    # Bayesian network of 19 nodes, 9*2 variables of network given 
    # Initial incomplete Bayesian model connected manually based on intuition
    print('Generating model.. ')
    initialModel =  BayesianModel({})
    initialModel.add_nodes_from(andPGM.img_features.columns[1:10].tolist())
    initialModel.add_edges_from([('f6_a' , 'f2_a'),\
                             ('f3_a' , 'f4_a') ,\
                             ('f5_a' , 'f9_a') ,\
                             ('f4_a' , 'f7_a') ])            
    
    # Use hill climb search algorithm to find network structure of initial 9 nodes
    hc = HillClimbSearch(data=andPGM.img_features.iloc[0:,1:10], \
                         scoring_method=BdeuScore(andPGM.img_features.iloc[0:,1:10], \
                                                  equivalent_sample_size=0.1*len(andPGM.img_features)), \
                         state_names = andPGM.states_9)
    # Get best estimated structure                     
    best_model = hc.estimate(start=initialModel)
    # Edges in the acquired graph
    print('model of 9 var: ', best_model.edges())
    
    # Create a Clone of generated Bayesian network structure
    clone_model = BayesianModel({})
    for edge in best_model.edges():
        new_edge = [edge[0][:-1]+'b', edge[1][:-1]+'b']
        clone_model.add_edges_from([new_edge])
    
    # Join together the Original and clone network through node 'same'
    multinetModel =  BayesianModel({})
    multinetModel.add_edges_from(best_model.edges() + clone_model.edges())
    multinetModel.add_node('same')
    multinetModel.add_edge('f5_a', 'same')
    multinetModel.add_edge('f9_a', 'same')
    multinetModel.add_edge('f5_b', 'same')
    multinetModel.add_edge('f9_b', 'same')
    print('Generating model.. Done')
    # Edges in the final structure
    print('Final model: ', multinetModel.edges())
    
    print('Fit data into model..')
    # fit the data to model to generate CPDs using maximum likelyhood estimation 
    multinetModel.fit(data=train_set, state_names=andPGM.states_all  ) 
    print('Fit data into model.. Done')
    print('CPDs generated: ')
    cpds = multinetModel.get_cpds()
    for cpd in cpds:
        print(cpd)
    # Inference using Variable Elimination
    print('Start inference..')
    inference = VariableElimination(multinetModel)
    train_set_same = train_set[train_set['same']==0]
    train_set_not_same = train_set[train_set['same']==1]
    
    # Accuracy of positive inferences
    acc_same = andPGM.chk_accuracy(train_set_same, inference,
                                   variables=train_set_same.columns[0:9].tolist(),
                                   evidence=train_set_same.columns[9:19].tolist() )
    print('accuracy of positives ', acc_same )
    
    # Accuracy of negative inferences
    acc_nt_same = andPGM.chk_accuracy(train_set_not_same, inference,
                                      variables=train_set_not_same.columns[0:9].tolist(),
                                      evidence=train_set_not_same.columns[9:19].tolist())
    print('accuracy of negatives', acc_nt_same)
    
  
if __name__ == "__main__":
    main()
    

