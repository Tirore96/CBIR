import numpy as np
import scipy.spatial.distance as spdist

class NaiveFeatures:
    def extractFeatures(self,img):      
         return np.mean(img)
        
    def distance(self,vec1,vec2):
         return abs(vec1-vec2)#spdist.euclidean(vec1,vec2)
  
    def extractFromBatch(self,batch):
        self.matrix = []
        for i in batch:
            self.matrix.append(self.extractFeatures(i))
     
    def query(self,query_img,k):
        if self.matrix is []:
            print("empty feature matrix")
            return []
        query_img_features = self.extractFeatures(query_img)
        distances_and_indices = []
        for index,database_img_features in enumerate(self.matrix):
            distance = self.distance(query_img_features,database_img_features)
            distances_and_indices.append((distance,index))
        distances_and_indices.sort(key=lambda pair:pair[0])
        relevant_dist_and_indices = distances_and_indices[:k]
        retval_indices = []
        retval_dist = []
        for i in relevant_dist_and_indices:
            dist,index = i
            retval_indices.append(index)
            retval_dist.append(dist)
        return retval_indices,retval_dist           