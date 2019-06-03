from abc import ABC, abstractmethod
import numpy as np
import pickle
import cv2
import scipy.spatial.distance as spdist
import mahotas as mh
import sklearn 
import scipy
import random
import skimage.feature as ft
from sklearn.metrics import roc_auc_score
from time import time
from functools import wraps
def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print ('Elapsed time: {}'.format(end-start))
        return result
    return wrapper

class ComparisonModel(ABC):
    @abstractmethod
    def extractFeatures(self,img): pass
    
    @abstractmethod
    def measureDistance(self,img_1_features,img_2_features): pass
    
    
class SimpleModel(ComparisonModel):
    def extractFeatures(self,img):
        return np.mean(img)
        
    def measureDistance(self,img_1_features,img_2_features):
        return abs(img_1_features-img_2_features)
    
    
    
    
class AKAZE:
    #Source: https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
    def extractFeatures(self,img,vector_size=32):
        try:
            alg = cv2.AKAZE_create()
            kps = alg.detect(img)
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
            kps, dsc = alg.compute(img, kps)
            if dsc is None:
                return np.zeros(vector_size*64)
            #print(dsc)
            dsc = dsc.flatten()
            needed_size = (vector_size * 64)
            if dsc.size < needed_size:
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        except cv2.error as e:
            print ('Error: ',e) 
            return None
        return dsc
            
    def extractFromBatch(self,batch,pickled_db_path="AKAZEfeatures.pck"):
        self.matrix = []
        for i in batch:
            self.matrix.append(self.extractAKAZEFeatures(i))
        
        self.matrix = np.asarray(self.matrix)
        with open(pickled_db_path,'wb') as fp:
            pickle.dump(self.matrix,fp)
            
    def loadFeatures(self,pickled_db_path="AKAZEfeatures.pck"):
        with open(pickled_db_path,'rb') as fp:
            self.matrix= pickle.load(fp)
 
    def cos_cdist(self,vector):
        v = vector.reshape(1,-1)
        return spdist.cdist(self.matrix,v,'cosine').reshape(-1)
    



    
    def query(self,img,k):
        features = self.extractFeatures(img)
        img_distances = self.cos_cdist(features)
        
        nearest_indices = np.argsort(img_distances)[:k].tolist()
        zero_lists = []
        for i in nearest_indices:
            if self.matrix[i][0] == 0:
                zero_lists.append(i)
        return nearest_indices, img_distances[nearest_indices].tolist(),zero_lists
                                                     


class Haralicks:
    def __init__(self,compute_dog=False):
        self.compute_dog = compute_dog
        self.matrix = []
        
        self.selected_features = [0,1,2,5,6]
        
        self.haralick_labels = ["Angular Second Moment",
                       "Contrast",
                       "Correlation",
                       "Sum of Squares: Variance",
                       "Inverse Difference Moment",
                       "Sum Average",
                       "Sum Variance",
                       "Sum Entropy",
                       "Entropy",
                       "Difference Variance",
                       "Difference Entropy",
                       "Information Measure of Correlation 1",
                       "Information Measure of Correlation 2",
                       "Maximal Correlation Coefficient"]
        
    
    def extractFeatures(self,img,feature_indices=None):      
        if self.compute_dog:
            img = mh.dog(img)
        features= mh.features.haralick(img).mean(0)
        if feature_indices == None:
            feature_indices = self.selected_features
        retval = [features[i] for i in feature_indices]
        return retval
    
    def distance(self,vec1,vec2):
        return scipy.spatial.distance.euclidean(vec1,vec2)
    
    def extractFromBatch(self,batch,feature_indices=None, pickled_db_path="haralickfeatures.pck"):
        matrix = []
        for i in batch:
            matrix.append(self.extractFeatures(i,feature_indices))
#            print(i.shape)
        with open(pickled_db_path,'wb') as fp:   
            pickle.dump(matrix,fp)      
        self.matrix = matrix
    
    def loadFeatures(self,pickled_db_path="haralickfeatures.pck"):
        with open(pickled_db_path,'rb') as fp:
            self.matrix = pickle.load(fp)
        
            
    def countClassificationRatio(self,label_indices,label_name,is_positive,label_value,labels):
        hit = 0
        for index in label_indices:
            sample_is_positive = labels[index][label_name] == label_value
            sample_and_query_same = sample_is_positive == is_positive
            if sample_and_query_same:
                hit += 1
                
        ratio = hit / (len(label_indices))
#        return ratio
        if is_positive:
            return ratio
        else:
            return 1.0 - ratio
#        if not is_positive:
#            ratio = 1 - ratio
#        return ratio              
#    def countClassificationRatio(self,label_indices,label_name,label_value,labels):
#        relevant_count = 0
#        non_relevant_count = 0 
#        for index in label_indices:
#            if labels[index][label_name] == label_value:
#                relevant_count += 1
#            else:
#                non_relevant_count += 1
#                
#        return relevant_count / (len(label_indices))
    
    def featureSelector(self,size,K,scans,labels,relevant_label,label_value):
        selection = []
        count = len(scans)
        

        binary_labels = [1 if i[relevant_label]==label_value else 0 for i in labels ]       
        best_auc = 0.0
        first_iter = True
        auc_up = None
        while len(selection) < size:
            index_auc = []
            for i in range(13):
#                print(i)
                if i in selection:
                    continue
                temp_selection = selection + [i]
                probabilities = np.zeros((count),dtype=np.float)
                #format database to same vector size
                self.extractFromBatch(scans,feature_indices=temp_selection)
                for m_iter in range(count):
                    retrieved_indices,distances = self.query(scans[m_iter],K+1,feature_indices=temp_selection)           
                    retrieved_indices = retrieved_indices[1:]           
        #            thresh_index = 0
        #            threshold = 0.0
        #            while threshold < 1.0:
                    ratio = self.countClassificationRatio(retrieved_indices,relevant_label,labels[m_iter][relevant_label]==label_value,label_value,labels)
#                    ratio = self.countClassificationRatio(retrieved_indices,relevant_label,labels[m_iter][relevant_label],labels)
#                    if not labels[m_iter][relevant_label] == label_value:
#                        ratio = 1-ratio
                    probabilities[m_iter] = ratio
                
#                for j in range(len(binary_labels)):
#                    print(probabilities[j],binary_labels[j])
                auc = roc_auc_score(binary_labels,probabilities)   
#                print(i,auc)
                index_auc.append((i,auc))
    
#                print(i,probabilities)
            index_auc.sort(reverse=True,key=lambda k: k[1])
#            print(index_auc)
            print(index_auc)
            best_feature_l = index_auc[0][0]
            local_best_auc_l = index_auc[0][1]
#            best_feature_r = index_auc[-1][0]
#            local_best_auc_r = index_auc[-1][1]

#            if first_iter:
#                if abs(best_auc- local_best_auc_l) >= abs(best_auc- local_best_auc_r):
#                    auc_up = True
#                else:
#                    auc_up = False
#                first_iter = False
#            if auc_up and local_best_auc_l <= best_auc or not auc_up and local_best_auc_r >= best_auc:
            if local_best_auc_l <= best_auc: # or not auc_up and local_best_auc_r >= best_auc:
                break
            else:
#                if auc_up:
                best_auc = local_best_auc_l
                selection.append(best_feature_l)
#
#                else:
#                    best_auc = local_best_auc_r
#                    selection.append(best_feature_r)
            
#        selection.sort()

        if len(selection) == 0:
            selection = [0]
        self.selected_features = selection
        self.matrix = []
        print(selection)
        print("finished Haralick Feature selection")
#               
        
        
    def query(self,query_img,k,feature_indices=None):
#        print(len(self.matrix))
        if self.matrix is []:
            print("empty feature matrix")
            return []
        query_img_features = self.extractFeatures(query_img,feature_indices)
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
            
        
class LBP3DFeatures:
    def __init__(self,params_args):
        self.lbp_xy = LBPFeatures(params_args[0][0],params_args[0][1])
        self.lbp_xz = LBPFeatures(params_args[1][0],params_args[1][1])       
        self.lbp_yz = LBPFeatures(params_args[2][0],params_args[2][1])              
        
    def extractFromBatch(self,scans,n_points=None,radius=None):
        retval = []
        for scan in scans:
            h = self.extractFeatures(scan,n_points,radius)
            retval.append(h)
        self.extracted = retval       
        
    def prepareSlices(self,scans):
        shape = scans[0].shape
        y_mid = round(shape[0]/2)
        x_mid = round(shape[1]/2)       
        z_mid = round(shape[2]/2)              
        imgs_xy = []
        imgs_xz = []       
        imgs_yz = []           
        for scan in scans:
            img_xy = scan[:,:,z_mid]
            img_xz = scan[y_mid,:,:]       
            img_yz = scan[:,x_mid,:]                            
            imgs_xy.append(img_xy)
            imgs_xz.append(img_xz)           
            imgs_yz.append(img_yz)               
        return imgs_xy,imgs_xz,imgs_yz
    def fit_parameters(self,K,scans,labels,relevant_label,label_value,radius_max=15,multiple_max=10):
        imgs_xy,imgs_xz,imgs_yz = self.prepareSlices(scans)

        self.lbp_xy.fit_parameters(K,imgs_xy,labels,relevant_label,label_value,radius_max,multiple_max)
        self.lbp_xz.fit_parameters(K,imgs_xz,labels,relevant_label,label_value,radius_max,multiple_max)       
        self.lbp_yz.fit_parameters(K,imgs_yz,labels,relevant_label,label_value,radius_max,multiple_max)              
        print("Finished LBP3D fitting")
        
    def extractFeatures(self,scan,n_points=None,radius=None):
        #[y,x,z]
        shape = scan.shape
        y_mid = round(shape[0]/2)
        x_mid = round(shape[1]/2)       
        z_mid = round(shape[2]/2)              
        img_xy = scan[:,:,z_mid]
        img_xz = scan[y_mid,:,:]       
        img_yz = scan[:,x_mid,:]              
        hist_xy = self.lbp_xy.extractFeatures(img_xy,n_points,radius)
        hist_xz = self.lbp_xz.extractFeatures(img_xz,n_points,radius)       
        hist_yz = self.lbp_yz.extractFeatures(img_yz,n_points,radius)              
        hist_concat = np.append(hist_xy,hist_xz)
        hist_concat = np.append(hist_concat,hist_yz)       
        return hist_concat
          
        
    def query(self,img,k,n_points=None,radius=None):
        index_distance = []
        h_query = self.extractFeatures(img,n_points,radius)
        for index,h_batch in enumerate(self.extracted):
            diff = self.kullback_leibler_divergence(h_query,h_batch)
            index_distance += [(index,diff)]
        index_distance.sort(key= lambda k:k[1])
        index = []
        distance = []
        for i_d in index_distance:
            i,d = i_d
            index += [i]
            distance += [d]
        return index[:k],distance[:k]              
        
    def kullback_leibler_divergence(self,p, q):
        p = np.asarray(p)
        q = np.asarray(q)
        min_length = min(len(p),len(q))
        p = p[:min_length]
        q = q[:min_length]
        filt = np.logical_and(p != 0, q != 0)
        return np.sum(p[filt] * np.log2(p[filt] / q[filt]))       
        
class LBPFeatures:
    def __init__(self,n_points=None,radius=None):
        self.radius = radius
        self.n_points = n_points

    #experiment with "uniform"
    @timing
    def fit_parameters(self,K,scans,labels,relevant_label,label_value,radius_max=15,multiple_max=10):
        binary_labels = [1 if i[relevant_label]==label_value else 0 for i in labels ]              
        params_auc = []
        count = len(scans)
        for radius in range(1,radius_max):
            for multiple in range(1,multiple_max):
                n_points = radius * multiple
                self.extractFromBatch(scans,n_points,radius)
                probabilities = np.zeros((count),dtype=np.float)               
                for m_iter in range(count):
                    retrieved_indices,distances = self.query(scans[m_iter],K+1,n_points=n_points,radius=radius)           
                    retrieved_indices = retrieved_indices[1:]           
        #            thresh_index = 0
        #            threshold = 0.0
        #            while threshold < 1.0:
                    ratio = self.countClassificationRatio(retrieved_indices,relevant_label,labels[m_iter][relevant_label]==label_value,label_value,labels)
#                    if not labels[m_iter][relevant_label] == label_value:
#                        ratio = 1-ratio

                    probabilities[m_iter] = ratio
                
#                print(probabilities)
                auc = roc_auc_score(binary_labels,probabilities)   
                print((n_points,radius),auc)
                params_auc.append(((n_points,radius),auc))
                if auc == 1.0:
                    break
            if auc == 1.0:
                break
        params_auc.sort(reverse=True,key=lambda k: k[1])
        print(params_auc)
        best_n_points,best_radius = params_auc[0][0]
        self.n_points = best_n_points
        self.radius = best_radius
        print("finished LBP fitting",self.radius,self.n_points,params_auc[0][1])
        self.extracted = []
    
    def countClassificationRatio(self,label_indices,label_name,is_positive,label_value,labels):
        hit = 0
        for index in label_indices:
            sample_is_positive = labels[index][label_name] == label_value
            sample_and_query_same = sample_is_positive == is_positive
            if sample_and_query_same:
                hit += 1
                
        ratio = hit / (len(label_indices))
        if is_positive:
            return ratio
        else:
            return 1.0 - ratio       
#        return ratio
#        if not is_positive:
#            ratio = 1 - ratio
#        return ratio   
    
#    def countClassificationRatio(self,label_indices,label_name,label_value,labels):
#        relevant_count = 0
#        non_relevant_count = 0 
#        for index in label_indices:
#            if labels[index][label_name] == label_value:
#                relevant_count += 1
#            else:
#                non_relevant_count += 1
#                
#        return relevant_count / (len(label_indices))               
                
                
    def extractFromBatch(self,scans,n_points=None,radius=None):
        if n_points == None:
            n_points = self.n_points
        if radius == None:
            radius = self.radius
        retval = []
        for scan in scans:
            h = self.extractFeatures(scan,n_points,radius)
            retval.append(h)
        self.extracted = retval       
    
    def extractFeatures(self,img,n_points,radius):
        eps = 0.001
        if n_points == None:
            n_points = self.n_points
        if radius == None:
            radius = self.radius       
#        print(n_points,radius)
        #http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
        lbp = ft.local_binary_pattern(img, n_points, radius, 'uniform')
        n_bins = n_points**2 + 1#int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))       
        hist = np.divide(hist,sum(hist))
        hist = np.add(hist,eps)
        hist = np.divide(hist,sum(hist))       
        return np.asarray(hist)
        
    
    #http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html   
#    def kullback_leibler_divergence(self,p, q):
#        p = np.asarray(p)
#        q = np.asarray(q)
#        #min_length = min(len(p),len(q))
#        #p = p[:min_length]
#        #q = q[:min_length]
#        filt = np.logical_and(p != 0, q != 0)
#        return np.sum(p[filt] * np.log2(p[filt] / q[filt]))
    
    def j_divergence(self,p,q):
        return self.kullback_leibler_divergence(p,q) + self.kullback_leibler_divergence(q,p)
    
    def kullback_leibler_divergence(self,p, q):
#        p = np.asarray(p)
#        q = np.asarray(q)
        #min_length = min(len(p),len(q))
        #p = p[:min_length]
        #q = q[:min_length]
        #filt = np.logical_and(p != 0, q != 0)
        return np.sum(p * np.log2(p / q))       
    #default 8 and 30
    def query(self,img,k,n_points=None,radius=None):
        if n_points == None:
            n_points = self.n_points
        if radius == None:
            radius = self.radius       
        index_distance = []
        h_query = self.extractFeatures(img,n_points,radius)
        for index,h_batch in enumerate(self.extracted):
             diff = self.j_divergence(h_query,h_batch)           
#            diff = self.kullback_leibler_divergence(h_query,h_batch)
             index_distance += [(index,diff)]
        index_distance.sort(key= lambda k:k[1])
        index = []
        distance = []
        for i_d in index_distance:
            i,d = i_d
            index += [i]
            distance += [d]
        return index[:k],distance[:k]              
        
        
        
class EdgeFeatureHelper:
    def __init__(self,compute_dog):
        self.compute_dog = compute_dog
        self.akaze = cv2.AKAZE_create()
        self.haralicks = Haralicks(compute_dog=compute_dog)
        
    def imgWithKeypoints(self,img):
        kpts = self.keypoints(img)
        for kpt in kpts:
            a,b = kpt.pt
            a = int(a)
            b = int(b)   
            img[a,b] = 255 
        return img,kpts
    
    def keypoints(self,img):
        img = np.asarray(img,dtype=np.uint8)
        kpts = self.akaze.detect(img, None)
        kpts.sort(key= lambda k: k.response,reverse=True)
        kpts = kpts[:10]   
        return kpts
 
    def haralickFromKeypoint(self,img,kpt):
        x,y = list(map(lambda val: int(val),kpt.pt))
        area = int(kpt.size//2)
        extract_region = img[x-area:x+area,y-area:y+area]
        return self.haralicks.extractFeatures(extract_region)                   
    
    def keypointDistance(self,kpts1,kpts2):
        difference = 0
        for i in range(len(kpts1)):
            resp1 = kpts1[i].response
            size1 = kpts1[i].size
            resp2 = kpts2[i].response
            size2 = kpts2[i].size           
            resp_diff = abs(resp1-resp2)
            size_diff = abs(size1-size2)
            temp = (resp_diff**2)**(0.5)
            difference= temp
        return difference   
 
    def euclideanDistance(self,vec1,vec2):
        return scipy.spatial.distance.euclidean(vec1,vec2)   
    
    def crossDistance(self,mat1,mat2):
        retval_distance = -1
        for row1 in mat1:
            for row2 in mat2:
                dist = scipy.spatial.distance.euclidean(row1,row2)                     
                if retval_distance == -1 or retval_distance > dist:
                    retval_distance = dist
        return retval_distance
    
    def extractFeaturesWithHaralick(self,img):
        kpts = self.keypoints(img)
        features = np.asarray(self.haralickFromKeypoint(img,kpts[0])).reshape(-1,5)
        if self.compute_dog:
            img = mh.dog(img)
        for kpt_index in range(1,len(kpts)):
            new_features = np.asarray(self.haralickFromKeypoint(img,kpts[kpt_index])).reshape(-1,5)
            features = np.append(features,new_features,axis=0)
        return features
    
    

class EdgeFeatures:
    def __init__(self,haralick_from_kpt=True,compute_dog=False):
        self.haralick_from_kpt = haralick_from_kpt 
        self.helper = EdgeFeatureHelper(compute_dog)
    
    def extractFeatures(self,img):
        if self.haralick_from_kpt:
            return self.helper.extractFeaturesWithHaralick(img)
        else:
            return self.helper.keypoints(img)
   
    def extractFromBatch(self,scans):
        self.extracted = [self.extractFeatures(scan) for scan in scans]

    
    def distance(self,f1,f2):
        if self.haralick_from_kpt:
            return self.helper.crossDistance(f1,f2)
        else:
            return self.helper.euclideanDistance(f1,f2)
        
    def query(self,img,k):
        index_distance = []
        f_query = self.extractFeatures(img)
        for index,f_batch in enumerate(self.extracted):
            diff = self.distance(f_query,f_batch)
            index_distance += [(index,diff)]
        index_distance.sort(key= lambda k:k[1])
        index_distance = index_distance[:k]
        index = []
        distance = []
        for i_d in index_distance:
            i,d = i_d
            index += [i]
            distance += [d]
        return index,distance  
    
class RandomFeatures:
    #Do nothing, just to share interface with the other feature objects
    def extractFromBatch(self,scans):
        self.scan_count = len(scans)
        return

    def query(self,img,k):
        #ignore img as query is random
        return [random.randint(0,self.scan_count) for _ in range(k)],[0 for _ in range(k)]
 
