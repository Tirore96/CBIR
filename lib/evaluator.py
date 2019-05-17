from lib.features import *
from lib.kneedata import *
from functools import wraps
from time import time
from sklearn.metrics import precision_recall_curve,roc_auc_score,roc_curve
import matplotlib.pyplot as plt

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print ('Elapsed time: {}'.format(end-start))
        return result
    return wrapper
#tasks
#make data scan attribute
#Provide even distribution of shapes

#make label attribute
#first simple labels, only shape, not position or size

#create categorizeImage() function
#Dictionary for each of the possible label values
class Evaluator:
    def __init__(self,data_obj,method_obj,load_as_slice=True):
        self.data = data_obj
        self.probabilities = None
        if load_as_slice:
            self.scans = self.data.getScansAtSlice()
        else:
            self.scans = self.data.scans
        self.labels = self.data.labels
        self.categorized_images = self.data.categorizeImages()
        
        self.method = method_obj
        self.method.extractFromBatch(self.scans)
        
        self.agerange = 10
        print("initialization finished")
        
    def query(self,m,k):
        indices,distances = self.method.query(self.scans[m],k)
        return indices,distances
        

    
    
    def avgPrecision(self,img_index,relevant_labels):
        relevant_images = self.relevantImages(self.labels[img_index],relevant_labels)           
        retrieved_indices,distances = self.query(img_index,self.data.count)                         
        avgP = 0
        for k_iter in range(1,self.data.count):
            avgP += self.precision(relevant_images,retrieved_indices[1:k_iter+1])
            
        return avgP /(self.data.count-1)
            
        
    #How many relevant of retrieved
    def precision(self,relevant_images,retrieved_images):
        #minus the query image       
        relevant_set = set(relevant_images)
        intersection = [img for img in retrieved_images if img in relevant_set]
        return len(intersection)/len(retrieved_images)
    
    #How many retrieved of all relevant
    def recall(self,relevant_images,retrieved_images):
        #minus the query image
        relevant_set = set(relevant_images)
        intersection = [img for img in retrieved_images if img in relevant_set]
        return len(intersection)/(len(relevant_images)-1)
    
    def relevantImages(self,labels,relevant_labels):
        retval = []
        for relevant_label in relevant_labels:
            if not relevant_label in labels:
                print("error: relevant label not found in image labels")
            if relevant_label == "age":
                age = labels[relevant_label]
                age_dict = self.categorized_images["age"]
                retval = retval + age_dict[age]
                for r in (1,self.agerange):
                    if age+r in age_dict:
                        retval = retval + age_dict[age+r]
                    if age-r in age_dict:
                        retval = retval + age_dict[age-r]  
            else:        
                key = relevant_label + str(labels[relevant_label])
                retval += self.categorized_images[key]
        #possibly dont remove duplicates to reward multi label match. Maybe even square the value if 2 or more labels both match
        unique = list(set(retval))        
        return unique
    
        
    def calcAuc(self,K,relevant_label,label_value):
        if self.probabilities is None:
            self.probabilities = self.calcClassificationProbabilities(K,relevant_label,label_value)                  
            probabilities = self.probabilities
        else:
            probabilities = self.probabilities
#        probabilities = self.calcClassificationProbabilities(K,relevant_label,label_value)
        binary_labels = [1 if i[relevant_label]==label_value else 0 for i in self.labels ]       
        return roc_auc_score(binary_labels,probabilities)
    
    def plotPrecisionRecall(self,img_index,relevant_labels):
        p,r = self.precisionRecallPoints(img_index,relevant_labels)
        plt.axis([0,1,0,1])
        plt.plot(r,p)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.show()       
            
    def countClassificationRatio(self,label_indices,label_name,label_value):
        relevant_count = 0
        non_relevant_count = 0 
        for index in label_indices:
            if self.labels[index][label_name] == label_value:
                relevant_count += 1
            else:
                non_relevant_count += 1
                
        return relevant_count / (len(label_indices))
    
    def calcClassificationProbabilities(self,K,relevant_label,label_value):
        probabilities = np.zeros((self.data.count),dtype=np.float)
        for m_iter in range(self.data.count):
            retrieved_indices,distances = self.query(m_iter,K+1)           
            retrieved_indices = retrieved_indices[1:]           
#            thresh_index = 0
#            threshold = 0.0
#            while threshold < 1.0:
            ratio = self.countClassificationRatio(retrieved_indices,relevant_label,self.labels[m_iter][relevant_label])
            probabilities[m_iter] = ratio
        self.probabilities = probabilities
        return probabilities
#       
    def ROCCurve(self,K,relevant_label,label_value,title=None,filename=None):
        if self.probabilities is None:
            self.probabilities = self.calcClassificationProbabilities(K,relevant_label,label_value)                  
            probabilities = self.probabilities
        else:
            probabilities = self.probabilities
            
        binary_labels = [1 if i[relevant_label]==label_value else 0 for i in self.labels ]
        fpr,tpr,thresh = roc_curve(binary_labels,probabilities)
        return fpr,tpr
        #plt.axis([0,1,0,1])
        #plt.plot(fpr,tpr)
        #plt.xlabel('false positive rate')
        #plt.ylabel('true positive rate')
        #if title != None:
        #    plt.title(title)
#       # plt.show()              
        #if filename != None:
        #    plt.savefig("results/"+filename)
        #plt.clf()       
    def precisionRecall(self,K,relevant_label,label_value,title=None,filename=None):
        if self.probabilities is None:
            self.probabilities = self.calcClassificationProbabilities(K,relevant_label,label_value)                  
            probabilities = self.probabilities
        else:
            probabilities = self.probabilities
#        probabilities = self.calcClassificationProbabilities(K,relevant_label,label_value)
        binary_labels = [1 if i[relevant_label]==label_value else 0 for i in self.labels ]
        precision,recall,_ = precision_recall_curve(binary_labels,probabilities)
        return precision,recall
        #plt.axis([0,1,0,1])
        #plt.plot(recall,precision)
        #plt.xlabel('recall')
        #plt.ylabel('precision')
        #if title != None:
        #    plt.title(title)
#       # plt.show()              
        #if filename != None:
        #    plt.savefig("results/"+filename)
        #plt.clf()
           
                 
                

                
            
            
    def precisionRecallPoints(self,img_index,relevant_labels):
        precision = []
        recall = []
        relevant_images = self.relevantImages(self.labels[img_index],relevant_labels)           
        retrieved_indices,distances = self.query(img_index,self.data.count)                  
        for k_iter in range(1,self.data.count):
            precision.append(self.precision(relevant_images,retrieved_indices[1:k_iter+1]))
            recall.append(self.recall(relevant_images,retrieved_indices[1:k_iter+1]))           
        
            
        return precision,recall
    
#    def avgPrecisionRecall(self,relevant_labels):
#        p = np.zeros(self.data.count)
#        r = np.zeros(self.data.count)       
    
     #if not sure what is preferred, make it a variable :P
    @timing
    def MAP(self,M,K,relevant_labels):
        retval = 0
        avgPrecisions = []
        #start 0 because it's an index
        for m_iter in range(0,M):
            retval_inner = 0
            
            relevant_images = self.relevantImages(self.labels[m_iter],relevant_labels)
            #start 1 because it's a count of retrieved images & K+1 because we delete an element we must add one more 
            retrieved_indices,distances = self.query(m_iter,K+1)           
            for k_iter in range(1,K+1):
                #ignore the first returned because it will be the same image
#                retrieved_indices,distances = self.query(m_iter,k_iter+1)
                retval_inner += self.precision(relevant_images,retrieved_indices[1:k_iter+1])
            retval += (retval_inner/K)
            avgPrecisions.append(retval_inner/K)
        return retval / M,np.std(avgPrecisions)
