import pickle
import numpy as np
import os

import h5py
class KneeData:
    def __init__(self):
        self.training_index = 0
        self.label_types = ["ishealthy","isright","age"]
        self.scan_start = 16#36
        self.scan_end = 240#219
        self.dim = self.scan_end - self.scan_start
        self.scans = []
        self.labels = []
        
    def getAge(self,filepath):
        filename = filepath.split('/')[-1]
        age = filename.split('-')[0]
        year = int(age[4:6])
        measurement_year = 104
        return measurement_year - year
    
    
    def normalize_img(self,img):
        #maybe it's important that range be within 256
        mean = np.mean(img)
        std = np.std(img)
        img = np.subtract(img,mean)
        img = np.divide(img,3*std)
        img = np.add(img,1)
        img = np.multiply(img,126)
        img = np.asarray(img,dtype=np.int32)
        return img
    
    def balanceData(self):
        label_index = [(label,index) for index,label in enumerate(self.labels)]
        label_index.sort(key=lambda k: k[0]["ishealthy"])
        scans = []
        labels = []
        for i in range(82):
            scans.append(self.scans[label_index[i][1]])
            labels.append(label_index[i][0])
        self.scans = scans
        self.labels = labels
        self.count = 82
        
                
    
    def loadData(self,from_original_files=False,pickled_db_path="scans_and_labels.pck",path="/home/dawit/Datalogi/BachelorProjekt/jub/data/"):
        if from_original_files:
            self.scans = []
            self.labels = []
            self.name_label = {}
#            self.ages = []
#            self.voxelsizes = []
            if from_original_files:
                for r, d, f in os.walk(path):
                    for file in f:
                        if '.mat' in file:

                            sample = h5py.File(os.path.join(r,file),'r')
                            age = self.getAge(file)

                            self.scans.append(self.process3DScan(sample["scan"]))
                            label = self.makeLabel(sample,age)                           
                            self.labels.append(label)
                            self.name_label[file] = label["ishealthy"]
#                            self.ages.append(age)
#                            self.voxelsizes.append(np.asarray(sample["voxelsize"]))
#                            print(file,self.labels[-1])

#            data = [self.scans,self.labels]#,self.ages,self.voxelsizes]
#            with open(pickled_db_path,'wb') as fp:   
#                pickle.dump(data,fp)      
        else:
            with open(pickled_db_path,'rb') as fp:   
                data = pickle.load(fp)             
                self.scans = data[0]
                self.labels = data[1]
#                self.ages = data[2]
#                self.voxelsizes = data[3]
        self.count = len(self.scans)
    
    def prepareTrainingData(self):
        imgs = self.getScansAtSlice()
        vector_labels = np.zeros((len(imgs),2))
        for index,label in enumerate(self.labels):
            if label["ishealthy"]:
                vector_labels[index] = [1,0]
            else:
                vector_labels[index] = [0,1]                              
        self.training_data = [imgs,vector_labels]
        
    def getTrainingData(self,count):
        if self.training_index + count <= len(self.scans):
            imgs = self.training_data[0][self.training_index:self.training_index+count]
            vector_labels = self.training_data[1][self.training_index:self.training_index+count]
            self.training_index += count
            return imgs,vector_labels
        else:
            print("Not enough training data")
            return -1
            
                
    def categorizeImages(self):
        if self.labels is None:
            print("error: labels aren't loaded")
            return False
        dictionary = {}
        for index,img_labels in enumerate(self.labels):
            for label in img_labels:
                if type(img_labels[label]) is bool:
                    key = str(label)+str(img_labels[label])
                    if key in dictionary:
                        dictionary[key].append(index)
                    else:
                        dictionary[key] = [index]
#        if "age" in self.label_types:
#            dictionary["age"] = self.makeAgeDictionary()
#            
#        ages = list(dictionary["age"].keys())
#        ages.sort()
#        median_age = ages[len(ages)//2]
#        dictionary["age_young"] = []
#        dictionary["age_old"] = []
#        for age in ages:
#            if age <= median_age:
#                dictionary["age_young"]+= dictionary["age"][age]
#            else:
#                dictionary["age_old"]+= dictionary["age"][age]               
        return dictionary
            
    def makeAgeDictionary(self):
        retval =  {}
        for index,age in enumerate(self.ages):
            if age in retval:
                retval[age].append(index)
            else:
                retval[age] = [index]
        return retval
        
    def getScansAtSlice(self,slice_index=None,three_slices=False):
#        print(self.scans[0])
        if slice_index == None:
            slice_index = round(self.scans[0].shape[2]/2)           
        if three_slices:
            retval = np.zeros((1,self.dim,self.dim,3),dtype=np.int32)                  
        else:
            retval = np.zeros((1,self.dim,self.dim),dtype=np.int32)       
        for scan in self.scans:
            if three_slices:
                a = 10
                b = int((2*a)/3)+1
                scan_slice = scan[:,:,slice_index-a:slice_index+a:b]
                scan_slice = np.asarray(scan_slice,dtype=np.int32).reshape(1,self.dim,self.dim,3)               
            else:
                scan_slice = scan[:,:,slice_index]           
                scan_slice = np.asarray(scan_slice,dtype=np.int32).reshape(1,self.dim,self.dim)
            retval = np.append(retval,scan_slice,axis=0)
        return retval[1:]
    
    def makeLabel(self,sample,age):
        labels = ["ishealthy","isright"]
        dictionary = {}
        for label in labels:
            dictionary[label] = True if int(sample[label][0][0]) == 1 else False
        dictionary["age"] = age
        return dictionary
    
    def process3DScan(self,scan):
        scan = scan[self.scan_start:self.scan_end,self.scan_start:self.scan_end]
        scan = self.normalize_img(scan)
        return scan    
