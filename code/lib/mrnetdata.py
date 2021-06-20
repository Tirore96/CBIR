#one class per task, each class training and test data
import numpy as np
import csv
import os
import pickle
class MRnetData:
    def __init__(self):
        self.training_index = 0       
        
        
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
    
    def getScansAtSlice(self,slice_index=None,three_slices=False):
        if slice_index == None:
            slice_index = round(self.scans[0].shape[2]/2)                  
#        retval = np.zeros((1,self.dim+24,self.dim+24),dtype=np.int32)       
        if three_slices:
            retval = np.zeros((1,self.dim+24,self.dim+24,3),dtype=np.int32)                  
        else:
            retval = np.zeros((1,self.dim+24,self.dim+24),dtype=np.int32)                  
        for scan in self.scans:
            if three_slices:
#                a = 10
#                b = int((2*a)/3)+2
#                print(a,b)
#                print(scan.shape)
#                scan_slice = scan[:,:,slice_index-a:slice_index+a:b]
                scan_slice = scan[:,:,slice_index-1:slice_index+2]
                scan_slice = np.asarray(scan_slice,dtype=np.int32).reshape(1,self.dim,self.dim,3)               
                scan_slice = np.pad(scan_slice,[(0,0),(12,12),(12,12),(0,0)],'constant')               
            else:
                scan_slice = scan[:,:,slice_index]           
                scan_slice = np.asarray(scan_slice,dtype=np.int32).reshape(1,self.dim,self.dim)
                scan_slice = np.pad(scan_slice,[(0,0),(12,12),(12,12)],'constant')
                
            retval = np.append(retval,scan_slice,axis=0)
        return retval[1:]       
    
    def prepareTrainingData(self):
        imgs = self.getScansAtSlice()
        vector_labels = np.zeros((len(imgs),2))
        for index,label in enumerate(self.labels):
            if label["value"]:
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
    
#    def genSameScansDifferentLabels(self,path_labels):
#        label_dict = self.loadLabels(path_labels)   
#        labels = []
#        for name in self.filenames:
#            label = label_dict[name]
#            labels.append({"value":label})
#        
#        mrnet = MRnetData()
#        mrnet.scans = self.scans
#        mrnet.labels = labels
#        mrnet.dim = self.dim
#        mrnet.training_data = self.training_data
#        mrnet.training_index = self.training_index
#        return mrnet
        
    def categorizeImages(self):
        if self.labels is None:
            print("error: labels aren't loaded")
            return False
        dictionary = {}
        for index,img_labels in enumerate(self.labels):
            for label in img_labels:
                key = str(label)+str(img_labels[label])
                if key in dictionary:
                    dictionary[key].append(index)
                else:
                    dictionary[key] = [index]
        return dictionary
    
    
    def process3DScan(self,scan):
        scan = self.normalize_img(scan)
        scan = np.swapaxes(scan,0,2)
        scan = np.swapaxes(scan,0,1)
        first_size,second_size,_ = scan.shape
        first_diff = (first_size - 200)/2
        second_diff = (second_size - 200)/2       
        if first_diff % 1.0 != 0.0:
            first_diff = round(first_diff)
            scan = scan[first_diff+1:-first_diff,:,:]
        
        else:
            first_diff = round(first_diff)           
            scan = scan[first_diff:-first_diff,:,:]           
        if second_diff % 1.0 != 0.0:
            second_diff = round(second_diff)
            scan = scan[:,second_diff+1:-second_diff,:]
        
        else:
            second_diff = round(second_diff)
            scan = scan[:,second_diff:-second_diff,:]                      
        
        return scan
    
    def loadLabels(self,path_labels):
        dictionary = {}
        with open(path_labels) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:       
                dictionary[str(row[0])+".npy"] = True if row[1] == '1' else False
        return dictionary
                
    def loadData(self,pickled_db_path,path_scans=None,path_labels=None,from_original_files=False):
        label_dict = self.loadLabels(path_labels)
        self.filenames = []
        if from_original_files:
            self.scans = []
            self.labels = []
            if from_original_files:
                for r, d, f in os.walk(path_scans):
                    for file in f:
                        if '.npy' in file:
                            sample = np.load(os.path.join(r,file))
                            processed_sample = self.process3DScan(sample)
                            self.scans.append(processed_sample)
                            label = label_dict[file]
                            self.labels.append({"value":label})
                            self.filenames.append(file)
            
#            data = [self.scans,self.labels,self.filenames]
#            with open(pickled_db_path,'wb') as fp:   
#                pickle.dump(data,fp)      
        else:
            with open(pickled_db_path,'rb') as fp:   
                data = pickle.load(fp)             
                self.scans = data[0]
                self.labels = data[1]
                self.filenames = data[2]
        self.count = len(self.scans)       
        self.dim = self.scans[0].shape[0]
