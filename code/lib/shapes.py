import numpy as np
import copy
import random as ran
from keras.applications.vgg16 import preprocess_input

pix_max = 255
class ShapeCreator:
    def __init__(self):
        self.circle = Circle()
    
    def makeCircle(self,img_width,r,x_offset,y_offset):
        return self.circle.makeCircle(img_width,r,x_offset,y_offset)
        
    def makeSquare(self,img_width,side_length, row_offset,col_offset):
        arr = np.zeros((img_width,img_width),dtype=np.uint8)
        row_start = int((img_width-side_length) * (row_offset/100))
        col_start = int((img_width-side_length) * (col_offset/100))
        for i in range(side_length):
            #top line
            arr[row_start][col_start+i] = pix_max
            #bottom line
            arr[row_start+side_length][col_start+i] = pix_max
            #right line
            arr[row_start + i][col_start+side_length] = pix_max 
            #left line
            arr[row_start + i][col_start] = pix_max
            
        arr[row_start+side_length][col_start+side_length] = pix_max 
        return np.asarray(arr)
    
    
    #img_width = 320
    def makeTriangle(self, img_width,side_length,row_offset,col_offset):
        height = side_length // 2 + 1
        arr = np.zeros((img_width,img_width),dtype=np.uint8)
        row_start = int((img_width-height) * (row_offset/100))
        col_start = int((img_width-side_length) * (col_offset/100))       
        for i in range(side_length+1):
            arr[row_start+height][col_start+i] = pix_max 
        for i in range(height):
            arr[row_start+height-i][col_start+i] = pix_max 
            arr[row_start+height-i][col_start+2*height-i-1] = pix_max 
        return np.asarray(arr)


            
class Circle:
    def makeCircle(self,img_width,r,x_offset,y_offset):
        arr = np.zeros((img_width,img_width),dtype=np.uint8)
        points = self.makeCirclePoints(img_width,r,x_offset,y_offset)
        for point in points:
            arr[point[1]][point[0]] = pix_max 
        return np.asarray(arr)
    
    
    def makeCirclePoints(self,img_width,r,x_offset,y_offset):
        #to make size compatible with other shapes
        r = r // 2
        x_offset_tweaker = x_offset / 50
        y_offset_tweaker = y_offset/ 50
        img_width_half = img_width // 2
        x_start = r*2 + int(((img_width-r*2)/2) * x_offset_tweaker)
        y_start = r+ int(((img_width-r*2)/2) * y_offset_tweaker)
        x_mid = x_start - r
        y_mid = y_start
        position = [x_start,y_start]
        positions = [position]
        while True:
            position = self.approxCirclePoint(x_mid,y_mid,position[0],position[1],r)
            if x_start == position[0] and y_start == position[1]:
                break
            positions.append(position)
        return positions
    
    def approxCirclePoint(self,x_c,y_c,x_p,y_p,r):
        positions = []
        circleEquation = np.sqrt((x_c-x_p)**2 + (y_c-y_p)**2)
        if x_c <= x_p:
            if y_c >= y_p:
                #Upper right (up,left-up,left)
                positions = [[x_p,y_p-1],[x_p-1,y_p-1],[x_p-1,y_p]]
            else:
                #Lower right (right,right-up,up)
                positions = [[x_p+1,y_p],[x_p+1,y_p-1],[x_p,y_p-1]]
        else: 
            if y_c >= y_p:
                #Upper left (left, left-down, down)
                positions = [[x_p-1,y_p],[x_p-1,y_p+1],[x_p,y_p+1]]
            else:
                #Lower left (down, down-right,right)
                positions = [[x_p,y_p+1],[x_p+1,y_p+1],[x_p+1,y_p]]
        return self.minCircleErrorPosition(x_c,y_c,r,positions)
    
    
    
    def minCircleErrorPosition(self,x_c,y_c,r,positions):
        position_min = []
        error_global = -1
        for position in positions:
            error_local = self.circleError(x_c,y_c,position[0],position[1],r)
            if error_global > error_local or error_global == -1:
                error_global = error_local
                position_min = position 
        return position_min
    
    def circleError(self,x_c,y_c,x_1,y_1,r):
        circleEquation_r = np.sqrt((x_c-x_1)**2 + (y_c-y_1)**2)
        return abs(circleEquation_r-r)

    
class SimpleShapeData:
    def __init__(self):
        self.creator = ShapeCreator()
        self.scans_train,self.labels_train,self.scans,self.labels= self.makeBatch()
#        self.scans,self.labels,_,_= self.makeBatch()
        self.count_train = len(self.scans_train)
        self.count= len(self.scans)       
        self.training_index = 0
        
    
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
    
    def fillImgTri(self,img):
        shape = img.shape
        for i in range(shape[0]):
            draw = False
            for j in range(shape[1]):
#                if img[i,j] == img[i,j] and img[i,j] == pix_max:
#                    continue
                if draw:
                    if img[i,j] == pix_max:
                        draw = False 
                else:
                    if img[i,j] == pix_max:
                        draw = True 
                if draw:
                    img[i,j] = pix_max
        return img   
    
    def fillImg(self,img):
        shape = img.shape
        for i in range(shape[0]):
            draw = False
            count = 0
            for j in range(shape[1]-10):
                if img[i,j] == img[i,j+10] and img[i,j] == pix_max:
                    break
                if draw:
                    if img[i,j] == pix_max:
                        break
                else:
                    if img[i,j] == pix_max:
                        if img[i,j+1] == pix_max and count < 5:
                            count+=1
                        else:
                            draw = True 
                if draw:
                    img[i,j] = pix_max
        return img      
    def makeBatch(self,offsets_arg=None,noises_arg=None):
        size = 224#200
        length = 70
        offsets = [[10,10],[90,10],[10,90],[80,80]]
        noises = [0,0.05,0.1,0.15]
        if not offsets_arg is None:
            offsets = offsets_arg
        if not noises_arg is None:
            noises = noises_arg
        args = [[offset,noise] for offset in offsets for noise in noises]
        id_incr = 0
        imgs_train = []
        labels_train = []
        imgs_test= []
        labels_test= []       
        fill_train = True
        funs = [self.creator.makeTriangle,self.creator.makeSquare,self.creator.makeCircle]
        shape_names = ["triangle","square","circle"]
        offset_names = ["top-left","bottom-left","top-right","bottom-right"]
        for shape_index,fun in enumerate(funs):
            for offset_index,offset in enumerate(offsets):
                for noise_index,noise in enumerate(noises):               
                    img = fun(size,length,offset[0],offset[1])                   
                    if shape_index == 0:
                        img = self.fillImgTri(img)
                    else:
                        img = self.fillImg(img)
                    img = self.addNoise(img,noise)
                    label = {"shape":shape_names[shape_index],"offset":offset_names[offset_index],"noise":noises[noise_index]}
                    if fill_train:
                        imgs_train.append(img)
                        labels_train.append(label)
                        fill_train = False
                    else:
                        imgs_test.append(img)
                        labels_test.append(label)
                        fill_train = True 
                        
        
        return  np.asarray(imgs_train),np.asarray(labels_test),np.asarray(imgs_test),np.asarray(labels_test) #triangles + squares # + circles
    
    def makeBatchOmitting(self,omitter):
        if omitter in self.shape_labels:
            index = self.shape_labels.index(omitter)
            batch = self.makeBatch()
            batch.pop(index)
            return batch
        else:
            print("omitter not part of list")
            return None
    
    #Inspiration: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    def addNoise(self,img,prob):
        img = copy.deepcopy(img)
        img = np.asarray(img)
        rows,cols = img.shape
        threshold = 1-prob
        for i in range(rows):
            for j in range(cols):
                rnd = ran.random()
                if rnd < prob:
                    img[i][j] = 0
                elif rnd > threshold:
                    img[i][j] = pix_max 
        return img
    
    def vggData(self,label_name,label_value):
        X = np.expand_dims(self.scans_train,3)
        X = np.repeat(X,3,3)
        X = np.array([preprocess_input(i) for i in X])
        Y = np.array([[1,0] if i[label_name] == label_value else [0,1] for i in self.labels_train])
        

#        for i in self.scans_train:
#            arr = np.expand_dims(i,axis=2)
#            arr = np.repeat(arr,3,2)
#            X = np.append(X,arr,0)
#        for i in self.labels_train:
#            val = 1 if i[label_name] == label_value else 0
#            Y = np.append(Y,val,0)
        return X,Y
    
    def prepareTrainingData(self,label_name,label_value):
#        imgs = self.getScansAtSlice(40)
        
#        label_name = "offset"
#        label_value = "top-left"
        vector_labels = np.zeros((len(self.scans_train),2))
        for index,label in enumerate(self.labels_train):
            if label[label_name] == label_value:
                vector_labels[index] = [1,0]
            else:
                vector_labels[index] = [0,1]                              
        self.training_data = [self.scans_train,vector_labels]
        
    def getTrainingData(self,count):
        if self.training_index + count <= len(self.scans_train):
            imgs = self.training_data[0][self.training_index:self.training_index+count]
            vector_labels = self.training_data[1][self.training_index:self.training_index+count]
            self.training_index += count
            return imgs,vector_labels
        else:
            print("Not enough training data")
            return -1
               
    
    
    