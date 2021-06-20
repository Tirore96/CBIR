import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
#        plt.figure(figsize=(20,10)) 
        plt.rcParams["figure.figsize"] = [16,9]

    def plotComparison(self,images,distances=None,original_img=None):
        if not original_img is None:
            print("Original Image")
            plt.imshow(original_img,cmap=plt.get_cmap('gray'))
            plt.show()
            print("__________________")
        for index,img in enumerate(images):
            plt.imshow(img,cmap=plt.get_cmap('gray'))
            plt.show()
            if not distances is None:
                print(distances[index])
                
    def plotSingleShape(self,img,dist=None):
        if dist!=None:
            print("Distance: ",dist)
        plt.imshow(img,cmap=plt.get_cmap('gray'))
        plt.show()
        
    def saveFig(self,img,filename):
        plt.imshow(img,cmap=plt.get_cmap('gray'))   
        plt.savefig(filename,bbox_inches='tight')

    
    def plotShapes(self,imgs,dists=None):
        for i,img in enumerate(imgs):
            if dists!=None:
                dist = dists[i]
            else:
                dist = None
            self.plotSingleShape(img,dist)
    
    def plotResults(self,batch,org_img_index,nearest_indices,dists):
        print("Original Image")
        org_img = batch[org_img_index]
        self.plotSingleShape(org_img)
        print("_____________")
        nearest_imgs = [batch[i] for i in nearest_indices]
        self.plotShapes(nearest_imgs,dists)
        
        
        