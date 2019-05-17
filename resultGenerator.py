import pickle
from lib.evaluator import *
from lib.features import *
from lib.shapes import *
from lib.MLMethod import *
from lib.mrnetdata import *
def pickleData():
    #training and test in same object
    toy_data = SimpleShapeData()
    toy_data.prepareTrainingData(label_name="shape",label_value="triangle")
    with open("pickled/toy_data.pck",'wb') as fp:   
        pickle.dump(toy_data,fp)         
    
    print("finished toy data")
    #Erik data###############################################3
    erik_train = KneeData()
    erik_train.loadData(from_original_files=True,pickled_db_path="pickled/erik_train.pck",path='/home/dawit/Datalogi/BachelorProjekt/data/erik/train_data/')
    erik_train.prepareTrainingData()
    with open("pickled/erik_train.pck",'wb') as fp:   
        pickle.dump(erik_train,fp)            
    
    erik_test = KneeData()
    erik_test.loadData(from_original_files=True,pickled_db_path="pickled/erik_test.pck",path='/home/dawit/Datalogi/BachelorProjekt/data/erik/test_data/')   
    with open("pickled/erik_test.pck",'wb') as fp:   
        pickle.dump(erik_test,fp)               
        
    print("finished erik data")
    
    miny_data = KneeData()
    miny_data.loadData(from_original_files=True,pickled_db_path="pickled/tiny_data.pck",path='/home/dawit/Datalogi/BachelorProjekt/data/tiny_data/')   
    with open("pickled/tiny_data.pck",'wb') as fp:   
        pickle.dump(miny_data,fp)                  
        
    #MRNet data#####################################################3
    tasks = ["abnormal","acl","meniscus"]
    for task in tasks: 
        mrnet_base_path = "/home/dawit/Datalogi/BachelorProjekt/data/"
        mrnet_train = MRnetData()
        mrnet_train.loadData(from_original_files=True,pickled_db_path=mrnet_base_path+"../jub/lib/mrnet/"+task+"_train.pck",path_scans=mrnet_base_path+"extracted_MR/"+task+"/train",\
                       path_labels=mrnet_base_path+"MRNet-v1.0/train-"+task+".csv")
        mrnet_train.prepareTrainingData()   
        with open("pickled/mrnet_"+task+"_train.pck",'wb') as fp:   
            pickle.dump(mrnet_train,fp)                  
            
        mrnet_test= MRnetData()
        mrnet_test.loadData(from_original_files=True,pickled_db_path=mrnet_base_path+"../jub/lib/mrnet/"+task+"_test.pck",path_scans=mrnet_base_path+"extracted_MR/"+task+"/test",\
                        path_labels=mrnet_base_path+"MRNet-v1.0/valid-"+task+".csv")
        mrnet_test.prepareTrainingData()   
        with open("pickled/mrnet_"+task+"_test.pck",'wb') as fp:   
            pickle.dump(mrnet_test,fp)                         
            
        print("finished "+task+" data")
    
    
    
    
class ResultGenerator:
    def __init__(self,pickle_paths,dataset_names):
        self.datasets= {}
        self.dataset_names = dataset_names
        for index,path in enumerate(pickle_paths):
            with open(path,'rb') as fp:   
                data = pickle.load(fp)                                
            print(dataset_names[index])
            self.datasets[dataset_names[index]] = data
    
    def genResultsForDataset(self,dataset_name_train,dataset_name_test,label_name,label_value):
        k = 5
        print("Generating results for "+dataset_name_test+", trained with "+dataset_name_train)
        train_data = self.datasets[dataset_name_train]
        test_data = self.datasets[dataset_name_test]       
        
        self.haralicks = Haralicks(compute_dog=False)
        self.haralicks3d = Haralicks(compute_dog=False)
        self.lbp= LBPFeatures(8,30)
        self.lbp3d= LBP3DFeatures([(8,30),(8,30),(8,30)])
        self.ml = ConvolutionMethod(load_prev_model=False)       
        
        self.fitMethods(train_data,label_name,label_value,dataset_name_train)
        
#        methods = [self.haralicks,self.haralicks3d,self.lbp,self.lbp3d,self.ml]
        evaluators = self.initEvaluators(test_data)
        precisionRecalls = []
        ROCs = []
        AUCs = []
        MAPs = []
        
        for evaluator_name in evaluators:
            evaluator = evaluators[evaluator_name]
            p,r = evaluator.precisionRecall(k,label_name,label_value,evaluator_name + " precision-recall curve, data: " + dataset_name_test,
                                      evaluator_name+"_"+dataset_name_test+"_PR.png")
            roc = evaluator.ROCCurve(k,label_name,label_value,evaluator_name + " ROC curve, data: " + dataset_name_test,
                                      evaluator_name+"_"+dataset_name_test+"_ROC.png")           
            auc = evaluator.calcAuc(k,label_name,label_value)
            map_std = evaluator.MAP(test_data.count,k,[label_name])           
            
            "flip precision and recall coordinates"
            precisionRecalls.append((r,p))
            ROCs.append(roc)
            AUCs.append(auc)
            MAPs.append(map_std)
            print("finished {}".format(evaluator_name))
        evaluator_names = list(evaluators.keys())
        with open("pickled/results_"+dataset_name_test+".pck",'wb') as fp:   
            pickle.dump([precisionRecalls,ROCs,AUCs,MAPs,evaluator_names],fp)                    
        self.generatePlots(dataset_name_test)

        
    def generatePlots(self,dataset):
        with open("pickled/results_"+dataset+".pck",'rb') as fp:   
            precisionRecalls,ROCs,AUCs,MAPs,evaluator_names = pickle.load(fp)                               
        roc_filename = dataset+"_ROC.png"
        self.plotCurves(ROCs,evaluator_names,"False positve rate","True positive rate","ROC curve, data: {}".format(dataset),roc_filename)
        pr_filename = dataset+"_PR.png"
        self.plotCurves(precisionRecalls,evaluator_names,"Recall","Precision","Precision-recall curve, data: {}".format(dataset),pr_filename)       
        self.plotAUC(AUCs,evaluator_names,dataset)       

        
    def plotAUC(self,aucs,names,dataset_name):
        l = len(aucs)
        plt.axis([0,l+1,0,1])
        plt.xlabel("methods")
#        names = list(evaluators.keys())
        plt.xticks(np.arange(1,l+1),names)
        
        plt.ylabel("AUC score")
        plt.title("AUC values across methods")                  
        for i in range(l):
            plt.bar(i+1,aucs[i])
            plt.text(i+1,aucs[i]+1,str(aucs[i]))
        plt.savefig("results/"+dataset_name+"_AUC.png")
        plt.clf()                  
    
#    def plotMAP(self,maps,evaluators,dataset_name):

    def plotCurves(self,curves,names,xlabel,ylabel,title,filename):
        plt.axis([0,1,0,1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)           
        for index,name in enumerate(names):
            x,y = curves[index]
            plt.plot(x,y,label=name)

#       # plt.show()              
        plt.legend()
        plt.savefig("results/"+filename)
        plt.clf()       
            
            #print("AUC: ",auc)
#            print("MAP: {}, std: {}".format(map_val,std_val))
#            #with open("results/"+evaluator_name+"_"+dataset_name_test+".txt","w+") as f:
#                auc_str = "AUC: %f \n" % (auc)
#                MAP,std = evaluator.MAP(test_data.count,k,[label_name])
#                map_std_str = "MAP: %f, with std: %f \n" % (map_val,std_val)
#                prob_arr = "Probability array: {}".format(evaluator.probabilities)
#                labels = "Labels array {}".format(evaluator.labels)
#                f.writelines([auc_str,map_std_str,prob_arr])

    def fitMethods(self,train_data,label_name,label_value,dataset_name_train):
        if type(train_data) is SimpleShapeData:
            slices = train_data.scans_train
            only_2d = True
        else:
            slices = train_data.getScansAtSlice()
            only_2d = False
        self.haralicks.featureSelector(5,5,slices,train_data.labels,label_name,label_value)       #5
        self.lbp.fit_parameters(5,slices,train_data.labels,label_name,label_value)
        self.ml.optimize(train_data,int((train_data.count)/5))#num_iterations=1)
#        with open("results/method_parameters"+dataset_name_train+".txt","w+") as f:
#            f.write("haralick selected features: "+ str(self.haralicks.selected_features)+"\n")
#            lbp_str = "lbp radius: %d, n_points: %d \n" % (self.lbp.radius,self.lbp.n_points)
#            f.write(lbp_str)
        
        fitted = []
        haralicks_fitted = self.haralicks.selected_features
        lbp_fitted = self.lbp.radius,self.lbp.n_points       
        fitted.append(haralicks_fitted)
        fitted.append(lbp_fitted)
        if not only_2d:
            self.haralicks3d.featureSelector(5,5,train_data.scans,train_data.labels,label_name,label_value)              #5
            self.lbp3d.fit_parameters(5,train_data.scans,train_data.labels,label_name,label_value)       
            haralicks3d_fitted = self.haralicks3d.selected_features
            lbp3d_fitted = (self.lbp3d.lbp_xy.radius,self.lbp3d.lbp_xy.n_points,self.lbp3d.lbp_xz.radius,self.lbp3d.lbp_xz.n_points,self.lbp3d.lbp_yz.radius,self.lbp3d.lbp_yz.n_points)
            fitted.append(haralicks3d_fitted)
            fitted.append(lbp3d_fitted)           
#            with open("results/method_parameters"+dataset_name_train+".txt","a") as f:
#                f.write("haralick3d selected features: "+ str(self.haralicks3d.selected_features)+"\n")
#                lbp_str = "lbp3d (xy radius: %d, n_points: %d),(xz radius: %d, n_points: %d),(yz radius: %d, n_points: %d) \n" %(self.lbp3d.lbp_xy.radius,self.lbp3d.lbp_xy.n_points,self.lbp3d.lbp_xz.radius,self.lbp3d.lbp_xz.n_points,self.lbp3d.lbp_yz.radius,self.lbp3d.lbp_yz.n_points)
#                f.write(lbp_str)           
#                f.write(lbp_str)
        print("Done fitting")


        with open("pickled/fitted_methods_"+dataset_name_train+".pck",'wb') as fp:   
             pickle.dump(fitted,fp)                               
        
    def initEvaluators(self,test_data):
        only_2d = type(test_data) is SimpleShapeData
        e_haralicks= Evaluator(data_obj=test_data,method_obj=self.haralicks,load_as_slice=True and not only_2d)
        e_lbp= Evaluator(data_obj=test_data,method_obj=self.lbp,load_as_slice=True and not only_2d)
        e_ml= Evaluator(data_obj=test_data,method_obj=self.ml,load_as_slice=True and not only_2d)
        eval_dict = {"haralicks2d":e_haralicks,"lbp2d":e_lbp,"ml":e_ml}
        if not only_2d:
            e_haralicks3d= Evaluator(data_obj=test_data,method_obj=self.haralicks3d,load_as_slice=False)
            e_lbp3d= Evaluator(data_obj=test_data,method_obj=self.lbp3d,load_as_slice=False)
            eval_dict["haralicks3d"] = e_haralicks3d
            eval_dict["lbp3d"] = e_lbp3d
        print("Done initializing evaluators")
        return eval_dict
    
