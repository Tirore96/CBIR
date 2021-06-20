import pickle
from lib.evaluator import *
from lib.features import *
from lib.shapes import *
from lib.MLMethod import *
from lib.mrnetdata import *
from lib.vggModel import *
def pickleData():
    #training and test in same object
    toy_data = SimpleShapeData()
    toy_data.prepareTrainingData(label_name="shape",label_value="triangle")
    with open("pickled/toy_data.pck",'wb') as fp:   
        pickle.dump(toy_data,fp)         
    toy_data.scans = toy_data.scans_train
    toy_data.labels = toy_data.labels_train
    with open("pickled/toy_data_train.pck",'wb') as fp:   
        pickle.dump(toy_data,fp)            
#    toy_data_train = SimpleShapeData()
#    toy_data_train.prepareTrainingData(label_name="shape",label_value="triangle")       
    
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
#        self.datasets= {}
        self.name_data_dict = {}
        for i in range(len(dataset_names)):
            self.name_data_dict[dataset_names[i]] = pickle_paths[i]

        self.k_vals = {"artificial_data_train":8,"OA_train":20,"mrnet_abnormal_train":25,"mrnet_meniscus_train":50,"mrnet_acl_train":50}
#        for index,path in enumerate(pickle_paths):
#            with open(path,'rb') as fp:   
#                data = pickle.load(fp)                                
#            print(dataset_names[index])
#            self.datasets[dataset_names[index]] = data
    
    def genResultsForDataset(self,dataset_name_train,dataset_name_test,label_name,label_value):
        k = self.k_vals[dataset_name_train]
        print("Generating results for "+dataset_name_test+", trained with "+dataset_name_train)
        with open(self.name_data_dict[dataset_name_train],'rb') as fp:   
            train_data = pickle.load(fp)                                
        with open(self.name_data_dict[dataset_name_test],'rb') as fp:   
            test_data = pickle.load(fp)                                

        print(dataset_name_test)

#        train_data = self.datasets[dataset_name_train]
#        test_data = self.datasets[dataset_name_test]       
        
        self.haralicks = Haralicks(compute_dog=False)
        self.haralicks3d = Haralicks(compute_dog=False)
        self.lbp= LBPFeatures(8,30)
        self.lbp3d= LBP3DFeatures([(8,30),(8,30),(8,30)])
        self.vgg = VGGModel()
        self.vgg_fitted = VGGModel()
#        self.ml = ConvolutionMethod(load_prev_model=False)       
        
        self.fitMethods(train_data,label_name,label_value,dataset_name_train)
        
#        methods = [self.haralicks,self.haralicks3d,self.lbp,self.lbp3d,self.ml]
        evaluators_test = self.initEvaluators(test_data)

        #evaluators = dict(evaluators,**evaluators_train)
        
        precisionRecalls_p_test = []
        precisionRecalls_n_test = []       
        ROCs_test = []
        AUCs_test = []
        prop_arrs_test = []
        for evaluator_name in evaluators_test:
#            print(evaluator_name)
            evaluator = evaluators_test[evaluator_name]
            pos,neg = evaluator.avgPrecisionRecallPoints([label_name],label_value)
            pos_p,pos_r = pos
            neg_p,neg_r = neg
            roc = evaluator.ROCCurve(k,label_name,label_value)           
            auc = evaluator.calcAuc(k,label_name,label_value)
            #"flip precision and recall coordinates"
            precisionRecalls_p_test.append((pos_r,pos_p))
            precisionRecalls_n_test.append((neg_r,neg_p))           
            ROCs_test.append(roc)
            AUCs_test.append(auc)
            prop_arrs_test.append(evaluator.probabilities)
#            MAPs.append(map_std)
            print("finished {}".format(evaluator_name))
        evaluator_names_test = list(evaluators_test.keys())
        

        evaluators_train = self.initEvaluators(train_data)
        precisionRecalls_p_train = []
        precisionRecalls_n_train = []       
        ROCs_train = []
        AUCs_train = []
        prop_arrs_train = []
        for evaluator_name in evaluators_train:
            evaluator = evaluators_train[evaluator_name]
            pos,neg = evaluator.avgPrecisionRecallPoints([label_name],label_value)
            pos_p,pos_r = pos
            neg_p,neg_r = neg           
#            p,r = evaluator.avgPrecisionRecallPoints([label_name])
            roc = evaluator.ROCCurve(k,label_name,label_value)           
            auc = evaluator.calcAuc(k,label_name,label_value)
            #"flip precision and recall coordinates"
            precisionRecalls_p_train.append((pos_r,pos_p))
            precisionRecalls_n_train.append((neg_r,neg_p))                      
#            precisionRecalls_train.append((r,p))
            ROCs_train.append(roc)
            AUCs_train.append(auc)
            prop_arrs_train.append(evaluator.probabilities)
#            MAPs.append(map_std)
            print("finished {}".format(evaluator_name))
        evaluator_names_train = list(evaluators_train.keys())       
        
        with open("pickled/results_"+dataset_name_test+".pck",'wb') as fp:   
            pickle.dump([precisionRecalls_p_test,precisionRecalls_n_test,ROCs_test,AUCs_test,evaluator_names_test,prop_arrs_test],fp)                    
        with open("pickled/results_"+dataset_name_train+".pck",'wb') as fp:   
            pickle.dump([precisionRecalls_p_train,precisionRecalls_n_train,ROCs_train,AUCs_train,evaluator_names_train,prop_arrs_train],fp)                               
            
        self.generatePlots(dataset_name_test)
        self.generatePlots(dataset_name_train)       

        
    def generatePlots(self,dataset):
        with open("pickled/results_"+dataset+".pck",'rb') as fp:   
            precisionRecalls_p,precisionRecalls_n,ROCs,AUCs,evaluator_names,prop_arrs = pickle.load(fp)                               
        roc_filename = dataset+"_ROC.png"
        self.plotCurves(ROCs,evaluator_names,"False positve rate","True positive rate","ROC curve, data: {}".format(dataset),roc_filename)
        pr_filename_p = dataset+"_PR_p.png"
        pr_filename_n = dataset+"_PR_n.png"       
        self.plotCurves(precisionRecalls_p,evaluator_names,"Recall","Precision","Positive precision-recall curve, data: {}".format(dataset),pr_filename_p)       
        self.plotCurves(precisionRecalls_n,evaluator_names,"Recall","Precision","Negative precision-recall curve, data: {}".format(dataset),pr_filename_n)              
        self.plotAUC(AUCs,evaluator_names,dataset)       

        
    def plotAUC(self,aucs,names,dataset_name):
        l = len(aucs)
        plt.axis([0,l+1,0,1])
        plt.xlabel("Methods")
#        names = list(evaluators.keys())
        plt.xticks(np.arange(1,l+1),names)
        
        plt.ylabel("AUC score")
        plt.title("AUC values across methods")                  
        for i in range(l):
            plt.bar(i+1,aucs[i])
            plt.text(i+0.7,round(aucs[i],3),"{0:.3f}".format(aucs[i]))
        plt.savefig("results/"+dataset_name+"_AUC.png")
        plt.clf()                  
    
    def plotLoss(self,history,dataset_name):
        plt.plot(history.accuracy)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Samples')
        plt.savefig("results/"+dataset_name+"_loss.png")
        plt.clf()                  
#    def plotLoss(self,losses,dataset_name):
#        l = len(losses)
##        plt.axis([0,l+1,0,1])
#        plt.xlabel("Epochs")
##        names = list(evaluators.keys())
##        plt.xticks(np.arange(1,l+1),names)
#        
#        plt.ylabel("Loss")
#        plt.title("Loss curve for CNN used on {}".format(dataset_name))                  
#        plt.plot(range(l),losses)
##        for i in range(l):
##            plt.bar(i+1,aucs[i])
##            plt.text(i+0.7,round(aucs[i]-0.05,3),"{0:.3f}".format(aucs[i]))
#        plt.savefig("results/"+dataset_name+"_loss.png")
#        plt.clf()                     
    
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
        k = self.k_vals[dataset_name_train]
        if type(train_data) is SimpleShapeData:
            slices = train_data.scans #scans_train
            only_2d = True
            slices_3 = slices
        else:
            slices = train_data.getScansAtSlice()
            only_2d = False
            slices_3 = train_data.getScansAtSlice(three_slices=True)
        self.haralicks.featureSelector(5,k,slices,train_data.labels,label_name,label_value)       #5
        self.lbp.fit_parameters(k,slices,train_data.labels,label_name,label_value)
        history = self.vgg_fitted.fit(slices_3,train_data.labels,label_name,label_value)
        self.plotLoss(history,dataset_name_train)
#        losses = self.ml.optimize(train_data,int((train_data.count)/5))#num_iterations=1)
#        self.plotLoss(losses,dataset_name_train)
        with open("results/method_parameters"+dataset_name_train+".txt","w+") as f:
            f.write("haralick selected features: "+ str(self.haralicks.selected_features)+"\n")
            lbp_str = "lbp radius: %d, n_points: %d \n" % (self.lbp.radius,self.lbp.n_points)
            f.write(lbp_str)
        
        fitted = []
        haralicks_fitted = self.haralicks.selected_features
        lbp_fitted = self.lbp.radius,self.lbp.n_points       
        fitted.append(haralicks_fitted)
        fitted.append(lbp_fitted)
        if not only_2d:
            self.haralicks3d.featureSelector(5,k,train_data.scans,train_data.labels,label_name,label_value)              #5
            self.lbp3d.fit_parameters(k,train_data.scans,train_data.labels,label_name,label_value)       
            haralicks3d_fitted = self.haralicks3d.selected_features
            lbp3d_fitted = (self.lbp3d.lbp_xy.radius,self.lbp3d.lbp_xy.n_points,self.lbp3d.lbp_xz.radius,self.lbp3d.lbp_xz.n_points,self.lbp3d.lbp_yz.radius,self.lbp3d.lbp_yz.n_points)
            fitted.append(haralicks3d_fitted)
            fitted.append(lbp3d_fitted)           
            with open("results/method_parameters"+dataset_name_train+".txt","a") as f:
                f.write("haralick3d selected features: "+ str(self.haralicks3d.selected_features)+"\n")
                lbp_str = "lbp3d (xy radius: %d, n_points: %d),(xz radius: %d, n_points: %d),(yz radius: %d, n_points: %d) \n" %(self.lbp3d.lbp_xy.radius,self.lbp3d.lbp_xy.n_points,self.lbp3d.lbp_xz.radius,self.lbp3d.lbp_xz.n_points,self.lbp3d.lbp_yz.radius,self.lbp3d.lbp_yz.n_points)
                f.write(lbp_str)           
#                f.write(lbp_str)
        print("Done fitting")


        with open("pickled/fitted_methods_"+dataset_name_train+".pck",'wb') as fp:   
             pickle.dump(fitted,fp)                               
        
    def initEvaluators(self,test_data):
        only_2d = type(test_data) is SimpleShapeData
        e_haralicks= Evaluator(data_obj=test_data,method_obj=self.haralicks,load_as_slice=True and not only_2d)
        #return {"haralicks2d":e_haralicks}
        e_lbp= Evaluator(data_obj=test_data,method_obj=self.lbp,load_as_slice=True and not only_2d)
        e_vgg = Evaluator(data_obj=test_data,method_obj=self.vgg,load_as_slice=True and not only_2d,three_slices=True)       
        e_vgg_fitted = Evaluator(data_obj=test_data,method_obj=self.vgg_fitted,load_as_slice=True and not only_2d,three_slices=True)              
        #e_ml= Evaluator(data_obj=test_data,method_obj=self.ml,load_as_slice=True and not only_2d)
        eval_dict = {"haralicks2d":e_haralicks,"lbp2d":e_lbp,"vgg":e_vgg,"vgg_f":e_vgg_fitted} #,"ml":e_ml}
        if not only_2d:
            e_haralicks3d= Evaluator(data_obj=test_data,method_obj=self.haralicks3d,load_as_slice=False)
            e_lbp3d= Evaluator(data_obj=test_data,method_obj=self.lbp3d,load_as_slice=False)
            eval_dict["haralicks3d"] = e_haralicks3d
            eval_dict["lbp3d"] = e_lbp3d
        print("Done initializing evaluators")
        return eval_dict
#        return {"haralicks2d":e_haralicks}
    
