from resultGenerator import *
import time
#pickleData()
start = time.time()
pickles = ["pickled/toy_data_train.pck","pickled/toy_data.pck","pickled/erik_train.pck","pickled/erik_test.pck",
           "pickled/mrnet_abnormal_train.pck","pickled/mrnet_abnormal_test.pck",
           "pickled/mrnet_meniscus_train.pck","pickled/mrnet_meniscus_test.pck",
           "pickled/mrnet_acl_train.pck","pickled/mrnet_acl_test.pck"]
names = ["artificial_data_train","artificial_data","OA_train","OA_test","mrnet_abnormal_train",
         "mrnet_abnormal_test","mrnet_meniscus_train","mrnet_meniscus_test",
         "mrnet_acl_train","mrnet_acl_test"]
resg = ResultGenerator(pickles,names)
#resg.generatePlots("toy_data_train")
#resg.generatePlots("toy_data")

resg.genResultsForDataset("artificial_data_train","artificial_data","shape","triangle")
resg.genResultsForDataset("OA_train","OA_test","ishealthy",True)
resg.genResultsForDataset("mrnet_abnormal_train","mrnet_abnormal_test","value",True)
resg.genResultsForDataset("mrnet_meniscus_train","mrnet_meniscus_test","value",True)
resg.genResultsForDataset("mrnet_acl_train","mrnet_acl_test","value",True)
end = time.time()
print(end-start)
