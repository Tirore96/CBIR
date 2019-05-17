from resultGenerator import *
import time
#resg = ResultGenerator(["pickled/tiny_data.pck"],["tiny"])
#resg.genResultsForDataset("tiny","tiny","ishealthy",True)
#resg.genResultsForDataset("tiny","tiny","ishealthy",True)

start = time.time()
pickles = ["pickled/toy_data.pck","pickled/erik_train.pck","pickled/erik_test.pck",
           "pickled/mrnet_abnormal_train.pck","pickled/mrnet_abnormal_test.pck",
           "pickled/mrnet_meniscus_train.pck","pickled/mrnet_meniscus_test.pck",
           "pickled/mrnet_acl_train.pck","pickled/mrnet_acl_test.pck"]
names = ["toy_data","erik_train","erik_test","mrnet_abnormal_train",
         "mrnet_abnormal_test","mrnet_meniscus_train","mrnet_meniscus_test",
         "mrnet_acl_train","mrnet_acl_test"]
resg = ResultGenerator(pickles,names)
#resg.genResultsForDataset("toy_data","toy_data","shape","triangle")
#resg.genResultsForDataset("erik_train","erik_test","ishealthy",True)
#resg.genResultsForDataset("mrnet_abnormal_train","mrnet_abnormal_test","value",True)
resg.genResultsForDataset("mrnet_meniscus_train","mrnet_meniscus_test","value",True)
resg.genResultsForDataset("mrnet_acl_train","mrnet_acl_test","value",True)
end = time.time()
print(end-start)
