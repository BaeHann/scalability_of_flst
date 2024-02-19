#from train_and_test import train
from train_and_test_extremely_large import train
from DataSource import *
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
target_label_dict = {}
source_label_dict = {}
target_train_dataset = TrainData("../Multivariate_ts", "EigenWorms/EigenWorms_TRAIN.ts",target_label_dict)
target_test_dataset = TestData("../Multivariate_ts", "EigenWorms/EigenWorms_TEST.ts",target_label_dict)
source_train_dataset = TrainData("../Multivariate_ts","MotorImagery/MotorImagery_TRAIN.ts",source_label_dict)
source_test_dataset = TestData("../Multivariate_ts", "MotorImagery/MotorImagery_TEST.ts",source_label_dict)
train(target_train_dataset,target_test_dataset,source_train_dataset,source_test_dataset,True)