from train_and_test import train
from DataSource import *
target_label_dict = {}
source_label_dict = {}
target_train_dataset = TrainData("../Multivariate_ts", "SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts",target_label_dict)
target_test_dataset = TestData("../Multivariate_ts", "SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts",target_label_dict)
source_train_dataset = TrainData("../Multivariate_ts","MotorImagery/MotorImagery_TRAIN.ts",source_label_dict)
source_test_dataset = TestData("../Multivariate_ts", "MotorImagery/MotorImagery_TEST.ts",source_label_dict)
train(target_train_dataset,target_test_dataset,source_train_dataset,source_test_dataset,True)