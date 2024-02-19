from DataSource import TrainData, TestData
from train_and_test import train

target_label_dict = {}
source_label_dict = {}
target_train_dataset = TrainData("Multivariate_ts", "SelfRegulationSCP1/SelfRegulationSCP1_TRAIN.ts",target_label_dict)
target_test_dataset = TestData("Multivariate_ts", "SelfRegulationSCP1/SelfRegulationSCP1_TEST.ts",target_label_dict)
source_train_dataset = TrainData("Multivariate_ts","SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts",source_label_dict) #地址问题可以结合着输出想
source_test_dataset = TestData("Multivariate_ts", "SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts",source_label_dict)
train(target_train_dataset,target_test_dataset,source_train_dataset,source_test_dataset,True)