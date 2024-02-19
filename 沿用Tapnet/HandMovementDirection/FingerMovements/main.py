from train_and_test import train
from DataSource import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
target_label_dict = {}
source_label_dict = {}
target_train_dataset = TrainData("../Multivariate_ts", "HandMovementDirection/HandMovementDirection_TRAIN.ts",target_label_dict)
target_test_dataset = TestData("../Multivariate_ts", "HandMovementDirection/HandMovementDirection_TEST.ts",target_label_dict)
source_train_dataset = TrainData("../Multivariate_ts","FingerMovements/FingerMovements_TRAIN.ts",source_label_dict)
source_test_dataset = TestData("../Multivariate_ts", "FingerMovements/FingerMovements_TEST.ts",source_label_dict)
train(target_train_dataset,target_test_dataset,source_train_dataset,source_test_dataset,True)