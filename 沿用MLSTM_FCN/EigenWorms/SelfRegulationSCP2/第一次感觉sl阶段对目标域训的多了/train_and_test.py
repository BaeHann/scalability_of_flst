import numpy as np
import os
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader

from Comparison.SLARDA.models import Seq_Transformer
from Comparison.SLARDA.train import CPC
from DataSource import TrainData, TestData
from C_DAN import RandomLayer, CDAN
from MLSTM_FCN import MLSTMfcn
from widgets import DimensionUnification, ProbTransfer, NoiseTransfer, AdversarialNetworkforCDAN, \
    FeatureDiscriminatorforSource, wgan_loss, init_weights, calc_coeff, grl_hook

def eval_model_testdata(target_feature_extraction_module, target_classification_module,test_dataloader,\
                        cur_epoch,with_nvidia=True):
    predict_list = np.array([])
    label_list = np.array([])
    for i, (x, y) in enumerate(test_dataloader):
        if with_nvidia:
            x = x.float().cuda()
            y_predict, _ = target_classification_module(target_feature_extraction_module(x))
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
        else:
            y_predict, _ = target_classification_module(target_feature_extraction_module(x))
            y_predict = y_predict.detach().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    #np.save("numpy_saved_with_accuracy/predicted_and_true_label/epoch_"+str(cur_epoch)+"_test_predict.npy",predict_list)
    #np.save("numpy_saved_with_accuracy/predicted_and_true_label/epoch_" + str(cur_epoch) + "_test_true.npy",label_list)
    str_out = "epoch_num:" + str(cur_epoch) + " accuracy_for_test:" + str(acc)
    with open("numpy_saved_with_accuracy/the_log.txt", "a", encoding='utf-8') as f:
        f.write(str_out + "\n")
        f.close()
    print(str_out)

def eval_model_traindata(target_feature_extraction_module, target_classification_module,train_dataloader,\
                         cur_epoch,with_nvidia=True):
    predict_list = np.array([])
    label_list = np.array([])
    for i, (x, y) in enumerate(train_dataloader):
        if with_nvidia:
            x = x.float().cuda()
            y_predict, _ = target_classification_module(target_feature_extraction_module(x))
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
        else:
            y_predict, _ = target_classification_module(target_feature_extraction_module(x))
            y_predict = y_predict.detach().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    #np.save("numpy_saved_with_accuracy/predicted_and_true_label/epoch_" + str(cur_epoch) + "_train_predict.npy",predict_list)
    #np.save("numpy_saved_with_accuracy/predicted_and_true_label/epoch_" + str(cur_epoch) + "_train_true.npy", label_list)
    str_out = "epoch_num:"+str(cur_epoch)+" accuracy_for_train:"+str(acc)
    with open("numpy_saved_with_accuracy/the_log.txt", "a", encoding='utf-8') as f:
        f.write(str_out + "\n")
        f.close()
    print(str_out)

def eval_source_model_traindata(source_feature_extraction_module,source_classification_module,train_dataloader,cur_epoch,with_nvidia=True):
    predict_list = np.array([])
    label_list = np.array([])
    for i, (x, y) in enumerate(train_dataloader):
        if with_nvidia:
            x = x.float().cuda()
            y_predict, _ = source_classification_module(source_feature_extraction_module(x))
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
        else:
            y_predict, _ = source_classification_module(source_feature_extraction_module(x))
            y_predict = y_predict.detach().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    #np.save("numpy_saved_with_accuracy/source_predicted_and_true_label/epoch_" + str(cur_epoch) + "_train_predict.npy",predict_list)
    #np.save("numpy_saved_with_accuracy/source_predicted_and_true_label/epoch_" + str(cur_epoch) + "_train_true.npy", label_list)
    str_out = "epoch_num:"+str(cur_epoch)+" accuracy_for_source_train:"+str(acc)
    with open("numpy_saved_with_accuracy/the_log.txt", "a", encoding='utf-8') as f:
        f.write(str_out + "\n")
        f.close()
    print(str_out)

def eval_source_model_testdata(source_feature_extraction_module, source_classification_module,test_dataloader,cur_epoch,with_nvidia=True):
    predict_list = np.array([])
    label_list = np.array([])
    for i, (x, y) in enumerate(test_dataloader):
        if with_nvidia:
            x = x.float().cuda()
            y_predict, _ = source_classification_module(source_feature_extraction_module(x))
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
        else:
            y_predict, _ = source_classification_module(source_feature_extraction_module(x))
            y_predict = y_predict.detach().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    #np.save("numpy_saved_with_accuracy/source_predicted_and_true_label/epoch_"+str(cur_epoch)+"_test_predict.npy",predict_list)
    #np.save("numpy_saved_with_accuracy/source_predicted_and_true_label/epoch_" + str(cur_epoch) + "_test_true.npy",label_list)
    str_out = "epoch_num:" + str(cur_epoch) + " accuracy_for_source_test:" + str(acc)
    with open("numpy_saved_with_accuracy/the_log.txt", "a", encoding='utf-8') as f:
        f.write(str_out + "\n")
        f.close()
    print(str_out)

def save_target_classification_modules(target_feature_extraction_module, target_classification_module,\
                                       cur_epoch):
    torch.save({
        'epoch': cur_epoch,
        'feature_extraction_state_dict': target_feature_extraction_module.state_dict(),
        'classification_state_dict': target_classification_module.state_dict(),
    }, "train_log/epoch_"+str(cur_epoch)+".tar")


def save_source_classification_modules(source_feature_extraction_module, source_classification_module,cur_epoch):
    torch.save({
        'epoch': cur_epoch,
        'feature_extraction_state_dict': source_feature_extraction_module.state_dict(),
        'classification_state_dict': source_classification_module.state_dict(),
    }, "train_log/epoch_"+str(cur_epoch)+"_source.tar")



def eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module,\
                                       target_dataloader,cur_epoch,whether_test=False,with_nvidia=True):
    predict_list = np.array([])
    label_list = np.array([])
    for i, (x, y) in enumerate(target_dataloader):
        if with_nvidia:
            x = x.float().cuda()
            y_predict, _ = target_classification_module(target_feature_extraction_module(x))
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
        else:
            y_predict, _ = target_classification_module(target_feature_extraction_module(x))
            y_predict = y_predict.detach().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    if whether_test==False:
        str_out = "epoch_num:"+str(cur_epoch)+" accuracy_for_train:"+str(acc)
    else:
        str_out = "epoch_num:" + str(cur_epoch) + " accuracy_for_test:" + str(acc)
    print(str_out)
def eval_source_model_being_pretrained(source_feature_extraction_module:DimensionUnification,\
                               source_classification_module,source_dataloader,cur_epoch,whether_test=False,with_nvidia=True):
    predict_list = np.array([])
    label_list = np.array([])
    for i, (x, y) in enumerate(source_dataloader):
        if with_nvidia:
            x = x.float().cuda()
            y_predict, _ = source_classification_module(source_feature_extraction_module(x))
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
        else:
            y_predict, _ = source_classification_module(source_feature_extraction_module(x))
            y_predict = y_predict.detach().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    if whether_test:
        str_out = "epoch_num:" + str(cur_epoch) + " accuracy_for_source_test:" + str(acc)
    else:
        str_out = "epoch_num:" + str(cur_epoch) + " accuracy_for_source_train:" + str(acc)
    print(str_out)
class Discriminator_ATT(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self, patch_size, att_hid_dim, depth, heads, mlp_dim):
        """Init discriminator."""
        self.patch_size =  patch_size
        self.hid_dim = att_hid_dim
        self.depth= depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        super(Discriminator_ATT, self).__init__()
        self.transformer1= Seq_Transformer(patch_size=self.patch_size, dim=att_hid_dim, depth=self.depth, heads= self.heads , mlp_dim=self.mlp_dim)
        self.DC = nn.Linear(att_hid_dim, 1)
        #self.apply(init_weights)
        self.iter_num = -1
        self.alpha = 100.0
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 20.0
        self.coeff = np.float(0.001)
    def forward(self, input):
        """Forward the discriminator."""
        if self.training:
            self.iter_num += 1
        if self.iter_num >= self.max_iter:
            self.iter_num = self.max_iter
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        self.coeff = coeff
        input = input * 1.0
        input.register_hook(grl_hook(coeff))
        # src_shape = [batch_size, seq_len, input_dim]
        input = input.view(input.size(0),-1, self.patch_size )
        features = self.transformer1(input)
        domain_output = self.DC(features)
        return domain_output
def train(target_train_dataset:TrainData,target_test_dataset:TestData,source_train_dataset:TrainData,source_test_dataset:TestData,\
          with_nvidia=False,epoch_num=650):
    #with torch.autograd.set_detect_anomaly(True):
        # 获取原始信号的长度、通道数以便后续使用
    target_original_length = target_train_dataset.time_length
    target_original_channel = target_train_dataset.in_channel
    target_num_class = target_train_dataset.num_class
    source_original_length = source_train_dataset.time_length
    source_original_channel = source_train_dataset.in_channel
    source_num_class = source_train_dataset.num_class
    """
    target_feature_extraction_module = nn.Sequential(
        nn.Linear(target_original_length,target_original_length//2),
        nn.ReLU(),
        nn.Linear(target_original_length//2,target_original_length),
    )
    """
    target_feature_extraction_module = DimensionUnification(target_original_channel,target_original_channel,\
                                                            target_original_length,target_original_length)
    target_classification_module = MLSTMfcn(num_classes=target_num_class,num_features=target_original_channel)
    #创建source数据集的模块与source到target转换模块的合并
    source_feature_extraction_module = DimensionUnification(source_original_channel,target_original_channel,\
                                                            source_original_length,target_original_length)

    source_classification_module = MLSTMfcn(num_classes=source_num_class,num_features=target_original_channel)
    #probtrasfer implemented thorugh the transformation of features before Linear
    feature_transfer_between_t_s = ProbTransfer(source_classification_module.length_before_classification)
    #modules required by CDAN in the target side
    input_length_for_cdan = 1024
    random_layer_for_cdan = RandomLayer([target_original_channel*target_original_length,target_num_class])
    #ad_net = AdversarialNetworkforCDAN(input_length_for_cdan,1024)
    ad_net = Discriminator_ATT(input_length_for_cdan, 128, 8, 8, 64).float()

    #modules required by probability discrimination in the source side
    #feature_discriminator_s = FeatureDiscriminatorforSource(source_classification_module.length_before_classification)
    classification_loss_module = nn.CrossEntropyLoss()
    #决定运算位置
    if with_nvidia:
        target_feature_extraction_module = target_feature_extraction_module.cuda()
        target_classification_module = target_classification_module.cuda()
        source_feature_extraction_module = source_feature_extraction_module.cuda()
        #source_to_target_feature_trans = source_to_target_feature_trans.cuda()
        source_classification_module = source_classification_module.cuda()
        feature_transfer_between_t_s = feature_transfer_between_t_s.cuda()
        #nf_for_transfer = nf_for_transfer.cuda()
        #nf_loss = nf_loss.cuda()
        #noise_confusion_for_nf = noise_confusion_for_nf.cuda()
        random_layer_for_cdan = random_layer_for_cdan.cuda() #无可学习参数，不用optimizer
        ad_net = ad_net.cuda()
        #feature_discriminator_s = feature_discriminator_s.cuda()
        classification_loss_module = classification_loss_module.cuda()
    #optimizer and scehduler
    optimizer_target_feature_extraction = torch.optim.RMSprop(target_feature_extraction_module.parameters(),lr=0.001)
    optimizer_target_classification = torch.optim.RMSprop(target_classification_module.parameters(),lr=0.003)
    optimizer_source_feature_extraction = torch.optim.RMSprop(source_feature_extraction_module.parameters(),lr=0.001)
    #optimizer_source_to_target_feature_trans = torch.optim.RMSprop(source_to_target_feature_trans.parameters(),lr=0.001)
    optimizer_source_classification = torch.optim.RMSprop(source_classification_module.parameters(),lr=0.003)
    optimizer_feature_transfer_between_t_s = torch.optim.RMSprop(feature_transfer_between_t_s.parameters(),lr=0.001)
    #optimizer_nf_for_transfer = torch.optim.RMSprop(nf_for_transfer.parameters(),lr=0.001)
    #optimizer_noise_confusion_for_nf = torch.optim.RMSprop(noise_confusion_for_nf.parameters(),lr=0.005)
    optimizer_ad_net = torch.optim.RMSprop(ad_net.parameters(),lr=0.001)
    #optimizer_feature_discriminator_s = torch.optim.RMSprop(feature_discriminator_s.parameters(),lr=0.001)
    optimizer_list = []
    optimizer_list.append(optimizer_target_feature_extraction)
    optimizer_list.append(optimizer_target_classification)
    optimizer_list.append(optimizer_source_feature_extraction)
    #optimizer_list.append(optimizer_source_to_target_feature_trans)
    optimizer_list.append(optimizer_source_classification)
    optimizer_list.append(optimizer_feature_transfer_between_t_s)
    #optimizer_list.append(optimizer_nf_for_transfer)
    #optimizer_list.append(optimizer_noise_confusion_for_nf)
    optimizer_list.append(optimizer_ad_net)
    #optimizer_list.append(optimizer_feature_discriminator_s)
    scheduler_target_feature_extraction = torch.optim.lr_scheduler.StepLR(optimizer_target_feature_extraction, step_size=25, gamma=0.8)
    scheduler_target_classification = torch.optim.lr_scheduler.StepLR(optimizer_target_classification, step_size=25, gamma=0.8)
    scheduler_source_feature_extraction = torch.optim.lr_scheduler.StepLR(optimizer_source_feature_extraction, step_size=25, gamma=0.8)
    #scheduler_source_to_target_feature_trans = torch.optim.lr_scheduler.StepLR(optimizer_source_to_target_feature_trans, step_size=25, gamma=0.8)
    scheduler_source_classification = torch.optim.lr_scheduler.StepLR(optimizer_source_classification, step_size=25, gamma=0.8)
    scheduler_feature_transfer_between_t_s = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_feature_transfer_between_t_s,\
                                                                                        'min',factor=0.7, min_lr=0.0001)
    #scheduler_nf_for_transfer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_nf_for_transfer,'min',factor=0.7, min_lr=0.0001)
    #scheduler_noise_confusion_for_nf = torch.optim.lr_scheduler.StepLR(optimizer_noise_confusion_for_nf, step_size=55,gamma=0.6)
    scheduler_ad_net = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ad_net,'min',factor=0.7, min_lr=0.0001)
    #scheduler_feature_discriminator_s = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_feature_discriminator_s,\
                                            #                                       'min',factor=0.7, min_lr=0.0001)
    SL_CPC = CPC(target_original_channel, 64, target_original_length // 2)
    SL_CPC = SL_CPC.cuda()
    optimizer_sl_cpc = torch.optim.Adam(SL_CPC.parameters(), lr=0.002)
    scheduler_sl_cpc = torch.optim.lr_scheduler.StepLR(optimizer_sl_cpc, step_size=25, gamma=0.7)

    target_train_loader = DataLoader(target_train_dataset, batch_size=20, shuffle=True)
    source_train_loader = DataLoader(source_train_dataset, batch_size=20, shuffle=True)
    target_test_loader = DataLoader(target_test_dataset, batch_size=20, shuffle=True)
    source_test_loader = DataLoader(source_test_dataset,batch_size=20,shuffle=True)

    #首先各自训练target和source的分类所需modules(训练多少个epoch最好先用test.py大概看一下)
    print("pretrain the target classification modules-----------------------------------------------------------")
    target_epoch_pretrain = 5
    #for cur_epoch in range(0):
    for cur_epoch in range(3):
        target_feature_extraction_module.train()
        target_classification_module.train()
        target_data = list(enumerate(target_train_loader))
        rounds_per_epoch = len(target_data)
        for batch_idx in range(rounds_per_epoch):
            _, (target_train, target_label) = target_data[batch_idx]
            if with_nvidia:
                target_train = target_train.float().cuda()
                target_label = target_label.cuda()
            target_feature = target_feature_extraction_module(target_train)
            t_sl_loss = SL_CPC(target_feature)
            target_classification_result, target_before_last_linear = target_classification_module(target_feature)
            target_classification_loss = classification_loss_module(target_classification_result, target_label)
            if with_nvidia:
                str_out = "Epoch:"+str(cur_epoch)+" batch_num:"+str(batch_idx)+" t_c_loss:"+str(target_classification_loss.data.cpu().numpy())\
                          + " t_sl_loss:"+str(t_sl_loss.data.cpu().numpy())
            else:
                str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_c_loss:" + str(target_classification_loss.data.numpy())\
                          + " t_sl_loss:"+str(t_sl_loss.data.numpy())
            t_total_Loss = target_classification_loss + t_sl_loss
            print(str_out)
            t_total_Loss.backward()
            optimizer_target_feature_extraction.step()
            optimizer_target_classification.step()
            optimizer_sl_cpc.step()
            optimizer_target_feature_extraction.zero_grad()
            optimizer_target_classification.zero_grad()
            optimizer_sl_cpc.zero_grad()
        scheduler_target_feature_extraction.step()
        scheduler_target_classification.step()
        scheduler_sl_cpc.step()
        target_feature_extraction_module.eval()
        target_classification_module.eval()
        eval_target_model_being_pretrained(target_feature_extraction_module,target_classification_module,target_train_loader,\
                                           cur_epoch)
        eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module,target_test_loader,\
                                           cur_epoch,True)
    print("pretrain the source classification modules-----------------------------------------------------------")
    source_epoch_pretrain = 120
    #for cur_epoch in range(0):
    for cur_epoch in range(source_epoch_pretrain):
        source_feature_extraction_module.train()
        #source_to_target_feature_trans.train()
        source_classification_module.train()
        source_data = list(enumerate(source_train_loader))
        rounds_per_epoch = len(source_data)
        for batch_idx in range(rounds_per_epoch):
            _, (source_train, source_label) = source_data[batch_idx]
            if with_nvidia:
                source_train = source_train.float().cuda()
                source_label = source_label.cuda()
            source_shape_changed_feature = source_feature_extraction_module(source_train)
            #source_shape_changed_feature = source_to_target_feature_trans(source_feature)
            source_classification_result, source_before_last_linear = source_classification_module(
                source_shape_changed_feature)
            source_classification_loss = classification_loss_module(source_classification_result, source_label)
            if with_nvidia:
                str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " s_c_loss:" + str(
                    source_classification_loss.data.cpu().numpy())
            else:
                str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " s_c_loss:" + str(
                    source_classification_loss.data.numpy())

            print(str_out)
            source_classification_loss.backward()
            optimizer_source_feature_extraction.step()
            #optimizer_source_to_target_feature_trans.step()
            optimizer_source_classification.step()
            optimizer_source_feature_extraction.zero_grad()
            #optimizer_source_to_target_feature_trans.zero_grad()
            optimizer_source_classification.zero_grad()
        scheduler_source_feature_extraction.step()
        #scheduler_source_to_target_feature_trans.step()
        scheduler_source_classification.step()
        source_feature_extraction_module.eval()
        #source_to_target_feature_trans.eval()
        source_classification_module.eval()
        eval_source_model_being_pretrained(source_feature_extraction_module,source_classification_module, \
                                           source_train_loader, cur_epoch)
        eval_source_model_being_pretrained(source_feature_extraction_module,source_classification_module, \
                                           source_test_loader, cur_epoch, True)
    print("self-supervised learning for target and source dataset-----------------------------------------------")
    #for cur_epoch in range(0):
    for cur_epoch in range(55 * target_epoch_pretrain):
        target_feature_extraction_module.train()
        target_classification_module.train()
        source_feature_extraction_module.train()
        #source_to_target_feature_trans.train()
        source_classification_module.train()
        SL_CPC.train()
        target_data = list(enumerate(target_train_loader))
        source_data = list(enumerate(source_train_loader))
        rounds_per_epoch = min(len(target_data), len(source_data))
        if cur_epoch % 50 == 0:
            for batch_idx in range(rounds_per_epoch):
                _, (target_train, target_label) = target_data[batch_idx]
                _, (source_train, source_label) = source_data[batch_idx]
                if with_nvidia:
                    target_train = target_train.float().cuda()
                    target_label = target_label.cuda()
                    source_train = source_train.float().cuda()
                    source_label = source_label.cuda()  # 这些label别习惯性地加上.float()
                target_feature = target_feature_extraction_module(target_train)
                t_sl_loss = SL_CPC(target_feature)
                target_classification_result, target_before_last_linear = target_classification_module(
                    target_feature)
                target_classification_loss = classification_loss_module(target_classification_result, target_label)
                source_shape_changed_feature = source_feature_extraction_module(source_train)
                #source_shape_changed_feature = source_to_target_feature_trans(source_feature)
                s_sl_loss = SL_CPC(source_shape_changed_feature)
                source_classification_result, source_before_last_linear = source_classification_module(
                    source_shape_changed_feature)
                source_classification_loss = classification_loss_module(source_classification_result, source_label)
                if with_nvidia:
                    str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_c_loss:" + str(
                        target_classification_loss.data.cpu().numpy()) \
                              + " t_sl_loss:" + str(t_sl_loss.data.cpu().numpy()) + " s_c_loss:" + str(
                        source_classification_loss.data.cpu().numpy()) \
                              + " s_sl_loss:" + str(s_sl_loss.data.cpu().numpy())
                else:
                    str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_c_loss:" + str(
                        target_classification_loss.data.numpy()) \
                              + " t_sl_loss:" + str(t_sl_loss.data.numpy()) + " s_c_loss:" + str(
                        source_classification_loss.data.numpy()) \
                              + " s_sl_loss:" + str(s_sl_loss.data.numpy())
                t_total_Loss = t_sl_loss + s_sl_loss + 0.8 * target_classification_loss + 1.2 * source_classification_loss
                print(str_out)
                t_total_Loss.backward()
                optimizer_target_feature_extraction.step()
                optimizer_target_classification.step()
                optimizer_sl_cpc.step()
                optimizer_source_feature_extraction.step()
                #optimizer_source_to_target_feature_trans.step()
                optimizer_source_classification.step()
                optimizer_source_feature_extraction.zero_grad()
                #optimizer_source_to_target_feature_trans.zero_grad()
                optimizer_source_classification.zero_grad()
                optimizer_target_feature_extraction.zero_grad()
                optimizer_target_classification.zero_grad()
                optimizer_sl_cpc.zero_grad()
            scheduler_target_feature_extraction.step()
            scheduler_target_classification.step()
            scheduler_sl_cpc.step()
            scheduler_source_feature_extraction.step()
            #scheduler_source_to_target_feature_trans.step()
            scheduler_source_classification.step()
            source_feature_extraction_module.eval()
            #source_to_target_feature_trans.eval()
            source_classification_module.eval()
            target_feature_extraction_module.eval()
            target_classification_module.eval()
            eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module,
                                               target_train_loader, cur_epoch)
            eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module,
                                               target_test_loader, cur_epoch, True)
            eval_source_model_being_pretrained(source_feature_extraction_module,source_classification_module, \
                                               source_train_loader, cur_epoch)
            eval_source_model_being_pretrained(source_feature_extraction_module,source_classification_module, \
                                               source_test_loader, cur_epoch, True)
        else:
            for batch_idx in range(rounds_per_epoch):
                _, (target_train, target_label) = target_data[batch_idx]
                _, (source_train, source_label) = source_data[batch_idx]
                if with_nvidia:
                    target_train = target_train.float().cuda()
                    target_label = target_label.cuda()
                    source_train = source_train.float().cuda()
                    source_label = source_label.cuda()  # 这些label别习惯性地加上.float()
                target_feature = target_feature_extraction_module(target_train)
                t_sl_loss = SL_CPC(target_feature)
                target_classification_result, target_before_last_linear = target_classification_module(
                    target_feature)
                target_classification_loss = classification_loss_module(target_classification_result, target_label)
                source_shape_changed_feature = source_feature_extraction_module(source_train)
                #source_shape_changed_feature = source_to_target_feature_trans(source_feature)
                s_sl_loss = SL_CPC(source_shape_changed_feature)
                source_classification_result, source_before_last_linear = source_classification_module(
                    source_shape_changed_feature)
                source_classification_loss = classification_loss_module(source_classification_result, source_label)
                if with_nvidia:
                    str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_c_loss:" + str(
                        target_classification_loss.data.cpu().numpy()) \
                              + " t_sl_loss:" + str(t_sl_loss.data.cpu().numpy()) + " s_c_loss:" + str(
                        source_classification_loss.data.cpu().numpy()) \
                              + " s_sl_loss:" + str(s_sl_loss.data.cpu().numpy())
                else:
                    str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_c_loss:" + str(
                        target_classification_loss.data.numpy()) \
                              + " t_sl_loss:" + str(t_sl_loss.data.numpy()) + " s_c_loss:" + str(
                        source_classification_loss.data.numpy()) \
                              + " s_sl_loss:" + str(s_sl_loss.data.numpy())
                t_total_Loss = t_sl_loss + s_sl_loss  # + 1.2*target_classification_loss + 1.2*source_classification_loss
                print(str_out)
                t_total_Loss.backward()
                optimizer_target_feature_extraction.step()
                # optimizer_target_classification.step()
                optimizer_sl_cpc.step()
                optimizer_source_feature_extraction.step()
                #optimizer_source_to_target_feature_trans.step()
                # optimizer_source_classification.step()
                optimizer_source_feature_extraction.zero_grad()
                #optimizer_source_to_target_feature_trans.zero_grad()
                # optimizer_source_classification.zero_grad()
                optimizer_target_feature_extraction.zero_grad()
                # optimizer_target_classification.zero_grad()
                optimizer_sl_cpc.zero_grad()
            scheduler_target_feature_extraction.step()
            # scheduler_target_classification.step()
            scheduler_sl_cpc.step()
            scheduler_source_feature_extraction.step()
            #scheduler_source_to_target_feature_trans.step()
            # scheduler_source_classification.step()
            source_feature_extraction_module.eval()
            #source_to_target_feature_trans.eval()
            source_classification_module.eval()
            target_feature_extraction_module.eval()
            target_classification_module.eval()
            eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module,
                                               target_train_loader, cur_epoch)
            eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module,
                                               target_test_loader, cur_epoch, True)
            eval_source_model_being_pretrained(source_feature_extraction_module, source_classification_module, \
                                               source_train_loader, cur_epoch)
            eval_source_model_being_pretrained(source_feature_extraction_module,source_classification_module, \
                                               source_test_loader, cur_epoch, True)
    torch.save({
        'feature_extraction_state_dict': target_feature_extraction_module.state_dict(),
        'classification_state_dict': target_classification_module.state_dict(),
    }, "train_log/target_classifier_itself.tar")
    torch.save({
        'feature_extraction_state_dict': source_feature_extraction_module.state_dict(),
        'classification_state_dict': source_classification_module.state_dict(),
    }, "train_log/source_classifier_itself.tar")

    print("jointly train in both target side and source side--------------------------------------")

    target_train_loader = DataLoader(target_train_dataset, batch_size=10, shuffle=True)
    source_train_loader = DataLoader(source_train_dataset, batch_size=10, shuffle=True)

    # 开始结合grad_norm进行整体训练
    # weights that grad_norm requires
    if with_nvidia:  # t和s各自来
        weights_grad_norm_t = nn.Parameter(torch.tensor([5]).float().cuda()) #！！！如果要调整数值的话下面更新处也要跟着变
        weights_grad_norm_s = nn.Parameter(torch.tensor([2, 4]).float().cuda())
    else:
        weights_grad_norm_t = nn.Parameter(torch.tensor([5]).float())
        weights_grad_norm_s = nn.Parameter(torch.tensor([2, 4]).float())
    optimizer_weights_t = torch.optim.Adam([weights_grad_norm_t], lr=0.0002)
    optimizer_weights_s = torch.optim.Adam([weights_grad_norm_s], lr=0.001)  # 暂时先不用调度器了

    initial_loss_t = None
    initial_loss_s = None
    alpha_for_grad_norm = 3
    # 调整成训练状态
    for cur_epoch in range(epoch_num):
        target_feature_extraction_module.train()
        target_classification_module.train()
        source_feature_extraction_module.train()
        #source_to_target_feature_trans.train()
        source_classification_module.train()
        feature_transfer_between_t_s.train()
        #nf_for_transfer.train()
        #nf_loss.train()
        #noise_confusion_for_nf.train()
        random_layer_for_cdan.train()
        ad_net.train()
        #feature_discriminator_s.train()

        # 保存各训练阶段的feature以便后续t-SNE
        """
        np_target_feature = None
        np_source_to_target_feature = None
        np_source_shape_changed_feature = None

        np_source_feature = None
        np_target_to_source_feature = None
        np_s2t2s_feature = None
        """
        target_data = list(enumerate(target_train_loader))
        source_data = list(enumerate(source_train_loader))
        rounds_per_epoch = min(len(target_data), len(source_data))
        #torch.backends.cudnn.enabled = False
        for batch_idx in range(rounds_per_epoch):
            _, (target_train, target_label) = target_data[batch_idx]
            _, (source_train, source_label) = source_data[batch_idx]
            if with_nvidia:
                target_train = target_train.float().cuda()
                target_label = target_label.cuda()
                source_train = source_train.float().cuda()
                source_label = source_label.cuda()  # 这些label别习惯性地加上.float()
            target_feature = target_feature_extraction_module(target_train)
            t_sl_loss = SL_CPC(target_feature)
            source_shape_changed_feature = source_feature_extraction_module(source_train)
            #source_shape_changed_feature = source_to_target_feature_trans(source_feature)
            s_sl_loss = SL_CPC(source_shape_changed_feature)


            """
            # 保存中间feature以便后续可视化
            if np_source_to_target_feature is None:
                if with_nvidia:
                    np_target_feature = target_feature.data.cpu().numpy()
                    np_source_to_target_feature = source_to_target_feature.data.cpu().numpy()
                    np_source_shape_changed_feature = source_shape_changed_feature.data.cpu().numpy()
                else:
                    np_target_feature = target_feature.data.numpy()
                    np_source_to_target_feature = source_to_target_feature.data.numpy()
                    np_source_shape_changed_feature = source_shape_changed_feature.data.numpy()
            else:
                if with_nvidia:
                    np_target_feature = np.concatenate((np_target_feature, target_feature.data.cpu().numpy()),
                                                       axis=0)
                    np_source_to_target_feature = np.concatenate(
                        (np_source_to_target_feature, source_to_target_feature.data.cpu().numpy()), axis=0)
                    np_source_shape_changed_feature = np.concatenate(
                        (np_source_shape_changed_feature, source_shape_changed_feature.data.cpu().numpy()), axis=0)
                else:
                    np_target_feature = np.concatenate((np_target_feature, target_feature.data.numpy()), axis=0)
                    np_source_to_target_feature = np.concatenate(
                        (np_source_to_target_feature, source_to_target_feature.data.numpy()), axis=0)
                    np_source_shape_changed_feature = np.concatenate(
                        (np_source_shape_changed_feature, source_shape_changed_feature.data.numpy()), axis=0)
            """
            target_classification_result, target_before_last_linear = target_classification_module(target_feature)
            #target_classification_module.eval()  # !!!
            target_classification_module.bn1.eval()
            target_classification_module.bn2.eval()
            target_classification_module.bn3.eval()
            source_to_target_classification_result, source_to_target_before_last_linear = target_classification_module(
                source_shape_changed_feature)
            #target_classification_module.train()
            target_classification_module.bn1.train()
            target_classification_module.bn2.train()
            target_classification_module.bn3.train()
            source_classification_result, source_before_last_linear = source_classification_module(
                source_shape_changed_feature)
            target_classification_loss = classification_loss_module(target_classification_result, target_label) # !!!
            # target_classification_loss = 100 * target_classification_loss #123123123
            source_classification_loss = classification_loss_module(source_classification_result,
                                                                    source_label)  # 不会有影响    #!!!
            # source_classification_loss = 100 * source_classification_loss #123123123
            # 涉及特征混合、迁移学习的损失函数
            cdan_loss = CDAN(target_feature, source_shape_changed_feature, target_classification_result, \
                             source_to_target_classification_result, ad_net, random_layer_for_cdan) # !!!
            # cdan_loss = 0.00001*cdan_loss   #123132131232132
            #transformed_target_before_last_linear = feature_transfer_between_t_s(target_before_last_linear)
            transformed_source_to_target_before = feature_transfer_between_t_s(source_to_target_before_last_linear)
            classification_result_s2t2s = source_classification_module.fc(transformed_source_to_target_before)
            s2t2s_classification_loss = classification_loss_module(classification_result_s2t2s, source_label)  # !!!
            # s2t2s_classification_loss = 50 * s2t2s_classification_loss  # 123123123
            """
            if np_target_to_source_feature is None:
                if with_nvidia:
                    np_source_feature = source_before_last_linear.data.cpu().numpy()
                    np_target_to_source_feature = transformed_target_before_last_linear.data.cpu().numpy()
                    np_s2t2s_feature = transformed_source_to_target_before.data.cpu().numpy()
                else:
                    np_source_feature = source_before_last_linear.data.numpy()
                    np_target_to_source_feature = transformed_target_before_last_linear.data.numpy()
                    np_s2t2s_feature = transformed_source_to_target_before.data.numpy()
            else:
                if with_nvidia:
                    np_source_feature = np.concatenate(
                        (np_source_feature, source_before_last_linear.data.cpu().numpy()), axis=0)
                    np_target_to_source_feature = np.concatenate(
                        (np_target_to_source_feature, transformed_target_before_last_linear.data.cpu().numpy()),
                        axis=0)
                    np_s2t2s_feature = np.concatenate(
                        (np_s2t2s_feature, transformed_source_to_target_before.data.cpu().numpy()), axis=0)
                else:
                    np_source_feature = np.concatenate((np_source_feature, source_before_last_linear.data.numpy()),
                                                       axis=0)
                    np_target_to_source_feature = np.concatenate(
                        (np_target_to_source_feature, transformed_target_before_last_linear.data.numpy()), axis=0)
                    np_s2t2s_feature = np.concatenate(
                        (np_s2t2s_feature, transformed_source_to_target_before.data.numpy()), axis=0)
            """
            # output the loss of the current batch
            if with_nvidia:
                str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_c_loss:" + str(
                    target_classification_loss.data.cpu().numpy()) \
                          + " t_sl_loss" + str(t_sl_loss.data.cpu().numpy()) + " s_c_loss:" + str(
                    source_classification_loss.data.cpu().numpy()) + \
                          " s_sl_loss:" + str(s_sl_loss.data.cpu().numpy()) + " cdan_loss:" + str(
                    cdan_loss.data.cpu().numpy()) + " s2t2s_c_loss:" + \
                          str(s2t2s_classification_loss.data.cpu().numpy()) + " weight_t" + \
                          str(weights_grad_norm_t.data.cpu().numpy()) + " weight_s" + str(
                    weights_grad_norm_s.data.cpu().numpy())
                # 不知道为什么不能在上方用target_nf_loss.data.cpu().numpy()[0]  这个[0]
            else:
                str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_c_loss:" + str(
                    target_classification_loss.data.numpy()[0]) \
                          + " t_sl_loss" + str(t_sl_loss.data.numpy()) + " s_c_loss:" + str(
                    source_classification_loss.data.numpy()[0]) + \
                          " s_sl_loss:" + str(s_sl_loss.data.numpy()) + " cdan_loss:" + str(
                    cdan_loss.data.numpy()[0]) + \
                          " s2t2s_c_loss:" + str(
                    s2t2s_classification_loss.data.numpy()[0]) + " weight_t" + \
                          str(weights_grad_norm_t.data.numpy()) + " weight_s" + str(
                    weights_grad_norm_s.data.numpy())
            print(str_out)
            with open("train_log/log.txt", "a", encoding='utf-8') as f:
                f.write(str_out + "\n")
                f.close()
            # 进行grad_norm辅助下的优化
            loss_t = []
            loss_s = []
            #loss_t.append(target_nf_loss)
            # loss_t.append(cdan_loss)   #对抗训练损失函数不定，先不参与grad_norm了
            loss_t.append(target_classification_loss)
            #loss_s.append(source_nf_loss)
            loss_s.append(source_classification_loss)
            # loss_s.append(feature_discriminator_s_loss)
            loss_s.append(s2t2s_classification_loss)
            loss_t_stacked = torch.stack(loss_t)
            loss_s_stacked = torch.stack(loss_s)
            if initial_loss_t is None:
                if with_nvidia:
                    initial_loss_t = 1 / (1 + np.exp(-loss_t_stacked.data.cpu().numpy()))
                    initial_loss_s = 1 / (1 + np.exp(-loss_s_stacked.data.cpu().numpy()))
                else:
                    initial_loss_t = 1 / (1 + np.exp(-loss_t_stacked.data.numpy()))
                    initial_loss_s = 1 / (1 + np.exp(-loss_s_stacked.data.numpy()))
            loss_total_without_ad = torch.sum(torch.mul(weights_grad_norm_t, loss_t_stacked)) + torch.sum(
                torch.mul(weights_grad_norm_s, loss_s_stacked))
            if cur_epoch < 12:
                loss_total = loss_total_without_ad + 3 * cdan_loss + 2 * t_sl_loss + 2 * s_sl_loss
            elif cur_epoch < 24:
                loss_total = loss_total_without_ad + 2 * cdan_loss + 1.8 * t_sl_loss + 1.5 * s_sl_loss
            elif cur_epoch < 50:
                loss_total = loss_total_without_ad + 1.5 * cdan_loss + 1.8 * t_sl_loss + 1.8 * s_sl_loss
            else:
                loss_total = loss_total_without_ad + 1.5 * cdan_loss + 2.5 * t_sl_loss + 2.5 * s_sl_loss
            for the_optim in optimizer_list:
                the_optim.zero_grad()
            optimizer_sl_cpc.zero_grad()
            optimizer_weights_s.zero_grad()
            optimizer_weights_t.zero_grad()
            loss_total.backward(retain_graph=True)
            optimizer_weights_s.zero_grad()
            optimizer_weights_t.zero_grad()
            #MLSTM_FCN记得改
            shared_t = target_feature_extraction_module
            shared_s = source_feature_extraction_module
            norms_t = []
            norms_s = []
            for i in range(len(loss_t_stacked)):
                grad_this_time = torch.autograd.grad(loss_t_stacked[i], shared_t.parameters(), retain_graph=True)
                norms_t.append(torch.cat(
                    [torch.norm(torch.mul(weights_grad_norm_t[i], g)).unsqueeze(0) for g in grad_this_time]).sum())
            for i in range(len(loss_s_stacked)):
                grad_this_time = torch.autograd.grad(loss_s_stacked[i], shared_s.parameters(), retain_graph=True)
                norms_s.append(torch.cat(
                    [torch.norm(torch.mul(weights_grad_norm_s[i], g)).unsqueeze(0) for g in grad_this_time]).sum())
            norms_t_stack = torch.stack(norms_t)
            norms_s_stack = torch.stack(norms_s)
            if with_nvidia:
                loss_ratio_t = (1 / (1 + np.exp(-loss_t_stacked.data.cpu().numpy()))) / initial_loss_t
                loss_ratio_s = (1 / (1 + np.exp(-loss_s_stacked.data.cpu().numpy()))) / initial_loss_s
            else:
                loss_ratio_t = (1 / (1 + np.exp(-loss_t_stacked.data.numpy()))) / initial_loss_t
                loss_ratio_s = (1 / (1 + np.exp(-loss_s_stacked.data.numpy()))) / initial_loss_s
            inverse_train_rate_t = loss_ratio_t / np.mean(loss_ratio_t)
            inverse_train_rate_s = loss_ratio_s / np.mean(loss_ratio_s)
            if with_nvidia:
                mean_norm_t = np.mean(norms_t_stack.data.cpu().numpy())
                mean_norm_s = np.mean(norms_s_stack.data.cpu().numpy())
            else:
                mean_norm_t = np.mean(norms_t_stack.data.numpy())
                mean_norm_s = np.mean(norms_s_stack.data.numpy())
            constant_term_t = torch.tensor(mean_norm_t * (inverse_train_rate_t ** alpha_for_grad_norm),
                                           requires_grad=False)
            constant_term_s = torch.tensor(mean_norm_s * (inverse_train_rate_s ** alpha_for_grad_norm),
                                           requires_grad=False)
            if with_nvidia:
                constant_term_t = constant_term_t.cuda()
                constant_term_s = constant_term_s.cuda()
            grad_norm_loss_t = torch.sum(torch.abs(norms_t_stack - constant_term_t))
            grad_norm_loss_s = torch.sum(torch.abs(norms_s_stack - constant_term_s))
            grad_for_weight_t = torch.autograd.grad(grad_norm_loss_t, weights_grad_norm_t)[0]
            grad_for_weight_s = torch.autograd.grad(grad_norm_loss_s, weights_grad_norm_s)[0]
            # optimizer_weights_t.step()
            # optimizer_weights_s.step()
            """
            if cur_epoch==0:  #第一个epoch不确定性太大，尽量那什么稳妥一些
                if with_nvidia:
                    initial_loss_t = 1/(1+np.exp(-loss_t_stacked.data.cpu().numpy()))
                    initial_loss_s = 1/(1+np.exp(-loss_s_stacked.data.cpu().numpy()))
                else:
                    initial_loss_t = 1/(1+np.exp(-loss_t_stacked.data.numpy()))
                    initial_loss_s = 1/(1+np.exp(-loss_s_stacked.data.numpy()))
            """
            if with_nvidia:
                saved_weights_grad_norm_t = weights_grad_norm_t.data.cpu().numpy()
                saved_weights_grad_norm_s = weights_grad_norm_s.data.cpu().numpy()
            else:
                saved_weights_grad_norm_t = weights_grad_norm_t.data.numpy()
                saved_weights_grad_norm_s = weights_grad_norm_s.data.numpy()
            # 清空计算图
            loss_total.data = loss_total.data * 0.0
            weights_grad_norm_t.data = weights_grad_norm_t.data * 0.0
            weights_grad_norm_s.data = weights_grad_norm_s.data * 0.0
            loss_t_stacked.data = loss_t_stacked.data * 0.0
            loss_s_stacked.data = loss_s_stacked.data * 0.0
            cdan_loss.data = cdan_loss.data * 0.0
            #feature_discriminator_s_loss.data = feature_discriminator_s_loss.data * 0.0
            loss_total.backward()
            if with_nvidia:
                weights_grad_norm_t.data = torch.tensor(saved_weights_grad_norm_t).cuda()
                weights_grad_norm_s.data = torch.tensor(saved_weights_grad_norm_s).cuda()
            else:
                weights_grad_norm_t.data = torch.tensor(saved_weights_grad_norm_t)
                weights_grad_norm_s.data = torch.tensor(saved_weights_grad_norm_s)
            weights_grad_norm_t.grad = grad_for_weight_t
            weights_grad_norm_s.grad = grad_for_weight_s  # 这么直接.data  .grad换是可以的，而且原来如果是require_grad那么还是
            optimizer_weights_t.step()
            optimizer_weights_s.step()
            for the_optim in optimizer_list:
                the_optim.step()
            # 进行weight_grad_norm的归一化
            optimizer_sl_cpc.step()
            weights_grad_norm_t.data[:].clamp_(min=0.0)
            normalize_coeff_t = 5 / torch.sum(weights_grad_norm_t.data, dim=0)
            weights_grad_norm_t.data = weights_grad_norm_t.data * normalize_coeff_t
            weights_grad_norm_s.data[:].clamp_(min=0.0)
            normalize_coeff_s = 6 / torch.sum(weights_grad_norm_s.data, dim=0)
            weights_grad_norm_s.data = weights_grad_norm_s.data * normalize_coeff_s
            # 进行WGAN判别器的参数大小截断
            for p in ad_net.parameters():
                p.data.clamp_(-0.0005, 0.0005)
            #for p in feature_discriminator_s.parameters():
             #   p.data.clamp_(-0.01, 0.01)
        scheduler_target_feature_extraction.step()
        scheduler_target_classification.step()
        scheduler_sl_cpc.step()
        scheduler_source_feature_extraction.step()
        #scheduler_source_to_target_feature_trans.step()
        scheduler_source_classification.step()
        scheduler_feature_transfer_between_t_s.step(s2t2s_classification_loss)  # 看一下科学不科学，是否会报错
        #scheduler_nf_for_transfer.step(target_nf_loss)
        #scheduler_noise_confusion_for_nf.step()
        scheduler_ad_net.step(cdan_loss)
        #scheduler_feature_discriminator_s.step(feature_discriminator_s_loss)
        if cur_epoch % 2 == 0:
            # 保存模型
            save_target_classification_modules(target_feature_extraction_module, target_classification_module,
                                               cur_epoch)
            save_source_classification_modules(source_feature_extraction_module,source_classification_module, cur_epoch)
            # 一定调成eval模式
            target_feature_extraction_module.eval()
            target_classification_module.eval()
            source_feature_extraction_module.eval()
            #source_to_target_feature_trans.eval()
            source_classification_module.eval()
            eval_model_traindata(target_feature_extraction_module, target_classification_module,
                                 target_train_loader, cur_epoch, with_nvidia)
            eval_model_testdata(target_feature_extraction_module, target_classification_module, target_test_loader,
                                cur_epoch, with_nvidia)
            eval_source_model_traindata(source_feature_extraction_module,source_classification_module, source_train_loader, cur_epoch, with_nvidia)
            eval_source_model_testdata(source_feature_extraction_module,source_classification_module, source_test_loader, cur_epoch, with_nvidia)
            """
            np.save(
                "numpy_saved_with_accuracy/feature_of_target_s2t/epoch_" + str(cur_epoch) + "target_feature.npy",
                np_target_feature)
            np.save("numpy_saved_with_accuracy/feature_of_target_s2t/epoch_" + str(cur_epoch) + "s2t_feature.npy",
                    np_source_to_target_feature)
            np.save(
                "numpy_saved_with_accuracy/feature_of_target_s2t/epoch_" + str(cur_epoch) + "source_feature.npy",
                np_source_shape_changed_feature)
            np.save(
                "numpy_saved_with_accuracy/feature_of_source_t2s/epoch_" + str(cur_epoch) + "source_feature.npy",
                np_source_feature)
            np.save(
                "numpy_saved_with_accuracy/feature_of_source_t2s/epoch_" + str(cur_epoch) + "target_feature.npy",
                np_target_to_source_feature)
            np.save("numpy_saved_with_accuracy/feature_of_source_t2s/epoch_" + str(cur_epoch) + "s2t2s_feature.npy",
                    np_s2t2s_feature)
            """