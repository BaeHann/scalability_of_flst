import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class TapNet(nn.Module): #这个代码要学习返回的这些为什么可以  还有  这种距离学习的方式为什么可行

    def __init__(self, nfeat, len_ts, nclass, dilation,dropout=0.2, filters=[256,256,128], kernels=[8,5,3],
                 layers=[500,300], use_rp=True,use_att=True, use_muse=False, use_lstm=True, use_cnn=True):
        super(TapNet, self).__init__()
        self.nclass = nclass
        self.dropout = dropout
        #self.use_metric = use_metric #这个在外面损失函数的时候用
        self.use_muse = use_muse
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.refresh = False
        self.round_num = 0 #记录累次分小batch时的次数，以决定权重
        self.whole = True #是否按照源代码把所有样本一个batch全送进来，这样实时准确性可以保证  #它和上面的两个参数全在外面更新
        # parameters for random projection
        self.use_rp = use_rp
        self.rp_group, self.rp_dim = [3, math.floor(nfeat * 2 / 3)]
        #self.proto_matrix = None

        if not self.use_muse:
            # LSTM
            self.channel = nfeat
            self.ts_length = len_ts

            self.lstm_dim = 128
            self.lstm = nn.LSTM(self.ts_length, self.lstm_dim)
            # self.dropout = nn.Dropout(0.8)

            # convolutional layer
            # features for each hidden layers
            # out_channels = [256, 128, 256]
            # filters = [256, 256, 128]
            # poolings = [2, 2, 2]
            paddings = [0, 0, 0]
            print("dilation", dilation)
            if self.use_rp:
                self.conv_1_models = nn.ModuleList()
                self.idx = []
                for i in range(self.rp_group):
                    self.conv_1_models.append(
                        nn.Conv1d(self.rp_dim, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1,
                                  padding=paddings[0]))
                    self.idx.append(np.random.permutation(nfeat)[0: self.rp_dim])
            else:
                self.conv_1 = nn.Conv1d(self.channel, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1,
                                        padding=paddings[0])
            # self.maxpool_1 = nn.MaxPool1d(poolings[0])
            self.conv_bn_1 = nn.BatchNorm1d(filters[0])

            self.conv_2 = nn.Conv1d(filters[0], filters[1], kernel_size=kernels[1], stride=1, padding=paddings[1])
            # self.maxpool_2 = nn.MaxPool1d(poolings[1])
            self.conv_bn_2 = nn.BatchNorm1d(filters[1])

            self.conv_3 = nn.Conv1d(filters[1], filters[2], kernel_size=kernels[2], stride=1, padding=paddings[2])
            # self.maxpool_3 = nn.MaxPool1d(poolings[2])
            self.conv_bn_3 = nn.BatchNorm1d(filters[2])

            # compute the size of input for fully connected layers
            fc_input = 0
            """
            if self.use_cnn:
                conv_size = len_ts
                for i in range(len(filters)):
                    conv_size = output_conv_size(conv_size, kernels[i], 1, paddings[i])
                fc_input += conv_size
                # * filters[-1]
            if self.use_lstm:
                fc_input += conv_size * self.lstm_dim
            """
            if self.use_rp:
                fc_input = self.rp_group * filters[2] + self.lstm_dim

        # Representation mapping function
        layers = [fc_input] + layers
        print("Layers", layers)
        self.mapping = nn.Sequential()
        for i in range(len(layers) - 2):
            self.mapping.add_module("fc_" + str(i), nn.Linear(layers[i], layers[i + 1]))
            self.mapping.add_module("bn_" + str(i), nn.BatchNorm1d(layers[i + 1]))
            self.mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # add last layer
        self.mapping.add_module("fc_" + str(len(layers) - 2), nn.Linear(layers[-2], layers[-1]))
        if len(layers) == 2:  # if only one layer, add batch normalization
            self.mapping.add_module("bn_" + str(len(layers) - 2), nn.BatchNorm1d(layers[-1]))
        self.length_before_classification = layers[-1]
        # Attention
        att_dim, semi_att_dim = 128, 128
        self.use_att = use_att
        if self.use_att:
            self.att_models = nn.ModuleList()
            for _ in range(nclass):
                att_model = nn.Sequential(
                    nn.Linear(layers[-1], att_dim),
                    nn.Tanh(),
                    nn.Linear(att_dim, 1)
                )
                self.att_models.append(att_model)


        #只有目标域能用上
        self.semi_att = nn.Sequential(
            nn.Linear(layers[-1], semi_att_dim),
            nn.Tanh(),
            nn.Linear(semi_att_dim, self.nclass)
        )
        self.proto_matrix = torch.zeros((nclass, self.length_before_classification)).float().cuda()#.cuda()不能少，整个模型.cuda只能对parameters起作用
    #对于TapNet处理s2t2s还是挺有挑战的
    def forward(self, input):
        x, labels, whether_train = input  # x is N * L, where L is the time-series feature dimension
        if whether_train: #一般训练输入
            if self.whole: #一开始的时候
                if not self.use_muse:
                    N = x.size(0)

                    # LSTM
                    if self.use_lstm:
                        x_lstm = self.lstm(x)[0]
                        x_lstm = x_lstm.mean(1)
                        x_lstm = x_lstm.view(N, -1)

                    if self.use_cnn:
                        # Covolutional Network
                        # input ts: # N * C * L
                        if self.use_rp:
                            for i in range(len(self.conv_1_models)):
                                # x_conv = x
                                x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                                x_conv = self.conv_bn_1(x_conv)
                                x_conv = F.leaky_relu(x_conv)

                                x_conv = self.conv_2(x_conv)
                                x_conv = self.conv_bn_2(x_conv)
                                x_conv = F.leaky_relu(x_conv)

                                x_conv = self.conv_3(x_conv)
                                x_conv = self.conv_bn_3(x_conv)
                                x_conv = F.leaky_relu(x_conv)

                                x_conv = torch.mean(x_conv, 2)

                                if i == 0:
                                    x_conv_sum = x_conv
                                else:
                                    x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)

                            x_conv = x_conv_sum
                        else:
                            x_conv = x
                            x_conv = self.conv_1(x_conv)  # N * C * L
                            x_conv = self.conv_bn_1(x_conv)
                            x_conv = F.leaky_relu(x_conv)

                            x_conv = self.conv_2(x_conv)
                            x_conv = self.conv_bn_2(x_conv)
                            x_conv = F.leaky_relu(x_conv)

                            x_conv = self.conv_3(x_conv)
                            x_conv = self.conv_bn_3(x_conv)
                            x_conv = F.leaky_relu(x_conv)

                            x_conv = x_conv.view(N, -1)

                    if self.use_lstm and self.use_cnn:
                        x = torch.cat([x_conv, x_lstm], dim=1)
                    elif self.use_lstm:
                        x = x_lstm
                    elif self.use_cnn:
                        x = x_conv
                    #

                # linear mapping to low-dimensional space
                x = self.mapping(x)

                # generate the class protocal with dimension C * D (nclass * dim)
                proto_list = []
                for i in range(self.nclass):
                    idx = (labels.squeeze() == i).nonzero().squeeze(1) #不知道会不会出错
                    if self.use_att:
                        # A = self.attention(x[idx_train][idx])  # N_k * 1
                        A = self.att_models[i](x[idx])  # N_k * 1
                        A = torch.transpose(A, 1, 0)  # 1 * N_k
                        A = F.softmax(A, dim=1)  # softmax over N_k
                        # print(A)
                        class_repr = torch.mm(A, x[idx])  # 1 * L
                        class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
                    else:  # if do not use attention, simply use the mean of training samples with the same labels.
                        class_repr = x[idx].mean(0)  # L * 1
                    proto_list.append(class_repr.view(1, -1))
                x_proto = torch.cat(proto_list, dim=0)
                if self.refresh:
                    self.proto_matrix = 0.6 * self.proto_matrix + 0.4 * x_proto.detach()
                else:
                    self.proto_matrix = x_proto.detach()
                # print(x_proto)
                # dists = euclidean_dist(x, x_proto)
                # log_dists = F.log_softmax(-dists * 1e7, dim=1)

                # prototype distance
                proto_dists = euclidean_dist(x_proto, x_proto)
                num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
                proto_dist = torch.sum(proto_dists) / num_proto_pairs

                #虽然不用了，但是可以学一学
                """
                if self.use_ss:
                    semi_A = self.semi_att(x[idx_test])  # N_test * c
                    semi_A = torch.transpose(semi_A, 1, 0)  # c * N_test
                    semi_A = F.softmax(semi_A, dim=1)  # softmax over N_test
                    x_proto_test = torch.mm(semi_A, x[idx_test])  # c * L
                    x_proto = (x_proto + x_proto_test) / 2
                """

                dists = euclidean_dist(x, x_proto)
                return -dists, proto_dist #最后一个x用作输出
            else: #后来联合起来，内存放不下了，只能这样了
                if not self.use_muse:
                    N = x.size(0)

                    # LSTM
                    if self.use_lstm:
                        x_lstm = self.lstm(x)[0]
                        x_lstm = x_lstm.mean(1)
                        x_lstm = x_lstm.view(N, -1)

                    if self.use_cnn:
                        # Covolutional Network
                        # input ts: # N * C * L
                        if self.use_rp:
                            for i in range(len(self.conv_1_models)):
                                # x_conv = x
                                x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                                x_conv = self.conv_bn_1(x_conv)
                                x_conv = F.leaky_relu(x_conv)

                                x_conv = self.conv_2(x_conv)
                                x_conv = self.conv_bn_2(x_conv)
                                x_conv = F.leaky_relu(x_conv)

                                x_conv = self.conv_3(x_conv)
                                x_conv = self.conv_bn_3(x_conv)
                                x_conv = F.leaky_relu(x_conv)

                                x_conv = torch.mean(x_conv, 2)

                                if i == 0:
                                    x_conv_sum = x_conv
                                else:
                                    x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)

                            x_conv = x_conv_sum
                        else:
                            x_conv = x
                            x_conv = self.conv_1(x_conv)  # N * C * L
                            x_conv = self.conv_bn_1(x_conv)
                            x_conv = F.leaky_relu(x_conv)

                            x_conv = self.conv_2(x_conv)
                            x_conv = self.conv_bn_2(x_conv)
                            x_conv = F.leaky_relu(x_conv)

                            x_conv = self.conv_3(x_conv)
                            x_conv = self.conv_bn_3(x_conv)
                            x_conv = F.leaky_relu(x_conv)

                            x_conv = x_conv.view(N, -1)

                    if self.use_lstm and self.use_cnn:
                        x = torch.cat([x_conv, x_lstm], dim=1)
                    elif self.use_lstm:
                        x = x_lstm
                    elif self.use_cnn:
                        x = x_conv
                    #

                # linear mapping to low-dimensional space
                x = self.mapping(x)
                if self.round_num == 0: #对于自监督阶段
                    alpha = 0.05
                elif self.round_num == 1: #对于联合训练阶段
                    alpha = 0.1
                else:   #为了适应单独数据集都不能一次batch全过完的情况设立的
                    alpha = 0.5
                # generate the class protocal with dimension C * D (nclass * dim)
                proto_list = []
                for i in range(self.nclass):
                    idx = (labels.squeeze() == i).nonzero().squeeze(1) #不知道会不会出错
                    if len(idx) == 0:
                        proto_list.append(torch.zeros((1,self.length_before_classification)).float().cuda())
                        continue
                    if self.use_att:
                        # A = self.attention(x[idx_train][idx])  # N_k * 1
                        A = self.att_models[i](x[idx])  # N_k * 1
                        A = torch.transpose(A, 1, 0)  # 1 * N_k
                        A = F.softmax(A, dim=1)  # softmax over N_k
                        # print(A)
                        class_repr = torch.mm(A, x[idx])  # 1 * L
                        class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
                    else:  # if do not use attention, simply use the mean of training samples with the same labels.
                        class_repr = x[idx].mean(0)  # L * 1
                    proto_list.append(class_repr.view(1, -1))
                x_proto = torch.cat(proto_list, dim=0)
                #x_proto = self.proto_matrix + alpha * x_proto #不知道会不会出错
                x_proto = (1-alpha) * self.proto_matrix + alpha * x_proto
                self.proto_matrix = x_proto.detach()

                # print(x_proto)
                # dists = euclidean_dist(x, x_proto)
                # log_dists = F.log_softmax(-dists * 1e7, dim=1)

                # prototype distance
                proto_dists = euclidean_dist(x_proto, x_proto)
                num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
                proto_dist = torch.sum(proto_dists) / num_proto_pairs

                #虽然不用了，但是可以学一学
                """
                if self.use_ss:
                    semi_A = self.semi_att(x[idx_test])  # N_test * c
                    semi_A = torch.transpose(semi_A, 1, 0)  # c * N_test
                    semi_A = F.softmax(semi_A, dim=1)  # softmax over N_test
                    x_proto_test = torch.mm(semi_A, x[idx_test])  # c * L
                    x_proto = (x_proto + x_proto_test) / 2
                """

                dists = euclidean_dist(x, x_proto)
                return -dists, proto_dist #最后一个x用作输出
        else: #测试输入
            if not self.use_muse:
                N = x.size(0)

                # LSTM
                if self.use_lstm:
                    x_lstm = self.lstm(x)[0]
                    x_lstm = x_lstm.mean(1)
                    x_lstm = x_lstm.view(N, -1)

                if self.use_cnn:
                    # Covolutional Network
                    # input ts: # N * C * L
                    if self.use_rp:
                        for i in range(len(self.conv_1_models)):
                            # x_conv = x
                            x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                            x_conv = self.conv_bn_1(x_conv)
                            x_conv = F.leaky_relu(x_conv)

                            x_conv = self.conv_2(x_conv)
                            x_conv = self.conv_bn_2(x_conv)
                            x_conv = F.leaky_relu(x_conv)

                            x_conv = self.conv_3(x_conv)
                            x_conv = self.conv_bn_3(x_conv)
                            x_conv = F.leaky_relu(x_conv)

                            x_conv = torch.mean(x_conv, 2)

                            if i == 0:
                                x_conv_sum = x_conv
                            else:
                                x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)

                        x_conv = x_conv_sum
                    else:
                        x_conv = x
                        x_conv = self.conv_1(x_conv)  # N * C * L
                        x_conv = self.conv_bn_1(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_2(x_conv)
                        x_conv = self.conv_bn_2(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_3(x_conv)
                        x_conv = self.conv_bn_3(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = x_conv.view(N, -1)

                if self.use_lstm and self.use_cnn:
                    x = torch.cat([x_conv, x_lstm], dim=1)
                elif self.use_lstm:
                    x = x_lstm
                elif self.use_cnn:
                    x = x_conv
                #

            # linear mapping to low-dimensional space
            x = self.mapping(x)

            # print(x_proto)
            # dists = euclidean_dist(x, x_proto)
            # log_dists = F.log_softmax(-dists * 1e7, dim=1)

            #虽然不用了，但是可以学一学
            """
            if self.use_ss:
                semi_A = self.semi_att(x[idx_test])  # N_test * c
                semi_A = torch.transpose(semi_A, 1, 0)  # c * N_test
                semi_A = F.softmax(semi_A, dim=1)  # softmax over N_test
                x_proto_test = torch.mm(semi_A, x[idx_test])  # c * L
                x_proto = (x_proto + x_proto_test) / 2
            """
            dists = euclidean_dist(x, self.proto_matrix)
            return -dists
    def process_s2t2s(self, input): #源域分类器才会用到
        x, labels = input
        alpha = 0.1
        # generate the class protocal with dimension C * D (nclass * dim)
        proto_list = []
        for i in range(self.nclass):
            idx = (labels.squeeze() == i).nonzero().squeeze(1)  # 不知道会不会出错
            if len(idx) == 0:
                proto_list.append(torch.zeros((1, self.length_before_classification)).float().cuda())
                continue
            if self.use_att:
                # A = self.attention(x[idx_train][idx])  # N_k * 1
                A = self.att_models[i](x[idx])  # N_k * 1
                A = torch.transpose(A, 1, 0)  # 1 * N_k
                A = F.softmax(A, dim=1)  # softmax over N_k
                # print(A)
                class_repr = torch.mm(A, x[idx])  # 1 * L
                class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
            else:  # if do not use attention, simply use the mean of training samples with the same labels.
                class_repr = x[idx].mean(0)  # L * 1
            proto_list.append(class_repr.view(1, -1))
        x_proto = torch.cat(proto_list, dim=0)
        #x_proto = self.proto_matrix + alpha * x_proto  # 不知道会不会出错
        x_proto = (1-alpha) * self.proto_matrix + alpha * x_proto
        self.proto_matrix = x_proto.detach()
        # print(x_proto)
        # dists = euclidean_dist(x, x_proto)
        # log_dists = F.log_softmax(-dists * 1e7, dim=1)

        # prototype distance
        proto_dists = euclidean_dist(x_proto, x_proto)
        num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
        proto_dist = torch.sum(proto_dists) / num_proto_pairs

        dists = euclidean_dist(x, x_proto)
        return -dists, proto_dist  # 最后一个x用作输出
    def process_s2t(self, input): #目标域分类器才会用到
        x = input
        if not self.use_muse:
            N = x.size(0)

            # LSTM
            if self.use_lstm:
                x_lstm = self.lstm(x)[0]
                x_lstm = x_lstm.mean(1)
                x_lstm = x_lstm.view(N, -1)

            if self.use_cnn:
                # Covolutional Network
                # input ts: # N * C * L
                if self.use_rp:
                    for i in range(len(self.conv_1_models)):
                        # x_conv = x
                        x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                        x_conv = self.conv_bn_1(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_2(x_conv)
                        x_conv = self.conv_bn_2(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_3(x_conv)
                        x_conv = self.conv_bn_3(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = torch.mean(x_conv, 2)

                        if i == 0:
                            x_conv_sum = x_conv
                        else:
                            x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)

                    x_conv = x_conv_sum
                else:
                    x_conv = x
                    x_conv = self.conv_1(x_conv)  # N * C * L
                    x_conv = self.conv_bn_1(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_2(x_conv)
                    x_conv = self.conv_bn_2(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_3(x_conv)
                    x_conv = self.conv_bn_3(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = x_conv.view(N, -1)

            if self.use_lstm and self.use_cnn:
                x = torch.cat([x_conv, x_lstm], dim=1)
            elif self.use_lstm:
                x = x_lstm
            elif self.use_cnn:
                x = x_conv
            #

            # linear mapping to low-dimensional space
        x = self.mapping(x)
        semi_A = self.semi_att(x)  # N_test * c
        semi_A = torch.transpose(semi_A, 1, 0)  # c * N_test
        semi_A = F.softmax(semi_A, dim=1)  # softmax over N_test
        x_proto_test = torch.mm(semi_A, x)  # c * L
        #x_proto = 0.2 * x_proto_test + self.proto_matrix
        x_proto = 0.2 * x_proto_test + 0.8 * self.proto_matrix
        self.proto_matrix = x_proto
        dists = euclidean_dist(x, x_proto)
        return -dists, x
        #return x