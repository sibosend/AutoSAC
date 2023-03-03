# -*- coding:utf-8 -*-
"""
Author:
    caritasem@foxmail.com
Reference:
"""
import torch
import torch.nn as nn

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import *
from ..layers import InteractingAnimaleLayer, CapsuleLayer


class AutoSAC(BaseModel):
    """Instantiates the AutoSAC Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param att_layer_num: int.The InteractingLayer number to be used.
    :param att_head_num: int.The head number in multi-head  self-attention network.
    :param att_res: bool.Whether or not use standard residual connections before output.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_use_bn:  bool. Whether use BatchNormalization before activation or not in DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, att_layer_num=3,
                 att_head_num=2, att_res=True, dnn_hidden_units=(256, 128), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cpu', gpus=None, digit_num_capsules=1):

        super(AutoSAC, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=0,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)

        self.use_fm = True
        self.with_capsule_layer = True

        if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
            raise ValueError("Either hidden_layer or att_layer_num must > 0")
        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0

        field_num = len(self.embedding_dict)
        # print("dnn_hidden_units.shape", dnn_hidden_units)
        embedding_size = self.embedding_size

        if len(dnn_hidden_units) and att_layer_num > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1] + \
                field_num * embedding_size
            # print("dnn_hidden_units[-1], field_num, embedding_size",
            #       dnn_hidden_units[-1], field_num, embedding_size)
        elif len(dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif att_layer_num > 0:
            dnn_linear_in_feature = field_num * embedding_size
        else:
            raise NotImplementedError

        if self.use_fm:
            self.fm = FM()

        # with capsule layer
        if self.with_capsule_layer:
            dnn_linear_in_feature = dnn_linear_in_feature + 1

        self.dnn_linear = nn.Linear(
            dnn_linear_in_feature, 1, bias=False).to(device)
        # print("self.dnn_linear", self.dnn_linear)
        self.dnn_hidden_units = dnn_hidden_units
        self.att_layer_num = att_layer_num
        # print("self.compute_input_dim(dnn_feature_columns)",
        #       self.compute_input_dim(dnn_feature_columns))
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)

        # load pretrained models, using ResNeSt-50 as an example
        # self.resnest = resnest50(pretrained=False, groups=2, num_classes=4)
        # self.resnest.fc = nn.Identity()

        self.int_layers = nn.ModuleList(
            [InteractingAnimaleLayer(embedding_size, att_head_num, att_res, device=device) for _ in range(att_layer_num)])

        carg_out_channel = 8
        carg_kernel_size = min(field_num, 3)
        carg_stride = 1
        self.primary_capsules = CapsuleLayer(num_capsules=field_num, num_route_nodes=-1, in_channels=1, out_channels=carg_out_channel,
                                             kernel_size=carg_kernel_size, stride=carg_stride, device=device)

        # Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
        # Output width = (Input width + padding width right + padding width left - kernel width) / (stride width) + 1

        carg_dig_route_height = (
            field_num + 0 + 0 - carg_kernel_size) / carg_stride + 1
        carg_dig_route_width = (embedding_size + 0 +
                                0 - carg_kernel_size) / carg_stride + 1
        carg_dig_route_num = int(carg_out_channel *
                                 carg_dig_route_height * carg_dig_route_width)

        self.digit_capsules = CapsuleLayer(num_capsules=digit_num_capsules, num_route_nodes=carg_dig_route_num, in_channels=field_num,
                                           out_channels=1, device=device)

        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        att_input = concat_fun(sparse_embedding_list, axis=1)

        for layer in self.int_layers:
            att_input = layer(att_input)

        att_output = torch.flatten(att_input, start_dim=1)

        """
        att_input.shape torch.Size([50, 15, 4])
        att_output.shape torch.Size([50, 60])
        """
        # print("att_input.shape, before flatten", att_input.shape)

        # mod_add_channel = att_input.unsqueeze(1)
        mod_add_channel = (concat_fun(
            sparse_embedding_list, axis=1)).unsqueeze(1)
        # p3d = (0, att_input.shape[1] - att_input.shape[2], 0, 0, 0, 0)
        # mod_add_column = F.pad(mod_add_channel, p3d, "constant", 0)
        # print("mod_add_column.shape", mod_add_column.shape)
        cap_p_out = self.primary_capsules(mod_add_channel)
        # print("capsule_out.shape", cap_p_out.shape)

        cap_d_out = self.digit_capsules(cap_p_out)
        cap_d_out = cap_d_out.squeeze(0)
        capsule_output = torch.flatten(cap_d_out, start_dim=1)
        # print("capsule_output.shape", capsule_output.shape)
        # print("capsule_out.flatten.shape", torch.flatten(
        #     capsule_out, start_dim=1).shape)

        # print("att_output.shape", att_output.shape)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        if len(self.dnn_hidden_units) > 0 and self.att_layer_num > 0:  # Deep & Interacting Layer
            # print("s1-1")
            deep_out = self.dnn(dnn_input)
            # print("before, att_output.shape", att_output.shape)
            # print("before, deep_out.shape", deep_out.shape)
            # stack_out = concat_fun([att_output, deep_out])
            if not self.with_capsule_layer:
                stack_out = concat_fun([att_output, deep_out])
            else:
                stack_out = concat_fun([att_output, deep_out, capsule_output])
            # print("after, stack_out.shape", stack_out.shape)
            logit += self.dnn_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            # print("s1-2")
            deep_out = self.dnn(dnn_input)

            # stack_out = deep_out

            if not self.with_capsule_layer:
                stack_out = deep_out
            else:
                stack_out = concat_fun([deep_out, capsule_output])

            logit += self.dnn_linear(stack_out)
        elif self.att_layer_num > 0:  # Only Interacting Layer
            # print("s1-3")
            # stack_out = att_output
            if not self.with_capsule_layer:
                stack_out = att_output
            else:
                stack_out = concat_fun([att_output, capsule_output])

            logit += self.dnn_linear(stack_out)
        else:  # Error
            pass

        y_pred = self.out(logit)

        return y_pred
