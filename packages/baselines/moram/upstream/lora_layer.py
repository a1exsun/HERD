# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from Artificer-AI-Lab/MoRAM @ aaa51d8ec67171e1bc5e1a1168351588561256c4
# (Apache-2.0), file TRACE/utils/my_peft/tuners/lora.py. Only the classes the MoRAM
# layer inherits are kept: `LoraLayer` and `Linear` (lines 477-667), copied verbatim,
# plus `transpose` from TRACE/utils/my_peft/utils/other.py (lines 180-181).
# Local adaptation: the module header imports below replace the package-relative
# imports of the unused LoraModel/LoraConfig/Embedding/8-bit code, which is omitted.
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


class LoraLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # self.lorapre_A = nn.ModuleDict({}) # modified
        # self.lorapre_B = nn.ModuleDict({}) # modified
        self.loranew_A = nn.ModuleDict({}) # modified
        self.loranew_B = nn.ModuleDict({}) # modified
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, r_sum): # modified 
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.loranew_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)})) # modified
            self.loranew_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)})) # modified
            self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r_sum, bias=False)})) # modified
            self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r_sum, self.out_features, bias=False)})) # modified
            # self.lorapre_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r_sum, bias=False)})) # modified
            # self.lorapre_B.update(nn.ModuleDict({adapter_name: nn.Linear(r_sum, self.out_features, bias=False)})) # modified
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.in_features)))})
            )
            self.lora_embedding_B.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.out_features, r)))})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):

        # modified
        if adapter_name in self.lora_A.keys(): 
            # initialize A and B to zero
            nn.init.zeros_(self.lora_A[adapter_name].weight)
            nn.init.zeros_(self.lora_B[adapter_name].weight)

        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

        # modified
        if adapter_name in self.loranew_A.keys(): 
            nn.init.kaiming_uniform_(self.loranew_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.loranew_B[adapter_name].weight)

        # # modified
        # if adapter_name in self.lorapre_A.keys(): 
        #     nn.init.zeros_(self.lorapre_A[adapter_name].weight)
        #     nn.init.zeros_(self.lorapre_B[adapter_name].weight)


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        r_sum: int = 0, # modified
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, r_sum) # modified
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_B[self.active_adapter].weight @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[self.active_adapter].weight.dtype)
            x = self.lora_dropout[self.active_adapter](x)

            result += (
                self.lora_B[self.active_adapter](
                    self.lora_A[self.active_adapter](x)
                )
                * self.scaling[self.active_adapter]
            )

            # modified
            result += (
                self.loranew_B[self.active_adapter](
                    self.loranew_A[self.active_adapter](x)
                )
                * self.scaling[self.active_adapter] 
            )
            
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result

