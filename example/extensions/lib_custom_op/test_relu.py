#!/usr/bin/env python3

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=arguments-differ

# This test checks dynamic loading of custom library into MXNet
# and checks end to end compute of a simple 2D gemm custom op

import mxnet as mx
import os
import time

#load library
if (os.name=='posix'):
    path = os.path.abspath('librelu_lib.so')
    mx.library.load(path)

class PyRelu(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        rand = mx.nd.random.normal(shape=in_data[0].shape, ctx=mx.gpu(0))
        x = mx.nd.relu(in_data[0] + rand)
        self.assign(out_data[0], req[0], x)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("pyrelu")
class PyReluProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(PyReluProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return PyRelu()

b = mx.nd.array([[-2,-1],[1,2]], ctx=mx.gpu())

print("--------start ndarray compute---------")
mx.random.seed(128, ctx=mx.gpu(0))
print(mx.nd.Custom(b, op_type='pyrelu'))
print(mx.nd.my_relu(b))
