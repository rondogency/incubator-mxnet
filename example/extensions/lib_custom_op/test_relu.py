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

a = mx.nd.ones(shape=(13,5), ctx=mx.cpu())
b = mx.nd.ones(shape=(13,5), ctx=mx.gpu())

print("--------start ndarray compute---------")
mx.random.seed(128, ctx=mx.cpu())
noise = mx.nd.random.normal(shape=a.shape, ctx=mx.cpu())
print(mx.nd.relu(a + noise))

mx.random.seed(128, ctx=mx.cpu())
print(mx.nd.my_noisy_relu(a))

mx.random.seed(128, ctx=mx.cpu())
print(mx.nd.my_noisy_relu(a))

mx.random.seed(128, ctx=mx.gpu())
noise = mx.nd.random.normal(shape=b.shape, ctx=mx.gpu())
print(mx.nd.relu(b + noise))

mx.random.seed(128, ctx=mx.gpu())
print(mx.nd.my_noisy_relu(b))

mx.random.seed(128, ctx=mx.gpu())
print(mx.nd.my_noisy_relu(b))
