/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2020 by Contributors
 * \file relu_lib.cu
 * \brief simple custom relu and noisy relu operator implemented using CUDA function
 */

#include <iostream>
#include "lib_api.h"

#define NumThreadPerBlock 256 // mxnet recommended cuda thread number per block

__global__ void relu_gpu_forward(float *out, float *in, int64_t N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        out[tid] = in[tid] > 0 ? in[tid] : 0;
}

__global__ void relu_gpu_backward(float *ingrad, float *outgrad, float *indata, int64_t N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        ingrad[tid] = indata[tid] > 0 ? 1 * outgrad[tid] : 0;
}

MXReturnValue forwardCPU(std::map<std::string, std::string> attrs,
                         std::vector<MXTensor> inputs,
                         std::vector<MXTensor> outputs,
                         OpResource res) {
    float* in_data = inputs[0].data<float>();
    float* out_data = outputs[0].data<float>();
    for (int i=0; i<inputs[0].size(); i++) {
        out_data[i] = in_data[i] ? in_data[i] : 0;
    }
    return MX_SUCCESS;
}

MXReturnValue backwardCPU(std::map<std::string, std::string> attrs,
                          std::vector<MXTensor> inputs,
                          std::vector<MXTensor> outputs,
                          OpResource res) {
    float* out_grad = inputs[0].data<float>();
    float* in_data = inputs[1].data<float>();
    float* in_grad = outputs[0].data<float>();
    for (int i=0; i<inputs[1].size(); i++) {
        in_grad[i] = in_data[i] > 0 ? 1 * out_grad[i] : 0;
    }
    return MX_SUCCESS;
}

MXReturnValue forwardGPU(std::map<std::string, std::string> attrs,
                         std::vector<MXTensor> inputs,
                         std::vector<MXTensor> outputs,
                         OpResource res) {
    float* in_data = inputs[0].data<float>();
    float* out_data = outputs[0].data<float>();

    mx_stream_t cuda_stream = res.get_cuda_stream();
    int64_t N = inputs[0].size();
    int num_block = (N + NumThreadPerBlock - 1) / NumThreadPerBlock;

    relu_gpu_forward<<<num_block,NumThreadPerBlock,0,cuda_stream>>>(out_data, in_data, N);

    return MX_SUCCESS;
}

MXReturnValue backwardGPU(std::map<std::string, std::string> attrs,
                          std::vector<MXTensor> inputs,
                          std::vector<MXTensor> outputs,
                          OpResource res) {
    float* out_grad = inputs[0].data<float>();
    float* in_data = inputs[1].data<float>();
    float* in_grad = outputs[0].data<float>();

    mx_stream_t cuda_stream = res.get_cuda_stream();
    int64_t N = inputs[0].size();
    int num_block = (N + NumThreadPerBlock - 1) / NumThreadPerBlock;

    relu_gpu_backward<<<num_block,NumThreadPerBlock,0,cuda_stream>>>(in_grad, out_grad, in_data, N);

    return MX_SUCCESS;
}

MXReturnValue parseAttrs(std::map<std::string, std::string> attrs, int* num_in, int* num_out) {
    *num_in = 1;
    *num_out = 1;
    return MX_SUCCESS;
}

MXReturnValue inferType(std::map<std::string, std::string> attrs,
                        std::vector<int> &intypes,
                        std::vector<int> &outtypes) {
    outtypes[0] = intypes[0];
    return MX_SUCCESS;
}

MXReturnValue inferShape(std::map<std::string, std::string> attrs,
                         std::vector<std::vector<unsigned int>> &inshapes,
                         std::vector<std::vector<unsigned int>> &outshapes) {
    outshapes[0] = inshapes[0];
    return MX_SUCCESS;
}

REGISTER_OP(my_relu)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setForward(forwardCPU, "cpu")
.setForward(forwardGPU, "gpu")
.setBackward(backwardCPU, "cpu")
.setBackward(backwardGPU, "gpu");

class MyStatefulReluCPU : public CustomStatefulOp {
public:
    explicit MyStatefulReluCPU() {}
    MXReturnValue Forward(std::vector<MXTensor> inputs,
                          std::vector<MXTensor> outputs,
                          OpResource op_res) {
        std::map<std::string, std::string> attrs;
        return forwardCPU(attrs, inputs, outputs, op_res);
    }
    MXReturnValue Backward(std::vector<MXTensor> inputs,
                           std::vector<MXTensor> outputs,
                           OpResource op_res) {
        std::map<std::string, std::string> attrs;
        return backwardCPU(attrs, inputs, outputs, op_res);
    }
    ~MyStatefulReluCPU() {}
};

class MyStatefulReluGPU : public CustomStatefulOp {
public:
    explicit MyStatefulReluGPU() {}
    MXReturnValue Forward(std::vector<MXTensor> inputs,
                          std::vector<MXTensor> outputs,
                          OpResource op_res) {
        std::map<std::string, std::string> attrs;
        return forwardGPU(attrs, inputs, outputs, op_res);
    }
    MXReturnValue Backward(std::vector<MXTensor> inputs,
                           std::vector<MXTensor> outputs,
                           OpResource op_res) {
        std::map<std::string, std::string> attrs;
        return backwardGPU(attrs, inputs, outputs, op_res);
    }
    ~MyStatefulReluGPU() {}
};

MXReturnValue createOpStateCPU(std::map<std::string, std::string> attrs,
                               CustomStatefulOp** op_inst) {
    *op_inst = new MyStatefulReluCPU();
    return MX_SUCCESS;
}

MXReturnValue createOpStateGPU(std::map<std::string, std::string> attrs,
                               CustomStatefulOp** op_inst) {
    *op_inst = new MyStatefulReluGPU();
    return MX_SUCCESS;
}

REGISTER_OP(my_state_relu)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setCreateOpState(createOpStateCPU, "cpu")
.setCreateOpState(createOpStateGPU, "gpu");



/* ------------------------ Below is noisy relu operator example ---------------------*/

#include <curand_kernel.h>
#include <random>

#define NumRandomPerThread 64 // mxnet recommended random numbers generated per thread

__global__ void noisy_relu_gpu_forward(float *out, float *in, int64_t N, void *states, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t* global_states = (curandStatePhilox4_32_10_t*)states;
    curandStatePhilox4_32_10_t thread_state = global_states[tid];

    int start = tid * step;
    int end = start + step;
    for (int i=start; i<end && i<N; ++i) {
        float noise = curand_normal(&thread_state);
        out[i] = in[i] + noise > 0 ? in[i] + noise : 0;
    }
}

MXReturnValue noisyForwardCPU(std::map<std::string, std::string> attrs,
                              std::vector<MXTensor> inputs,
                              std::vector<MXTensor> outputs,
                              OpResource res) {
    float* in_data = inputs[0].data<float>();
    float* out_data = outputs[0].data<float>();

    std::mt19937 *states = static_cast<std::mt19937*>(res.get_cpu_rand_states());
    std::normal_distribution<float> dist_normal;

    for (int i=0; i<inputs[0].size(); ++i) {
        float noise = dist_normal(*states);
        out_data[i] = in_data[i] + noise > 0 ? in_data[i] + noise : 0;
    }
    return MX_SUCCESS;
}

MXReturnValue noisyForwardGPU(std::map<std::string, std::string> attrs,
                              std::vector<MXTensor> inputs,
                              std::vector<MXTensor> outputs,
                              OpResource res) {
    float* in_data = inputs[0].data<float>();
    float* out_data = outputs[0].data<float>();

    mx_stream_t cuda_stream = res.get_cuda_stream();
    int64_t N = inputs[0].size();

    int num_thread_need = (N + NumRandomPerThread - 1) / NumRandomPerThread;
    // each cuda thread processes [step * tid, step * id + step) snippet of input tensor
    int step = (N + num_thread_need - 1) / num_thread_need;
    int num_block = (num_thread_need + NumThreadPerBlock - 1) / NumThreadPerBlock;
    void *global_states = res.get_gpu_rand_states();

    noisy_relu_gpu_forward<<<num_block,NumThreadPerBlock,0,cuda_stream>>>(out_data, in_data, N, global_states, step);

    return MX_SUCCESS;
}

REGISTER_OP(my_noisy_relu)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setForward(noisyForwardCPU, "cpu")
.setForward(noisyForwardGPU, "gpu")
.setBackward(backwardCPU, "cpu")
.setBackward(backwardGPU, "gpu");

MXReturnValue initialize(int version) {
    if (version >= 10400) {
        std::cout << "MXNet version " << version << " supported" << std::endl;
        return MX_SUCCESS;
    } else {
        std::cout << "MXNet version " << version << " not supported" << std::endl;
        return MX_FAIL;
    }
}
