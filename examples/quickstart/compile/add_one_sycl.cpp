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

// [example.begin]
// File: compile/add_one_sycl.cpp
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <sycl/sycl.hpp>

namespace tvm_ffi_example_sycl {

void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  int64_t n = x.size(0);
  float* x_data = static_cast<float*>(x.data_ptr());
  float* y_data = static_cast<float*>(y.data_ptr());
  sycl::queue& queue =
      *static_cast<sycl::queue*>(TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { y_data[i] = x_data[i] + 1; });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_sycl, tvm_ffi_example_sycl::AddOne);
}  // namespace tvm_ffi_example_sycl
// [example.end]
