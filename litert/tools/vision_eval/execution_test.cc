// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/tools/vision_eval/execution.h"

#include <cmath>

#include <gtest/gtest.h>
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/options/litert_arm_vulkan_ml_options.h"

namespace litert::vision {
namespace {
TEST(BackendTest, CpuBaselineUsesReferenceKernelsWithoutVendorOptions) {
  EvaluationConfig config;
  config.set_baseline_accelerator("cpu");
  auto* backend = config.mutable_backend();
  backend->set_accelerator("npu,gpu");
  backend->set_num_threads(4);
  backend->set_xnnpack_only(true);
  backend->set_arm_vulkan_ml_enable_jit(true);
  backend->set_compiler_plugin_library_dir("/plugins");
  backend->set_dispatch_library_dir("/dispatch");
  backend->set_runtime_library_dir("/runtime");
  backend->set_openvino_device("npu");
  backend->set_gpu_use_ir_model("true");
  backend->set_gpu_backend("opencl");
  backend->set_require_full_acceleration(true);
  ASSERT_TRUE(ValidateBackendOptions(*backend));
  auto baseline = BaselineOptions(config);
  EXPECT_EQ(baseline.accelerator(), "cpu");
  EXPECT_EQ(baseline.num_threads(), 1);
  EXPECT_FALSE(baseline.arm_vulkan_ml_enable_jit());
  EXPECT_TRUE(baseline.compiler_plugin_library_dir().empty());
  EXPECT_TRUE(baseline.dispatch_library_dir().empty());
  EXPECT_EQ(baseline.runtime_library_dir(), "/runtime");
  EXPECT_TRUE(baseline.openvino_device().empty());
  EXPECT_EQ(baseline.gpu_use_ir_model(), "default");
  EXPECT_FALSE(baseline.require_full_acceleration());
  auto options = CreateOptions(baseline, true);
  ASSERT_TRUE(options);
  auto cpu = options->GetCpuOptions();
  ASSERT_TRUE(cpu);
  auto mode = cpu->GetKernelMode();
  ASSERT_TRUE(mode);
  EXPECT_EQ(*mode, kLiteRtCpuKernelModeReference);
}

TEST(BackendTest, InheritedBaselinePreservesOptionsAndArmJitIsOptIn) {
  EvaluationConfig config;
  config.mutable_backend()->set_accelerator("npu");
  config.mutable_backend()->set_arm_vulkan_ml_enable_jit(true);
  auto baseline = BaselineOptions(config);
  EXPECT_EQ(baseline.SerializeAsString(), config.backend().SerializeAsString());
  auto enabled = CreateOptions(baseline);
  ASSERT_TRUE(enabled);
  baseline.set_arm_vulkan_ml_enable_jit(false);
  auto disabled = CreateOptions(baseline);
  ASSERT_TRUE(disabled);
  auto arm = enabled->GetOptions<arm_vulkan_ml::ArmVulkanMLOptions>();
  ASSERT_TRUE(arm);
  ASSERT_TRUE(arm->GetEnableJustInTime());
  auto jit_enabled = arm->GetEnableJustInTime();
  EXPECT_TRUE(*jit_enabled);
  auto other = disabled->GetOptions<arm_vulkan_ml::ArmVulkanMLOptions>();
  ASSERT_TRUE(other);
  ASSERT_TRUE(other->GetEnableJustInTime());
  auto jit_disabled = other->GetEnableJustInTime();
  EXPECT_FALSE(*jit_disabled);
}

TEST(BackendTest, RejectsMalformedAcceleratorOptions) {
  BackendOptions backend;
  EXPECT_TRUE(ValidateBackendOptions(backend));
  backend.set_accelerator("cpu,,gpu");
  EXPECT_FALSE(ValidateBackendOptions(backend));
  backend.set_accelerator("cpu");
  backend.set_arm_vulkan_ml_enable_jit(true);
  EXPECT_FALSE(ValidateBackendOptions(backend));
  backend.set_arm_vulkan_ml_enable_jit(false);
  backend.set_num_threads(-1);
  EXPECT_FALSE(ValidateBackendOptions(backend));
  backend.set_num_threads(0);
  backend.set_gpu_use_ir_model("maybe");
  EXPECT_FALSE(ValidateBackendOptions(backend));
}
TEST(BackendTest, NonfiniteInputAndMissingQuantizationAreRejected) {
  TensorSpec tensor;
  tensor.add_shape(1);
  EXPECT_FALSE(EncodeImage({std::nanf("")}, tensor));
  tensor.set_type(INT8);
  EXPECT_FALSE(EncodeImage({1}, tensor));
  tensor.set_scale(0);
  EXPECT_FALSE(EncodeImage({1}, tensor));
  tensor.set_scale(.25f);
  auto quantized = EncodeImage({1}, tensor);
  ASSERT_TRUE(quantized);
  EXPECT_EQ(quantized->at(0), 4);
}
}  // namespace
}  // namespace litert::vision
