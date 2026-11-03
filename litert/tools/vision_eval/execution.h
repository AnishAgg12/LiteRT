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

#ifndef LITERT_TOOLS_VISION_EVAL_EXECUTION_H_
#define LITERT_TOOLS_VISION_EVAL_EXECUTION_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/tools/vision_eval/data.h"

namespace litert::vision {
Expected<void> ValidateBackendOptions(const BackendOptions& options);
BackendOptions BaselineOptions(const EvaluationConfig& config);
Expected<Options> CreateOptions(const BackendOptions& options,
                                bool reference_cpu = false);
DeviceIdentity DeviceInfo();
Expected<std::vector<FileIdentity>> RuntimeLibraries(const EvaluationConfig& config);
Expected<std::vector<uint8_t>> EncodeImage(const std::vector<float>& input,
                                           const TensorSpec& spec);

// Own one compiled model and reuse its buffers across samples. The enclosing
// Environment must outlive this object.
class Execution {
 public:
  static Expected<std::unique_ptr<Execution>> Create(
      Environment& environment, const BackendOptions& backend,
      const std::string& path, int signature_index, bool reference_cpu,
      ExecutionIdentity& identity);
  Expected<std::vector<std::vector<float>>> Invoke(
      const std::vector<std::vector<uint8_t>>& inputs, int warmups,
      double& milliseconds);
  const std::vector<TensorSpec>& inputs() const { return input_specs_; }
  const std::vector<TensorSpec>& outputs() const { return output_specs_; }

 private:
  explicit Execution(CompiledModel model, int signature)
      : model_(std::move(model)), signature_(signature) {}
  // Buffers are destroyed before the model that owns their requirements.
  CompiledModel model_;
  std::vector<TensorBuffer> inputs_;
  std::vector<TensorBuffer> outputs_;
  std::vector<TensorSpec> input_specs_;
  std::vector<TensorSpec> output_specs_;
  int signature_;
};
}  // namespace litert::vision

#endif  // LITERT_TOOLS_VISION_EVAL_EXECUTION_H_
