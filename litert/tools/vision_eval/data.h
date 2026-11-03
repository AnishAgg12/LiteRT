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

#ifndef LITERT_TOOLS_VISION_EVAL_DATA_H_
#define LITERT_TOOLS_VISION_EVAL_DATA_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "litert/cc/litert_expected.h"
#include "litert/tools/vision_eval/evaluation.pb.h"
#include "litert/tools/vision_eval/image.h"
#include "google/protobuf/message.h"  // from @com_google_protobuf

namespace litert::vision {
Error Invalid(const std::string& message);
Expected<void> ReadProto(const std::string& path, google::protobuf::Message& message);
Expected<void> WriteProto(const std::string& path,
                          const google::protobuf::Message& message,
                          ProtoFormat format = BINARY_PROTO);
Expected<void> WriteBytes(const std::string& path, const void* data, size_t size);
Expected<size_t> TensorBytes(const TensorSpec& spec);
Expected<std::vector<uint8_t>> ReadTensor(const std::string& path,
                                          const TensorSpec& spec);
Expected<std::vector<float>> ReadFloats(const std::string& path,
                                        const TensorSpec& spec);
Expected<FileIdentity> Identify(const std::string& path);
Expected<void> VerifyFile(const FileIdentity& file);
bool SafeName(const std::string& value);
std::string Resolve(const std::string& root, const std::string& path);
Expected<DatasetManifest> LoadManifest(const EvaluationConfig& config);
Expected<void> ValidateManifest(const DatasetManifest& manifest, Task task);
Expected<ImagePreprocessing> ImageOptions(const Preprocessing& preprocessing);
Expected<void> ValidateConfig(const EvaluationConfig& config);
Expected<EvaluationConfig> ParseArgs(int argc, char** argv);
void PrintUsage();
}  // namespace litert::vision

#endif  // LITERT_TOOLS_VISION_EVAL_DATA_H_
