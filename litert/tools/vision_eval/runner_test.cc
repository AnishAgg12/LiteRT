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

#include "litert/tools/vision_eval/runner.h"

#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "litert/core/filesystem.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::vision {
namespace {
namespace fs = ::litert::internal;

class RunnerTest : public testing::Test {
 protected:
  void SetUp() override {
    const auto* test = testing::UnitTest::GetInstance()->current_test_info();
    root_ = fs::Join({testing::TempDir(), test->name()});
    ASSERT_TRUE(fs::MkDir(root_));
    tflite::ModelT model;
    model.version = 3;
    model.buffers.push_back(std::make_unique<tflite::BufferT>());
    model.operator_codes.push_back(std::make_unique<tflite::OperatorCodeT>());
    model.operator_codes[0]->builtin_code = tflite::BuiltinOperator_ADD;
    auto graph = std::make_unique<tflite::SubGraphT>();
    graph->inputs = {0, 1};
    graph->outputs = {2, 3};
    for (const char* name : {"left", "right", "depth", "segmentation"}) {
      auto tensor = std::make_unique<tflite::TensorT>();
      tensor->name = name;
      tensor->type = tflite::TensorType_FLOAT32;
      tensor->shape = {1, 1, 1, 3};
      graph->tensors.push_back(std::move(tensor));
    }
    for (int output : {2, 3}) {
      auto op = std::make_unique<tflite::OperatorT>();
      op->opcode_index = 0;
      op->inputs = {0, 1};
      op->outputs = {output};
      op->builtin_options.Set(tflite::AddOptionsT());
      graph->operators.push_back(std::move(op));
    }
    model.subgraphs.push_back(std::move(graph));
    auto signature = std::make_unique<tflite::SignatureDefT>();
    signature->signature_key = "serving_default";
    for (int i = 0; i < 4; ++i) {
      auto map = std::make_unique<tflite::TensorMapT>();
      map->name = model.subgraphs[0]->tensors[i]->name;
      map->tensor_index = i;
      (i < 2 ? signature->inputs : signature->outputs).push_back(std::move(map));
    }
    model.signature_defs.push_back(std::move(signature));
    flatbuffers::FlatBufferBuilder builder;
    auto packed = tflite::Model::Pack(builder, &model);
    tflite::FinishModelBuffer(builder, packed);
    ASSERT_TRUE(WriteBytes(fs::Join({root_, "model.tflite"}), builder.GetBufferPointer(), builder.GetSize()));
    for (const char* binding : {"left", "right"}) {
      auto* input = manifest_.add_inputs();
      input->set_name(std::string("data_") + binding);
      input->set_binding_name(binding);
      for (int dim : {1, 1, 1, 3}) input->add_shape(dim);
    }
    for (bool depth : {true, false}) {
      auto* output = manifest_.add_outputs();
      output->mutable_tensor()->set_name(depth ? "depth" : "segmentation");
      for (int dim : {1, 1, 1, 3}) output->mutable_tensor()->add_shape(dim);
      output->mutable_scoring()->set_task(depth ? DEPTH : SEGMENTATION);
      output->mutable_scoring()->set_classes(1);
    }
    float inputs[] = {1, 2, 3};
    float truth[] = {2, 4, 6};
    uint8_t labels[] = {0, 0, 0};
    ASSERT_TRUE(WriteBytes(fs::Join({root_, "input.raw"}), inputs, sizeof(inputs)));
    ASSERT_TRUE(WriteBytes(fs::Join({root_, "depth.raw"}), truth, sizeof(truth)));
    ASSERT_TRUE(WriteBytes(fs::Join({root_, "labels.raw"}), labels, sizeof(labels)));
    for (int i = 0; i < 3; ++i) {
      auto* sample = manifest_.add_samples();
      sample->set_id("sample_" + std::to_string(i));
      for (const auto& spec : manifest_.inputs()) {
        auto* input = sample->add_inputs();
        input->set_name(spec.name());
        input->mutable_file()->set_path("input.raw");
      }
      for (bool depth : {true, false}) {
        auto* target = sample->add_targets();
        target->set_name(depth ? "depth" : "segmentation");
        target->mutable_file()->set_path(depth ? "depth.raw" : "labels.raw");
        target->mutable_tensor()->set_type(depth ? FLOAT32 : UINT8);
        target->mutable_tensor()->add_shape(1);
        target->mutable_tensor()->add_shape(3);
      }
    }
    ASSERT_TRUE(WriteProto(fs::Join({root_, "manifest.txtpb"}), manifest_, TEXT_PROTO));
    config_.set_model((fs::Join({root_, "model.tflite"})));
    config_.set_manifest((fs::Join({root_, "manifest.txtpb"})));
    config_.set_output_dir((fs::Join({root_, "results"})));
    config_.mutable_backend()->set_accelerator("cpu");
    config_.mutable_backend()->set_num_threads(1);
    config_.set_dump_predictions(true);
    config_.set_dump_inputs(true);
  }
  void QuantizedModel(const std::string& path, bool unsigned_input, int zero_point) {
    std::ifstream input(fs::Join({root_, "model.tflite"}), std::ios::binary);
    ASSERT_TRUE(input);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    std::unique_ptr<tflite::ModelT> model(tflite::GetModel(bytes.data())->UnPack());
    for (auto& tensor : model->subgraphs[0]->tensors) {
      tensor->type = unsigned_input ? tflite::TensorType_UINT8 : tflite::TensorType_INT8;
      tensor->quantization = std::make_unique<tflite::QuantizationParametersT>();
      tensor->quantization->scale = {.25f};
      tensor->quantization->zero_point = {zero_point};
    }
    model->operator_codes[0]->version = 2;
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, model.get()));
    ASSERT_TRUE(WriteBytes(path, builder.GetBufferPointer(), builder.GetSize()));
  }
  Expected<void> Run(RunSummary& summary) {
    // The test's model is a stable stand-in for the executable identity.
    return Evaluate(config_, summary, config_.model());
  }
  std::string root_;
  EvaluationConfig config_;
  DatasetManifest manifest_;
};

TEST_F(RunnerTest, NamedMultipleInputsAndOutputsBaselineAndResume) {
  config_.set_baseline_model(config_.model());
  config_.set_baseline_accelerator("cpu");
  config_.set_require_output_parity(true);
  config_.set_warmup_runs(1);
  RunSummary first;
  auto status = Run(first);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(first.status(), PASSED);
  EXPECT_EQ(first.completed(), 3);
  EXPECT_EQ(first.metrics_size(), 2);
  EXPECT_EQ(first.baseline_metrics_size(), 2);
  EXPECT_EQ(first.comparisons_size(), 2);
  EXPECT_TRUE(first.comparisons(0).allclose());
  EXPECT_TRUE(first.identity().candidate().has_model());
  EXPECT_EQ(first.identity().candidate().inputs_size(), 2);
  EXPECT_TRUE(fs::Exists(fs::Join({root_, "results/samples/sample_0/depth.raw"})));
  RunSummary second;
  status = Run(second);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(second.resumed(), 3);
  EXPECT_EQ(second.metrics(0).SerializeAsString(), first.metrics(0).SerializeAsString());
  EXPECT_EQ(second.provenance_sha256(), first.provenance_sha256());
}

TEST_F(RunnerTest, FailedSampleLeavesCompletedSamplesResumable) {
  manifest_.mutable_samples(1)->mutable_inputs(0)->mutable_file()->set_path("missing.raw");
  ASSERT_TRUE(WriteProto(fs::Join({root_, "manifest.txtpb"}), manifest_, TEXT_PROTO));
  RunSummary first;
  EXPECT_FALSE(Run(first));
  EXPECT_EQ(first.status(), FAILED);
  EXPECT_EQ(first.completed(), 1);
  EXPECT_TRUE(fs::Exists(fs::Join({root_, "results/samples/sample_0/result.binpb"})));
  EXPECT_FALSE(fs::Exists(fs::Join({root_, "results/samples/sample_1/result.binpb"})));
  float inputs[] = {1, 2, 3};
  ASSERT_TRUE(WriteBytes(fs::Join({root_, "missing.raw"}), inputs, sizeof(inputs)));
  RunSummary second;
  auto status = Run(second);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(second.completed(), 3);
  EXPECT_EQ(second.resumed(), 1);
}

TEST_F(RunnerTest, ChangedProvenanceDoesNotOverwriteExistingSummary) {
  RunSummary first;
  auto status = Run(first);
  ASSERT_TRUE(status) << status.Error().Message();
  auto original = HashFile((fs::Join({root_, "results/summary.txtpb"})));
  ASSERT_TRUE(original);
  config_.set_rtol(.2);
  RunSummary second;
  EXPECT_FALSE(Run(second));
  auto after = HashFile((fs::Join({root_, "results/summary.txtpb"})));
  ASSERT_TRUE(after);
  EXPECT_EQ(*original, *after);
}

TEST_F(RunnerTest, ModifiedCompletedInputIsRejected) {
  RunSummary first;
  auto status = Run(first);
  ASSERT_TRUE(status) << status.Error().Message();
  float changed[] = {4, 2, 3};
  ASSERT_TRUE(WriteBytes(fs::Join({root_, "input.raw"}), changed, sizeof(changed)));
  RunSummary second;
  EXPECT_FALSE(Run(second));
  EXPECT_EQ(second.status(), FAILED);
  EXPECT_EQ(second.resumed(), 0);
}
TEST_F(RunnerTest, QuantizedBaselinesWithDifferentZeroPointsReceiveSameRealInputs) {
  for (bool unsigned_input : {false, true}) {
    const int zero = unsigned_input ? 128 : 7;
    const auto candidate = fs::Join({root_, (unsigned_input ? "u8.tflite" : "i8.tflite")});
    const auto baseline = fs::Join({root_, (unsigned_input ? "base_u8.tflite" : "base_i8.tflite")});
    ASSERT_NO_FATAL_FAILURE(QuantizedModel(candidate, unsigned_input, zero));
    ASSERT_NO_FATAL_FAILURE(QuantizedModel(baseline, unsigned_input, zero - 10));
    config_.set_model(candidate);
    config_.set_baseline_model(baseline);
    config_.set_baseline_accelerator("cpu");
    config_.set_require_output_parity(true);
    config_.set_output_dir((fs::Join({root_, (unsigned_input ? "results_u8" : "results_i8")})));
    const auto type = unsigned_input ? UINT8 : INT8;
    for (auto& input : *manifest_.mutable_inputs()) {
      input.set_type(type);
      input.set_scale(.25f);
      input.set_zero_point(zero);
    }
    for (auto& output : *manifest_.mutable_outputs()) {
      output.mutable_tensor()->set_type(type);
      output.mutable_tensor()->set_scale(.25f);
      output.mutable_tensor()->set_zero_point(zero);
    }
    const std::vector<uint8_t> input = {static_cast<uint8_t>(zero + 4),
                                       static_cast<uint8_t>(zero + 8),
                                       static_cast<uint8_t>(zero + 12)};
    ASSERT_TRUE(WriteBytes(fs::Join({root_, "input.raw"}), input.data(), input.size()));
    ASSERT_TRUE(WriteProto(fs::Join({root_, "manifest.txtpb"}), manifest_, TEXT_PROTO));
    RunSummary summary;
    auto status = Run(summary);
    ASSERT_TRUE(status) << status.Error().Message();
    ASSERT_EQ(summary.comparisons_size(), 2);
    EXPECT_TRUE(summary.comparisons(0).allclose());
    EXPECT_TRUE(summary.comparisons(1).allclose());
    EXPECT_EQ(summary.metrics(0).depth().abs_rel(), 0);
  }
}

TEST_F(RunnerTest, NonfinitePreparedInputsFailBeforeInvocation) {
  float input[] = {std::numeric_limits<float>::quiet_NaN(), 2, 3};
  ASSERT_TRUE(WriteBytes(fs::Join({root_, "input.raw"}), input, sizeof(input)));
  RunSummary summary;
  EXPECT_FALSE(Run(summary));
  EXPECT_EQ(summary.status(), FAILED);
  EXPECT_EQ(summary.completed(), 0);
  EXPECT_FALSE(fs::Exists(fs::Join({root_, "results/samples/sample_0/result.binpb"})));
}

TEST_F(RunnerTest, SemanticallyCorruptCheckpointIsRejected) {
  RunSummary first;
  auto status = Run(first);
  ASSERT_TRUE(status) << status.Error().Message();
  const auto path = fs::Join({root_, "results/samples/sample_0/result.binpb"});
  SampleResult row;
  ASSERT_TRUE(ReadProto(path, row));
  row.mutable_metrics(0)->mutable_depth()->set_images(2);
  ASSERT_TRUE(WriteProto(path, row));
  RunSummary second;
  EXPECT_FALSE(Run(second));
  EXPECT_EQ(second.status(), FAILED);
}
}  // namespace
}  // namespace litert::vision
