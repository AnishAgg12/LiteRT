/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstddef>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/transforms/optimization/Passes.h"  // from @stablehlo
#include "tflite/converter/stablehlo/transforms/stablehlo_passes.h"

namespace mlir {
namespace odml {
namespace {

#define GEN_PASS_DEF_SIMPLIFYHLOPASS
#include "tflite/converter/stablehlo/transforms/stablehlo_passes.h.inc"

// Debug name of StableHLO's constant transpose folder, which is excluded below.
constexpr llvm::StringLiteral kFoldTransposeOpPattern =
    "::FoldTransposeOpPattern";

class SimplifyHloPass : public impl::SimplifyHloPassBase<SimplifyHloPass> {
 public:
  void runOnOperation() override;
};

void SimplifyHloPass::runOnOperation() {
  MLIRContext* context = &getContext();
  RewritePatternSet patterns(context);

  // Folder float rewrites that may change numerics are disabled.
  stablehlo::StablehloAggressiveFolderPassOptions folder_options;
  folder_options.optimizeFloat = false;
  stablehlo::populateStablehloAggressiveFolderPatterns(context, &patterns,
                                                       folder_options);

  // Unlike `mhlo.transpose` (which only folded splats and identity
  // permutations), StableHLO's transpose folder materializes the transpose of
  // any constant element by element, ignoring `foldOpElementLimit`. On large
  // weights (e.g. `F.linear` kernels) that dominates conversion time, so drop
  // it: splats are folded by prepare-hlo's `FoldSplatTranspose` and identity
  // permutations by the canonicalization patterns below. The pattern is found
  // by its debug name, so fail loudly if that ever stops matching.
  auto& native_patterns = patterns.getNativePatterns();
  const size_t num_patterns = native_patterns.size();
  llvm::erase_if(
      native_patterns, [](const std::unique_ptr<RewritePattern>& pattern) {
        return pattern->getDebugName().ends_with(kFoldTransposeOpPattern);
      });
  const size_t num_erased = num_patterns - native_patterns.size();
  if (num_erased != 1) {
    getOperation().emitError()
        << "expected to exclude exactly one StableHLO pattern named '*"
        << kFoldTransposeOpPattern << "', but excluded " << num_erased;
    return signalPassFailure();
  }

  stablehlo::populateStablehloCanonicalizationPatterns(context, &patterns);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateSimplifyHloPass() {
  return std::make_unique<SimplifyHloPass>();
}

static PassRegistration<SimplifyHloPass> pass;

}  // namespace odml
}  // namespace mlir
