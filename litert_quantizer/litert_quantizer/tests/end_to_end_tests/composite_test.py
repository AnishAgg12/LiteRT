# Copyright 2024 Google LLC
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
# ==============================================================================

"""E2E tests for the quantizer for model with add."""

from absl.testing import absltest
from absl.testing import parameterized
from flatbuffers import flexbuffers
import numpy as np

from ai_edge_litert.tools import flatbuffer_utils
from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import test_utils
from litert_quantizer.utils import tfl_flatbuffer_utils
from litert_quantizer.utils import tfl_interpreter_utils

_RNG = np.random.default_rng(42)


def _create_layer_norm_composite_model() -> bytearray:
  """Creates an in-memory LayerNorm (group_norm) composite model.

  Built in-memory from `simple_composite.tflite` instead of checking in a new
  `.tflite` binary file because only
  `test_layer_norm_composite_with_decomposition` uses it.

  Returns:
    Serialized TFLite model bytearray.
  """
  model_path = test_utils.get_path_to_datafile(
      '../models/simple_composite.tflite'
  )
  model = tfl_flatbuffer_utils.read_model(model_path)

  # Configure main subgraph: composite group_norm with inputs
  # (x [32, 32], gamma [32], beta [32]) and output [32, 32].
  main_sg = model.subgraphs[0]
  main_sg.tensors[1].shape = [32]
  beta_tensor = qtyping.TensorT()
  beta_tensor.name = b'arg2'
  beta_tensor.shape = [32]
  beta_tensor.type = qtyping.TensorType.FLOAT32
  beta_tensor.buffer = 0
  main_sg.tensors.append(beta_tensor)
  main_sg.inputs = [0, 1, 3]

  composite_op = main_sg.operators[0]
  composite_op.inputs = [0, 1, 3]
  composite_opts = composite_op.builtinOptions2
  composite_opts.name = b'od' + b'ml.group_norm'
  composite_opts.decompositionSubgraphIndex = 1
  # LayerNorm uses the group_norm composite with one group and sub_type 1.
  composite_opts.compositeAttributes = np.frombuffer(
      flexbuffers.Dumps({
          'channel_axis': -1,
          'epsilon': 1e-5,
          'num_groups': 1,
          'sub_type': 1,
      }),
      dtype=np.uint8,
  )

  sig_input_beta = flatbuffer_utils.schema_fb.TensorMapT()
  sig_input_beta.name = b'arg2'
  sig_input_beta.tensorIndex = 3
  model.signatureDefs[0].inputs.append(sig_input_beta)

  # Configure decomposition subgraph: (decomp_arg0 * decomp_arg1) + decomp_arg2.
  mul_op_code = qtyping.OperatorCodeT()
  mul_op_code.builtinCode = qtyping.BuiltinOperator.MUL
  mul_op_code.deprecatedBuiltinCode = qtyping.BuiltinOperator.MUL
  mul_opcode_index = len(model.operatorCodes)
  model.operatorCodes.append(mul_op_code)

  decomp_sg = model.subgraphs[1]
  decomp_sg.tensors[1].shape = [32]
  decomp_beta = qtyping.TensorT()
  decomp_beta.name = b'decomp_arg2'
  decomp_beta.shape = [32]
  decomp_beta.type = qtyping.TensorType.FLOAT32
  decomp_beta.buffer = 0
  decomp_sg.tensors.append(decomp_beta)

  decomp_mul_out = qtyping.TensorT()
  decomp_mul_out.name = b'tfl.mul'
  decomp_mul_out.shape = [32, 32]
  decomp_mul_out.type = qtyping.TensorType.FLOAT32
  decomp_mul_out.buffer = 0
  decomp_sg.tensors.append(decomp_mul_out)
  decomp_sg.inputs = [0, 1, 3]

  mul_op = qtyping.OperatorT()
  mul_op.opcodeIndex = mul_opcode_index
  mul_op.inputs = [0, 1]
  mul_op.outputs = [4]
  mul_op.builtinOptionsType = qtyping.BuiltinOptions.MulOptions
  mul_op.builtinOptions = qtyping.MulOptionsT()

  add_op = decomp_sg.operators[0]
  add_op.inputs = [4, 3]
  add_op.outputs = [2]
  decomp_sg.operators = [mul_op, add_op]

  return flatbuffer_utils.convert_object_to_bytearray(model)


# test_with_decomposition.mlir
# func @main(%arg0, %arg1) ->  {
#   %0 = composite "npu_call" %arg0, %arg1 {decomposition = @decomp}
#   return %0
# }
# func private @decomp(%arg0: , %arg1: ) ->  {
#   %0 = tfl.add %arg0, %arg1
#   return %0
# }

# test_with_kernel.mlir (subset)
# %12 = stablehlo.composite "scaled_dot_product_attention" ...


def _get_calibration_data(
    model: qtyping.ModelT,
    name_pref: str,
    subgraph_index: int = 0,
    num_samples: int = 2,
):
  sg: qtyping.SubGraphT = model.subgraphs[subgraph_index]
  inputs = [sg.tensors[i] for i in sg.inputs]

  data = []

  for _ in range(num_samples):
    samp_data = {}
    for i, inp in enumerate(inputs):
      inp: qtyping.TensorT = inp
      name = f'{name_pref}{i}'
      samp_data[name] = _RNG.uniform(size=inp.shape).astype(np.float32)

    data.append(samp_data)

  return {
      tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: data,
  }


class CompositeTest(parameterized.TestCase):

  @property
  def output_tolerance(self) -> float:
    return 1e-4

  def assertNotQuantiezedType(self, tensor: qtyping.TensorT):
    self.assertIn(
        tensor.type, [qtyping.TensorType.FLOAT32, qtyping.TensorType.INT32]
    )

  @parameterized.parameters(
      'default_a8w8',
      'default_a16w8',
  )
  def test_composite_with_decomposition(self, recipe_name):
    model = test_utils.get_path_to_datafile('../models/simple_composite.tflite')
    qt = quantizer.Quantizer(model)
    qt.load_quantization_recipe(recipe_name)

    f_model = tfl_flatbuffer_utils.read_model(model)
    self.assertTrue(qt.need_calibration)
    calib_data = _get_calibration_data(f_model, 'arg')
    calibration_result = qt.calibrate(calib_data)

    result = qt.quantize(calibration_result)
    q_model = tfl_flatbuffer_utils.read_model(result.quantized_model)
    for sg in q_model.subgraphs:
      for tensor in sg.tensors:
        self.assertIsNotNone(tensor.quantization)

    test_data = _get_calibration_data(f_model, 'arg')
    comparison = qt.validate(
        error_metrics=[quantizer.ValidationErrorMetric.MSE], test_data=test_data
    )
    _all_results = comparison.get_all_tensor_results()
    metric = 'mean_squared_difference'
    with self.subTest(error_metric=metric):
      results = {k: v.get(metric, 0.0) for k, v in _all_results.items()}
      output_mse = results['output']
      self.assertLess(output_mse, self.output_tolerance)

  @parameterized.parameters(
      'default_a8w8',
      'default_a16w8',
  )
  def test_composite_with_kernel(self, recipe_name):
    model = test_utils.get_path_to_datafile('../models/sdpa_composite.tflite')
    qt = quantizer.Quantizer(model)
    qt.load_quantization_recipe(recipe_name)
    self.assertTrue(qt.need_calibration)

    f_model = tfl_flatbuffer_utils.read_model(model)
    calib_data = _get_calibration_data(f_model, 'args_')
    calibration_result = qt.calibrate(calib_data)

    result = qt.quantize(calibration_result)
    q_model = tfl_flatbuffer_utils.read_model(result.quantized_model)

    main_sg = q_model.subgraphs[0]
    for op in main_sg.operators:
      if 'COMPOSITE' in flatbuffer_utils.opcode_to_name(
          q_model, op.opcodeIndex
      ):
        for inp in op.inputs:
          tensor = main_sg.tensors[inp]
          self.assertNotQuantiezedType(tensor)
        for out in op.outputs:
          tensor = main_sg.tensors[out]
          self.assertNotQuantiezedType(tensor)

    sdpa_sg = q_model.subgraphs[1]
    for tensor in sdpa_sg.tensors:
      self.assertNotQuantiezedType(tensor)

  @parameterized.named_parameters(
      dict(testcase_name='a8w8', recipe_name='default_a8w8'),
      dict(testcase_name='a16w8', recipe_name='default_a16w8'),
  )
  def test_layer_norm_composite_with_decomposition(self, recipe_name: str):
    model = _create_layer_norm_composite_model()
    qt = quantizer.Quantizer(model)
    qt.load_quantization_recipe(recipe_name)

    f_model = tfl_flatbuffer_utils.read_model(model)
    self.assertTrue(qt.need_calibration)
    calib_data = _get_calibration_data(f_model, 'arg')
    calibration_result = qt.calibrate(calib_data)

    result = qt.quantize(calibration_result)
    q_model = tfl_flatbuffer_utils.read_model(result.quantized_model)
    for sg in q_model.subgraphs:
      for tensor in sg.tensors:
        self.assertIsNotNone(tensor.quantization)

    test_data = _get_calibration_data(f_model, 'arg')
    comparison = qt.validate(
        error_metrics=[quantizer.ValidationErrorMetric.MSE], test_data=test_data
    )
    all_results = comparison.get_all_tensor_results()
    metric = 'mean_squared_difference'
    with self.subTest(error_metric=metric):
      results = {k: v.get(metric, 0.0) for k, v in all_results.items()}
      output_mse = results['output']
      self.assertLess(output_mse, self.output_tolerance)


if __name__ == '__main__':
  absltest.main()
