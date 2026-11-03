# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare C++ scoring against the NumPy protocols from PR #11544.

The Python implementation is a test oracle, never an evaluation dependency.
"""

import os
from pathlib import Path
import subprocess

from google.protobuf import text_format
import numpy as np

from absl.testing import absltest as googletest
from litert.tools.vision_eval import evaluation_pb2 as pb
from litert.tools.vision_eval import metrics_reference as reference


class MetricParityTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.root = Path(self.create_tempdir().full_path)
    self.binary = (
        Path(os.environ['TEST_SRCDIR'])
        / os.environ['TEST_WORKSPACE']
        / os.environ['LITERT_VISION_EVAL_BIN']
    )
    self.manifest = pb.DatasetManifest(version=1)
    self.manifest.inputs.add(name='image', shape=[1])
    (self.root / 'input.raw').write_bytes(np.zeros(1, dtype='<f4').tobytes())
    self.oracles = {
        'seg': reference.SegmentationMetrics(classes=3),
        'depth': reference.DepthMetrics(),
        'normals': reference.NormalMetrics(),
    }
    for name, channels, task in [('seg', 3, pb.SEGMENTATION),
                                 ('depth', 1, pb.DEPTH),
                                 ('normals', 3, pb.NORMALS)]:
      output = self.manifest.outputs.add()
      output.tensor.CopyFrom(pb.TensorSpec(name=name, shape=[1, channels, 2, 2]))
      output.scoring.task = task
      output.scoring.classes = 3
    rng = np.random.default_rng(31)
    for i in range(3):
      sample = self.manifest.samples.add(id=f'image_{i}')
      source = sample.inputs.add(name='image')
      source.file.path = 'input.raw'
      directory = self.root / 'predictions' / sample.id
      directory.mkdir(parents=True)
      for name in self.oracles:
        channels = 1 if name == 'depth' else 3
        prediction = rng.normal(size=(channels, 2, 2)).astype('<f4')
        if name == 'depth':
          prediction = np.abs(prediction) * 3
          target = rng.uniform(0, 10, size=(3, 4)).astype('<f4')
          target[0, :3] = [0, np.nan, .001]
        elif name == 'seg':
          target = rng.integers(0, 3, size=(3, 4), dtype=np.uint8)
          target[0, 0] = 255
        else:
          target = rng.normal(size=(3, 3, 4)).astype('<f4')
          target[:, 0, 0] = 0
          target[0, 0, 1] = np.nan
          prediction[:, 0, 0] = 0
        prediction.tofile(directory / f'{name}.raw')
        target.tofile(self.root / f'{sample.id}_{name}.raw')
        label = sample.targets.add(name=name)
        label.file.path = f'{sample.id}_{name}.raw'
        label.tensor.CopyFrom(pb.TensorSpec(
            shape=target.shape, type=pb.UINT8 if name == 'seg' else pb.FLOAT32))
        self.oracles[name].update(prediction[0] if name == 'depth' else prediction, target)
    self.manifest_path = self.root / 'manifest.txtpb'
    self.manifest_path.write_text(text_format.MessageToString(self.manifest))

  def command(self, output='results', binary=False):
    return [str(self.binary), '--score_only=true',
            f'--manifest={self.manifest_path}',
            f'--predictions_dir={self.root / "predictions"}',
            f'--output_dir={self.root / output}',
            '--summary_format=binaryproto' if binary else '--summary_format=textproto']

  def summary(self, directory='results', binary=False):
    summary = pb.RunSummary()
    if binary:
      summary.ParseFromString((self.root / directory / 'summary.binpb').read_bytes())
    else:
      text_format.Parse((self.root / directory / 'summary.txtpb').read_text(), summary)
    return summary

  def test_cpp_metrics_match_numpy_and_resume(self):
    subprocess.run(self.command(), check=True, capture_output=True)
    summary = self.summary()
    self.assertEqual(summary.status, pb.PASSED)
    self.assertEqual(summary.completed, 3)
    for row in summary.metrics:
      expected = self.oracles[row.name].result()
      actual = getattr(row, row.WhichOneof('metrics'))
      for name, value in expected.items():
        if name == 'confusion':
          self.assertEqual(list(actual.confusion), np.asarray(value).ravel().tolist())
        elif name == 'within_11.25_degrees':
          self.assertAlmostEqual(actual.within_11_25_degrees, value, places=7)
        elif name == 'within_22.5_degrees':
          self.assertAlmostEqual(actual.within_22_5_degrees, value, places=7)
        elif value is None:
          self.assertFalse(actual.HasField(name))
        else:
          np.testing.assert_allclose(getattr(actual, name), value, atol=2e-5, rtol=2e-6)
    subprocess.run(self.command(), check=True, capture_output=True)
    resumed = self.summary()
    self.assertEqual(resumed.resumed, 3)
    self.assertEqual(resumed.metrics, summary.metrics)
    subprocess.run(self.command('binary', binary=True), check=True, capture_output=True)
    self.assertEqual(self.summary('binary', binary=True).metrics, summary.metrics)

  def test_modified_prediction_and_corrupt_checkpoint_are_rejected(self):
    subprocess.run(self.command(), check=True, capture_output=True)
    path = self.root / 'predictions/image_0/depth.raw'
    np.zeros((2, 2), dtype='<f4').tofile(path)
    status = subprocess.run(self.command(), check=False, capture_output=True)
    self.assertNotEqual(status.returncode, 0)
    self.assertEqual(self.summary().status, pb.FAILED)
    subprocess.run(self.command('fresh'), check=True, capture_output=True)
    (self.root / 'fresh/samples/image_0/result.binpb').write_bytes(b'broken checkpoint')
    status = subprocess.run(self.command('fresh'), check=False, capture_output=True)
    self.assertNotEqual(status.returncode, 0)


if __name__ == '__main__':
  googletest.main()
