import unittest
from pathlib import Path

import numpy as np
import os
import torch

import crepe

# this data contains a sine sweep
file = Path(__file__).parent / 'sweep.wav'
f0_file = Path(__file__).parent / 'sweep.f0.csv'


def verify_f0():
    result = np.loadtxt(f0_file, delimiter=',', skiprows=1)

    # it should be confident enough about the presence of pitch in every frame
    assert np.mean(result[:, 2] > 0.5) > 0.98

    # the frequencies should be linear
    assert np.corrcoef(result[:, 1]) > 0.99

    os.remove(f0_file)


class TestSweep(unittest.TestCase):
    # def test_sweep(self):
    #     crepe.process_file(str(file))
    #     verify_f0()
    #
    # def test_to_chainer(self):
    #     output_dir = Path(__file__).parent
    #     for model_name in ['tiny', 'small', 'medium', 'large', 'full']:
    #         model = crepe.core.build_and_load_model(model_name)
    #         with open(output_dir / f'{model_name}.json', mode='w') as f:
    #             f.write(model.to_json())

    def test_pytorch(self):
        # from converted_pytorch import KitModel
        # model = torch.load("converted_pytorch")

        crepe.process_file(str(file))


# def test_sweep_cli(self):
    #     assert os.system("crepe {}".format(file)) == 0
    #     verify_f0()
