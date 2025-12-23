import os

import openmc
from openmc.utility_funcs import change_directory
from openmc.examples import random_ray_pin_cell
import pytest

from tests.testing_harness import TolerantPyAPITestHarnessTD


class MGXSTestHarnessTD(TolerantPyAPITestHarnessTD):
    def _cleanup(self):
        super()._cleanup()
        f = 'mgxs.h5'
        if os.path.exists(f):
            os.remove(f)


@pytest.mark.parametrize("time_method", ["ti",
                                         "sdp"])
def test_random_ray_time_dependent(time_method):
    with change_directory(time_method):
        openmc.reset_auto_ids()
        model = random_ray_pin_cell(time_dependent=True)
        model.settings.time_dependent['n_timesteps'] = 5
        model.settings.random_ray['time_method'] = time_method
        model.settings.batches = 400
        model.settings.inactive = 200 
        harness = MGXSTestHarnessTD(model, 6)
        harness.main()
