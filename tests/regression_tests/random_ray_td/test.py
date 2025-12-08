import os

import openmc
from openmc.utility_funcs import change_directory
from openmc.examples import random_ray_lattice
import pytest

from tests.testing_harness import TolerantPyAPITestHarnessTD


class MGXSTestHarnessTD(TolerantPyAPITestHarnessTD):
    def _cleanup(self):
        super()._cleanup()
        f = 'mgxs.h5'
        if os.path.exists(f):
            os.remove(f)


@pytest.mark.parametrize("time_dependent_method", ["ti",
                                       "sdp"])
def test_random_ray_time_dependent(time_dependent_method):
    with change_directory(time_dependent_method):
        openmc.reset_auto_ids()
        model = random_ray_lattice(time_dependent=True)
        model.settings.random_ray['time_mode'] = time_dependent_method
        harness = MGXSTestHarnessTD(model, 3)
        harness.main()
