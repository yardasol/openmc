import os

from openmc.examples import random_ray_lattice

from tests.testing_harness import TolerantPyAPITestHarnessTD


class MGXSTestHarnessTD(TolerantPyAPITestHarnessTD):
    def _cleanup(self):
        super()._cleanup()
        f = 'mgxs.h5'
        if os.path.exists(f):
            os.remove(f)


def test_random_ray_basic():
    model = random_ray_lattice(time_dependent=True)
    harness = MGXSTestHarnessTD(model, 3)
    harness.main()
