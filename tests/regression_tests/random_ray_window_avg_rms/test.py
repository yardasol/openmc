import os

from openmc.examples import random_ray_lattice

from tests.testing_harness import TolerantPyAPITestHarness


class MGXSTestHarness(TolerantPyAPITestHarness):
    def _cleanup(self):
        super()._cleanup()
        f = 'mgxs.h5'
        if os.path.exists(f):
            os.remove(f)


def test_random_ray_basic():
    model = random_ray_lattice()
    model.settings.convergence_method = 'window-averaged rms'
    model.settings.convergence_window_size = 100
    model.settings.max_source_convergence_batches = 500
    model.settings.source_convergence_threshold = 5e-2
    harness = MGXSTestHarness('statepoint.106.h5', model)
    harness.main()
