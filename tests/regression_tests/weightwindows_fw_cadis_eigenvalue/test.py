import os

import openmc
from openmc.examples import reflected_absorbing_jezebel

from tests.testing_harness import WeightWindowPyAPITestHarness


class MGXSTestHarness(WeightWindowPyAPITestHarness):
    def _cleanup(self):
        super()._cleanup()
        f = 'mgxs.h5'
        if os.path.exists(f):
            os.remove(f)


def test_weight_windows_fw_cadis_eigenvalue():
    model = reflected_absorbing_jezebel()

    # Fast spectrum ony
    groups = openmc.mgxs.EnergyGroups([2.231e6, 3.679e6, 6.0655e6, 2.e7])

    model.convert_to_multigroup(
        method='material_wise', energy_groups=groups, nparticles=10000,
        overwrite_mgxs_library=True, mgxs_path="mgxs.h5"
    )

    # Convert to a random ray model
    model.convert_to_random_ray()

    # Set the number of particles
    model.settings.particles = 1000
    model.settings.batches = 400
    model.settings.inactive = 200
    model.settings.random_ray['distance_inactive'] = 15.0
    model.settings.random_ray['distance_active'] = 150.0

    # Overlay a basic 2x2 mesh
    n = 40
    mesh = openmc.RegularMesh()
    mesh.dimension = (n, n)
    bbox = model.geometry.bounding_box
    mesh.lower_left = (bbox.lower_left[0], bbox.lower_left[1])
    mesh.upper_right = (bbox.upper_right[0], bbox.upper_right[1])
    model.settings.random_ray['source_region_meshes'] = [
        (mesh, [model.geometry.root_universe])]

    wwg = openmc.WeightWindowGenerator(
        method="fw_cadis", mesh=mesh, max_realizations=model.settings.batches)
    model.settings.weight_window_generators = wwg

    harness = MGXSTestHarness('statepoint.400.h5', model)
    harness.main()
