import os

import openmc
from openmc import RegularMesh
from openmc.examples import reflected_absorbing_jezebel

model = reflected_absorbing_jezebel()

convert = False
weight_windows = False
if convert:
    # Fast spectrum ony
    ebounds = [2.231e6, 3.679e6, 6.0655e6, 2.e7]
    groups = openmc.mgxs.EnergyGroups(ebounds)

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

    # Overlay a mesh
    mesh = model.tallies[0].filters[0].mesh
    model.settings.random_ray['source_region_meshes'] = [
        (mesh, [model.geometry.root_universe])]

    wwg = openmc.WeightWindowGenerator(
        method="fw_cadis", mesh=mesh, max_realizations=model.settings.batches)
    model.settings.weight_window_generators = wwg
if weight_windows:
    model.settings.weight_window_checkpoints = {'collision': True, 'surface': True}
    model.settings.survival_biasing = False
    model.settings.weight_windows_file = "weight_windows.h5"
    model.settings.weight_windows_on = True

model.export_to_model_xml()
