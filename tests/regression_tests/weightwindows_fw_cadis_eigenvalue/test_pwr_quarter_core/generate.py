import os

import openmc
from openmc import RegularMesh
from openmc.examples import pwr_2d_quarter_core
model = pwr_2d_quarter_core()
model.settings.batches = 300
model.settings.inactive = 50
model.settings.particles = 10000

convert = True
weight_windows = False
if convert:
    ebounds = [1e-5, 0.05, 0.15, 0.275, 0.625, 3, 1.7e4, 8.2e5, 2.e7]
    groups = openmc.mgxs.EnergyGroups(ebounds)

    model.convert_to_multigroup(
        domain_type='universe', domains=[1, 2, 3, 4, 5], energy_groups=groups,
        nparticles=10000, overwrite_mgxs_library=False, mgxs_path="mgxs.h5"
    )

    # Convert to a random ray model
    model.convert_to_random_ray()

    # Set the number of particles
    model.settings.particles = 650
    model.settings.batches = 10000
    model.settings.inactive = 3500
    model.settings.random_ray['distance_inactive'] = 15.0
    model.settings.random_ray['distance_active'] = 150.0

    # Overlay a mesh
    n = 10
    pitch = 1.26
    mesh = RegularMesh()
    mesh.dimension = (n, n)
    mesh.lower_left = (-pitch / 2, -pitch / 2)
    mesh.upper_right = (pitch / 2, pitch / 2)
    #mesh = model.tallies[0].filters[0].mesh
    model.settings.random_ray['source_region_meshes'] = [
        (mesh, [model.geometry.root_universe])]

    wwg = openmc.WeightWindowGenerator(
        method="fw_cadis", mesh=mesh, max_realizations=model.settings.batches - model.settings.inactive)
    model.settings.weight_window_generators = wwg
if weight_windows:
    model.settings.weight_window_checkpoints = {'collision': True, 'surface': True}
    model.settings.survival_biasing = False
    model.settings.weight_windows_file = "weight_windows.h5"
    model.settings.weight_windows_on = True

model.export_to_model_xml()
