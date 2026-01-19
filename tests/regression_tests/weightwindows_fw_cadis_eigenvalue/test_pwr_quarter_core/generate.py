import os

import openmc
from openmc import RegularMesh
from openmc.examples import pwr_2d_quarter_core

convert = True
weight_windows = False
homog_pin_cell = False

model = pwr_2d_quarter_core(homog_pin_cell)

# settings from Wagner paper
model.settings.batches = 600
model.settings.inactive = 50
model.settings.particles = 15000
model.materials.cross_sections = '/home/olek/projects/cross-section-libraries/endfb71_hdf5/cross_sections.xml'
if convert:
    ebounds = [1e-5, 0.05, 0.15, 0.275, 0.625, 3, 1.7e4, 8.2e5, 2.e7]
    groups = openmc.mgxs.EnergyGroups(ebounds)

    model.convert_to_multigroup(
        domain_type='universe', domains=[1, 2, 3], energy_groups=groups,
        nparticles = model.settings.particles,
        overwrite_mgxs_library=False, correction='P0'
    )

    # Convert to a random ray model
    model.convert_to_random_ray()

    # Set the number of particles
    model.settings.particles = 10000
    model.settings.batches = 10000
    model.settings.inactive = 3500
    model.settings.random_ray['distance_inactive'] = 25.0
    model.settings.random_ray['distance_active'] = 820.0

    # Overlay a mesh
    base_mesh = model.tallies[0].filters[0].mesh
    m = 4
    x = int(17 * 10.5 + 0.5)
    n = m * x
    pitch = 1.26 / m
    mesh = RegularMesh(mesh_id=2)
    mesh.dimension = (n, n)
    mesh.lower_left = (-pitch / 2, -pitch / 2)
    mesh.upper_right = (1.26 * 17 * 10.5, 1.26 * 17 * 10.5)
    model.settings.random_ray['source_region_meshes'] = [
        (mesh, [model.geometry.root_universe])]
    model.settings.random_ray['source_shape'] = 'linear'

    ww_mesh = model.tallies[0].filters[0].mesh
    wwg = openmc.WeightWindowGenerator(energy_bounds=ebounds,
        method="fw_cadis", mesh=ww_mesh, max_realizations=model.settings.batches - model.settings.inactive)
    model.settings.weight_window_generators = wwg
if weight_windows:
    model.settings.weight_window_checkpoints = {'collision': True, 'surface': True}
    model.settings.survival_biasing = False
    model.settings.weight_windows_file = "weight_windows.h5"
    model.settings.weight_windows_on = True
    model.materials.cross_sections = '/home/olek/projects/cross-section-libraries/endfb71_hdf5/cross_sections.xml'

model.export_to_model_xml()
# Repace the line <cell id="20" material="3" universe="1"/>
# with            <cell id="20" material="1" universe="1"/>

