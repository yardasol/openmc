#include "openmc/random_ray/random_ray_simulation.h"
#include "openmc/random_ray/bd_utilities.h"

#include "openmc/eigenvalue.h"
#include "openmc/geometry.h"
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/output.h"
#include "openmc/plot.h"
#include "openmc/random_ray/flat_source_domain.h"
#include "openmc/random_ray/random_ray.h"
#include "openmc/simulation.h"
#include "openmc/source.h"
#include "openmc/tallies/filter.h"
#include "openmc/tallies/tally.h"
#include "openmc/tallies/tally_scoring.h"
#include "openmc/timer.h"

namespace openmc {

//==============================================================================
// Non-member functions
//==============================================================================

void openmc_run_random_ray()
{
  //////////////////////////////////////////////////////////
  // Run forward simulation
  //////////////////////////////////////////////////////////

  // Check if adjoint calculation is needed. If it is, we will run the forward
  // calculation first and then the adjoint calculation later.
  bool adjoint_needed = FlatSourceDomain::adjoint_;

  // Configure the domain for forward simulation
  FlatSourceDomain::adjoint_ = false;

  // If we're going to do an adjoint simulation afterwards, report that this is
  // the initial forward flux solve.
  if (adjoint_needed && mpi::master)
    header("FORWARD FLUX SOLVE", 3);

  // Initialize OpenMC general data structures
  openmc_simulation_init();

  // Validate that inputs meet requirements for random ray mode
  if (mpi::master)
    validate_random_ray_inputs();

  // Initialize Random Ray Simulation Object
  RandomRaySimulation sim;

  // Initialize fixed sources, if present
  sim.prepare_fixed_sources();

  // Begin main simulation timer
  simulation::time_total.start();

  // Execute random ray simulation
  sim.simulate();

  // End main simulation timer
  simulation::time_total.stop();

  // Normalize and save the final forward flux
  sim.domain()->normalize_final_quantities();

  // Finalize OpenMC
  openmc_simulation_finalize();

  // Reduce variables across MPI ranks
  sim.reduce_simulation_statistics();

  // Output all simulation results
  sim.output_simulation_results();

  // Extract flux and source for initial condition for time-dependent
  // simulation
  if (settings::is_initial_condition) {
    previous_k_eff = simulation::keff;
    sim.domain()->store_time_step_quantities(false);

    // We can reuse the maps and source regions found during the
    // initial condition simulation. This is actually required
    // to properly set the initial conditions.
    switch (RandomRay::source_shape_) {
    case RandomRaySourceShape::FLAT:
      source_domain = move(sim.domain_);
      break;
    case RandomRaySourceShape::LINEAR:
    case RandomRaySourceShape::LINEAR_XY:
      fatal_error("Time-dependent simulations do not currently suppot linear "
                  "source regions.");
      break;
    default:
      fatal_error("Unknown random ray source shape");
    }
  }

  //////////////////////////////////////////////////////////
  // Run adjoint simulation (if enabled)
  //////////////////////////////////////////////////////////
  if (!adjoint_needed) {
    return;
  }
  reset_timers();

  // Configure the domain for adjoint simulation
  FlatSourceDomain::adjoint_ = true;

  if (mpi::master)
    header("ADJOINT FLUX SOLVE", 3);

  // Initialize OpenMC general data structures
  openmc_simulation_init();

  sim.k_eff_ = 1.0;

  // Initialize adjoint fixed sources, if present
  sim.prepare_fixed_sources_adjoint();

  // Transpose scattering matrix
  sim.domain()->transpose_scattering_matrix();

  // Swap nu_sigma_f and chi
  sim.domain()->nu_sigma_f_.swap(sim.domain()->chi_);

  // Begin main simulation timer
  simulation::time_total.start();

  // Execute random ray simulation
  sim.simulate();

  // End main simulation timer
  simulation::time_total.stop();

  // Finalize OpenMC
  openmc_simulation_finalize();

  // Reduce variables across MPI ranks
  sim.reduce_simulation_statistics();

  // Output all simulation results
  sim.output_simulation_results();
}

//==============================================================================
// Time-dependent global variables
//==============================================================================
double previous_k_eff;
unique_ptr<FlatSourceDomain> source_domain;

void openmc_run_random_ray_time_dependent()
{
  warning("Time-dependent explicit void treatment has not yet been "
          "implemented. Use caution when interpreting results from models with "
          "voids, as they may contain large inaccuracies.");
  // Criticality solve to get initial condition
  settings::run_mode = RunMode::EIGENVALUE;
  openmc_run_random_ray();
  rename_statepoint_file(0);
  if (settings::output_tallies) {
    rename_tallies_file(0);
  }

  /////////////////////////////////
  // Settings for timestepping loop
  /////////////////////////////////

  reset_timers();
  simulation::time_initialize_td.start();
  RandomRaySimulation sim_td(false);
  sim_td.domain_ = move(source_domain);
  set_time_dependent_settings();
  simulation::time_initialize_td.stop();

  // Timestepping loop
  // TODO: Add support for time-dependent restart
  for (int i = 0; i < settings::n_timesteps; i++) {
    settings::current_timestep = i + 1;

    // Print simulation information
    if (mpi::master) {
      // Offset to resolve steady state
      std::string message =
        fmt::format("TIME DEPENDENT SOLVE {0}", settings::current_timestep);
      const char* msg = message.c_str();
      header(msg, 3);
    }

    if (settings::current_timestep > 1)
      reset_timers();

    // Initialize OpenMC general data structures
    openmc_simulation_init();

    simulation::time_initialize_td.start();

    sim_td.k_eff_ = previous_k_eff;
    sim_td.domain()->source_regions_.adjoint_reset();
    sim_td.domain()->propagate_final_quantities();
    sim_td.domain()->source_regions_.time_step_reset();

    // Compute RHS backward differences to be used later
    sim_td.domain()->compute_rhs_bd_quantities();

    // Update time dependent cross section based on the density
    sim_td.domain()->update_material_density(i);

    simulation::time_initialize_td.stop();

    // Begin main simulation timer
    simulation::time_total.start();

    // Execute random ray simulation
    sim_td.simulate();

    // End main simulation timer
    simulation::time_total.stop();

    // Finalize OpenMC
    openmc_simulation_finalize();

    // Reduce variables across MPI ranks
    sim_td.reduce_simulation_statistics();

    // Output all simulation results
    sim_td.output_simulation_results();

    // Rename statepoint and tallies file
    rename_statepoint_file(settings::current_timestep);
    if (settings::output_tallies) {
      rename_tallies_file(settings::current_timestep);
    }

    // Save the converged keff in previous_k_eff
    previous_k_eff = simulation::keff;

    // Normalize and store final quantities for next time step
    sim_td.domain()->normalize_final_quantities();
    sim_td.domain()->store_time_step_quantities();

    // Advance time
    simulation::current_time += settings::dt;
  }
}

void set_time_dependent_settings()
{
  // Reset flags
  settings::run_mode = RunMode::TIME_DEPENDENT;
  settings::is_initial_condition = false;

  simulation::current_time = settings::dt;
}

void rename_statepoint_file(int i)
{
  // Rename statepoint file
  std::string old_filename_ = fmt::format(
    "{0}statepoint.{1}.h5", settings::path_output, settings::n_batches);
  std::string new_filename_ =
    fmt::format("{0}openmc_td_simulation_{1}.h5", settings::path_output, i);

  const char* old_fname = old_filename_.c_str();
  const char* new_fname = new_filename_.c_str();
  std::rename(old_fname, new_fname);
}

void rename_tallies_file(int i)
{
  // Rename tallies file
  std::string old_filename_ =
    fmt::format("{0}tallies.out", settings::path_output);
  std::string new_filename_ =
    fmt::format("{0}tallies_{1}.out", settings::path_output, i);

  const char* old_fname = old_filename_.c_str();
  const char* new_fname = new_filename_.c_str();
  std::rename(old_fname, new_fname);
}

// Enforces restrictions on inputs in random ray mode.  While there are
// many features that don't make sense in random ray mode, and are therefore
// unsupported, we limit our testing/enforcement operations only to inputs
// that may cause erroneous/misleading output or crashes from the solver.
void validate_random_ray_inputs()
{
  // Validate tallies
  ///////////////////////////////////////////////////////////////////
  for (auto& tally : model::tallies) {

    // Validate score types
    for (auto score_bin : tally->scores_) {
      switch (score_bin) {
      case SCORE_FLUX:
      case SCORE_TOTAL:
      case SCORE_FISSION:
      case SCORE_NU_FISSION:
      case SCORE_EVENTS:
        break;

      case SCORE_PROMPT_NU_FISSION:
      case SCORE_DELAYED_NU_FISSION:
      case SCORE_PRECURSORS: {
        if (settings::run_mode == RunMode::TIME_DEPENDENT ||
            settings::is_initial_condition) {
          break;
        } else {
          fatal_error(
            "Invalid score specified in tallies.xml. Time-dependent "
            "random ray mode must be active to score prompt nu-fission, "
            "delayed nu-fission, and precursors.");
        }
      }
      default:
        fatal_error(
          "Invalid score specified. Only flux, total, fission, nu-fission, and "
          "event scores are supported in random ray mode. (prompt nu-fission, "
          "delayed nu-fission, and precursors are supported in time-dependent "
          "random ray mode).");
      }
    }

    // Validate filter types
    for (auto f : tally->filters()) {
      auto& filter = *model::tally_filters[f];

      switch (filter.type()) {
      case FilterType::CELL:
      case FilterType::CELL_INSTANCE:
      case FilterType::DISTRIBCELL:
      case FilterType::ENERGY:
      case FilterType::MATERIAL:
      case FilterType::MESH:
      case FilterType::UNIVERSE:
      case FilterType::PARTICLE:
        break;
      case FilterType::DELAYED_GROUP:
        if (settings::run_mode == RunMode::TIME_DEPENDENT ||
            settings::is_initial_condition) {
          break;
        } else {
          fatal_error(
            "Invalid score specified in tallies.xml. Time-dependent "
            "random ray mode must be active to score prompt nu-fission, "
            "delayed nu-fission, and precursors.");
        }
      default:
        fatal_error("Invalid filter specified. Only cell, cell_instance, "
                    "distribcell, energy, material, mesh, and universe filters "
                    "are supported in random ray mode (delayed_group is "
                    "supported in time-dependent mode).");
      }
    }
  }

  // Validate MGXS data
  ///////////////////////////////////////////////////////////////////
  for (auto& material : data::mg.macro_xs_) {
    if (!material.is_isotropic) {
      fatal_error("Anisotropic MGXS detected. Only isotropic XS data sets "
                  "supported in random ray mode.");
    }
    if (material.get_xsdata().size() > 1) {
      fatal_error("Non-isothermal MGXS detected. Only isothermal XS data sets "
                  "supported in random ray mode.");
    }
    for (int g = 0; g < data::mg.num_energy_groups_; g++) {
      if (material.exists_in_model) {
        // Temperature and angle indices, if using multiple temperature
        // data sets and/or anisotropic data sets.
        // TODO: Currently assumes we are only using single temp/single angle
        // data.
        const int t = 0;
        const int a = 0;
        double sigma_t =
          material.get_xs(MgxsType::TOTAL, g, NULL, NULL, NULL, t, a);
        if (sigma_t <= 0.0) {
          fatal_error("No zero or negative total macroscopic cross sections "
                      "allowed in random ray mode. If the intention is to make "
                      "a void material, use a cell fill of 'None' instead.");
        }
      }
    }
  }

  // Validate ray source
  ///////////////////////////////////////////////////////////////////

  // Check for independent source
  IndependentSource* is =
    dynamic_cast<IndependentSource*>(RandomRay::ray_source_.get());
  if (!is) {
    fatal_error("Invalid ray source definition. Ray source must provided and "
                "be of type IndependentSource.");
  }

  // Check for box source
  SpatialDistribution* space_dist = is->space();
  SpatialBox* sb = dynamic_cast<SpatialBox*>(space_dist);
  if (!sb) {
    fatal_error(
      "Invalid ray source definition -- only box sources are allowed.");
  }

  // Check that box source is not restricted to fissionable areas
  if (sb->only_fissionable()) {
    fatal_error(
      "Invalid ray source definition -- fissionable spatial distribution "
      "not allowed.");
  }

  // Check for isotropic source
  UnitSphereDistribution* angle_dist = is->angle();
  Isotropic* id = dynamic_cast<Isotropic*>(angle_dist);
  if (!id) {
    fatal_error("Invalid ray source definition -- only isotropic sources are "
                "allowed.");
  }

  // Validate external sources
  ///////////////////////////////////////////////////////////////////
  if (settings::run_mode == RunMode::FIXED_SOURCE) {
    if (model::external_sources.size() < 1) {
      fatal_error("Must provide a particle source (in addition to ray source) "
                  "in fixed source random ray mode.");
    }

    for (int i = 0; i < model::external_sources.size(); i++) {
      Source* s = model::external_sources[i].get();

      // Check for independent source
      IndependentSource* is = dynamic_cast<IndependentSource*>(s);

      if (!is) {
        fatal_error(
          "Only IndependentSource external source types are allowed in "
          "random ray mode");
      }

      // Check for isotropic source
      UnitSphereDistribution* angle_dist = is->angle();
      Isotropic* id = dynamic_cast<Isotropic*>(angle_dist);
      if (!id) {
        fatal_error(
          "Invalid source definition -- only isotropic external sources are "
          "allowed in random ray mode.");
      }

      // Validate that a domain ID was specified
      if (is->domain_ids().size() == 0) {
        fatal_error("Fixed sources must be specified by domain "
                    "id (cell, material, or universe) in random ray mode.");
      }

      // Check that a discrete energy distribution was used
      Distribution* d = is->energy();
      Discrete* dd = dynamic_cast<Discrete*>(d);
      if (!dd) {
        fatal_error(
          "Only discrete (multigroup) energy distributions are allowed for "
          "external sources in random ray mode.");
      }
    }
  }

  // Validate plotting files
  ///////////////////////////////////////////////////////////////////
  for (int p = 0; p < model::plots.size(); p++) {

    // Get handle to OpenMC plot object
    Plot* openmc_plot = dynamic_cast<Plot*>(model::plots[p].get());

    // Random ray plots only support voxel plots
    if (!openmc_plot) {
      warning(fmt::format(
        "Plot {} will not be used for end of simulation data plotting -- only "
        "voxel plotting is allowed in random ray mode.",
        p));
      continue;
    } else if (openmc_plot->type_ != Plot::PlotType::voxel) {
      warning(fmt::format(
        "Plot {} will not be used for end of simulation data plotting -- only "
        "voxel plotting is allowed in random ray mode.",
        p));
      continue;
    }
  }

  // Warn about slow MPI domain replication, if detected
  ///////////////////////////////////////////////////////////////////
#ifdef OPENMC_MPI
  if (mpi::n_procs > 1) {
    warning(
      "Domain replication in random ray is supported, but suffers from poor "
      "scaling of source all-reduce operations. Performance may severely "
      "degrade beyond just a few MPI ranks. Domain decomposition may be "
      "implemented in the future to provide efficient scaling.");
  }
#endif
}

void write_random_ray_hdf5(hid_t group)
{
  hid_t random_ray_group = create_group(group, "random_ray");
  switch (RandomRay::source_shape_) {
  case RandomRaySourceShape::FLAT:
    write_dataset(random_ray_group, "source_shape", "flat");
    break;
  case RandomRaySourceShape::LINEAR:
    write_dataset(random_ray_group, "source_shape", "linear");
    break;
  case RandomRaySourceShape::LINEAR_XY:
    write_dataset(random_ray_group, "source_shape", "linear xy");
    break;
  default:
    break;
  }

  switch (FlatSourceDomain::volume_estimator_) {
  case RandomRayVolumeEstimator::SIMULATION_AVERAGED:
    write_dataset(random_ray_group, "volume_estimator", "simulation averaged");
    break;
  case RandomRayVolumeEstimator::NAIVE:
    write_dataset(random_ray_group, "volume_estimator", "naive");
    break;
  case RandomRayVolumeEstimator::HYBRID:
    write_dataset(random_ray_group, "volume_estimator", "hybrid");
    break;
  default:
    break;
  }

  write_dataset(
    random_ray_group, "distance_active", RandomRay::distance_active_);
  write_dataset(
    random_ray_group, "distance_inactive", RandomRay::distance_inactive_);
  write_dataset(random_ray_group, "volume_normalized_flux_tallies",
    FlatSourceDomain::volume_normalized_flux_tallies_);
  write_dataset(random_ray_group, "adjoint_mode", FlatSourceDomain::adjoint_);

  write_dataset(random_ray_group, "avg_miss_rate", RandomRay::avg_miss_rate_);
  write_dataset(
    random_ray_group, "n_source_regions", RandomRay::n_source_regions_);
  write_dataset(random_ray_group, "n_external_source_regions",
    RandomRay::n_external_source_regions_);
  write_dataset(random_ray_group, "n_geometric_intersections",
    RandomRay::total_geometric_intersections_);
  int64_t n_integrations =
    RandomRay::total_geometric_intersections_ * data::mg.num_energy_groups_;
  write_dataset(random_ray_group, "n_integrations", n_integrations);

  if (settings::run_mode == RunMode::TIME_DEPENDENT) {
    write_dataset(random_ray_group, "bd_order", RandomRay::bd_order_);
    switch (RandomRay::time_method_) {
    case RandomRayTimeMethod::TI:
      write_dataset(random_ray_group, "time_method", "ti");
      break;
    case RandomRayTimeMethod::SDP:
      write_dataset(random_ray_group, "time_method", "sdp");
      break;
    default:
      break;
    }
  }
  close_group(random_ray_group);
}

//==============================================================================
// RandomRaySimulation implementation
//==============================================================================

RandomRaySimulation::RandomRaySimulation(bool generate_source_domain)
  : negroups_(data::mg.num_energy_groups_),
    ndgroups_(data::mg.num_delayed_groups_)
{
  // There are no source sites in random ray mode, so be sure to disable to
  // ensure we don't attempt to write source sites to statepoint
  settings::source_write = false;

  // Random ray mode does not have an inner loop over generations within a
  // batch, so set the current gen to 1
  simulation::current_gen = 1;

  if (generate_source_domain) {
    switch (RandomRay::source_shape_) {
    case RandomRaySourceShape::FLAT:
      domain_ = make_unique<FlatSourceDomain>();
      break;
    case RandomRaySourceShape::LINEAR:
    case RandomRaySourceShape::LINEAR_XY:
      domain_ = make_unique<LinearSourceDomain>();
      break;
    default:
      fatal_error("Unknown random ray source shape");
    }

    // Convert OpenMC native MGXS into a more efficient format
    // internal to the random ray solver
    domain_->flatten_xs();
  }
}

void RandomRaySimulation::prepare_fixed_sources()
{
  if (settings::run_mode == RunMode::FIXED_SOURCE) {
    // Transfer external source user inputs onto random ray source regions
    domain_->convert_external_sources();
    domain_->count_external_source_regions();
  }
}

void RandomRaySimulation::prepare_fixed_sources_adjoint()
{
  domain_->source_regions_.adjoint_reset();
  if (settings::run_mode == RunMode::FIXED_SOURCE) {
    domain_->set_adjoint_sources();
  }
}

void RandomRaySimulation::simulate()
{
  // Random ray power iteration loop
  while (simulation::current_batch < settings::n_batches) {

    // Initialize the current batch
    initialize_batch();
    initialize_generation();

    // Reset total starting particle weight used for normalizing tallies
    simulation::total_weight = 1.0;

    // TODO: add update source convenience function
    // domain_->compute_neutron_source()
    // Update source term (scattering + fission)
    domain_->update_neutron_source(k_eff_);
    if (settings::run_mode == RunMode::TIME_DEPENDENT) {
      domain_->update_neutron_source_td(k_eff_);
      if (RandomRay::time_method_ == RandomRayTimeMethod::SDP) {
        domain_->compute_neutron_source_time_derivative();
        domain_->compute_scalar_flux_time_derivative_2();
      }
    }

    // Reset scalar fluxes, iteration volume tallies, and region hit flags to
    // zero
    domain_->batch_reset();

    // Start timer for transport
    simulation::time_transport.start();

// Transport sweep over all random rays for the iteration
#pragma omp parallel for schedule(dynamic)                                     \
  reduction(+ : total_geometric_intersections_)
    for (int i = 0; i < simulation::work_per_rank; i++) {
      RandomRay ray(i, domain_.get());
      total_geometric_intersections_ +=
        ray.transport_history_based_single_ray();
    }

    simulation::time_transport.stop();

    // If using multiple MPI ranks, perform all reduce on all transport results
    domain_->all_reduce_replicated_source_regions();

    // Normalize scalar flux and update volumes
    domain_->normalize_scalar_flux_and_volumes(
      settings::n_particles * RandomRay::distance_active_);

    // Add source to scalar flux, compute number of FSR hits
    int64_t n_hits = domain_->add_source_to_scalar_flux(); 

    if (settings::run_mode == RunMode::EIGENVALUE ||
        settings::run_mode == RunMode::TIME_DEPENDENT) {
      // Compute random ray k-eff
      k_eff_ = domain_->compute_k_eff(k_eff_);

      // Store random ray k-eff into OpenMC's native k-eff variable
      global_tally_tracklength = k_eff_;
    }

    // Compute precursors
    domain_->compute_precursors(k_eff_);

    // Execute all tallying tasks, if the source is converged
    if (simulation::current_batch > settings::n_inactive) {

      // Add this iteration's estimates (flux, precursors, etc.) to final
      // accumulated estimate
      domain_->accumulate_iteration_quantities();

      if (mpi::master) {
        // Generate mapping between source regions and tallies
        if (!domain_->mapped_all_tallies_) {
          domain_->convert_source_regions_to_tallies();
        }

        // Use above mapping to contribute FSR flux data to appropriate tallies
        domain_->random_ray_tally();
      }
    }

    // Set phi_old = phi_new
    domain_->flux_swap();
    if (settings::run_mode == RunMode::TIME_DEPENDENT) {
      domain_->flux_td_swap();
      domain_->precursors_swap();
    }

    // Check for any obvious insabilities/nans/infs
    instability_check(n_hits, k_eff_, avg_miss_rate_);

    RandomRay::avg_miss_rate_ = avg_miss_rate_ / settings::n_batches;
    RandomRay::total_geometric_intersections_ = total_geometric_intersections_;
    RandomRay::n_external_source_regions_ = domain_->n_external_source_regions_;
    RandomRay::n_source_regions_ = domain_->n_source_regions_;

    // Finalize the current batch
    finalize_generation();
    finalize_batch();
  } // End random ray power iteration loop
}

void RandomRaySimulation::reduce_simulation_statistics()
{
  // Reduce number of intersections
#ifdef OPENMC_MPI
  if (mpi::n_procs > 1) {
    uint64_t total_geometric_intersections_reduced = 0;
    MPI_Reduce(&total_geometric_intersections_,
      &total_geometric_intersections_reduced, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
      mpi::intracomm);
    total_geometric_intersections_ = total_geometric_intersections_reduced;
  }
#endif
}

void RandomRaySimulation::output_simulation_results() const
{
  // Print random ray results
  if (mpi::master) {
    print_results_random_ray();
    if (model::plots.size() > 0) {
      domain_->output_to_vtk();
    }
  }
}

// Apply a few sanity checks to catch obvious cases of numerical instability.
// Instability typically only occurs if ray density is extremely low.
void RandomRaySimulation::instability_check(
  int64_t n_hits, double k_eff, double& avg_miss_rate) const
{
  double percent_missed = ((domain_->n_source_regions_ - n_hits) /
                            static_cast<double>(domain_->n_source_regions_)) *
                          100.0;
  avg_miss_rate += percent_missed;

  if (mpi::master) {
    if (percent_missed > 10.0) {
      warning(fmt::format(
        "Very high FSR miss rate detected ({:.3f}%). Instability may occur. "
        "Increase ray density by adding more rays and/or active distance.",
        percent_missed));
    } else if (percent_missed > 1.0) {
      warning(
        fmt::format("Elevated FSR miss rate detected ({:.3f}%). Increasing "
                    "ray density by adding more rays and/or active "
                    "distance may improve simulation efficiency.",
          percent_missed));
    }

    if (k_eff > 10.0 || k_eff < 0.01 || !(std::isfinite(k_eff))) {
      fatal_error("Instability detected");
    }
  }
}

// Print random ray simulation results
void RandomRaySimulation::print_results_random_ray() const
{
  using namespace simulation;

  if (settings::verbosity >= 6) {
    double total_integrations =
      RandomRay::total_geometric_intersections_ * negroups_;
    double time_per_integration =
      simulation::time_transport.elapsed() / total_integrations;
    double misc_time = time_total.elapsed() - time_update_src.elapsed() -
                       time_transport.elapsed() - time_tallies.elapsed() -
                       time_bank_sendrecv.elapsed();
    if (settings::run_mode == RunMode::TIME_DEPENDENT) {
      misc_time -=
        time_initialize_td.elapsed() + time_update_bd_vectors_td.elapsed();
    }
    header("Simulation Statistics", 4);
    fmt::print(
      " Total Iterations                  = {}\n", settings::n_batches);
    fmt::print(" Flat Source Regions (FSRs)        = {}\n",
      RandomRay::n_source_regions_);
    fmt::print(" FSRs Containing External Sources  = {}\n",
      RandomRay::n_external_source_regions_);
    fmt::print(" Total Geometric Intersections     = {:.4e}\n",
      static_cast<double>(RandomRay::total_geometric_intersections_));
    fmt::print("   Avg per Iteration               = {:.4e}\n",
      static_cast<double>(RandomRay::total_geometric_intersections_) /
        settings::n_batches);
    fmt::print("   Avg per Iteration per FSR       = {:.2f}\n",
      static_cast<double>(RandomRay::total_geometric_intersections_) /
        static_cast<double>(settings::n_batches) /
        RandomRay::n_source_regions_);
    fmt::print(" Avg FSR Miss Rate per Iteration   = {:.4f}%\n",
      RandomRay::avg_miss_rate_);
    fmt::print(" Energy Groups                     = {}\n", negroups_);
    if (settings::run_mode == RunMode::TIME_DEPENDENT ||
        settings::is_initial_condition)
      fmt::print(" Delay Groups                      = {}\n", ndgroups_);
    fmt::print(
      " Total Integrations                = {:.4e}\n", total_integrations);
    fmt::print("   Avg per Iteration               = {:.4e}\n",
      total_integrations / settings::n_batches);

    std::string estimator;
    switch (domain_->volume_estimator_) {
    case RandomRayVolumeEstimator::SIMULATION_AVERAGED:
      estimator = "Simulation Averaged";
      break;
    case RandomRayVolumeEstimator::NAIVE:
      estimator = "Naive";
      break;
    case RandomRayVolumeEstimator::HYBRID:
      estimator = "Hybrid";
      break;
    default:
      fatal_error("Invalid volume estimator type");
    }
    fmt::print(" Volume Estimator Type             = {}\n", estimator);

    std::string adjoint_true = (FlatSourceDomain::adjoint_) ? "ON" : "OFF";
    fmt::print(" Adjoint Flux Mode                 = {}\n", adjoint_true);

    if (settings::run_mode == RunMode::TIME_DEPENDENT) {
      std::string time_method =
        (RandomRay::time_method_ == RandomRayTimeMethod::TI) ? "TI" : "SDP";
      fmt::print(" Time Method                       = {}\n", time_method);
      fmt::print(
        " Backwards Difference Order        = {}\n", RandomRay::bd_order_);
    }

    header("Timing Statistics", 4);
    show_time("Total time for initialization", time_initialize.elapsed());
    show_time("Reading cross sections", time_read_xs.elapsed(), 1);
    show_time("Total simulation time", time_total.elapsed());
    show_time("Transport sweep only", time_transport.elapsed(), 1);
    show_time("Source update only", time_update_src.elapsed(), 1);
    if (settings::run_mode == RunMode::TIME_DEPENDENT) {
      show_time(
        "Time-dependent source update only", time_update_src_td.elapsed(), 1);
      misc_time -= time_update_src_td.elapsed();
    }
    if (settings::is_initial_condition ||
        settings::run_mode == RunMode::TIME_DEPENDENT) {
      show_time(
        "Precursor computation only", time_compute_precursors.elapsed(), 1);
      misc_time -= time_compute_precursors.elapsed();
    }

    show_time("Tally conversion only", time_tallies.elapsed(), 1);
    show_time("MPI source reductions only", time_bank_sendrecv.elapsed(), 1);
    show_time("Other iteration routines", misc_time, 1);
    if (settings::run_mode == RunMode::EIGENVALUE ||
        settings::run_mode == RunMode::TIME_DEPENDENT)
      show_time("Time in inactive batches", time_inactive.elapsed());
    show_time("Time in active batches", time_active.elapsed());
    show_time("Time writing statepoints", time_statepoint.elapsed());
    show_time("Total time for finalization", time_finalize.elapsed());
    show_time("Time per integration", time_per_integration);
  }

  if (settings::verbosity >= 4 && settings::run_mode == RunMode::EIGENVALUE) {
    header("Results", 4);
    fmt::print(" k-effective                       = {:.5f} +/- {:.5f}\n",
      simulation::keff, simulation::keff_std);
  }
}

//------------------------------------------------------------------------------
// Time Dependent Methods

} // namespace openmc
