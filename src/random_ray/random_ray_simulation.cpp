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

  // Declare forward flux so that it can be saved for later adjoint simulation
  vector<double> forward_flux;
  {
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
    sim.domain()->serialize_final_fluxes(forward_flux);

    double normalization_factor =
      1.0 / (settings::n_batches - settings::n_inactive);
    double source_normalization_factor =
      sim.domain()->compute_fixed_source_normalization_factor() *
      normalization_factor;

    normalize_serialized_vector(forward_flux, source_normalization_factor);

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
      previous_scalar_flux = forward_flux;

      sim.domain()->serialize_final_precursors(previous_precursors);
      normalize_serialized_vector(previous_precursors, normalization_factor);
      if (RandomRay::time_mode_ == RandomRayTimeMode::SDP) {
        sim.domain()->serialize_final_sources(previous_source);
        normalize_serialized_vector(previous_source, normalization_factor);
      }
      if (RandomRay::precursor_mode_ == RandomRayPrecursorMode::ANALYTIC) {
        sim.domain()->serialize_final_delayed_fission_source(
          previous_delayed_fission_source);
        normalize_serialized_vector(
          previous_delayed_fission_source, normalization_factor);
      }
    }
  }

  //////////////////////////////////////////////////////////
  // Run adjoint simulation (if enabled)
  //////////////////////////////////////////////////////////

  if (adjoint_needed) {
    reset_timers();

    // Configure the domain for adjoint simulation
    FlatSourceDomain::adjoint_ = true;

    if (mpi::master)
      header("ADJOINT FLUX SOLVE", 3);

    // Initialize OpenMC general data structures
    openmc_simulation_init();

    // Initialize Random Ray Simulation Object
    RandomRaySimulation adjoint_sim;

    // Initialize adjoint fixed sources, if present
    adjoint_sim.prepare_fixed_sources_adjoint(forward_flux);

    // Transpose scattering matrix
    adjoint_sim.domain()->transpose_scattering_matrix();

    // Swap nu_sigma_f and chi
    adjoint_sim.domain()->nu_sigma_f_.swap(adjoint_sim.domain()->chi_);

    // Begin main simulation timer
    simulation::time_total.start();

    // Execute random ray simulation
    adjoint_sim.simulate();

    // End main simulation timer
    simulation::time_total.stop();

    // Finalize OpenMC
    openmc_simulation_finalize();

    // Reduce variables across MPI ranks
    adjoint_sim.reduce_simulation_statistics();

    // Output all simulation results
    adjoint_sim.output_simulation_results();
  }
}

//==============================================================================
// Time-dependent global variables
//==============================================================================
// 2D time solution arrays
vector<double> scalar_flux_bd;
vector<double> source_bd;
vector<double> precursors_bd;
vector<double> delayed_fission_source_bd;

// 1D RHS BD arrays
vector<double> scalar_flux_rhs_bd;

vector<double> source_rhs_bd;
vector<double> scalar_flux_rhs_bd_2;

vector<double> precursors_rhs_bd;

vector<double> precursors_im1;
vector<double> delayed_fission_source_im1;
vector<double> delayed_fission_source_im2;

double previous_k_eff;
// TODO: remove?
vector<double> previous_scalar_flux;
vector<double> previous_scalar_flux_td;
vector<double> previous_precursors;
vector<double> previous_source;
vector<double> previous_delayed_fission_source;

void initialize_bd_vector(int64_t vector_size, int n_timesteps,
  vector<double>& bd_vector, vector<double>& vector)
{
  bd_vector.assign(vector_size * n_timesteps, 0.0);
#pragma omp parallel for
  for (int t = 0; t < n_timesteps - 1; t++) {
    for (int i = 0; i < vector_size; i++)
      bd_vector[t * vector_size + i] = vector[i];
  } 
}

void compute_rhs_backward_difference(int64_t vector_size,
  int bd_order, vector<double>& bd_vector,
  vector<double>& rhs_bd_vector, int derivative_order)
{
  rhs_bd_vector.assign(vector_size, 0.0);
#pragma omp parallel for
  for (int i = 0; i < vector_size; i++)
    rhs_bd_vector[i] = rhs_backwards_difference(bd_vector, vector_size, i, bd_order, settings::dt, derivative_order);
}

void increment_bd_vector(int64_t vector_size, vector<double>* bd_vector)
{
  vector<double> vector_blank;
  vector_blank.assign(vector_size, 0.0);
  update_bd_vector(bd_vector, vector_blank, true);
}

void get_bd_vector_slice(int64_t vector_size, vector<double>& storage_vector,
  vector<double>& bd_vector, int neg_timestep_index)
{
  storage_vector.assign(vector_size, 0.0);
#pragma omp parallel for
  for (int i = 0; i < vector_size; i++)
    storage_vector[i] = bd_vector[vector_size * neg_timestep_index + i];
}

void normalize_serialized_vector(
  vector<double>& vector, double normalization_factor)
{
#pragma omp parallel for
  for (uint64_t i = 0; i < vector.size(); i++)
    vector[i] *= normalization_factor;
}

void openmc_run_random_ray_time_dependent()
{
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

  simulation::time_initialize_td.start();
  reset_timers();

  int64_t n_source_elements = previous_scalar_flux.size();
  int64_t n_source_regions = n_source_elements / data::mg.num_energy_groups_;
  int64_t n_delay_elements = n_source_regions * data::mg.num_delayed_groups_;

  initialize_bd_vector(n_source_elements, RandomRaySimulation::bd_order_ + 2,
    scalar_flux_bd, previous_scalar_flux);
  initialize_bd_vector(n_delay_elements, RandomRaySimulation::bd_order_ + 1,
    precursors_bd, previous_precursors);
  if (RandomRay::time_mode_ == RandomRayTimeMode::SDP)
    initialize_bd_vector(n_source_elements, RandomRaySimulation::bd_order_ + 1,
      source_bd, previous_source);

  if (RandomRay::precursor_mode_ == RandomRayPrecursorMode::ANALYTIC)
    initialize_bd_vector(n_delay_elements, 3, delayed_fission_source_bd,
      previous_delayed_fission_source);

  set_time_dependent_settings();
  simulation::time_initialize_td.stop();

  // Timestepping loop
  for (int i = 0; i < settings::n_timesteps; i++) {
    settings::current_timestep = i + 1;

    // Print simulation information
    if (mpi::master) {
      // Offset to resolve steady state
      std::string message = fmt::format("TIME DEPENDENT SOLVE {0}", i);
      const char* msg = message.c_str();
      header(msg, 3);
    }

    if (settings::current_timestep > 1)
    reset_timers();

    // Initialize OpenMC general data structures
    openmc_simulation_init();

    RandomRaySimulation sim_td;
    sim_td.domain()->bd_order_ = RandomRaySimulation::bd_order_;
    sim_td.k_eff_ = previous_k_eff;
    sim_td.domain()->set_initial_condition(
      previous_scalar_flux, scalar_flux_bd, previous_precursors);

    // Increment BD vectors to a zero-valued solution to be filled in.
    sim_td.increment_bd_vectors(n_source_elements, n_delay_elements);

    // Compute RHS backward differences to be used later
    sim_td.compute_rhs_bd_vectors(n_source_elements, n_delay_elements);
    sim_td.store_rhs_bd_vectors();

    // Update time dependent cross section based on the density
    sim_td.domain()->update_material_density(i);

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
    rename_statepoint_file(i + 1);
    if (settings::output_tallies) {
      rename_tallies_file(i + 1);
    }

    // Save the converged keff in previous_k_eff
    previous_k_eff = simulation::keff;

    // Serialize quantities
    sim_td.normalize_and_store_quantities();
  }
}

void set_time_dependent_settings()
{
  // Reset flags
  settings::run_mode = RunMode::TIME_DEPENDENT;
  settings::is_initial_condition = false;

  // Set batches
  settings::n_batches = settings::n_batches_td;
  settings::n_inactive = settings::n_inactive_td;
  settings::n_max_batches = settings::n_batches_td;

  // Reset k_generation and entropy
  int m = settings::n_max_batches * settings::gen_per_batch;
  simulation::k_generation.clear();
  simulation::k_generation.reserve(m);
  simulation::entropy.clear();
  simulation::entropy.reserve(m);

  // Reset statepoint_batch for statepoint writing
  settings::statepoint_batch.clear();
  settings::statepoint_batch.insert(settings::n_batches);
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

//==============================================================================
// RandomRaySimulation implementation
//==============================================================================

// Static variable declaration
int RandomRaySimulation::bd_order_ {1};

RandomRaySimulation::RandomRaySimulation()
  : negroups_(data::mg.num_energy_groups_),
    ndgroups_(data::mg.num_delayed_groups_)
{
  // There are no source sites in random ray mode, so be sure to disable to
  // ensure we don't attempt to write source sites to statepoint
  settings::source_write = false;

  // Random ray mode does not have an inner loop over generations within a
  // batch, so set the current gen to 1
  simulation::current_gen = 1;

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

void RandomRaySimulation::prepare_fixed_sources()
{
  if (settings::run_mode == RunMode::FIXED_SOURCE) {
    // Transfer external source user inputs onto random ray source regions
    domain_->convert_external_sources();
    domain_->count_external_source_regions();
  }
}

void RandomRaySimulation::prepare_fixed_sources_adjoint(
  vector<double>& forward_flux)
{
  if (settings::run_mode == RunMode::FIXED_SOURCE) {
    domain_->set_adjoint_sources(forward_flux);
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
      if (RandomRay::time_mode_ == RandomRayTimeMode::SDP) {
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

    // Determine if the source is converged
    bool converged;
    if (simulation::current_batch >= settings::convergence_window_size) {
      compute_and_store_batch_fission_source();
      converged = compare_window_averaged_rms_error();
    } else {
      compute_and_store_batch_fission_source(false);
      converged = false;
    }

    // Execute all tallying tasks, if this is an active batch
    //    if (simulation::current_batch > settings::n_inactive) {
    if (!converged) {
      // TODO: increment current batch?
      // TODO: increment total batches?
      settings::statepoint_batch.clear();
      settings::statepoint_batch.insert(settings::n_batches);
    } else {

      // TODO: add helper function
      // domain_->accumulation_iteration_quantities()
      // Add this iteration's scalar flux estimate to final accumulated estimate
      domain_->accumulate_iteration_flux();
      if (settings::run_mode == RunMode::TIME_DEPENDENT ||
          settings::is_initial_condition)
        domain_->accumulate_iteration_precursors();
      if (settings::run_mode == RunMode::TIME_DEPENDENT)
        domain_->accumulate_iteration_flux_td();

      if (RandomRay::time_mode_ == RandomRayTimeMode::SDP) {
        if (settings::run_mode == RunMode::TIME_DEPENDENT) {
          domain_->accumulate_iteration_source_td();
        } else if (settings::is_initial_condition) {
          domain_->accumulate_iteration_source();
        }
      }

      if (RandomRay::precursor_mode_ == RandomRayPrecursorMode::ANALYTIC) {
        domain_->accumulate_iteration_delayed_fission_source();
      }

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
    print_results_random_ray(total_geometric_intersections_,
      avg_miss_rate_ / settings::n_batches, negroups_,
      domain_->n_source_regions_, domain_->n_external_source_regions_);
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
void RandomRaySimulation::print_results_random_ray(
  uint64_t total_geometric_intersections, double avg_miss_rate, int negroups,
  int64_t n_source_regions, int64_t n_external_source_regions) const
{
  using namespace simulation;

  if (settings::verbosity >= 6) {
    double total_integrations = total_geometric_intersections * negroups;
    double time_per_integration =
      simulation::time_transport.elapsed() / total_integrations;
    double misc_time = time_total.elapsed() - time_update_src.elapsed() -
                       time_transport.elapsed() - time_tallies.elapsed() -
                       time_bank_sendrecv.elapsed();
    header("Simulation Statistics", 4);
    fmt::print(
      " Total Iterations                  = {}\n", settings::n_batches);
    fmt::print(" Flat Source Regions (FSRs)        = {}\n", n_source_regions);
    fmt::print(
      " FSRs Containing External Sources  = {}\n", n_external_source_regions);
    fmt::print(" Total Geometric Intersections     = {:.4e}\n",
      static_cast<double>(total_geometric_intersections));
    fmt::print("   Avg per Iteration               = {:.4e}\n",
      static_cast<double>(total_geometric_intersections) / settings::n_batches);
    fmt::print("   Avg per Iteration per FSR       = {:.2f}\n",
      static_cast<double>(total_geometric_intersections) /
        static_cast<double>(settings::n_batches) / n_source_regions);
    fmt::print(" Avg FSR Miss Rate per Iteration   = {:.4f}%\n", avg_miss_rate);
    fmt::print(" Energy Groups                     = {}\n", negroups);
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
    if (RandomRay::time_mode_ == RandomRayTimeMode::SDP) {
      show_time("Source time derivaitve computation only",
        time_compute_neutron_source_time_derivative.elapsed(), 1);
      misc_time -= time_compute_neutron_source_time_derivative.elapsed();
      show_time("Scalar flux time derivative computation only",
        time_compute_scalar_time_derivative_2.elapsed(), 1);
      misc_time -= time_compute_scalar_time_derivative_2.elapsed();
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

void RandomRaySimulation::compute_and_store_batch_fission_source(
  bool shift_window)
{
  // Compute the window-averaged RMS error
  for (int64_t sr = 0; sr < domain_->n_source_regions(); sr++) {
    int material = source_regions_.material(sr);
    double F_sr = 0.0;
#pragma omp parallel for
    for (int g = 0; g < negroups_; g++) {
      double sigma_t;
      double flux;
      if (settings::is_initial_condition) {
        sigma_f = domain_->sigma_f_[material * negroups_ + g];
        flux = domain_->source_regions_.scalar_flux_old(sr, g);
      } else {
        sigma_f = domain_->sigma_f_td_[material * negroups_ + g];
        flux = domain_->source_regions_.scalar_flux_td_old(sr, g);
      }
      F_sr += sigma_t * flux;
    }
    domain_->batchwise_fission_source(sr).push_front(F_sr);
    // Remove the first n_source_regions_ * negroups_ elements
    // if we move the window
    if (shift_window)
      domain_->batchwise_fission_source(sr).pop_back();
  }
}

void RandomRaySimulation::compare_window_averaged_rms_error()
{
  // Compute the window-averaged RMS error
  double W_fissile = 0.0;
  if (!all_fissile_regions_found) {
#pragma omp parallel for
    for (int64_t sr = 0; sr < domain_->n_source_regions_; sr++) {
      int material = source_regions_.material(sr);
      // TODO: make it so we only have to look for all fissile regions once
      if (domain_->sigma_f_[material * negroups_] != 0.0)
        W_fissile += 1;
      else
        continue;
    }
  }
  int half_window = int(0.5 * settings::convergence_window_size);
  double rms = 0.0;
#pragma omp parallel for
  for (int64_t sr = 0; sr < domain_->n_source_regions_; sr++) {
    int material = source_regions_.material(sr);
    double F_sr_new = 0.0;
    double F_sr_old = 0.0;
    for (int b = 0; b < half_window; b++)
      F_sr_new += domain_->source_regions_.batchwise_fission_source(sr)[b];
    for (int b = half_window; b < B_w; b++)
      F_sr_old += domain_->source_regions_.batchwise_fission_source(sr)[b];
  }
    F_sr_new /= half_window;
    F_sr_old /= half_window;
    rms += pow((F_sr_new - F_sr_old) / F_sr_new, 2);
  }
  rms /= W_fissile;
  rms = sqrt(rms);

  if (rms <= settings::source_convergence_threshold)
    return true;
  else
    return false;
}

// TODO: Remove the zero-valued solution from the BD vectors
// TODO: Perform bd_vector shifting inside the source region. This would
void RandomRaySimulation::increment_bd_vectors(
  int64_t n_source_elements, int64_t n_delay_elements)
{
  simulation::time_update_bd_vectors_td.start();
  increment_bd_vector(n_source_elements, &scalar_flux_bd);
  increment_bd_vector(n_delay_elements, &precursors_bd);
  if (RandomRay::time_mode_ == RandomRayTimeMode::SDP)
    increment_bd_vector(n_source_elements, &source_bd);
  if (RandomRay::precursor_mode_ == RandomRayPrecursorMode::ANALYTIC)
    increment_bd_vector(n_delay_elements, &delayed_fission_source_bd);
  simulation::time_update_bd_vectors_td.stop();
}

// TODO: do this inside of a source region
void RandomRaySimulation::compute_rhs_bd_vectors(
  int64_t n_source_elements, int64_t n_delay_elements)
{
  compute_rhs_backward_difference(n_source_elements,
    RandomRaySimulation::bd_order_, scalar_flux_bd, scalar_flux_rhs_bd, 1);

  if (RandomRay::time_mode_ == RandomRayTimeMode::SDP) {
    simulation::time_compute_neutron_source_time_derivative.start();
    compute_rhs_backward_difference(n_source_elements,
      RandomRaySimulation::bd_order_, source_bd, source_rhs_bd, 1);
    simulation::time_compute_neutron_source_time_derivative.stop();

    simulation::time_compute_scalar_time_derivative_2.start();
    compute_rhs_backward_difference(n_source_elements,
      RandomRaySimulation::bd_order_, scalar_flux_bd, scalar_flux_rhs_bd_2, 2);
    simulation::time_compute_scalar_time_derivative_2.stop();
  }
  if (RandomRay::precursor_mode_ == RandomRayPrecursorMode::ANALYTIC) {
    get_bd_vector_slice(n_delay_elements, precursors_im1, precursors_bd, 1);
    get_bd_vector_slice(
      n_delay_elements, delayed_fission_source_im1, precursors_bd, 1);
    get_bd_vector_slice(
      n_delay_elements, delayed_fission_source_im2, precursors_bd, 2);
  } else {
    compute_rhs_backward_difference(n_delay_elements,
      RandomRaySimulation::bd_order_, precursors_bd, precursors_rhs_bd, 1);
  }
}
void RandomRaySimulation::store_rhs_bd_vectors()
{
#pragma omp for
  for (int64_t sr = 0; sr < domain_->n_source_regions_; sr++) {
    for (int g = 0; g < negroups_; g++) {
      domain_->source_regions_.scalar_flux_rhs_bd(sr, g) =
        scalar_flux_rhs_bd[sr * negroups_ + g];
      if (RandomRay::time_mode_ == RandomRayTimeMode::SDP) {
        domain_->source_regions_.source_rhs_bd(sr, g) =
          source_rhs_bd[sr * negroups_ + g];
        domain_->source_regions_.scalar_flux_rhs_bd_2(sr, g) =
          scalar_flux_rhs_bd_2[sr * negroups_ + g];
      }
    }
    for (int dg = 0; dg < ndgroups_; dg++) {
      if (RandomRay::precursor_mode_ == RandomRayPrecursorMode::ANALYTIC) {
        domain_->source_regions_.precursors_im1(sr, dg) =
          precursors_im1[sr * ndgroups_ + dg];
        domain_->source_regions_.delayed_fission_source_im1(sr, dg) =
          delayed_fission_source_im1[sr * ndgroups_ + dg];
        domain_->source_regions_.delayed_fission_source_im2(sr, dg) =
          delayed_fission_source_im2[sr * ndgroups_ + dg];
      } else {
        domain_->source_regions_.precursors_rhs_bd(sr, dg) =
          precursors_rhs_bd[sr * ndgroups_ + dg];
      }
    }
  }
}

// TODO: Perform normalization inside of the source regions
void RandomRaySimulation::normalize_and_store_quantities()
{
  // Compute normalization factors
  double normalization_factor =
    1.0 / (settings::n_batches - settings::n_inactive);
  double source_normalization_factor =
    domain_->compute_fixed_source_normalization_factor() * normalization_factor;

  domain_->serialize_final_fluxes(previous_scalar_flux);
  normalize_serialized_vector(
    previous_scalar_flux, source_normalization_factor);

  domain_->serialize_final_td_fluxes(previous_scalar_flux_td);
  normalize_serialized_vector(
    previous_scalar_flux_td, source_normalization_factor);
  update_bd_vector(&scalar_flux_bd, previous_scalar_flux_td, false);

  domain_->serialize_final_precursors(previous_precursors);
  normalize_serialized_vector(previous_precursors, normalization_factor);
  update_bd_vector(&precursors_bd, previous_precursors, false);

  if (RandomRay::time_mode_ == RandomRayTimeMode::SDP) {
    domain_->serialize_final_td_sources(previous_source);
    normalize_serialized_vector(previous_source, normalization_factor);
    update_bd_vector(&source_bd, previous_source, false);
  }
  if (RandomRay::precursor_mode_ == RandomRayPrecursorMode::ANALYTIC) {
    domain_->serialize_final_delayed_fission_source(
      previous_delayed_fission_source);
    normalize_serialized_vector(
      previous_delayed_fission_source, normalization_factor);
    update_bd_vector(
      &delayed_fission_source_bd, previous_delayed_fission_source, false);
  }

} // namespace openmc
