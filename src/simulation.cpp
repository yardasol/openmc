#include "openmc/simulation.h"

#include "openmc/bank.h"
#include "openmc/capi.h"
#include "openmc/collision_track.h"
#include "openmc/container_util.h"
#include "openmc/eigenvalue.h"
#include "openmc/error.h"
#include "openmc/event.h"
#include "openmc/geometry_aux.h"
#include "openmc/ifp.h"
#include "openmc/material.h"
#include "openmc/math_functions.h"
#include "openmc/message_passing.h"
#include "openmc/nuclide.h"
#include "openmc/output.h"
#include "openmc/particle.h"
#include "openmc/photon.h"
#include "openmc/physics_common.h"
#include "openmc/random_lcg.h"
#include "openmc/reaction.h"
#include "openmc/settings.h"
#include "openmc/source.h"
#include "openmc/state_point.h"
#include "openmc/tallies/derivative.h"
#include "openmc/tallies/filter.h"
#include "openmc/tallies/tally.h"
#include "openmc/tallies/trigger.h"
#include "openmc/timer.h"
#include "openmc/track_output.h"
#include "openmc/weight_windows.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include "openmc/tensor.h"

#ifdef OPENMC_MPI
#include <mpi.h>
#endif

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <string>

//==============================================================================
// C API functions
//==============================================================================

// OPENMC_RUN encompasses all the main logic where iterations are performed
// over the batches, generations, and histories in a fixed source or
// k-eigenvalue calculation.

int openmc_run()
{
  openmc::simulation::time_total.start();
  openmc_simulation_init();

  // Ensure that a batch isn't executed in the case that the maximum number of
  // batches has already been run in a restart statepoint file
  int status = 0;
  if (openmc::simulation::current_batch >= openmc::settings::n_max_batches) {
    status = openmc::STATUS_EXIT_MAX_BATCH;
  }

  int err = 0;
  while (status == 0 && err == 0) {
    err = openmc_next_batch(&status);
  }

  openmc_simulation_finalize();
  openmc::simulation::time_total.stop();

  // Kinetic batches for eigenvalue simulations
  if (openmc::settings::kinetic_simulation &&
      openmc::settings::run_mode == openmc::RunMode::EIGENVALUE) {

    // No longer in initial condition
    openmc::simulation::is_initial_condition = false;

    // Store steady state k-eff
    openmc::simulation::initial_keff = openmc::simulation::keff;
    // Store various eigenvalues quatnties, as well as overall_generation and
    // n_realizations. These are needed to properly compute k_eff if using
    // decorrelating generations for TDMC simulations
    if (openmc::settings::n_decorrelate_generations > 0) {
      using namespace openmc;
      using namespace simulation;

      // Used for consistent eigenvalue calculations and particle IDs + seeds
      initial_overall_generation =
        overall_generation() - 1; // off by one correction
      // Used for consistent eigenvalue calculations
      initial_n_realizations = n_realizations;
      initial_gen_per_batch = settings::gen_per_batch;
      store_initial_k_eigenvalue_quantities();
    }

    // Copy the criticality source bank. We will want to run
    // some criticality generations to decorrelate the batches.
    openmc::simulation::initial_source_bank = openmc::simulation::source_bank;
    if (openmc::settings::forced_decay)
      openmc::simulation::initial_precursor_source_bank =
        openmc::simulation::precursor_source_bank;

    // No need to reinitialize every data structure. We can just reset global
    // vars and set the initialized flag back to true
    openmc_reset_global_variables();
    openmc_hard_reset();
    openmc::simulation::initialized = true;
    openmc::simulation::current_batch = 0;
    openmc::settings::n_inactive = 0;
    if (openmc::settings::n_decorrelate_generations > 0) {
      openmc::set_initial_k_eigenvalue_quantities();
    }
    status = 0;
    openmc_display_header();

    // Run the kinetic simulation
    openmc::simulation::time_total.start();
    while (status == 0 && err == 0) {
      err = openmc_next_batch(&status);
    }
    openmc_simulation_finalize();
    openmc::simulation::time_total.stop();
  }
  return err;
}

int openmc_simulation_init()
{
  using namespace openmc;

  // Skip if simulation has already been initialized
  if (simulation::initialized)
    return 0;

  // Initialize nuclear data (energy limits, log grid)
  if (settings::run_CE) {
    initialize_data();
  }

  // Determine how much work each process should do
  calculate_work(
    settings::n_particles, simulation::work_per_rank, simulation::work_index);

  if (settings::kinetic_simulation) {
    // Precursor + neutron work per rank
    simulation::combined_work_per_rank = simulation::work_per_rank;

    // Create precursor work
    if (settings::forced_decay) {
      calculate_work(settings::n_precursor_particles,
        simulation::precursor_work_per_rank, simulation::precursor_work_index);
      simulation::combined_work_per_rank += simulation::precursor_work_per_rank;
    }

    // Create precursor + neutron work_index
    simulation::combined_work_index.resize(mpi::n_procs + 1);
    for (int i = 0; i < mpi::n_procs; ++i) {
      simulation::combined_work_index[i + 1] = simulation::work_index[i + 1];
      if (settings::forced_decay)
        simulation::combined_work_index[i + 1] +=
          simulation::precursor_work_index[i + 1] -
          simulation::precursor_work_index[i];
    }
  }

  // Allocate source, fission and surface source banks.
  allocate_banks();

  // Create track file if needed
  if (!settings::track_identifiers.empty() || settings::write_all_tracks) {
    open_track_file();
  }

  // If doing an event-based simulation, intialize the particle buffer
  // and event queues
  if (settings::event_based) {
    int64_t event_buffer_length =
      std::min(simulation::work_per_rank, settings::max_particles_in_flight);
    init_event_queues(event_buffer_length);
  }

  // Allocate tally results arrays if they're not allocated yet
  for (auto& t : model::tallies) {
    t->set_strides();
    t->init_results();
  }

  // Set up material nuclide index mapping
  for (auto& mat : model::materials) {
    mat->init_nuclide_index();
  }

  // Reset global variables and tallies -- this is done before loading state
  // point (as that will potentially populate k_generation and entropy)
  openmc_reset_global_variables();
  openmc_reset();

  // If this is a restart run, load the state point data and binary source
  // file
  if (settings::restart_run) {
    load_state_point();
    write_message("Resuming simulation...", 6);
  } else {
    // Only initialize primary source bank for eigenvalue simulations
    if (settings::run_mode == RunMode::EIGENVALUE &&
        settings::solver_type == SolverType::MONTE_CARLO) {
      initialize_source();
    }
  }

  // Display header
  openmc_display_header();

  // load weight windows from file
  if (!settings::weight_windows_file.empty()) {
    openmc_weight_windows_import(settings::weight_windows_file.c_str());
  }

  // Set flag indicating initialization is done
  simulation::initialized = true;
  return 0;
}

void openmc_reset_global_variables()
{
  using namespace openmc;

  simulation::current_batch = 0;
  simulation::ct_current_file = 1;
  simulation::ssw_current_file = 1;
  simulation::k_generation.clear();
  simulation::entropy.clear();
  reset_source_rejection_counters();
}

void openmc_display_header()
{
  using namespace openmc;

  if (mpi::master) {
    if (settings::kinetic_simulation) {
      if (simulation::is_initial_condition) {
        if (simulation::source_correction)
          header("KINETIC SIMULATION INITIAL CONDITION (SOURCE CORRECTION)", 3);
        else
          header("KINETIC SIMULATION INITIAL CONDITION", 3);
      } else {
        std::string message = "KINETIC SIMULATION";
        if (settings::solver_type == SolverType::RANDOM_RAY) {
          message = fmt::format(
            "{0} TIME STEP {1}", message, simulation::current_timestep);
        }
        const char* msg = message.c_str();
        header(msg, 3);
      }
    }
    if (settings::run_mode == RunMode::FIXED_SOURCE) {
      if (settings::solver_type == SolverType::MONTE_CARLO) {
        header("FIXED SOURCE TRANSPORT SIMULATION", 3);
      } else if (settings::solver_type == SolverType::RANDOM_RAY) {
        header("FIXED SOURCE TRANSPORT SIMULATION (RANDOM RAY SOLVER)", 3);
      }
    } else if (settings::run_mode == RunMode::EIGENVALUE) {
      if (settings::solver_type == SolverType::MONTE_CARLO) {
        header("K EIGENVALUE SIMULATION", 3);
      } else if (settings::solver_type == SolverType::RANDOM_RAY) {
        header("K EIGENVALUE SIMULATION (RANDOM RAY SOLVER)", 3);
      }
      if (settings::verbosity >= 7)
        print_columns();
    }
  }
}

int openmc_simulation_finalize()
{
  using namespace openmc;

  // Skip if simulation was never run
  if (!simulation::initialized)
    return 0;

  // Stop active batch timer and start finalization timer
  simulation::time_active.stop();
  simulation::time_finalize.start();

  // Clear material nuclide mapping
  for (auto& mat : model::materials) {
    mat->mat_nuclide_index_.clear();
  }

  // Close track file if open
  if (!settings::track_identifiers.empty() || settings::write_all_tracks) {
    close_track_file();
  }

  // Increment total number of generations
  simulation::total_gen += simulation::current_batch * settings::gen_per_batch;

#ifdef OPENMC_MPI
  broadcast_results();
#endif

  // Write tally results to tallies.out
  if (settings::output_tallies && mpi::master)
    write_tallies();

  // If weight window generators are present in this simulation,
  // write a weight windows file
  if (variance_reduction::weight_windows_generators.size() > 0) {
    openmc_weight_windows_export();
  }

  // Deactivate all tallies
  for (auto& t : model::tallies) {
    t->active_ = false;
  }

  // Stop timers and show timing statistics
  simulation::time_finalize.stop();
  simulation::time_total.stop();
  if (mpi::master) {
    if (settings::solver_type != SolverType::RANDOM_RAY) {
      if (settings::verbosity >= 6)
        print_runtime();
      if (settings::verbosity >= 4)
        print_results();
    }
  }
  if (settings::check_overlaps)
    print_overlap_check();

  // Reset flags
  simulation::initialized = false;
  return 0;
}

int openmc_next_batch(int* status)
{
  using namespace openmc;
  using openmc::simulation::current_gen;

  // Make sure simulation has been initialized
  if (!simulation::initialized) {
    set_errmsg("Simulation has not been initialized yet.");
    return OPENMC_E_ALLOCATE;
  }

  initialize_batch();

  // Initialize kinetic batch for a kinetic MC simulation
  initialize_kinetic_batch();

  // =======================================================================
  // LOOP OVER GENERATIONS (THESE ARE TIME STEPS FOR KINETIC SIMULATION)
  for (current_gen = 1; current_gen <= settings::gen_per_batch; ++current_gen) {
    openmc_simulate_generation();
  }

  finalize_batch();

  // Check simulation ending criteria
  if (status) {
    if (simulation::current_batch >= settings::n_max_batches) {
      *status = STATUS_EXIT_MAX_BATCH;
    } else if (simulation::satisfy_triggers) {
      *status = STATUS_EXIT_ON_TRIGGER;
    } else {
      *status = STATUS_EXIT_NORMAL;
    }
  }
  return 0;
}

void openmc_simulate_generation()
{
  using namespace openmc;

  initialize_generation();

  // Start timer for transport
  simulation::time_transport.start();

  // Transport loop
  if (settings::event_based) {
    transport_event_based();
  } else {
    transport_history_based();
  }

  // Accumulate time for transport
  simulation::time_transport.stop();

  finalize_generation();
}

void openmc_relax_kinetic_batch()
{
  using namespace openmc;
  using openmc::simulation::current_gen;

  // Deactivate tallies for relaxation generations
  deactivate_tallies();

  // Set number of relaxation generations to run
  settings::gen_per_batch = settings::n_inactive_timesteps;

  if (mpi::master)
    write_message(fmt::format(
      " Batch {0} relaxation generations", simulation::current_batch));
  for (current_gen = 1; current_gen <= settings::gen_per_batch; ++current_gen) {
    openmc_simulate_generation();
  }

  activate_tallies();
}

bool openmc_is_statepoint_batch()
{
  using namespace openmc;
  using openmc::simulation::current_gen;

  if (!simulation::initialized)
    return false;
  else
    return contains(settings::statepoint_batch, simulation::current_batch);
}

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace simulation {

int ct_current_file;
int current_batch;
int current_gen;
bool initialized {false};
double keff {1.0};
double keff_std;
double k_col_abs {0.0};
double k_col_tra {0.0};
double k_abs_tra {0.0};
double log_spacing;
int n_lost_particles {0};
bool need_depletion_rx {false};
int restart_batch;
bool satisfy_triggers {false};
int ssw_current_file;
int total_gen {0};
double total_weight;
double total_weight_end;
int64_t work_per_rank;

const RegularMesh* entropy_mesh {nullptr};
const RegularMesh* ufs_mesh {nullptr};

vector<double> k_generation;
vector<int64_t> work_index;

bool is_initial_condition {true};

bool is_decorrelation_generation {false};
int initial_overall_generation {0};
int32_t initial_n_realizations {0};
int32_t initial_gen_per_batch {1};
vector<double> initial_k_generation;
array<double, 2> initial_k_sum {0.0, 0.0};
double initial_k_col {0.0};
double initial_k_abs {0.0};
double initial_k_tra {0.0};
double initial_k_col_abs {0.0};
double initial_k_col_tra {0.0};
double initial_k_abs_tra {0.0};

int current_timestep;
double current_time {0.0};
bool source_correction {false};
double initial_keff;

int64_t combined_work_per_rank;
vector<int64_t> combined_work_index;

// Precursor Particle Variables
int64_t precursor_work_per_rank {0};
vector<int64_t> precursor_work_index;

double average_neutron_weight;
double average_precursor_weight;

bool weighted_comb {false};

} // namespace simulation

//==============================================================================
// Non-member functions
//==============================================================================

void allocate_banks()
{
  if (settings::run_mode == RunMode::EIGENVALUE &&
      settings::solver_type == SolverType::MONTE_CARLO) {
    // Allocate source bank
    simulation::source_bank.resize(simulation::work_per_rank);

    // Allocate bank for fission census
    init_census_bank(simulation::fission_bank, simulation::progeny_per_particle,
      simulation::work_per_rank);

    // Allocate IFP bank
    if (settings::ifp_on)
      resize_simulation_ifp_banks();

    if (settings::kinetic_simulation) {
      // Allocate source bank copy
      simulation::initial_source_bank.resize(simulation::work_per_rank);

      // Allocate bank for time census
      init_census_bank(simulation::time_census_bank,
        simulation::time_progeny_per_particle,
        simulation::combined_work_per_rank);

      if (settings::forced_decay) {
        simulation::precursor_source_bank.resize(
          simulation::precursor_work_per_rank);
        simulation::initial_precursor_source_bank.resize(
          simulation::precursor_work_per_rank);
        // We use work_per_rank here because each simulated particle can
        // potentially create a precursor and the shared bank needs the size to
        // reflect that
        init_census_bank(simulation::precursor_shared_bank,
          simulation::precursor_progeny_per_particle,
          simulation::combined_work_per_rank);
      }
    }
  }

  if (settings::surf_source_write) {
    // Allocate surface source bank
    simulation::surf_source_bank.reserve(settings::ssw_max_particles);
  }

  if (settings::collision_track) {
    // Allocate collision track bank
    collision_track_reserve_bank();
  }
}

void activate_tallies()
{
  simulation::time_inactive.stop();
  simulation::time_active.start();
  for (auto& t : model::tallies) {
    t->active_ = true;
  }
  setup_active_tallies();
}

void deactivate_tallies()
{
  simulation::time_active.stop();
  simulation::time_inactive.start();
  for (auto& t : model::tallies) {
    t->active_ = false;
  }
  setup_active_tallies();
}

void initialize_batch()
{
  // Increment current batch
  ++simulation::current_batch;
  if (settings::run_mode == RunMode::FIXED_SOURCE) {
    if (settings::solver_type == SolverType::RANDOM_RAY &&
        simulation::current_batch < settings::n_inactive + 1) {
      write_message(
        6, "Simulating batch {:<4} (inactive)", simulation::current_batch);
    } else {
      write_message(6, "Simulating batch {}", simulation::current_batch);
    }
  }

  // Reset total starting particle weight used for normalizing tallies
  simulation::total_weight = 0.0;
  simulation::total_weight_end = 0.0;

  // Determine if this batch is the first inactive or active batch.
  bool first_inactive = false;
  bool first_active = false;
  if (!settings::restart_run) {
    first_inactive = settings::n_inactive > 0 && simulation::current_batch == 1;
    first_active = simulation::current_batch == settings::n_inactive + 1;
  } else if (simulation::current_batch == simulation::restart_batch + 1) {
    first_inactive = simulation::restart_batch < settings::n_inactive;
    first_active = !first_inactive;
  }

  // Manage active/inactive timers and activate tallies if necessary.
  if (first_inactive) {
    simulation::time_inactive.start();
  } else if (first_active) {
    simulation::time_inactive.stop();
    simulation::time_active.start();
    for (auto& t : model::tallies) {
      t->active_ = true;
    }
  }

  // Add user tallies to active tallies list
  setup_active_tallies();
}

void initialize_kinetic_batch()
{
  if (settings::kinetic_simulation && !simulation::is_initial_condition &&
      settings::run_mode == RunMode::EIGENVALUE &&
      settings::solver_type == SolverType::MONTE_CARLO) {
    // Run some criticality generations to decorrelate batches for eigenvalue
    // simulations
    if (settings::n_decorrelate_generations > 0) {
      // Run decorrelation generatiosn
      decorrelate_kinetic_eigenvalue_batch();
    } else {
      simulation::source_bank = simulation::initial_source_bank;
      simulation::keff = simulation::initial_keff;

      if (settings::forced_decay)
        simulation::precursor_source_bank =
          simulation::initial_precursor_source_bank;
    }
    // Force all particles to have a time of zero
    set_bank_times_to_zero();

    if (settings::n_inactive_timesteps > 0) {
      // Relax kinetic generation time domain
      openmc_relax_kinetic_batch();

      // Reset bank times
      set_bank_times_to_zero();
    }

    // Prepare for loop over time census boundaries, generations are now time
    // steps. Subtract one to account for the infinity boundary at the zeroth
    // index
    settings::gen_per_batch = settings::time_census_boundaries.size() - 1;
    if (mpi::master)
      write_message(fmt::format(
        " Batch {0} active time step generations", simulation::current_batch));
  }
  return;
}

void finalize_batch()
{
  // Reduce tallies onto master process and accumulate
  simulation::time_tallies.start();
  accumulate_tallies();
  simulation::time_tallies.stop();

  // update weight windows if needed
  for (const auto& wwg : variance_reduction::weight_windows_generators) {
    wwg->update();
  }

  // Reset global tally results
  if (simulation::current_batch <= settings::n_inactive) {
    simulation::global_tallies.fill(0.0);
    simulation::n_realizations = 0;
  }

  // Check_triggers
  if (mpi::master)
    check_triggers();
#ifdef OPENMC_MPI
  MPI_Bcast(&simulation::satisfy_triggers, 1, MPI_C_BOOL, 0, mpi::intracomm);
#endif
  if (simulation::satisfy_triggers ||
      (settings::trigger_on &&
        simulation::current_batch == settings::n_max_batches)) {
    settings::statepoint_batch.insert(simulation::current_batch);
  }

  // Write out state point if it's been specified for this batch and is not
  // a CMFD run instance
  if (contains(settings::statepoint_batch, simulation::current_batch) &&
      !settings::cmfd_run) {
    if (contains(settings::sourcepoint_batch, simulation::current_batch) &&
        settings::source_write && !settings::source_separate) {
      bool b = (settings::run_mode == RunMode::EIGENVALUE);
      openmc_statepoint_write(nullptr, &b);
    } else {
      bool b = false;
      openmc_statepoint_write(nullptr, &b);
    }
  }

  if (settings::run_mode == RunMode::EIGENVALUE) {
    // Write out a separate source point if it's been specified for this batch
    if (contains(settings::sourcepoint_batch, simulation::current_batch) &&
        settings::source_write && settings::source_separate) {

      // Determine width for zero padding
      int w = std::to_string(settings::n_max_batches).size();
      std::string source_point_filename = fmt::format("{0}source.{1:0{2}}",
        settings::path_output, simulation::current_batch, w);
      span<SourceSite> bankspan(simulation::source_bank);
      write_source_point(source_point_filename, bankspan,
        simulation::work_index, settings::source_mcpl_write);
    }

    // Write a continously-overwritten source point if requested.
    if (settings::source_latest) {
      auto filename = settings::path_output + "source";
      span<SourceSite> bankspan(simulation::source_bank);
      write_source_point(filename, bankspan, simulation::work_index,
        settings::source_mcpl_write);
    }
  }

  // Write out surface source if requested.
  if (settings::surf_source_write &&
      simulation::ssw_current_file <= settings::ssw_max_files) {
    bool last_batch = (simulation::current_batch == settings::n_batches);
    if (simulation::surf_source_bank.full() || last_batch) {
      // Determine appropriate filename
      auto filename = fmt::format("{}surface_source.{}", settings::path_output,
        simulation::current_batch);
      if (settings::ssw_max_files == 1 ||
          (simulation::ssw_current_file == 1 && last_batch)) {
        filename = settings::path_output + "surface_source";
      }

      // Get span of source bank and calculate parallel index vector
      auto surf_work_index = mpi::calculate_parallel_index_vector(
        simulation::surf_source_bank.size());
      span<SourceSite> surfbankspan(simulation::surf_source_bank.begin(),
        simulation::surf_source_bank.size());

      // Write surface source file
      write_source_point(
        filename, surfbankspan, surf_work_index, settings::surf_mcpl_write);

      // Reset surface source bank and increment counter
      simulation::surf_source_bank.clear();
      if (!last_batch && settings::ssw_max_files >= 1) {
        simulation::surf_source_bank.reserve(settings::ssw_max_particles);
      }
      ++simulation::ssw_current_file;
    }
  }
  // Write collision track file if requested
  if (settings::collision_track) {
    collision_track_flush_bank();
  }
}

void initialize_generation()
{
  if (settings::run_mode == RunMode::EIGENVALUE &&
      (simulation::is_initial_condition ||
        !simulation::is_initial_condition &&
          simulation::is_decorrelation_generation)) {
    // Only clear out the fission bank if running a steady state simulation, the
    // IC of a kinetic simulation, or decorrelateing a kinetic simulation batch
    simulation::fission_bank.resize(0);

    // Count source sites if using uniform fission source weighting
    if (settings::ufs_on)
      ufs_count_sites();

    // Store current value of tracklength k
    simulation::keff_generation = simulation::global_tallies(
      GlobalTally::K_TRACKLENGTH, TallyResult::VALUE);
  }

  if (settings::kinetic_simulation &&
      settings::solver_type == SolverType::MONTE_CARLO &&
      !simulation::is_initial_condition) {
    // Reset total starting particle weight used for normalizing tallies for
    // decorrleation generations AND for time census generations
    simulation::total_weight = 0.0;
    simulation::total_weight_end = 0.0;
  }

  if (settings::kinetic_simulation &&
      settings::solver_type == SolverType::MONTE_CARLO) {
    // Clear out the time census bank
    simulation::time_census_bank.resize(0);
    if (settings::forced_decay) {
      simulation::precursor_shared_bank.resize(0);
    }
  }
}

void finalize_generation()
{
  auto& gt = simulation::global_tallies;

  // Update global tallies with the accumulation variables
  if (settings::run_mode == RunMode::EIGENVALUE &&
      (simulation::is_initial_condition ||
        !simulation::is_initial_condition &&
          simulation::is_decorrelation_generation)) {
    gt(GlobalTally::K_COLLISION, TallyResult::VALUE) += global_tally_collision;
    gt(GlobalTally::K_ABSORPTION, TallyResult::VALUE) +=
      global_tally_absorption;
    gt(GlobalTally::K_TRACKLENGTH, TallyResult::VALUE) +=
      global_tally_tracklength;
  }
  if (settings::kinetic_simulation && !simulation::is_initial_condition &&
      !simulation::is_decorrelation_generation) {
    gt(GlobalTally::K_ABSORPTION, TallyResult::VALUE) +=
      global_tally_absorption;
    gt(GlobalTally::K_PRODUCTION, TallyResult::VALUE) +=
      global_tally_production;
  }
  gt(GlobalTally::LEAKAGE, TallyResult::VALUE) += global_tally_leakage;

  // reset tallies
  if (settings::run_mode == RunMode::EIGENVALUE &&
      (simulation::is_initial_condition ||
        !simulation::is_initial_condition &&
          simulation::is_decorrelation_generation)) {
    global_tally_collision = 0.0;
    global_tally_absorption = 0.0;
    global_tally_tracklength = 0.0;
  }
  if (settings::kinetic_simulation && !simulation::is_initial_condition &&
      !simulation::is_decorrelation_generation) {
    global_tally_production = 0.0;
    global_tally_absorption = 0.0;
  }
  global_tally_leakage = 0.0;

  if (settings::run_mode == RunMode::EIGENVALUE &&
      settings::solver_type == SolverType::MONTE_CARLO) {
    // If using shared memory, stable sort the fission bank (by parent IDs)
    // so as to allow for reproducibility regardless of which order particles
    // are run in.
    if (simulation::is_initial_condition ||
        (!simulation::is_initial_condition &&
          simulation::is_decorrelation_generation)) {
      // Only sort fission bank if running a steady state simulation, the IC
      // of a kinetic simulation, or decorrelateing a kinetic simulation batch
      sort_census_bank(simulation::fission_bank,
        simulation::progeny_per_particle, simulation::work_index);

      // Distribute fission bank across processors evenly
      synchronize_bank(simulation::fission_bank, simulation::source_bank,
        settings::n_particles, simulation::work_per_rank,
        simulation::work_index, simulation::average_neutron_weight);
    }
  }

  if (settings::solver_type == SolverType::MONTE_CARLO &&
      settings::kinetic_simulation) {

    // Time census only after the initial condition
    if (!simulation::is_initial_condition &&
        !simulation::is_decorrelation_generation) {
      if (settings::neutron_weighted_comb)
        simulation::weighted_comb = true;
      // If using shared memory, stable sort the time census bank (by parent
      // IDs) so as to allow for reproducibility regardless of which order
      // particles are run in.
      sort_census_bank(simulation::time_census_bank,
        simulation::time_progeny_per_particle, simulation::combined_work_index);

      // Distribute time census bank across processors evenly
      synchronize_bank(simulation::time_census_bank, simulation::source_bank,
        settings::n_particles, simulation::combined_work_per_rank,
        simulation::work_index, simulation::average_neutron_weight);
      if (settings::neutron_weighted_comb)
        simulation::weighted_comb = false;
    }

    if (settings::forced_decay) {
      if (settings::precursor_weighted_comb &&
          !simulation::is_initial_condition &&
          !simulation::is_decorrelation_generation)
        simulation::weighted_comb = true;
      // The precursor bank should also be sorted
      sort_census_bank(simulation::precursor_shared_bank,
        simulation::precursor_progeny_per_particle,
        simulation::combined_work_index);
      // Distribute also precursors source sites
      synchronize_bank(simulation::precursor_shared_bank,
        simulation::precursor_source_bank, settings::n_precursor_particles,
        simulation::combined_work_per_rank, simulation::precursor_work_index,
        simulation::average_precursor_weight);
      if (settings::precursor_weighted_comb &&
          !simulation::is_initial_condition &&
          !simulation::is_decorrelation_generation)
        simulation::weighted_comb = false;
    }
  }

  // TODO: will this prevent dynamic keff?
  if (settings::run_mode == RunMode::EIGENVALUE) {

    // Calculate shannon entropy
    if (settings::entropy_on &&
        settings::solver_type == SolverType::MONTE_CARLO)
      shannon_entropy();

    // Collect results and statistics
    // Don't calculate k during kinetic simulations
    if (simulation::is_initial_condition ||
        (!simulation::is_initial_condition &&
          simulation::is_decorrelation_generation)) {
      calculate_generation_keff();
      calculate_average_keff();
    }

    // Write generation output
    if (mpi::master && settings::verbosity >= 7) {
      print_generation();
    }
  }
}

void initialize_history(Particle& p, int64_t index_source, bool from_precursor)
{
  // set defaults
  if (!from_precursor) {
    if (settings::run_mode == RunMode::EIGENVALUE) {
      // set defaults for eigenvalue simulations from primary bank
      p.from_source(&simulation::source_bank[index_source - 1]);
    } else if (settings::run_mode == RunMode::FIXED_SOURCE) {
      // TODO: support fixed source sims with forced decay (required overhauling
      // the particle indexing) initialize random number seed
      int64_t id = (simulation::total_gen + overall_generation() - 1) *
                     settings::n_particles +
                   simulation::work_index[mpi::rank] + index_source;
      uint64_t seed = init_seed(id, STREAM_SOURCE);
      // sample from external source distribution or custom library then set
      auto site = sample_external_source(&seed);
      p.from_source(&site);
    }
  } else {
    SourceSite precursor_site =
      simulation::precursor_source_bank[index_source - 1];
    p.from_source(&precursor_site);
  }
  // TODO: this is for IFP, disable for kinetic simulations
  p.current_work() = index_source;

  // set identifier for particle
  if (!simulation::is_initial_condition &&
      !simulation::is_decorrelation_generation) {
    p.id() = simulation::combined_work_index[mpi::rank] + index_source;
    if (from_precursor)
      // Adjust precursor ID by work_per_rank to prevent duplicate particle IDs
      p.id() += simulation::work_per_rank;
  } else {
    p.id() = simulation::work_index[mpi::rank] + index_source;
  }

  // set progeny count to zero
  p.n_progeny() = 0;

  // Reset particle event counter
  p.n_event() = 0;

  // Reset split counter
  p.n_split() = 0;

  // Reset weight window ratio
  p.ww_factor() = 0.0;

  // set particle history start weight
  p.wgt_born() = p.wgt();

  // Reset pulse_height_storage
  std::fill(p.pht_storage().begin(), p.pht_storage().end(), 0);

  // set random number seed
  int64_t particle_seed =
    (simulation::total_gen + overall_generation() - 1) * settings::n_particles +
    p.id();
  if (!simulation::is_initial_condition &&
      !simulation::is_decorrelation_generation) {
    // Adjust the particle seed by the number of precursor particles simulated
    // to prevent duplicate seeds
    particle_seed += (simulation::total_gen + overall_generation() - 1) *
                     settings::n_precursor_particles;
  }

  init_particle_seeds(particle_seed, p.seeds());

  // set particle trace
  // TODO: Will this mess up for TD sims?
  p.trace() = false;
  if (simulation::current_batch == settings::trace_batch &&
      simulation::current_gen == settings::trace_gen &&
      p.id() == settings::trace_particle)
    p.trace() = true;

  // Set particle track.
  p.write_track() = check_track_criteria(p);

  // Set the particle's initial weight window value.
  p.wgt_ww_born() = -1.0;
  apply_weight_windows(p);

  // Check particle time index
  int expected_tb_idx;
  if (openmc::settings::kinetic_simulation &&
      !simulation::is_initial_condition &&
      !simulation::is_decorrelation_generation) {
    expected_tb_idx = openmc::simulation::current_gen;
  } else {
    expected_tb_idx = 0;
  }
  if (p.time_bound_idx() != expected_tb_idx) {
    std::string err = fmt::format("Expected time bound index was {0} but "
                                  " particle time bound was {1}",
      expected_tb_idx, p.time_bound_idx());
    fatal_error(err);
  }

  // Display message if high verbosity or trace is on
  if (settings::verbosity >= 9 || p.trace()) {
    write_message("Simulating Particle {}", p.id());
  }

  // Add particle's starting weight to count for normalizing tallies later (skip
  // for forced decay)
  if (!from_precursor) {
#pragma omp atomic
    simulation::total_weight += p.wgt();
  }

  if (openmc::settings::kinetic_simulation &&
      !simulation::is_initial_condition &&
      !simulation::is_decorrelation_generation) {
    if (simulation::current_gen == 1) {
      double sigma_t = p.macro_xs().total;
      p.wgt() /= p.speed() * sigma_t;
    }
  }

  // Force calculation of cross-sections by setting last energy to zero
  if (settings::run_CE) {
    p.invalidate_neutron_xs();
  }

  // Prepare to write out particle track.
  if (p.write_track())
    add_particle_track(p);
}

int overall_generation()
{
  using namespace simulation;
  int overall_gen = settings::gen_per_batch * (current_batch - 1) + current_gen;
  if (is_decorrelation_generation)
    // Increment the overall generation when decorrelating TD batches
    overall_gen += initial_overall_generation;
  return overall_gen;
}

void calculate_work(
  int64_t n_particles, int64_t& work_per_rank, vector<int64_t>& work_index)
{
  // Determine minimum amount of particles to simulate on each processor
  int64_t min_work = n_particles / mpi::n_procs;

  // Determine number of processors that have one extra particle (or precursor
  // particle)
  int64_t remainder = n_particles % mpi::n_procs;

  int64_t i_bank = 0;
  work_index.resize(mpi::n_procs + 1);
  work_index[0] = 0;
  for (int i = 0; i < mpi::n_procs; ++i) {
    // Number of particles for rank i
    int64_t work_i = i < remainder ? min_work + 1 : min_work;

    // Set number of particles
    if (mpi::rank == i)
      work_per_rank = work_i;

    // Set index into source bank for rank i
    i_bank += work_i;
    work_index[i + 1] = i_bank;
  }
}

void initialize_data()
{
  // Determine minimum/maximum energy for incident neutron/photon data
  data::energy_max = {INFTY, INFTY, INFTY, INFTY};
  data::energy_min = {0.0, 0.0, 0.0, 0.0};

  for (const auto& nuc : data::nuclides) {
    if (nuc->grid_.size() >= 1) {
      int neutron = ParticleType::neutron().transport_index();
      data::energy_min[neutron] =
        std::max(data::energy_min[neutron], nuc->grid_[0].energy.front());
      data::energy_max[neutron] =
        std::min(data::energy_max[neutron], nuc->grid_[0].energy.back());
    }
  }

  if (settings::photon_transport) {
    for (const auto& elem : data::elements) {
      if (elem->energy_.size() >= 1) {
        int photon = ParticleType::photon().transport_index();
        int n = elem->energy_.size();
        data::energy_min[photon] =
          std::max(data::energy_min[photon], std::exp(elem->energy_(1)));
        data::energy_max[photon] =
          std::min(data::energy_max[photon], std::exp(elem->energy_(n - 1)));
      }
    }

    if (settings::electron_treatment == ElectronTreatment::TTB) {
      // Determine if minimum/maximum energy for bremsstrahlung is greater/less
      // than the current minimum/maximum
      if (data::ttb_e_grid.size() >= 1) {
        int photon = ParticleType::photon().transport_index();
        int electron = ParticleType::electron().transport_index();
        int positron = ParticleType::positron().transport_index();
        int n_e = data::ttb_e_grid.size();

        const std::vector<int> charged = {electron, positron};
        for (auto t : charged) {
          data::energy_min[t] = std::exp(data::ttb_e_grid(1));
          data::energy_max[t] = std::exp(data::ttb_e_grid(n_e - 1));
        }

        data::energy_min[photon] =
          std::max(data::energy_min[photon], data::energy_min[electron]);

        data::energy_max[photon] =
          std::min(data::energy_max[photon], data::energy_max[electron]);
      }
    }
  }

  // Show which nuclide results in lowest energy for neutron transport
  for (const auto& nuc : data::nuclides) {
    // If a nuclide is present in a material that's not used in the model, its
    // grid has not been allocated
    if (nuc->grid_.size() > 0) {
      double max_E = nuc->grid_[0].energy.back();
      int neutron = ParticleType::neutron().transport_index();
      if (max_E == data::energy_max[neutron]) {
        write_message(7, "Maximum neutron transport energy: {} eV for {}",
          data::energy_max[neutron], nuc->name_);
        if (mpi::master && data::energy_max[neutron] < 20.0e6) {
          warning("Maximum neutron energy is below 20 MeV. This may bias "
                  "the results.");
        }
        break;
      }
    }
  }

  // Set up logarithmic grid for nuclides
  for (auto& nuc : data::nuclides) {
    nuc->init_grid();
  }
  int neutron = ParticleType::neutron().transport_index();
  simulation::log_spacing =
    std::log(data::energy_max[neutron] / data::energy_min[neutron]) /
    settings::n_log_bins;
}

#ifdef OPENMC_MPI
void broadcast_results()
{
  // Broadcast tally results so that each process has access to results
  for (auto& t : model::tallies) {
    // Create a new datatype that consists of all values for a given filter
    // bin and then use that to broadcast. This is done to minimize the
    // chance of the 'count' argument of MPI_BCAST exceeding 2**31
    auto& results = t->results_;

    auto shape = results.shape();
    int count_per_filter = shape[1] * shape[2];
    MPI_Datatype result_block;
    MPI_Type_contiguous(count_per_filter, MPI_DOUBLE, &result_block);
    MPI_Type_commit(&result_block);
    MPI_Bcast(results.data(), shape[0], result_block, 0, mpi::intracomm);
    MPI_Type_free(&result_block);
  }

  // Also broadcast global tally results
  auto& gt = simulation::global_tallies;
  MPI_Bcast(gt.data(), gt.size(), MPI_DOUBLE, 0, mpi::intracomm);

  // These guys are needed so that non-master processes can calculate the
  // combined estimate of k-effective
  double temp[] {
    simulation::k_col_abs, simulation::k_col_tra, simulation::k_abs_tra};
  MPI_Bcast(temp, 3, MPI_DOUBLE, 0, mpi::intracomm);
  simulation::k_col_abs = temp[0];
  simulation::k_col_tra = temp[1];
  simulation::k_abs_tra = temp[2];
}

#endif

void free_memory_simulation()
{
  simulation::k_generation.clear();
  simulation::entropy.clear();
}

void transport_history_based_single_particle(Particle& p)
{
  while (p.alive()) {
    p.event_calculate_xs();
    if (p.alive()) {
      p.event_advance();
    }
    if (p.alive()) {
      if (p.collision_distance() > p.boundary().distance()) {
        p.event_cross_surface();
      } else if (p.alive()) {
        p.event_collide();
      }
    }
    p.event_revive_from_secondary();
  }
  p.event_death();
}

void transport_history_based()
{
  double old_weight_survive;
  if (settings::kinetic_simulation && !simulation::is_initial_condition &&
      !simulation::is_decorrelation_generation) {
    // Zero out precursor_progeny_per_particle
    if (settings::forced_decay) {
      std::fill(simulation::precursor_progeny_per_particle.begin(),
        simulation::precursor_progeny_per_particle.end(), 0);
    }
    // Set surival weight to average weight of neutrons if using branchless
    // collisions, as the average weight will change over time.
    if (settings::branchless_collision) {
      old_weight_survive = settings::weight_survive;
      if (simulation::current_gen > 1)
        settings::weight_survive = simulation::average_neutron_weight;
    }
  }

#pragma omp parallel for schedule(runtime)
  for (int64_t i_work = 1; i_work <= simulation::work_per_rank; ++i_work) {
    Particle p;
    initialize_history(p, i_work);
    transport_history_based_single_particle(p);
  }
  if (settings::kinetic_simulation && !simulation::is_initial_condition &&
      !simulation::is_decorrelation_generation) {
    // Only use forced decay in the transient part of a kinetic simulation
    if (settings::forced_decay) {
      if (simulation::current_gen == 1)
        simulation::average_neutron_weight =
          simulation::total_weight / settings::n_particles;
#pragma omp parallel for schedule(runtime)
      for (int64_t i_work = 1; i_work <= simulation::precursor_work_per_rank;
           ++i_work) {
        Particle p;
        initialize_history(p, i_work, true);
        forced_precursor_decay(p, i_work);
        transport_history_based_single_particle(p);
      }
    }
    if (settings::branchless_collision)
      settings::weight_survive = old_weight_survive;
  }
}

void transport_event_based()
{
  int64_t remaining_work = simulation::work_per_rank;
  int64_t source_offset = 0;

  // To cap the total amount of memory used to store particle object data, the
  // number of particles in flight at any point in time can bet set. In the case
  // that the maximum in flight particle count is lower than the total number
  // of particles that need to be run this iteration, the event-based transport
  // loop is executed multiple times until all particles have been completed.
  while (remaining_work > 0) {
    // Figure out # of particles to run for this subiteration
    int64_t n_particles =
      std::min(remaining_work, settings::max_particles_in_flight);

    // Initialize all particle histories for this subiteration
    process_init_events(n_particles, source_offset);

    // Event-based transport loop
    while (true) {
      // Determine which event kernel has the longest queue
      int64_t max = std::max({simulation::calculate_fuel_xs_queue.size(),
        simulation::calculate_nonfuel_xs_queue.size(),
        simulation::advance_particle_queue.size(),
        simulation::surface_crossing_queue.size(),
        simulation::collision_queue.size()});

      // Execute event with the longest queue
      if (max == 0) {
        break;
      } else if (max == simulation::calculate_fuel_xs_queue.size()) {
        process_calculate_xs_events(simulation::calculate_fuel_xs_queue);
      } else if (max == simulation::calculate_nonfuel_xs_queue.size()) {
        process_calculate_xs_events(simulation::calculate_nonfuel_xs_queue);
      } else if (max == simulation::advance_particle_queue.size()) {
        process_advance_particle_events();
      } else if (max == simulation::surface_crossing_queue.size()) {
        process_surface_crossing_events();
      } else if (max == simulation::collision_queue.size()) {
        process_collision_events();
      }
    }

    // Execute death event for all particles
    process_death_events(n_particles);

    // Adjust remaining work and source offset variables
    remaining_work -= n_particles;
    source_offset += n_particles;
  }
}

// Consecutive batch decorrelation scheme from "New kinetic simulation
// capabilites for Tripoli-4: Methods and applications", M. Faucher et
// al. (2018)
void decorrelate_kinetic_eigenvalue_batch()
{
  using openmc::simulation::current_gen;

  // Deactivate tallies for decorrelation generations
  deactivate_tallies();

  // Set number of decorrelation generations to run
  settings::gen_per_batch = settings::n_decorrelate_generations;

  // Set source banks to steady state copies
  simulation::source_bank = simulation::initial_source_bank;
  if (settings::forced_decay)
    simulation::precursor_source_bank =
      simulation::initial_precursor_source_bank;

  simulation::is_decorrelation_generation = true;
  if (mpi::master)
    write_message(fmt::format(
      " Batch {0} decorrelation generations", simulation::current_batch));
  for (current_gen = 1; current_gen <= settings::gen_per_batch; ++current_gen) {
    openmc_simulate_generation();
  }
  simulation::is_decorrelation_generation = false;

  // Save the decorrelated source banks
  simulation::initial_source_bank = simulation::source_bank;
  if (settings::forced_decay)
    simulation::initial_precursor_source_bank =
      simulation::precursor_source_bank;

  // Reactivate tallies for the kinetic simulation
  activate_tallies();
}

void store_initial_k_eigenvalue_quantities()
{
  using namespace simulation;
  auto& gt = simulation::global_tallies;

  initial_k_generation = k_generation;
  initial_k_col = gt(GlobalTally::K_COLLISION, TallyResult::VALUE);
  initial_k_abs = gt(GlobalTally::K_ABSORPTION, TallyResult::VALUE);
  initial_k_tra = gt(GlobalTally::K_TRACKLENGTH, TallyResult::VALUE);
  initial_k_sum = k_sum;
  initial_k_col_abs = k_col_abs;
  initial_k_col_tra = k_col_tra;
  initial_k_abs_tra = k_abs_tra;
}

void set_initial_k_eigenvalue_quantities()
{
  using namespace simulation;
  auto& gt = simulation::global_tallies;

  k_generation = initial_k_generation;
  gt(GlobalTally::K_COLLISION, TallyResult::VALUE) = initial_k_col;
  gt(GlobalTally::K_ABSORPTION, TallyResult::VALUE) = initial_k_abs;
  gt(GlobalTally::K_TRACKLENGTH, TallyResult::VALUE) = initial_k_tra;
  k_sum = initial_k_sum;
  k_col_abs = initial_k_col_abs;
  k_col_tra = initial_k_col_tra;
  k_abs_tra = initial_k_abs_tra;
}

// EDGE CASE TODO: This will affect generation of precursor particles
// during the decorrelation generations.. will the effect matter?
void set_bank_times_to_zero()
{
#pragma omp parallel for schedule(runtime)
  for (int64_t i_work = 1; i_work <= simulation::work_per_rank; ++i_work) {
    simulation::source_bank[i_work - 1].time = 0.0;
    simulation::source_bank[i_work - 1].time_bound_idx = 1;
    simulation::source_bank[i_work - 1].wgt = 1.0;
  }
  if (settings::forced_decay) {
#pragma omp parallel for schedule(runtime)
    for (int64_t i_work = 1; i_work <= simulation::precursor_work_per_rank;
         ++i_work) {
      SourceSite& precursor_site =
        simulation::precursor_source_bank[i_work - 1];
      precursor_site.time = 0.0;
      precursor_site.time_born = 0.0;
      precursor_site.time_bound_idx = 1;
      precursor_site.wgt = 1.0;
    }
  }
}

void forced_precursor_decay(Particle& p, int64_t i_work)
{
  // Set the stream to allow using random numbers
  p.stream() = STREAM_TRACKING;

  SourceSite& precursor_site = simulation::precursor_source_bank[i_work - 1];
  uint64_t* seed = p.current_seed();
  const auto& nuc {data::nuclides[precursor_site.i_nuclide]};

  auto& rx = *nuc->fission_rx_[precursor_site.i_fission_rx];

  int group = sample_forced_decay(p, precursor_site);

  // Add forced decay particle's starting weight to count for normalizing
  // tallies later
  double wgt = p.wgt();
  simulation::total_weight += wgt;

  // Sample energy out
  // TODO: sample E_in?
  double E_in =
    precursor_site.E; // Banked energy of incoming fissioning neutron
  double E_out;
  double mu;
  rx.products_[group].sample(E_in, E_out, mu, seed);

  // Apply angle out
  p.u() = rotate_angle(p.u(), mu, nullptr, seed);

  // Split or russian roulette precursor particle
  // TODO: add control flow to prevent pathological spliting rouletting?
  apply_russian_roulette(p);
  apply_splitting(p);

  // Subtract weight from total if eliminated by russian roulette
  if (p.wgt() == 0.0)
    simulation::total_weight -= wgt;

  // Progeny vector adjustment to allow sorting algorithm to function properly
  // This happens after all banked neutrons have already been transported, so
  // we need to adjust the precursor_progeny_per_particle when adding the
  // precursor site to the shared bank.
  int64_t offset =
    precursor_site.parent_id - 1 - simulation::work_index[mpi::rank];
  precursor_site.progeny_id =
    simulation::precursor_progeny_per_particle[offset];
  simulation::precursor_progeny_per_particle[offset] += 1;

  // Add this precursor site to the shared precursor bank
  simulation::precursor_shared_bank.thread_safe_append(precursor_site);
}

int sample_forced_decay(Particle& p, SourceSite& precursor_site)
{
  uint64_t* seed = p.current_seed();
  const auto& nuc {data::nuclides[precursor_site.i_nuclide]};
  auto& rx = *nuc->fission_rx_[precursor_site.i_fission_rx];

  // Sample decay time
  double dt = settings::time_census_boundaries[p.time_bound_idx()] - p.time();
  p.time() += dt * prn(seed);

  double lambda_b = compute_lambda_b(precursor_site, rx);

  double neutron_sum;
  double precursor_sum;
  // Compute weight factors for forced decay neutron and precursor
  compute_forced_decay_weight_factors(
    rx, precursor_site, p, lambda_b, neutron_sum, precursor_sum);

  int group;
  if (settings::combined_precursor) {
    // Sample delay group of combined precursor particle
    group = sample_precursor_delay_group(
      rx, precursor_site, p, lambda_b, neutron_sum, seed);
  } else {
    group = precursor_site.delayed_group;
  }

  p.wgt() = precursor_site.wgt * neutron_sum;
  // LSH Approach
  // if (settings::combined_precursor) {
  //  double dt = settings::time_census_boundaries[p.time_bound_idx()] -
  //              settings::time_census_boundaries[p.time_bound_idx() - 1];
  //  p.wgt() *= dt;
  //}

  p.wgt_last() = p.wgt();

  precursor_site.wgt *= precursor_sum;
  precursor_site.time = settings::time_census_boundaries[p.time_bound_idx()];
  precursor_site.time_bound_idx += 1;

  return group;
}

double compute_lambda_b(SourceSite& precursor_site, const Reaction& rx)
{
  const auto& nuc {data::nuclides[precursor_site.i_nuclide]};
  double nu_d_tot = nuc->nu(precursor_site.E, Nuclide::EmissionMode::delayed);
  double& E_in = precursor_site.E;

  // compute lambda_b if needed (precursor site is from equilibrium calculation)
  double lambda_b;
  if (precursor_site.time_born == 0.0 && settings::combined_precursor) {
    for (int group = 1; group < nuc->n_precursor_; ++group) {
      double decay_rate = rx.products_[group].decay_rate_;
      double nu_d = nuc->nu(E_in, Nuclide::EmissionMode::delayed, group);
      lambda_b += nu_d / decay_rate;
    }
    return nu_d_tot / lambda_b;
  } else {
    return 1.0;
  }
}

void compute_forced_decay_weight_factors(const Reaction& rx,
  SourceSite& precursor_site, Particle& p, double lambda_b, double& neutron_sum,
  double& precursor_sum)
{
  double dt = p.time() - precursor_site.time_born;
  double& E_in = precursor_site.E;

  // Compute weight factors for forced decay neutron and precursor
  if (settings::combined_precursor) {
    const auto& nuc {data::nuclides[precursor_site.i_nuclide]};
    double nu_d_tot = nuc->nu(E_in, Nuclide::EmissionMode::delayed);
    for (int group = 1; group < nuc->n_precursor_; ++group) {
      double decay_rate = rx.products_[group].decay_rate_;
      double exp = std::exp(-1.0 * dt * decay_rate);
      double nu_d = nuc->nu(E_in, Nuclide::EmissionMode::delayed, group);
      double gamma_i = nu_d / nu_d_tot;
      if (precursor_site.time_born == 0.0) {
        gamma_i *= lambda_b / decay_rate;
      }
      // LSH Approach
      // neutron_sum += gamma_i * decay_rate * exp;
      // precursor_sum += gamma_i * exp;
      neutron_sum += gamma_i * (1.0 - exp);
      precursor_sum += gamma_i * exp;
    }
  } else {
    double decay_rate = rx.products_[precursor_site.delayed_group].decay_rate_;
    double exp = std::exp(-1.0 * dt * decay_rate);
    neutron_sum = 1.0 - exp;
    precursor_sum = exp;
  }
}

int sample_precursor_delay_group(const Reaction& rx, SourceSite& precursor_site,
  Particle& p, double lambda_b, double neutron_sum, uint64_t* seed)
{
  double& E_in = precursor_site.E;
  const auto& nuc {data::nuclides[precursor_site.i_nuclide]};
  double nu_d_tot = nuc->nu(E_in, Nuclide::EmissionMode::delayed);

  double dt = p.time() - precursor_site.time_born;
  double xi = prn(seed) * neutron_sum;
  double prob = 0.0;
  int group;
  for (group = 1; group < nuc->n_precursor_; ++group) {
    double decay_rate = rx.products_[group].decay_rate_;
    double exp = std::exp(-1.0 * dt * decay_rate);
    double nu_d = nuc->nu(E_in, Nuclide::EmissionMode::delayed, group);
    double gamma_i = nu_d / nu_d_tot;
    if (precursor_site.time_born = 0.0) {
      gamma_i *= lambda_b / decay_rate;
    }
    prob += gamma_i * decay_rate * exp;
    if (xi < prob)
      break;
  }
  return group;
}

} // namespace openmc
