#ifndef OPENMC_RANDOM_RAY_SIMULATION_H
#define OPENMC_RANDOM_RAY_SIMULATION_H

#include "openmc/random_ray/flat_source_domain.h"
#include "openmc/random_ray/linear_source_domain.h"

namespace openmc {

/*
 * The RandomRaySimulation class encompasses data and methods for running a
 * random ray simulation.
 */

class RandomRaySimulation {
public:
  //----------------------------------------------------------------------------
  // Constructors
  RandomRaySimulation(bool generate_source_domain = true);

  //----------------------------------------------------------------------------
  // Methods
  void compute_segment_correction_factors();
  void prepare_fixed_sources();
  void prepare_fixed_sources_adjoint();
  void simulate();
  void reduce_simulation_statistics();
  void output_simulation_results() const;
  void instability_check(
    int64_t n_hits, double k_eff, double& avg_miss_rate) const;
  void print_results_random_ray(uint64_t total_geometric_intersections,
    double avg_miss_rate, int negroups, int64_t n_source_regions,
    int64_t n_external_source_regions) const;

  //----------------------------------------------------------------------------
  // Time Dependent Methods
  void find_all_fissile_regions();
  void compute_and_store_batch_fission_source(bool shift_window = false);

  double compute_window_averaged_rms_error();
  void is_window_avg_rms_source_converged(); // Determine if the window-averaged
                                             // RMS of the source is converg

  //----------------------------------------------------------------------------
  // Accessors
  FlatSourceDomain* domain() const { return domain_.get(); }

  //---------------------------------------------------------------------------
  // Data members

  // Random ray eigenvalue
  double k_eff_ {1.0};

  // Contains all flat source region data
  unique_ptr<FlatSourceDomain> domain_;

private:
  //----------------------------------------------------------------------------
  // Data members

  // Tracks the average FSR miss rate for analysis and reporting
  double avg_miss_rate_ {0.0};

  // Tracks the total number of geometric intersections by all rays for
  // reporting
  uint64_t total_geometric_intersections_ {0};

  // Number of energy groups
  int negroups_;

  // Number of delay groups
  int ndgroups_;

  // Flag to determine if every fissile region has been found
  bool all_fissile_regions_found_ {false};
  vector<int> fissile_region_srs_;

  // Flag to track if the source has converged
  bool source_converged_ {false};

}; // class RandomRaySimulation

//============================================================================
//! Non-member functions
//============================================================================

void openmc_run_random_ray();
void validate_random_ray_inputs();

void openmc_run_random_ray_time_dependent();
void set_time_dependent_settings();

void rename_statepoint_file(int i);
void rename_tallies_file(int i);
void increment_batches(); // Incremenet the number of batches and related
                          // arrays for an unconverged simulation. Only used
                          // for window-averged RMS convergence.
void fix_batches(); // If a simulation is converged but hasn't reached the set
                    // number of inactive batches, reduce the number of batches
                    // and related arrays so the tallying and statepoint
                    // machinery works as intended.

//==============================================================================
// Time-dependent global variables
//==============================================================================
extern double previous_k_eff;
extern unique_ptr<FlatSourceDomain> source_domain;

} // namespace openmc

#endif // OPENMC_RANDOM_RAY_SIMULATION_H
