#ifndef OPENMC_RANDOM_RAY_SIMULATION_H
#define OPENMC_RANDOM_RAY_SIMULATION_H

#include "openmc/random_ray/flat_source_domain.h"
#include "openmc/random_ray/linear_source_domain.h"

namespace openmc {

//==============================================================================
// Global variable declarations
//==============================================================================
namespace random_ray_td {

//Initial conditions for time-depedent simulations
vector<double> precursor_init;
vector<double> scalar_flux_init;
vector<float> source_init;

} // namespace random_ray_td

/*
 * The RandomRaySimulation class encompasses data and methods for running a
 * random ray simulation.
 */

class RandomRaySimulation {
public:
  //----------------------------------------------------------------------------
  // Constructors
  RandomRaySimulation();

  //----------------------------------------------------------------------------
  // Methods
  void compute_segment_correction_factors();
  void simulate();
  void reduce_simulation_statistics();
  void output_simulation_results() const;
  void instability_check(
    int64_t n_hits, double k_eff, double& avg_miss_rate) const;
  void print_results_random_ray(uint64_t total_geometric_intersections,
    double avg_miss_rate, int negroups, int64_t n_source_regions,
    int64_t n_external_source_regions) const;

  vector<double> get_precursor_initial_condition();
  vector<double> get_scalar_flux_initial_condition();
  vector<float> get_source_initial_condition();

  //----------------------------------------------------------------------------
  // Data members
private:
  // Contains all flat source region data
  unique_ptr<FlatSourceDomain> domain_;

  // Random ray eigenvalue
  double k_eff_ {1.0};

  // Tracks the average FSR miss rate for analysis and reporting
  double avg_miss_rate_ {0.0};

  // Tracks the total number of geometric intersections by all rays for
  // reporting
  uint64_t total_geometric_intersections_ {0};

  // Number of energy groups
  int negroups_;
  int ndgroups_;

}; // class RandomRaySimulation

//============================================================================
//! Non-member functions
//============================================================================

void openmc_run_random_ray(bool initial_condition = false);
void openmc_run_random_ray_time_dependent();
void validate_random_ray_inputs();

} // namespace openmc

#endif // OPENMC_RANDOM_RAY_SIMULATION_H
