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
  RandomRaySimulation();

  //----------------------------------------------------------------------------
  // Methods
  void compute_segment_correction_factors();
  void apply_fixed_sources_and_mesh_domains();
  void prepare_fixed_sources_adjoint();
  void set_initial_condition();
  void set_rhs_bd_vectors();
  void simulate();
  void output_simulation_results() const;
  void instability_check(
    int64_t n_hits, double k_eff, double& avg_miss_rate) const;
  void print_results_random_ray(uint64_t total_geometric_intersections,
    double avg_miss_rate, int negroups, int ndgroups, int64_t n_source_regions,
    int64_t n_external_source_regions) const;

  //----------------------------------------------------------------------------
  // Accessors
  FlatSourceDomain* domain() const { return domain_.get(); }

private:
  //----------------------------------------------------------------------------
  // Data members

  // Contains all flat source region data
  unique_ptr<FlatSourceDomain> domain_;

  // Tracks the average FSR miss rate for analysis and reporting
  double avg_miss_rate_ {0.0};

  // Tracks the total number of geometric intersections by all rays for
  // reporting
  uint64_t total_geometric_intersections_ {0};

  // Number of energy groups
  int negroups_;

  // Number of delay groups
  int ndgroups_;

}; // class RandomRaySimulation

//============================================================================
//! Non-member functions
//============================================================================

void openmc_run_random_ray();
void validate_random_ray_inputs();
void openmc_reset_random_ray();

void openmc_run_random_ray_time_dependent();
void rename_statepoint_file(int i);

void fill_bd_vector(
  int64_t vector_size, int n_timesteps, vector<double>& bd_vector);

void compute_rhs_backward_difference(int64_t vector_size,
  int bd_order, vector<double>& bd_vector,
  vector<double>& rhs_bd_vector, int derivative_order);

void increment_bd_vector(int64_t vector_size, vector<double>* bd_vector);
void get_bd_vector_slice(int64_t vector_size, vector<double>& storage_vector,
  vector<double>& bd_vector, int neg_timestep_index);

//==============================================================================
// Time-dependent global variables
//==============================================================================
extern vector<double> scalar_flux_bd;
extern vector<double> precursors_bd;
extern vector<double> source_bd;
extern vector<double> delayed_fission_source_bd;

extern vector<double> scalar_flux_rhs_bd;

extern vector<double> source_rhs_bd;
extern vector<double> scalar_flux_rhs_bd_2;

extern vector<double> precursors_rhs_bd;

extern vector<double> precursors_im1;
extern vector<double> delayed_fission_source_im1;
extern vector<double> delayed_fission_source_im2;

extern double previous_k_eff;

} // namespace openmc

#endif // OPENMC_RANDOM_RAY_SIMULATION_H
