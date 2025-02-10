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
  void prepare_fixed_sources();
  void prepare_fixed_sources_adjoint(vector<double>& forward_flux);
  void simulate(bool td = false);
  void reduce_simulation_statistics();
  void output_simulation_results() const;
  void instability_check(
    int64_t n_hits, double k_eff, double& avg_miss_rate) const;
  void print_results_random_ray(uint64_t total_geometric_intersections,
    double avg_miss_rate, int negroups, int ndgroups, int64_t n_source_regions,
    int64_t n_external_source_regions) const;

  //----------------------------------------------------------------------------
  // Accessors
  FlatSourceDomain* domain() const { return domain_.get(); }

  //---------------------------------------------------------------------------
  // Data members

  // Maximum order for BDF approximation.
  static int bdf_order_max_;

  // Random ray eigenvalue
  double k_eff_ {1.0};

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

void openmc_run_random_ray(bool initial_condition = false);
void validate_random_ray_inputs();

void openmc_run_random_ray_time_dependent();
void rename_statepoint_file(int i);
void initialize_bdf_vectors(int64_t n_source_elements, int64_t n_delay_elements, int bdf_order_max, vector<double>* scalar_flux_bdf, vector<float>* source_bdf, vector<double>* precursors_bdf, vector<double>* criticality_scalar_flux, vector<float>* criticality_source);
void increment_bdf_vectors(int64_t n_source_elements, int64_t n_delay_elements, vector<double>* scalar_flux_bdf, vector<float>* source_bdf, vector<double>* precursors_bdf);
void rename_statepoint_file(int i);

//==============================================================================
// Time-dependent global variables
//==============================================================================
extern vector<double> scalar_flux_bdf;
extern vector<float> source_bdf;
extern vector<double> precursors_bdf;

extern vector<double> criticality_scalar_flux;
extern vector<float> criticality_source;

} // namespace openmc

#endif // OPENMC_RANDOM_RAY_SIMULATION_H
