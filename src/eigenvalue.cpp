#include "openmc/eigenvalue.h"

#include "openmc/tensor.h"

#include "openmc/array.h"
#include "openmc/bank.h"
#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/ifp.h"
#include "openmc/math_functions.h"
#include "openmc/mesh.h"
#include "openmc/message_passing.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/tallies/tally.h"
#include "openmc/timer.h"

#include <algorithm> // for min
#include <cmath>     // for sqrt, abs, pow
#include <iterator>  // for back_inserter
#include <limits>    //for infinity
#include <string>

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace simulation {

double keff_generation;
array<double, 2> k_sum;
vector<double> entropy;
tensor::Tensor<double> source_frac;

} // namespace simulation

//==============================================================================
// Non-member functions
//==============================================================================

void calculate_generation_keff()
{
  const auto& gt = simulation::global_tallies;

  // Get keff for this generation by subtracting off the starting value
  simulation::keff_generation =
    gt(GlobalTally::K_TRACKLENGTH, TallyResult::VALUE) -
    simulation::keff_generation;

  double keff_reduced;
#ifdef OPENMC_MPI
  if (settings::solver_type != SolverType::RANDOM_RAY) {
    // Combine values across all processors
    MPI_Allreduce(&simulation::keff_generation, &keff_reduced, 1, MPI_DOUBLE,
      MPI_SUM, mpi::intracomm);
  } else {
    // If using random ray, MPI parallelism is provided by domain replication.
    // As such, all fluxes will be reduced at the end of each transport sweep,
    // such that all ranks have identical scalar flux vectors, and will all
    // independently compute the same value of k. Thus, there is no need to
    // perform any additional MPI reduction here.
    keff_reduced = simulation::keff_generation;
  }
#else
  keff_reduced = simulation::keff_generation;
#endif

  // Normalize single batch estimate of k
  // TODO: This should be normalized by total_weight, not by n_particles
  if (settings::solver_type != SolverType::RANDOM_RAY) {
    keff_reduced /= settings::n_particles;
  }

  simulation::k_generation.push_back(keff_reduced);
}

void calculate_average_keff()
{
  // Determine overall generation and number of active generations
  int i = overall_generation() - 1;
  int n;
  if (simulation::current_batch > settings::n_inactive) {
    n = settings::gen_per_batch * simulation::n_realizations +
        simulation::current_gen;
  } else {
    n = 0;
  }

  if (n <= 0) {
    // For inactive generations, use current generation k as estimate for next
    // generation
    simulation::keff = simulation::k_generation[i];
  } else {
    // Sample mean of keff
    simulation::k_sum[0] += simulation::k_generation[i];
    simulation::k_sum[1] += std::pow(simulation::k_generation[i], 2);

    // Determine mean
    simulation::keff = simulation::k_sum[0] / n;

    if (n > 1) {
      double t_value;
      if (settings::confidence_intervals) {
        // Calculate t-value for confidence intervals
        double alpha = 1.0 - CONFIDENCE_LEVEL;
        t_value = t_percentile(1.0 - alpha / 2.0, n - 1);
      } else {
        t_value = 1.0;
      }

      // Standard deviation of the sample mean of k
      simulation::keff_std =
        t_value *
        std::sqrt(
          (simulation::k_sum[1] / n - std::pow(simulation::keff, 2)) / (n - 1));

      // In some cases (such as an infinite medium problem), random ray
      // may estimate k exactly and in an unvarying manner between iterations.
      // In this case, the floating point roundoff between the division and the
      // power operations may cause an extremely small negative value to occur
      // inside the sqrt operation, leading to NaN. If this occurs, we check for
      // it and set the std dev to zero.
      if (!std::isfinite(simulation::keff_std)) {
        simulation::keff_std = 0.0;
      }
    }
  }
}

int openmc_get_keff(double* k_combined)
{
  k_combined[0] = 0.0;
  k_combined[1] = 0.0;

  // Special case for n <=3. Notice that at the end,
  // there is a N-3 term in a denominator.
  if (simulation::n_realizations <= 3 ||
      settings::solver_type == SolverType::RANDOM_RAY) {
    k_combined[0] = simulation::keff;
    k_combined[1] = simulation::keff_std;
    if (simulation::n_realizations <= 1) {
      k_combined[1] = std::numeric_limits<double>::infinity();
    }
    return 0;
  }

  // Initialize variables
  int64_t n = simulation::n_realizations;

  // Copy estimates of k-effective and its variance (not variance of the mean)
  const auto& gt = simulation::global_tallies;

  array<double, 3> kv {};
  tensor::Tensor<double> cov = tensor::zeros<double>({3, 3});
  kv[0] = gt(GlobalTally::K_COLLISION, TallyResult::SUM) / n;
  kv[1] = gt(GlobalTally::K_ABSORPTION, TallyResult::SUM) / n;
  kv[2] = gt(GlobalTally::K_TRACKLENGTH, TallyResult::SUM) / n;
  cov(0, 0) =
    (gt(GlobalTally::K_COLLISION, TallyResult::SUM_SQ) - n * kv[0] * kv[0]) /
    (n - 1);
  cov(1, 1) =
    (gt(GlobalTally::K_ABSORPTION, TallyResult::SUM_SQ) - n * kv[1] * kv[1]) /
    (n - 1);
  cov(2, 2) =
    (gt(GlobalTally::K_TRACKLENGTH, TallyResult::SUM_SQ) - n * kv[2] * kv[2]) /
    (n - 1);

  // Calculate covariances based on sums with Bessel's correction
  cov(0, 1) = (simulation::k_col_abs - n * kv[0] * kv[1]) / (n - 1);
  cov(0, 2) = (simulation::k_col_tra - n * kv[0] * kv[2]) / (n - 1);
  cov(1, 2) = (simulation::k_abs_tra - n * kv[1] * kv[2]) / (n - 1);
  cov(1, 0) = cov(0, 1);
  cov(2, 0) = cov(0, 2);
  cov(2, 1) = cov(1, 2);

  // Check to see if two estimators are the same; this is guaranteed to happen
  // in MG-mode with survival biasing when the collision and absorption
  // estimators are the same, but can theoretically happen at anytime.
  // If it does, the standard estimators will produce floating-point
  // exceptions and an expression specifically derived for the combination of
  // two estimators (vice three) should be used instead.

  // First we will identify if there are any matching estimators
  int i, j;
  bool use_three = false;
  if ((std::abs(kv[0] - kv[1]) / kv[0] < FP_REL_PRECISION) &&
      (std::abs(cov(0, 0) - cov(1, 1)) / cov(0, 0) < FP_REL_PRECISION)) {
    // 0 and 1 match, so only use 0 and 2 in our comparisons
    i = 0;
    j = 2;

  } else if ((std::abs(kv[0] - kv[2]) / kv[0] < FP_REL_PRECISION) &&
             (std::abs(cov(0, 0) - cov(2, 2)) / cov(0, 0) < FP_REL_PRECISION)) {
    // 0 and 2 match, so only use 0 and 1 in our comparisons
    i = 0;
    j = 1;

  } else if ((std::abs(kv[1] - kv[2]) / kv[1] < FP_REL_PRECISION) &&
             (std::abs(cov(1, 1) - cov(2, 2)) / cov(1, 1) < FP_REL_PRECISION)) {
    // 1 and 2 match, so only use 0 and 1 in our comparisons
    i = 0;
    j = 1;

  } else {
    // No two estimators match, so set boolean to use all three estimators.
    use_three = true;
  }

  if (use_three) {
    // Use three estimators as derived in the paper by Urbatsch

    // Initialize variables
    double g = 0.0;
    array<double, 3> S {};

    for (int l = 0; l < 3; ++l) {
      // Permutations of estimates
      int k;
      switch (l) {
      case 0:
        // i = collision, j = absorption, k = tracklength
        i = 0;
        j = 1;
        k = 2;
        break;
      case 1:
        // i = absortion, j = tracklength, k = collision
        i = 1;
        j = 2;
        k = 0;
        break;
      case 2:
        // i = tracklength, j = collision, k = absorption
        i = 2;
        j = 0;
        k = 1;
        break;
      }

      // Calculate weighting
      double f = cov(j, j) * (cov(k, k) - cov(i, k)) - cov(k, k) * cov(i, j) +
                 cov(j, k) * (cov(i, j) + cov(i, k) - cov(j, k));

      // Add to S sums for variance of combined estimate
      S[0] += f * cov(0, l);
      S[1] += (cov(j, j) + cov(k, k) - 2.0 * cov(j, k)) * kv[l] * kv[l];
      S[2] += (cov(k, k) + cov(i, j) - cov(j, k) - cov(i, k)) * kv[l] * kv[j];

      // Add to sum for combined k-effective
      k_combined[0] += f * kv[l];
      g += f;
    }

    // Complete calculations of S sums
    for (auto& S_i : S) {
      S_i *= (n - 1);
    }
    S[0] *= (n - 1) * (n - 1);

    // Calculate combined estimate of k-effective
    k_combined[0] /= g;

    // Calculate standard deviation of combined estimate
    g *= (n - 1) * (n - 1);
    k_combined[1] =
      std::sqrt(S[0] / (g * n * (n - 3)) * (1 + n * ((S[1] - 2 * S[2]) / g)));

  } else {
    // Use only two estimators
    // These equations are derived analogously to that done in the paper by
    // Urbatsch, but are simpler than for the three estimators case since the
    // block matrices of the three estimator equations reduces to scalars here

    // Store the commonly used term
    double f = kv[i] - kv[j];
    double g = cov(i, i) + cov(j, j) - 2.0 * cov(i, j);

    // Calculate combined estimate of k-effective
    k_combined[0] = kv[i] - (cov(i, i) - cov(i, j)) / g * f;

    // Calculate standard deviation of combined estimate
    k_combined[1] = (cov(i, i) * cov(j, j) - cov(i, j) * cov(i, j)) *
                    (g + n * f * f) / (n * (n - 2) * g * g);
    k_combined[1] = std::sqrt(k_combined[1]);
  }
  return 0;
}

void shannon_entropy()
{
  // Get source weight in each mesh bin
  bool sites_outside;
  tensor::Tensor<double> p =
    simulation::entropy_mesh->count_sites(simulation::fission_bank.data(),
      simulation::fission_bank.size(), &sites_outside);

  // display warning message if there were sites outside entropy box
  if (sites_outside) {
    if (mpi::master)
      warning("Fission source site(s) outside of entropy box.");
  }

  if (mpi::master) {
    // Normalize to total weight of bank sites
    p /= p.sum();

    // Sum values to obtain Shannon entropy
    double H = 0.0;
    for (auto p_i : p) {
      if (p_i > 0.0) {
        H -= p_i * std::log2(p_i);
      }
    }

    // Add value to vector
    simulation::entropy.push_back(H);
  }
}

void ufs_count_sites()
{
  if (simulation::current_batch == 1 && simulation::current_gen == 1) {
    // On the first generation, just assume that the source is already evenly
    // distributed so that effectively the production of fission sites is not
    // biased

    std::size_t n = simulation::ufs_mesh->n_bins();
    double vol_frac = simulation::ufs_mesh->volume_frac_;
    simulation::source_frac = tensor::Tensor<double>({n}, vol_frac);

  } else {
    // count number of source sites in each ufs mesh cell
    bool sites_outside;
    simulation::source_frac =
      simulation::ufs_mesh->count_sites(simulation::source_bank.data(),
        simulation::source_bank.size(), &sites_outside);

    // Check for sites outside of the mesh
    if (mpi::master && sites_outside) {
      fatal_error("Source sites outside of the UFS mesh!");
    }

#ifdef OPENMC_MPI
    // Send source fraction to all processors
    int n_bins = simulation::ufs_mesh->n_bins();
    MPI_Bcast(
      simulation::source_frac.data(), n_bins, MPI_DOUBLE, 0, mpi::intracomm);
#endif

    // Normalize to total weight to get fraction of source in each cell
    double total = simulation::source_frac.sum();
    simulation::source_frac /= total;

    // Since the total starting weight is not equal to n_particles, we need to
    // renormalize the weight of the source sites
    for (int i = 0; i < simulation::work_per_rank; ++i) {
      simulation::source_bank[i].wgt *= settings::n_particles / total;
    }
  }
}

double ufs_get_weight(const Particle& p)
{
  // Determine indices on ufs mesh for current location
  int mesh_bin = simulation::ufs_mesh->get_bin(p.r());
  if (mesh_bin < 0) {
    p.write_restart();
    fatal_error("Source site outside UFS mesh!");
  }

  if (simulation::source_frac(mesh_bin) != 0.0) {
    return simulation::ufs_mesh->volume_frac_ /
           simulation::source_frac(mesh_bin);
  } else {
    return 1.0;
  }
}

void write_eigenvalue_hdf5(hid_t group)
{
  write_dataset(group, "n_inactive", settings::n_inactive);
  write_dataset(group, "generations_per_batch", settings::gen_per_batch);
  write_dataset(group, "k_generation", simulation::k_generation);
  if (settings::entropy_on) {
    write_dataset(group, "entropy", simulation::entropy);
  }
  write_dataset(group, "k_col_abs", simulation::k_col_abs);
  write_dataset(group, "k_col_tra", simulation::k_col_tra);
  write_dataset(group, "k_abs_tra", simulation::k_abs_tra);
  array<double, 2> k_combined;
  openmc_get_keff(k_combined.data());
  write_dataset(group, "k_combined", k_combined);
}

void read_eigenvalue_hdf5(hid_t group)
{
  read_dataset(group, "generations_per_batch", settings::gen_per_batch);
  int n = simulation::restart_batch * settings::gen_per_batch;
  simulation::k_generation.resize(n);
  read_dataset(group, "k_generation", simulation::k_generation);
  if (settings::entropy_on) {
    read_dataset(group, "entropy", simulation::entropy);
  }
  read_dataset(group, "k_col_abs", simulation::k_col_abs);
  read_dataset(group, "k_col_tra", simulation::k_col_tra);
  read_dataset(group, "k_abs_tra", simulation::k_abs_tra);
}

} // namespace openmc
