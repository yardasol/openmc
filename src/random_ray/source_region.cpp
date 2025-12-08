#include "openmc/random_ray/source_region.h"
#include "openmc/random_ray/random_ray.h"

#include "openmc/error.h"
#include "openmc/message_passing.h"

namespace openmc {

//==============================================================================
// SourceRegion implementation
//==============================================================================
SourceRegion::SourceRegion(int negroups, int ndgroups, bool is_linear)
{
  if (settings::run_mode == RunMode::EIGENVALUE) {
    // If in eigenvalue mode, set starting flux to guess of 1
    scalar_flux_old_.assign(negroups, 1.0);
    if (settings::is_initial_condition) {
      precursors_old_.assign(ndgroups, 0.0);
      precursors_new_.assign(ndgroups, 0.0);
      precursors_final_.assign(ndgroups, 0.0);
      tally_delay_task_.resize(ndgroups);
    }
  } else if (settings::run_mode == RunMode::FIXED_SOURCE) {
    // If in fixed source mode, set starting flux to guess of zero
    // and initialize external source arrays
    scalar_flux_old_.assign(negroups, 0.0);
    external_source_.assign(negroups, 0.0);
  }

  if (settings::run_mode == RunMode::TIME_DEPENDENT ||
      settings::is_initial_condition) {
    // If in time dependent mode, set starting flux to guess of 1
    scalar_flux_old_.assign(negroups, 1.0);

    scalar_flux_td_old_.assign(negroups, 1.0);
    scalar_flux_td_new_.assign(negroups, 0.0);
    source_td_.resize(negroups);
    scalar_flux_td_final_.assign(negroups, 0.0);

    delayed_fission_source_.assign(ndgroups, 0.0);
    precursors_old_.assign(ndgroups, 0.0);
    precursors_new_.assign(ndgroups, 0.0);
    precursors_final_.assign(ndgroups, 0.0);
    tally_delay_task_.resize(ndgroups);

    scalar_flux_bd_;
    scalar_flux_rhs_bd_.resize(negroups);

    // SDP arrays
    if (RandomRay::time_method_ == RandomRayTimeMethod::SDP) {
      source_final_.assign(negroups, 0.0);

      source_td_final_.assign(negroups, 0.0);
      source_time_derivative_.assign(negroups, 0.0);
      scalar_flux_time_derivative_2_.assign(negroups, 0.0);

      source_bd_;
      source_rhs_bd_.resize(negroups);
      scalar_flux_rhs_bd_2_.resize(negroups);
    }

    // Analytic precursor integration arrays
    if (RandomRay::precursor_method_ == RandomRayPrecursorMethod::INTEGRATION) {
      delayed_fission_source_final_.assign(ndgroups, 0.0);

      precursors_im1_.resize(ndgroups);
      delayed_fission_source_im1_.resize(ndgroups);
      delayed_fission_source_im2_.resize(ndgroups);
    } else {
      precursors_bd_;
      precursors_rhs_bd_.resize(ndgroups);
    }
  }

  scalar_flux_new_.assign(negroups, 0.0);
  source_.resize(negroups);
  scalar_flux_final_.assign(negroups, 0.0);

  tally_task_.resize(negroups);

  if (settings::source_convergence_method ==
      SourceConvergenceMethod::WINDOW_AVG_RMS)
    batchwise_fission_source_;

  if (is_linear) {
    source_gradients_.resize(negroups);
    flux_moments_old_.resize(negroups);
    flux_moments_new_.resize(negroups);
    flux_moments_t_.resize(negroups);
  }
}

//==============================================================================
// SourceRegionContainer implementation
//==============================================================================
void SourceRegionContainer::push_back(const SourceRegion& sr)
{
  n_source_regions_++;

  // Scalar fields
  material_.push_back(sr.material_);
  lock_.push_back(sr.lock_);
  volume_.push_back(sr.volume_);
  volume_t_.push_back(sr.volume_t_);
  volume_naive_.push_back(sr.volume_naive_);
  position_recorded_.push_back(sr.position_recorded_);
  external_source_present_.push_back(sr.external_source_present_);
  position_.push_back(sr.position_);
  volume_task_.push_back(sr.volume_task_);

  if (settings::source_convergence_method ==
      SourceConvergenceMethod::WINDOW_AVG_RMS)
    batchwise_fission_source_.push_back(sr.batchwise_fission_source_);

  // Only store these fields if is_linear_ is true
  if (is_linear_) {
    centroid_.push_back(sr.centroid_);
    centroid_iteration_.push_back(sr.centroid_iteration_);
    centroid_t_.push_back(sr.centroid_t_);
    mom_matrix_.push_back(sr.mom_matrix_);
    mom_matrix_t_.push_back(sr.mom_matrix_t_);
  }

  // Energy-dependent fields
  for (int g = 0; g < negroups_; ++g) {
    scalar_flux_old_.push_back(sr.scalar_flux_old_[g]);
    scalar_flux_new_.push_back(sr.scalar_flux_new_[g]);
    scalar_flux_final_.push_back(sr.scalar_flux_final_[g]);
    source_.push_back(sr.source_[g]);

    if (settings::run_mode == RunMode::FIXED_SOURCE) {
      external_source_.push_back(sr.external_source_[g]);
    }

    if (settings::run_mode == RunMode::TIME_DEPENDENT ||
        settings::is_initial_condition) {
      scalar_flux_td_old_.push_back(sr.scalar_flux_td_old_[g]);
      scalar_flux_td_new_.push_back(sr.scalar_flux_td_new_[g]);
      scalar_flux_td_final_.push_back(sr.scalar_flux_td_final_[g]);
      source_td_.push_back(sr.source_td_[g]);

      scalar_flux_bd_.push_back(sr.scalar_flux_bd_);
      scalar_flux_rhs_bd_.push_back(sr.scalar_flux_rhs_bd_[g]);

      // SDP arrays
      if (RandomRay::time_method_ == RandomRayTimeMethod::SDP) {
        source_final_.push_back(sr.source_final_[g]);

        source_td_final_.push_back(sr.source_final_[g]);
        source_time_derivative_.push_back(sr.source_time_derivative_[g]);
        scalar_flux_time_derivative_2_.push_back(
          sr.scalar_flux_time_derivative_2_[g]);

        source_bd_.push_back(sr.source_bd_);
        source_rhs_bd_.push_back(sr.source_rhs_bd_[g]);
        scalar_flux_rhs_bd_2_.push_back(sr.scalar_flux_rhs_bd_2_[g]);
      }
    }

    // Only store these fields if is_linear_ is true
    if (is_linear_) {
      source_gradients_.push_back(sr.source_gradients_[g]);
      flux_moments_old_.push_back(sr.flux_moments_old_[g]);
      flux_moments_new_.push_back(sr.flux_moments_new_[g]);
      flux_moments_t_.push_back(sr.flux_moments_t_[g]);
    }

    // Tally tasks
    tally_task_.emplace_back(sr.tally_task_[g]);
  }

  // Precursor-dependent fields
  if (settings::run_mode == RunMode::TIME_DEPENDENT ||
      settings::is_initial_condition) {
    for (int dg = 0; dg < ndgroups_; dg++) {
      delayed_fission_source_.push_back(sr.delayed_fission_source_[dg]);
      precursors_old_.push_back(sr.precursors_old_[dg]);
      precursors_new_.push_back(sr.precursors_new_[dg]);
      precursors_final_.push_back(sr.precursors_final_[dg]);
      tally_delay_task_.emplace_back(sr.tally_delay_task_[dg]);

      // Analytic precursor integration arrays
      if (RandomRay::precursor_method_ == RandomRayPrecursorMethod::INTEGRATION) {
        delayed_fission_source_final_.push_back(
          sr.delayed_fission_source_final_[dg]);

        precursors_im1_.push_back(sr.precursors_im1_[dg]);
        delayed_fission_source_im1_.push_back(
          sr.delayed_fission_source_im1_[dg]);
        delayed_fission_source_im2_.push_back(
          sr.delayed_fission_source_im2_[dg]);
        // Backward difference arrays
      } else {
        precursors_bd_.push_back(sr.precursors_bd_);
        precursors_rhs_bd_.push_back(sr.precursors_rhs_bd_[dg]);
      }
    }
  }
}

void SourceRegionContainer::assign(
  int n_source_regions, const SourceRegion& source_region)
{
  // Clear existing data
  n_source_regions_ = 0;
  material_.clear();
  lock_.clear();
  volume_.clear();
  volume_t_.clear();
  volume_naive_.clear();
  position_recorded_.clear();
  external_source_present_.clear();
  position_.clear();

  if (is_linear_) {
    centroid_.clear();
    centroid_iteration_.clear();
    centroid_t_.clear();
    mom_matrix_.clear();
    mom_matrix_t_.clear();
  }

  scalar_flux_old_.clear();
  scalar_flux_new_.clear();
  scalar_flux_final_.clear();
  source_.clear();
  external_source_.clear();

  if (is_linear_) {
    source_gradients_.clear();
    flux_moments_old_.clear();
    flux_moments_new_.clear();
    flux_moments_t_.clear();
  }

  if (settings::run_mode == RunMode::TIME_DEPENDENT ||
      settings::is_initial_condition) {
    scalar_flux_td_old_.clear();
    scalar_flux_td_new_.clear();
    scalar_flux_td_final_.clear();
    source_td_.clear();

    scalar_flux_bd_.clear();
    scalar_flux_rhs_bd_.clear();

    if (RandomRay::time_method_ == RandomRayTimeMethod::SDP) {
      source_final_.clear();

      source_time_derivative_.clear();
      scalar_flux_time_derivative_2_.clear();

      source_bd_.clear();
      source_rhs_bd_.clear();
      scalar_flux_rhs_bd_2_.clear();
    }

    if (RandomRay::precursor_method_ == RandomRayPrecursorMethod::INTEGRATION) {
      delayed_fission_source_final_.clear();

      precursors_im1_.clear();
      delayed_fission_source_im1_.clear();
      delayed_fission_source_im2_.clear();
    } else {
      precursors_bd_.clear();
      precursors_rhs_bd_.clear();
    }

    delayed_fission_source_.clear();
    precursors_old_.clear();
    precursors_new_.clear();
    precursors_final_.clear();
    tally_delay_task_.clear();
  }

  tally_task_.clear();
  volume_task_.clear();

  if (settings::source_convergence_method ==
      SourceConvergenceMethod::WINDOW_AVG_RMS)
    batchwise_fission_source_.clear();

  // Fill with copies of source_region
  for (int i = 0; i < n_source_regions; ++i) {
    push_back(source_region);
  }
}

void SourceRegionContainer::flux_swap()
{
  scalar_flux_old_.swap(scalar_flux_new_);
  if (is_linear_) {
    flux_moments_old_.swap(flux_moments_new_);
  }
}

void SourceRegionContainer::mpi_sync_ranks(bool reduce_position)
{
#ifdef OPENMC_MPI

  // The "position_recorded" variable needs to be allreduced (and maxed),
  // as whether or not a cell was hit will affect some decisions in how the
  // source is calculated in the next iteration so as to avoid dividing
  // by zero. We take the max rather than the sum as the hit values are
  // expected to be zero or 1.
  MPI_Allreduce(MPI_IN_PLACE, position_recorded_.data(), 1, MPI_INT, MPI_MAX,
    mpi::intracomm);

  // The position variable is more complicated to reduce than the others,
  // as we do not want the sum of all positions in each cell, rather, we
  // want to just pick any single valid position. Thus, we perform a gather
  // and then pick the first valid position we find for all source regions
  // that have had a position recorded. This operation does not need to
  // be broadcast back to other ranks, as this value is only used for the
  // tally conversion operation, which is only performed on the master rank.
  // While this is expensive, it only needs to be done for active batches,
  // and only if we have not mapped all the tallies yet. Once tallies are
  // fully mapped, then the position vector is fully populated, so this
  // operation can be skipped.

  // Then, we perform the gather of position data, if needed
  if (reduce_position) {

    // Master rank will gather results and pick valid positions
    if (mpi::master) {
      // Initialize temporary vector for receiving positions
      vector<vector<Position>> all_position;
      all_position.resize(mpi::n_procs);
      for (int i = 0; i < mpi::n_procs; i++) {
        all_position[i].resize(n_source_regions_);
      }

      // Copy master rank data into gathered vector for convenience
      all_position[0] = position_;

      // Receive all data into gather vector
      for (int i = 1; i < mpi::n_procs; i++) {
        MPI_Recv(all_position[i].data(), n_source_regions_ * 3, MPI_DOUBLE, i,
          0, mpi::intracomm, MPI_STATUS_IGNORE);
      }

      // Scan through gathered data and pick first valid cell posiiton
      for (int64_t sr = 0; sr < n_source_regions_; sr++) {
        if (position_recorded_[sr] == 1) {
          for (int i = 0; i < mpi::n_procs; i++) {
            if (all_position[i][sr].x != 0.0 || all_position[i][sr].y != 0.0 ||
                all_position[i][sr].z != 0.0) {
              position_[sr] = all_position[i][sr];
              break;
            }
          }
        }
      }
    } else {
      // Other ranks just send in their data
      MPI_Send(position_.data(), n_source_regions_ * 3, MPI_DOUBLE, 0, 0,
        mpi::intracomm);
    }
  }

  // For the rest of the source region data, we simply perform an all reduce,
  // as these values will be needed on all ranks for transport during the
  // next iteration.
  MPI_Allreduce(MPI_IN_PLACE, volume_.data(), n_source_regions_, MPI_DOUBLE,
    MPI_SUM, mpi::intracomm);

  MPI_Allreduce(MPI_IN_PLACE, scalar_flux_new_.data(),
    n_source_regions_ * negroups_, MPI_DOUBLE, MPI_SUM, mpi::intracomm);

  if (settings::run_mode == RunMode::TIME_DEPENDENT) {
    MPI_Allreduce(MPI_IN_PLACE, scalar_flux_td_new_.data(),
      n_source_regions_ * negroups_, MPI_DOUBLE, MPI_SUM, mpi::intracomm);
  }

  if (is_linear_) {
    // We are going to assume we can safely cast Position, MomentArray,
    // and MomentMatrix to contiguous arrays of doubles for the MPI
    // allreduce operation. This is a safe assumption as typically
    // compilers will at most pad to 8 byte boundaries. If a new FP32
    // MomentArray type is introduced, then there will likely be padding, in
    // which case this function will need to become more complex.
    if (sizeof(MomentArray) != 3 * sizeof(double) ||
        sizeof(MomentMatrix) != 6 * sizeof(double)) {
      fatal_error(
        "Unexpected buffer padding in linear source domain reduction.");
    }

    MPI_Allreduce(MPI_IN_PLACE, static_cast<void*>(flux_moments_new_.data()),
      n_source_regions_ * negroups_ * 3, MPI_DOUBLE, MPI_SUM, mpi::intracomm);
    MPI_Allreduce(MPI_IN_PLACE, static_cast<void*>(mom_matrix_.data()),
      n_source_regions_ * 6, MPI_DOUBLE, MPI_SUM, mpi::intracomm);
    MPI_Allreduce(MPI_IN_PLACE, static_cast<void*>(centroid_iteration_.data()),
      n_source_regions_ * 3, MPI_DOUBLE, MPI_SUM, mpi::intracomm);
  }

#endif
}

void SourceRegionContainer::adjoint_reset()
{
  std::fill(volume_.begin(), volume_.end(), 0.0);
  std::fill(volume_t_.begin(), volume_t_.end(), 0.0);
  std::fill(volume_naive_.begin(), volume_naive_.end(), 0.0);
  std::fill(
    external_source_present_.begin(), external_source_present_.end(), 0);
  std::fill(external_source_.begin(), external_source_.end(), 0.0);
  std::fill(centroid_.begin(), centroid_.end(), Position {0.0, 0.0, 0.0});
  std::fill(centroid_iteration_.begin(), centroid_iteration_.end(),
    Position {0.0, 0.0, 0.0});
  std::fill(centroid_t_.begin(), centroid_t_.end(), Position {0.0, 0.0, 0.0});
  std::fill(mom_matrix_.begin(), mom_matrix_.end(),
    MomentMatrix {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  std::fill(mom_matrix_t_.begin(), mom_matrix_t_.end(),
    MomentMatrix {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  if (settings::run_mode == RunMode::FIXED_SOURCE) {
    std::fill(scalar_flux_old_.begin(), scalar_flux_old_.end(), 0.0);
  } else {
    std::fill(scalar_flux_old_.begin(), scalar_flux_old_.end(), 1.0);
  }
  std::fill(scalar_flux_new_.begin(), scalar_flux_new_.end(), 0.0);
  std::fill(source_.begin(), source_.end(), 0.0f);
  std::fill(external_source_.begin(), external_source_.end(), 0.0f);
  std::fill(source_gradients_.begin(), source_gradients_.end(),
    MomentArray {0.0, 0.0, 0.0});
  std::fill(flux_moments_old_.begin(), flux_moments_old_.end(),
    MomentArray {0.0, 0.0, 0.0});
  std::fill(flux_moments_new_.begin(), flux_moments_new_.end(),
    MomentArray {0.0, 0.0, 0.0});
  std::fill(flux_moments_t_.begin(), flux_moments_t_.end(),
    MomentArray {0.0, 0.0, 0.0});

  // Time-dependent arrays
  if (settings::run_mode == RunMode::TIME_DEPENDENT) {
    std::fill(scalar_flux_td_old_.begin(), scalar_flux_td_old_.end(), 0.0);
    std::fill(scalar_flux_td_new_.begin(), scalar_flux_td_new_.end(), 0.0);
    std::fill(precursors_old_.begin(), precursors_old_.end(), 0.0);
    std::fill(precursors_new_.begin(), precursors_new_.end(), 0.0);

    // BD Vectors
    std::fill(scalar_flux_rhs_bd_.begin(), scalar_flux_rhs_bd_.end(), 0.0);

    if (RandomRay::time_method_ == RandomRayTimeMethod::SDP) {
      std::fill(
        source_time_derivative_.begin(), source_time_derivative_.end(), 0.0);
      std::fill(scalar_flux_time_derivative_2_.begin(),
        scalar_flux_time_derivative_2_.end(), 0.0);

      std::fill(source_rhs_bd_.begin(), source_rhs_bd_.end(), 0.0);
      std::fill(
        scalar_flux_rhs_bd_2_.begin(), scalar_flux_rhs_bd_2_.end(), 0.0);
    }
    if (RandomRay::precursor_method_ == RandomRayPrecursorMethod::INTEGRATION) {
      std::fill(
        delayed_fission_source_.begin(), delayed_fission_source_.end(), 0.0);

      std::fill(precursors_im1_.begin(), precursors_im1_.end(), 0.0);
      std::fill(delayed_fission_source_im1_.begin(),
        delayed_fission_source_im1_.end(), 0.0);
      std::fill(delayed_fission_source_im2_.begin(),
        delayed_fission_source_im2_.end(), 0.0);
    } else {
      std::fill(precursors_rhs_bd_.begin(), precursors_rhs_bd_.end(), 0.0);
    }
  }
}

// Time-dependent methods
void SourceRegionContainer::flux_td_swap()
{
  scalar_flux_td_old_.swap(scalar_flux_td_new_);
  // TODO: Add support for linear source regions
}

void SourceRegionContainer::precursors_swap()
{
  precursors_old_.swap(precursors_new_);
}

void SourceRegionContainer::time_step_reset()
{
  std::fill(scalar_flux_final_.begin(), scalar_flux_final_.end(), 0.0);
  std::fill(scalar_flux_td_final_.begin(), scalar_flux_td_final_.end(), 0.0);
  std::fill(precursors_final_.begin(), precursors_final_.end(), 0.0);
  if (RandomRay::time_method_ == RandomRayTimeMethod::SDP)
    std::fill(source_td_final_.begin(), source_td_final_.end(), 0.0);
  if (RandomRay::precursor_method_ == RandomRayPrecursorMethod::INTEGRATION)
    std::fill(delayed_fission_source_final_.begin(), source_final_.end(), 0.0);
}

} // namespace openmc
  
