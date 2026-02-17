#include "openmc/bank.h"
#include "openmc/capi.h"
#include "openmc/error.h"
#include "openmc/ifp.h"
#include "openmc/message_passing.h"
#include "openmc/simulation.h"
#include "openmc/timer.h"
#include "openmc/vector.h"

#include <cstdint>
#include <numeric>

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace simulation {

vector<SourceSite> source_bank;

vector<SourceSite> precursor_source_bank;

SharedArray<SourceSite> surf_source_bank;

SharedArray<CollisionTrackSite> collision_track_bank;

// The fission bank is allocated as a SharedArray, rather than a vector, as it
// will be shared by all threads in the simulation. It will be allocated to a
// fixed maximum capacity in the init_fission_bank() function. Then, Elements
// will be added to it by using SharedArray's special thread_safe_append()
// function.
SharedArray<SourceSite> fission_bank;

vector<vector<int>> ifp_source_delayed_group_bank;

vector<vector<double>> ifp_source_lifetime_bank;

vector<vector<int>> ifp_fission_delayed_group_bank;

vector<vector<double>> ifp_fission_lifetime_bank;

vector<SourceSite> initial_source_bank;

// The time census bank is allocated as a SharedArray, rather than a vector, as
// it will be shared by all threads in the simulation. It will be allocated to a
// fixed maximum capacity in the init_fission_bank() function. Then, Elements
// will be added to it by using SharedArray's special thread_safe_append()
// function.
SharedArray<SourceSite> time_census_bank;

// The future bank tracks particles that have times greater than the current
// census time boundary
SharedArray<SourceSite> future_bank;

// The precursor particle bank tracks precursor particles that serve as source
// sites for delayed neutrons when using forced decay (Which is only done in
// kinetic simuations)
SharedArray<SourceSite> precursor_shared_bank;

// Each entry in this vector corresponds to the number of neutrons produced
// this generation for the particle located at that index. This vector is
// used to efficiently sort the census bank after each iteration.
vector<int64_t> progeny_per_particle;

// Each entry in this vector corresponds to the number of precursors produced
// this generation for the particle located at that index. This vector is
// used to efficiently sort the precursor bank after each time step.
vector<int64_t> precursors_per_particle;

} // namespace simulation

//==============================================================================
// Non-member functions
//==============================================================================

void free_memory_bank()
{
  simulation::source_bank.clear();
  simulation::surf_source_bank.clear();
  simulation::collision_track_bank.clear();
  simulation::fission_bank.clear();
  simulation::progeny_per_particle.clear();
  simulation::ifp_source_delayed_group_bank.clear();
  simulation::ifp_source_lifetime_bank.clear();
  simulation::ifp_fission_delayed_group_bank.clear();
  simulation::ifp_fission_lifetime_bank.clear();
  simulation::time_census_bank.clear();
  simulation::precursor_source_bank.clear();
  simulation::precursor_shared_bank.clear();
  simulation::precursors_per_particle.clear();
}

void init_census_bank(SharedArray<SourceSite>& census_bank,
  vector<int64_t>& progeny_per_particle, int64_t& work_per_rank)
{
  census_bank.reserve(3 * work_per_rank);
  progeny_per_particle.resize(work_per_rank);
}

// Performs an O(n) sort on a census bank, by leveraging
// the parent_id and progeny_id fields of banked particles. See the following
// paper for more details:
// "Reproducibility and Monte Carlo Eigenvalue Calculations," F.B. Brown and
// T.M. Sutton, 1992 ANS Annual Meeting, Transactions of the American Nuclear
// Society, Volume 65, Page 235.
void sort_census_bank(SharedArray<SourceSite>& census_bank,
  vector<int64_t>& progeny_per_particle, vector<int64_t>& work_index)
{
  // Ensure we don't read off the end of the array if we ran with 0 particles
  if (progeny_per_particle.size() == 0) {
    return;
  }

  // Perform exclusive scan summation to determine starting indices in census
  // bank for each parent particle id
  std::exclusive_scan(progeny_per_particle.begin(), progeny_per_particle.end(),
    progeny_per_particle.begin(), 0);

  // We need a scratch vector to make permutation of the census bank into
  // sorted order easy. Under normal usage conditions, the census bank is
  // over provisioned, so we can use that as scratch space.
  SourceSite* sorted_bank;
  vector<SourceSite> sorted_bank_holder;
  vector<vector<int>> sorted_ifp_delayed_group_bank;
  vector<vector<double>> sorted_ifp_lifetime_bank;

  // If there is not enough space, allocate a temporary vector and point to it
  if (census_bank.size() > census_bank.capacity() / 2) {
    sorted_bank_holder.resize(census_bank.size());
    sorted_bank = sorted_bank_holder.data();
  } else { // otherwise, point sorted_bank to unused portion of the census bank
    sorted_bank = &census_bank[census_bank.size()];
  }

  // TODO: DISABLE FOR TIME CENSUS
  if (settings::ifp_on) {
    allocate_temporary_vector_ifp(
      sorted_ifp_delayed_group_bank, sorted_ifp_lifetime_bank);
  }

  // Use parent and progeny indices to sort census bank
  for (int64_t i = 0; i < census_bank.size(); i++) {
    const auto& site = census_bank[i];
    int64_t offset = site.parent_id - 1 - work_index[mpi::rank];
    int64_t idx = progeny_per_particle[offset] + site.progeny_id;
    if (idx >= census_bank.size()) {
      fatal_error("Mismatch detected between sum of all particle progeny and "
                  "shared census bank size.");
    }
    sorted_bank[idx] = site;
    // TODO: disable for time census bank?
    if (settings::ifp_on) {
      copy_ifp_data_from_fission_banks(
        i, sorted_ifp_delayed_group_bank[idx], sorted_ifp_lifetime_bank[idx]);
    }
  }

  // Copy sorted bank into the census bank
  std::copy(sorted_bank, sorted_bank + census_bank.size(), census_bank.data());
  // TODO: disable for time census bank?
  if (settings::ifp_on) {
    copy_ifp_data_to_fission_banks(
      sorted_ifp_delayed_group_bank.data(), sorted_ifp_lifetime_bank.data());
  }
}

// TODO replace settings::n_particles with target_size, and work_index with
// another vector...
void synchronize_bank(SharedArray<SourceSite>& census_bank,
  vector<SourceSite>& source_bank, int64_t& n_particles, int64_t& work_per_rank,
  vector<int64_t>& work_index)
{
  simulation::time_bank.start();

  // In order to properly understand the census bank algorithm, you need to
  // think of the census and source bank as being one global array divided
  // over multiple processors. At the start, each processor has a random amount
  // of census bank sites -- each processor needs to know the total number of
  // sites in order to figure out the probability for selecting
  // sites. Furthermore, each proc also needs to know where in the 'global'
  // census bank its own sites starts in order to ensure reproducibility by
  // skipping ahead to the proper seed.

#ifdef OPENMC_MPI
  int64_t start = 0;
  int64_t n_bank = census_bank.size();
  MPI_Exscan(&n_bank, &start, 1, MPI_INT64_T, MPI_SUM, mpi::intracomm);

  // While we would expect the value of start on rank 0 to be 0, the MPI
  // standard says that the receive buffer on rank 0 is undefined and not
  // significant
  if (mpi::rank == 0)
    start = 0;

  int64_t finish = start + census_bank.size();
  int64_t total = finish;
  MPI_Bcast(&total, 1, MPI_INT64_T, mpi::n_procs - 1, mpi::intracomm);

#else
  int64_t start = 0;
  int64_t finish = census_bank.size();
  int64_t total = finish;
#endif

  // If there are not that many particles per generation, it's possible that no
  // census sites were created at all on a single processor. Rather than add
  // extra logic to treat this circumstance, we really want to ensure the user
  // runs enough particles to avoid this in the first place.

  if (census_bank.size() == 0) {
    fatal_error(
      "No census sites banked on MPI rank " + std::to_string(mpi::rank));
  }

  simulation::time_bank_sample.start();

  // Allocate temporary source bank -- we don't really know how many census
  // sites were created, so overallocate by a factor of 3
  int64_t index_temp = 0;

  vector<SourceSite> temp_sites(3 * work_per_rank);

  // Temporary banks for IFP
  vector<vector<int>> temp_delayed_groups;
  vector<vector<double>> temp_lifetimes;
  // TODO: DISABLE FOR TIME CENSUS
  if (settings::ifp_on) {
    resize_ifp_data(temp_delayed_groups, temp_lifetimes, 3 * work_per_rank);
  }

  // ==========================================================================
  // SAMPLE N_PARTICLES FROM CENSUS BANK AND PLACE IN TEMP_SITES

  // We use Uniform Combing method to exactly get the targeted particle size
  // [https://doi.org/10.1080/00295639.2022.2091906]

  // Make sure all processors use the same random number seed.
  int64_t id = simulation::total_gen + overall_generation();
  uint64_t seed = init_seed(id, STREAM_TRACKING);

  // Comb specification
  double teeth_distance = static_cast<double>(total) / n_particles;
  double teeth_offset = prn(&seed) * teeth_distance;

  // First and last hitting tooth
  int64_t end = start + census_bank.size();
  int64_t tooth_start = std::ceil((start - teeth_offset) / teeth_distance);
  int64_t tooth_end = std::floor((end - teeth_offset) / teeth_distance) + 1;

  // Locally comb particles in census_bank
  double tooth = tooth_start * teeth_distance + teeth_offset;
  for (int64_t i = tooth_start; i < tooth_end; i++) {
    int64_t idx = std::floor(tooth) - start;
    temp_sites[index_temp] = census_bank[idx];
    // TODO: disable for time census bank
    if (settings::ifp_on) {
      copy_ifp_data_from_fission_banks(
        idx, temp_delayed_groups[index_temp], temp_lifetimes[index_temp]);
    }
    ++index_temp;

    // Next tooth
    tooth += teeth_distance;
  }

  // At this point, the sampling of source sites is done and now we need to
  // figure out where to send source sites. Since it is possible that one
  // processor's share of the source bank spans more than just the immediate
  // neighboring processors, we have to perform an ALLGATHER to determine the
  // indices for all processors

#ifdef OPENMC_MPI
  // First do an exclusive scan to get the starting indices for
  start = 0;
  MPI_Exscan(&index_temp, &start, 1, MPI_INT64_T, MPI_SUM, mpi::intracomm);
  finish = start + index_temp;

  // TODO: protect for MPI_Exscan at rank 0

  // Allocate space for bank_position if this hasn't been done yet
  int64_t bank_position[mpi::n_procs];
  MPI_Allgather(
    &start, 1, MPI_INT64_T, bank_position, 1, MPI_INT64_T, mpi::intracomm);
#else
  start = 0;
  finish = index_temp;
#endif

  simulation::time_bank_sample.stop();
  simulation::time_bank_sendrecv.start();

#ifdef OPENMC_MPI
  // ==========================================================================
  // SEND BANK SITES TO NEIGHBORS

  // IFP number of generation
  int ifp_n_generation;
  // TODO: DISABLE FOR TIME CENSUS
  if (settings::ifp_on) {
    broadcast_ifp_n_generation(
      ifp_n_generation, temp_delayed_groups, temp_lifetimes);
  }

  int64_t index_local = 0;
  vector<MPI_Request> requests;

  // IFP send buffers
  vector<int> send_delayed_groups;
  vector<double> send_lifetimes;

  if (start < n_particles) {
    // Determine the index of the processor which has the first part of the
    // source_bank for the local processor
    int neighbor =
      upper_bound_index(work_index.begin(), work_index.end(), start);

    // Resize IFP send buffers
    // TODO: DISABLE FOR TIME CENSUS
    if (settings::ifp_on && mpi::n_procs > 1) {
      resize_ifp_data(send_delayed_groups, send_lifetimes,
        ifp_n_generation * 3 * work_per_rank);
    }

    while (start < finish) {
      // Determine the number of sites to send
      int64_t n = std::min(work_index[neighbor + 1], finish) - start;

      // Initiate an asynchronous send of source sites to the neighboring
      // process
      if (neighbor != mpi::rank) {
        requests.emplace_back();
        MPI_Isend(&temp_sites[index_local], static_cast<int>(n),
          mpi::source_site, neighbor, mpi::rank, mpi::intracomm,
          &requests.back());

        // TODO: DISABLE FOR TIME CENSUS
        if (settings::ifp_on) {
          // Send IFP data
          if (is_beta_effective_or_both())
            send_ifp_info(index_local, n, ifp_n_generation, neighbor, requests,
              temp_delayed_groups, send_delayed_groups);
          if (is_generation_time_or_both())
            send_ifp_info(index_local, n, ifp_n_generation, neighbor, requests,
              temp_lifetimes, send_lifetimes);
        }
      }

      // Increment all indices
      start += n;
      index_local += n;
      ++neighbor;

      // Check for sites out of bounds -- this only happens in the rare
      // circumstance that a processor close to the end has so many sites that
      // it would exceed the bank on the last processor
      if (neighbor > mpi::n_procs - 1)
        break;
    }
  }

  // ==========================================================================
  // RECEIVE BANK SITES FROM NEIGHBORS OR TEMPORARY BANK

  start = work_index[mpi::rank];
  index_local = 0;

  // IFP receive buffers
  vector<int> recv_delayed_groups;
  vector<double> recv_lifetimes;
  vector<DeserializationInfo> deserialization_info;

  // Determine what process has the source sites that will need to be stored at
  // the beginning of this processor's source bank.

  int neighbor;
  if (start >= bank_position[mpi::n_procs - 1]) {
    neighbor = mpi::n_procs - 1;
  } else {
    neighbor =
      upper_bound_index(bank_position, bank_position + mpi::n_procs, start);
  }

  // Resize IFP receive buffers
  // TODO: DISABLE FOR TIME CENSUS
  if (settings::ifp_on && mpi::n_procs > 1) {
    resize_ifp_data(
      recv_delayed_groups, recv_lifetimes, ifp_n_generation * work_per_rank);
  }

  while (start < work_index[mpi::rank + 1]) {
    // Determine how many sites need to be received
    int64_t n;
    if (neighbor == mpi::n_procs - 1) {
      n = work_index[mpi::rank + 1] - start;
    } else {
      n = std::min(bank_position[neighbor + 1], work_index[mpi::rank + 1]) -
          start;
    }

    if (neighbor != mpi::rank) {
      // If the source sites are not on this processor, initiate an
      // asynchronous receive for the source sites

      requests.emplace_back();
      MPI_Irecv(&source_bank[index_local], static_cast<int>(n),
        mpi::source_site, neighbor, neighbor, mpi::intracomm, &requests.back());

      // TODO: DISABLE FOR TIME CENSUS
      if (settings::ifp_on) {
        // Receive IFP data
        if (is_beta_effective_or_both())
          receive_ifp_data(index_local, n, ifp_n_generation, neighbor, requests,
            recv_delayed_groups, deserialization_info);
        if (is_generation_time_or_both())
          receive_ifp_data(index_local, n, ifp_n_generation, neighbor, requests,
            recv_lifetimes, deserialization_info);
      }

    } else {
      // If the source sites are on this processor, we can simply copy them
      // from the temp_sites bank

      index_temp = start - bank_position[mpi::rank];
      std::copy(&temp_sites[index_temp], &temp_sites[index_temp + n],
        &source_bank[index_local]);

      // TODO: DISABLE FOR TIME CENSUS
      if (settings::ifp_on) {
        copy_partial_ifp_data_to_source_banks(
          index_temp, n, index_local, temp_delayed_groups, temp_lifetimes);
      }
    }

    // Increment all indices
    start += n;
    index_local += n;
    ++neighbor;
  }

  // Since we initiated a series of asynchronous ISENDs and IRECVs, now we have
  // to ensure that the data has actually been communicated before moving on to
  // the next generation

  int n_request = requests.size();
  MPI_Waitall(n_request, requests.data(), MPI_STATUSES_IGNORE);

  // TODO: DISABLE FOR TIME CENSUS
  if (settings::ifp_on) {
    if (is_beta_effective_or_both())
      deserialize_ifp_info(ifp_n_generation, recv_delayed_groups,
        simulation::ifp_source_delayed_group_bank, deserialization_info);
    if (is_generation_time_or_both())
      deserialize_ifp_info(ifp_n_generation, recv_lifetimes,
        simulation::ifp_source_lifetime_bank, deserialization_info);
  }

#else
  std::copy(
    temp_sites.data(), temp_sites.data() + n_particles, source_bank.begin());
  // TODO: DISABLE FOR TIME CENSUS
  if (settings::ifp_on) {
    copy_complete_ifp_data_to_source_banks(temp_delayed_groups, temp_lifetimes);
  }
#endif

  simulation::time_bank_sendrecv.stop();
  simulation::time_bank.stop();
}

//==============================================================================
// C API
//==============================================================================

extern "C" int openmc_source_bank(void** ptr, int64_t* n)
{
  if (!ptr || !n) {
    set_errmsg("Received null pointer.");
    return OPENMC_E_INVALID_ARGUMENT;
  }

  if (simulation::source_bank.size() == 0) {
    set_errmsg("Source bank has not been allocated.");
    return OPENMC_E_ALLOCATE;
  } else {
    *ptr = simulation::source_bank.data();
    *n = simulation::source_bank.size();
    return 0;
  }
}

extern "C" int openmc_fission_bank(void** ptr, int64_t* n)
{
  if (!ptr || !n) {
    set_errmsg("Received null pointer.");
    return OPENMC_E_INVALID_ARGUMENT;
  }

  if (simulation::fission_bank.size() == 0) {
    set_errmsg("Fission bank has not been allocated.");
    return OPENMC_E_ALLOCATE;
  } else {
    *ptr = simulation::fission_bank.data();
    *n = simulation::fission_bank.size();
    return 0;
  }
}

} // namespace openmc
