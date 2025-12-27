#ifndef OPENMC_MCPL_INTERFACE_H
#define OPENMC_MCPL_INTERFACE_H

#include "openmc/particle_data.h"
#include "openmc/span.h"
#include "openmc/vector.h"

#include <string>

namespace openmc {

//==============================================================================
// Constants
//==============================================================================

extern "C" const bool MCPL_ENABLED;

//==============================================================================
// Functions
//==============================================================================

//! Get a vector of source sites from an MCPL file
//
//! \param[in] path  Path to MCPL file
//! \return  Vector of source sites
vector<SourceSite> mcpl_source_sites(std::string path);

//! Write an MCPL source file
//
//! \param[in] filename     Path to MCPL file
//! \param[in] source_bank  Vector of SourceSites to write to file for this
//!                         MPI rank. Note that this can't be const due to
//!                         it being used as work space by MPI.
//! \param[in] bank_indx    Pointer to vector of site index ranges over all
//!                         MPI ranks. This can be computed by calling
//!                         calculate_parallel_index_vector on
//!                         source_bank.size().
void write_mcpl_source_point(const char* filename, span<SourceSite> source_bank,
  const vector<int64_t>& bank_index);

//! Write an MCPL collision track file
//!
//! This function writes collision track data to an MCPL file. Additional
//! collision-specific metadata (such as energy deposition, material info, etc.)
//! is stored in the file header as blob data.
//!
//! \param[in] filename     Path to MCPL file
//! \param[in] collision_track_bank  Vector of CollisionTrackSites to write to
//!                         file for this MPI rank.
//! \param[in] bank_index   Pointer to vector of site index ranges over all
//!                         MPI ranks.
void write_mcpl_collision_track(const char* filename,
  span<CollisionTrackSite> collision_track_bank,
  const vector<int64_t>& bank_index);

//! Check if MCPL functionality is available
bool is_mcpl_interface_available();

//! Initialize the MCPL interface
void initialize_mcpl_interface_if_needed();

} // namespace openmc

#endif // OPENMC_MCPL_INTERFACE_H
