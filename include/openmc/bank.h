#ifndef OPENMC_BANK_H
#define OPENMC_BANK_H

#include <cstdint>

#include "openmc/particle.h"
#include "openmc/position.h"
#include "openmc/shared_array.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace simulation {

extern vector<SourceSite> source_bank;

extern vector<SourceSite> precursor_source_bank;

extern SharedArray<SourceSite> surf_source_bank;

extern SharedArray<CollisionTrackSite> collision_track_bank;

extern SharedArray<SourceSite> fission_bank;

extern vector<vector<int>> ifp_source_delayed_group_bank;

extern vector<vector<double>> ifp_source_lifetime_bank;

extern vector<vector<int>> ifp_fission_delayed_group_bank;

extern vector<vector<double>> ifp_fission_lifetime_bank;

extern SharedArray<SourceSite> time_census_bank;

extern SharedArray<SourceSite> precursor_shared_bank;

extern vector<int64_t> progeny_per_particle;

extern vector<int64_t> precursors_per_particle;

} // namespace simulation

//==============================================================================
// Non-member functions
//==============================================================================

void sort_census_bank(SharedArray<SourceSite>& census_bank, vector<int64_t>& progeny_per_particle, vector<int64_t>& work_index);

void free_memory_bank();

void init_census_bank(SharedArray<SourceSite>& census_bank, int64_t max, vector<int64_t>& progeny_per_particle, int64_t& work_per_rank);

//! Sample/redistribute source sites from accumulated fission sites
void synchronize_bank(SharedArray<SourceSite>& census_bank);

} // namespace openmc

#endif // OPENMC_BANK_H
