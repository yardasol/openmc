//! \file physics_common.h
//! A collection of physics methods common to MG, CE, photon, etc.

#ifndef OPENMC_PHYSICS_COMMON_H
#define OPENMC_PHYSICS_COMMON_H

#include "openmc/particle.h"

namespace openmc {

//! \brief Performs the russian roulette operation for a particle
//! \param[in,out] p  Particle object
//! \param[in] weight_survive Weight assigned to particles that survive
void russian_roulette(Particle& p, double weight_survive);

//! \brief Performs the split operation for a particle
//! \param[in,out] p  Particle object
//! \param[in] target_weight Weight assigned to split particles
//! \param[in] max_split Maximum possible particles produced with splitting
void split(Particle& p, double target_weight, int max_split);

//! \brief Performs the global russian roulette operation on a particle
//! \param[in,out] p  Particle object
void apply_russian_roulette(Particle& p);

} // namespace openmc
#endif // OPENMC_PHYSICS_COMMON_H
