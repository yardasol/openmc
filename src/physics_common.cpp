#include "openmc/physics_common.h"

#include "openmc/random_lcg.h"
#include "openmc/settings.h"

namespace openmc {

//==============================================================================
// RUSSIAN_ROULETTE
//==============================================================================

void russian_roulette(Particle& p, double weight_survive)
{
  if (weight_survive * prn(p.current_seed()) < p.wgt()) {
    p.wgt() = weight_survive;
  } else {
    p.wgt() = 0.;
  }
}

void apply_russian_roulette(Particle& p)
{
  // Exit if survival biasing or branchless collision is turned off
  if (!(settings::survival_biasing || settings::branchless_collision))
    return;

  // if survival normalization is on, use normalized weight cutoff and
  // normalized weight survive
  if (settings::survival_normalization) {
    if (p.wgt() < settings::roulette_weight_cutoff * p.wgt_born()) {
      russian_roulette(p, settings::weight_survive * p.wgt_born());
    }
  } else if (p.wgt() < settings::roulette_weight_cutoff) {
    russian_roulette(p, settings::weight_survive);
  }
}

void split(Particle& p, double target_weight, int max_split)
{
  double weight = p.wgt();
  // do not further split the particle if above the history splitting limit
  if (p.n_split() >= settings::max_history_splits)
    return;

  int n_split = std::ceil(weight / target_weight);
  n_split = std::min(n_split, max_split);

  p.n_split() += n_split;

  // Create secondaries and divide weight evenly among all particles
  int i_split = std::round(n_split);
  for (int l = 0; l < i_split - 1; l++) {
    p.split(weight / n_split);
  }
  // remaining weight is applied to current particle
  p.wgt() = weight / n_split;
}

void apply_splitting(Particle& p)
{
  // Exit if survival biasing or branchless collision is turned off
  if (!(settings::survival_biasing || settings::branchless_collision))
    return;

  if (p.wgt() > settings::splitting_weight_cutoff) {
    split(p, settings::weight_split, settings::max_split);
  }
}
} // namespace openmc
