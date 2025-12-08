#include "openmc/random_ray/random_ray.h"

#include "openmc/constants.h"
#include "openmc/geometry.h"
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/random_ray/flat_source_domain.h"
#include "openmc/random_ray/linear_source_domain.h"
#include "openmc/search.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/source.h"

namespace openmc {

//==============================================================================
// Non-method functions
//==============================================================================

// returns 1 - exp(-tau)
// Equivalent to -(_expm1f(-tau)), but faster
// Written by Colin Josey.
double cjosey_exponential(double tau)
{
  constexpr double c1n = -1.0000013559236386308;
  constexpr double c2n = 0.23151368626911062025;
  constexpr double c3n = -0.061481916409314966140;
  constexpr double c4n = 0.0098619906458127653020;
  constexpr double c5n = -0.0012629460503540849940;
  constexpr double c6n = 0.00010360973791574984608;
  constexpr double c7n = -0.000013276571933735820960;

  constexpr double c0d = 1.0;
  constexpr double c1d = -0.73151337729389001396;
  constexpr double c2d = 0.26058381273536471371;
  constexpr double c3d = -0.059892419041316836940;
  constexpr double c4d = 0.0099070188241094279067;
  constexpr double c5d = -0.0012623388962473160860;
  constexpr double c6d = 0.00010361277635498731388;
  constexpr double c7d = -0.000013276569500666698498;

  double x = -tau;

  double den = c7d;
  den = den * x + c6d;
  den = den * x + c5d;
  den = den * x + c4d;
  den = den * x + c3d;
  den = den * x + c2d;
  den = den * x + c1d;
  den = den * x + c0d;

  double num = c7n;
  num = num * x + c6n;
  num = num * x + c5n;
  num = num * x + c4n;
  num = num * x + c3n;
  num = num * x + c2n;
  num = num * x + c1n;
  num = num * x;

  return num / den;
}

// The below two functions (exponentialG and exponentialG2) were developed
// by Colin Josey. The implementation of these functions is closely based
// on the OpenMOC versions of these functions. The OpenMOC license is given
// below:

// Copyright (C) 2012-2023 Massachusetts Institute of Technology and OpenMOC
// contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Computes y = 1/x-(1-exp(-x))/x**2 using a 5/6th order rational
// approximation. It is accurate to 2e-7 over [0, 1e5]. Developed by Colin
// Josey using Remez's algorithm, with original implementation in OpenMOC at:
// https://github.com/mit-crpg/OpenMOC/blob/develop/src/exponentials.h
double exponentialG(double tau)
{
  // Numerator coefficients in rational approximation for 1/x - (1 - exp(-x)) /
  // x^2
  constexpr double d0n = 0.5;
  constexpr double d1n = 0.176558112351595;
  constexpr double d2n = 0.04041584305811143;
  constexpr double d3n = 0.006178333902037397;
  constexpr double d4n = 0.0006429894635552992;
  constexpr double d5n = 0.00006064409107557148;

  // Denominator coefficients in rational approximation for 1/x - (1 - exp(-x))
  // / x^2
  constexpr double d0d = 1.0;
  constexpr double d1d = 0.6864462055546078;
  constexpr double d2d = 0.2263358514260129;
  constexpr double d3d = 0.04721469893686252;
  constexpr double d4d = 0.006883236664917246;
  constexpr double d5d = 0.0007036272419147752;
  constexpr double d6d = 0.00006064409107557148;

  double x = tau;

  double num = d5n;
  num = num * x + d4n;
  num = num * x + d3n;
  num = num * x + d2n;
  num = num * x + d1n;
  num = num * x + d0n;

  double den = d6d;
  den = den * x + d5d;
  den = den * x + d4d;
  den = den * x + d3d;
  den = den * x + d2d;
  den = den * x + d1d;
  den = den * x + d0d;

  return num / den;
}

// Computes G2 : y = 2/3 - (1 + 2/x) * (1/x + 0.5 - (1 + 1/x) * (1-exp(-x)) /
// x) using a 5/5th order rational approximation. It is accurate to 1e-6 over
// [0, 1e6]. Developed by Colin Josey using Remez's algorithm, with original
// implementation in OpenMOC at:
// https://github.com/mit-crpg/OpenMOC/blob/develop/src/exponentials.h
double exponentialG2(double tau)
{

  // Coefficients for numerator in rational approximation
  constexpr double g1n = -0.08335775885589858;
  constexpr double g2n = -0.003603942303847604;
  constexpr double g3n = 0.0037673183263550827;
  constexpr double g4n = 0.00001124183494990467;
  constexpr double g5n = 0.00016837426505799449;

  // Coefficients for denominator in rational approximation
  constexpr double g1d = 0.7454048371823628;
  constexpr double g2d = 0.23794300531408347;
  constexpr double g3d = 0.05367250964303789;
  constexpr double g4d = 0.006125197988351906;
  constexpr double g5d = 0.0010102514456857377;

  double x = tau;

  double num = g5n;
  num = num * x + g4n;
  num = num * x + g3n;
  num = num * x + g2n;
  num = num * x + g1n;
  num = num * x;

  double den = g5d;
  den = den * x + g4d;
  den = den * x + g3d;
  den = den * x + g2d;
  den = den * x + g1d;
  den = den * x + 1.0;

  return num / den;
}

//==============================================================================
// RandomRay implementation
//==============================================================================

// Static Variable Declarations
double RandomRay::distance_inactive_;
double RandomRay::distance_active_;
double RandomRay::avg_miss_rate_;
int RandomRay::bd_order_ {1};
unique_ptr<Source> RandomRay::ray_source_;
RandomRaySourceShape RandomRay::source_shape_ {RandomRaySourceShape::FLAT};
RandomRayTimeMethod RandomRay::time_method_ {RandomRayTimeMethod::TI};
RandomRayPrecursorMethod RandomRay::precursor_method_ {RandomRayPrecursorMethod::BD};
int64_t RandomRay::n_source_regions_;
int64_t RandomRay::n_external_source_regions_;
uint64_t RandomRay::total_geometric_intersections_;

RandomRay::RandomRay()
  : angular_flux_(data::mg.num_energy_groups_),
    delta_psi_(data::mg.num_energy_groups_),
    negroups_(data::mg.num_energy_groups_),
    angular_flux_td_(data::mg.num_energy_groups_),
    delta_psi_td_(data::mg.num_energy_groups_),
    angular_flux_td_prime_(data::mg.num_energy_groups_),
    delta_psi_td_prime_(data::mg.num_energy_groups_)
{
  if (source_shape_ == RandomRaySourceShape::LINEAR ||
      source_shape_ == RandomRaySourceShape::LINEAR_XY) {
    delta_moments_.resize(negroups_);
  }
}

RandomRay::RandomRay(uint64_t ray_id, FlatSourceDomain* domain) : RandomRay()
{
  initialize_ray(ray_id, domain);
}

// Transports ray until termination criteria are met
uint64_t RandomRay::transport_history_based_single_ray()
{
  using namespace openmc;
  while (alive()) {
    event_advance_ray();
    if (!alive())
      break;
    event_cross_surface();
  }

  return n_event();
}

// Transports ray across a single source region
void RandomRay::event_advance_ray()
{
  // Find the distance to the nearest boundary
  boundary() = distance_to_boundary(*this);
  double distance = boundary().distance;

  if (distance <= 0.0) {
    mark_as_lost("Negative transport distance detected for particle " +
                 std::to_string(id()));
    return;
  }

  bool td_transport = settings::run_mode == RunMode::TIME_DEPENDENT;
  if (is_active_) {
    // If the ray is in the active length, need to check if it has
    // reached its maximum termination distance. If so, reduce
    // the ray traced length so that the ray does not overrun the
    // maximum numerical length (so as to avoid numerical bias).
    if (distance_travelled_ + distance >= distance_active_) {
      distance = distance_active_ - distance_travelled_;
      wgt() = 0.0;
    }

    distance_travelled_ += distance;
    attenuate_flux(distance, true, td_transport);
  } else {
    // If the ray is still in the dead zone, need to check if it
    // has entered the active phase. If so, split into two segments (one
    // representing the final part of the dead zone, the other representing the
    // first part of the active length) and attenuate each. Otherwise, if the
    // full length of the segment is within the dead zone, attenuate as normal.
    if (distance_travelled_ + distance >= distance_inactive_) {
      is_active_ = true;
      double distance_dead = distance_inactive_ - distance_travelled_;
      attenuate_flux(distance_dead, false, td_transport);

      double distance_alive = distance - distance_dead;

      // Ensure we haven't travelled past the active phase as well
      if (distance_alive > distance_active_) {
        distance_alive = distance_active_;
        wgt() = 0.0;
      }

      attenuate_flux(distance_alive, true, td_transport);
      distance_travelled_ = distance_alive;
    } else {
      distance_travelled_ += distance;
      attenuate_flux(distance, false, td_transport);
    }
  }

  // Advance particle
  for (int j = 0; j < n_coord(); ++j) {
    coord(j).r += distance * coord(j).u;
  }
}

void RandomRay::attenuate_flux(double distance, bool is_active, bool td_transport)
{
  switch (source_shape_) {
  case RandomRaySourceShape::FLAT:
    attenuate_flux_flat_source(distance, is_active, td_transport);
    break;
  case RandomRaySourceShape::LINEAR:
  case RandomRaySourceShape::LINEAR_XY:
    //TODO: implement td transport for linear sources
    if (td_transport) {
      fatal_error("LINEAR and LINEAR_XY source shapes unimplemented for time-dependent transport.");
    } else {
      attenuate_flux_linear_source(distance, is_active);
    }
    break;
  default:
    fatal_error("Unknown source shape for random ray transport.");
  }
}

// This function forms the inner loop of the random ray transport process.
// It is responsible for several tasks. Based on the incoming angular flux
// of the ray and the source term in the region, the outgoing angular flux
// is computed. The delta psi between the incoming and outgoing fluxes is
// contributed to the estimate of the total scalar flux in the source region.
// Additionally, the contribution of the ray path to the stochastically
// estimated volume is also kept track of. All tasks involving writing
// to the data for the source region are done with a lock over the entire
// source region.  Locks are used instead of atomics as all energy groups
// must be written, such that locking once is typically much more efficient
// than use of many atomic operations corresponding to each energy group
// individually (at least on CPU). Several other bookkeeping tasks are also
// performed when inside the lock.
void RandomRay::attenuate_flux_flat_source(double distance, bool is_active, bool td_transport)
{
  // The number of geometric intersections is counted for reporting purposes
  n_event()++;

  // Determine source region index etc.
  int i_cell = lowest_coord().cell;

  // The source region is the spatial region index
  int64_t sr = domain_->source_region_offsets_[i_cell] + cell_instance();

  // The source element is the energy-specific region index
  int material = this->material();

  // MOC incoming flux attenuation + source contribution/attenuation equation
  for (int g = 0; g < negroups_; g++) {
    double sigma_t = domain_->sigma_t_[material * negroups_ + g];
    double tau = sigma_t * distance;
    double exponential = cjosey_exponential(tau); // exponential = 1 - exp(-tau)
    double new_delta_psi =
      (angular_flux_[g] - domain_->source_regions_.source(sr, g) / sigma_t) * exponential;
    delta_psi_[g] = new_delta_psi;
    angular_flux_[g] -= new_delta_psi;
    if (td_transport) {
      double sigma_t_td = domain_->sigma_t_td_[material * negroups_ + g];
      double tau_td = sigma_t_td * distance;
      double exponential_td = cjosey_exponential(tau_td); // exponential = 1 - exp(-tau)
      double new_delta_psi_td =
        (angular_flux_td_[g] -
          domain_->source_regions_.source_td(sr, g) / sigma_t_td) *
        exponential_td;
      if (RandomRay::time_method_ == RandomRayTimeMethod::SDP) {
        double source_derivative =
          domain_->source_regions_.source_time_derivative(sr, g);
        double flux_derivative_2 =
          domain_->source_regions_.scalar_flux_time_derivative_2(sr, g);
        double T1 = (source_derivative - flux_derivative_2) / sigma_t_td;

        // SDP terms for characteristic equation
        double inverse_vbar = domain_->inverse_vbar_[material * negroups_ + g];
        new_delta_psi_td += T1 * inverse_vbar * exponential_td / sigma_t_td;
        new_delta_psi_td += distance * inverse_vbar *
                            (angular_flux_td_prime_[g] - T1) *
                            (1 - exponential_td);

        // Time Derivative Characteristic Equation
        double new_delta_psi_td_prime =
          (angular_flux_td_prime_[g] - T1) * exponential_td;
        delta_psi_td_prime_[g] = new_delta_psi_td_prime;
        angular_flux_td_prime_[g] -= new_delta_psi_td_prime;
      }

      delta_psi_td_[g] = new_delta_psi_td;
      angular_flux_td_[g] -= new_delta_psi_td;
    }
  }

  // If ray is in the active phase (not in dead zone), make contributions to
  // source region bookkeeping
  if (is_active) {

    // Aquire lock for source region
    domain_->source_regions_.lock(sr).lock();

    // Accumulate delta psi into new estimate of source region flux for
    // this iteration
    for (int g = 0; g < negroups_; g++) {
      domain_->source_regions_.scalar_flux_new(sr, g) += delta_psi_[g];
      if (td_transport)
        domain_->source_regions_.scalar_flux_td_new(sr, g) += delta_psi_td_[g];
    }

    // Accomulate volume (ray distance) into this iteration's estimate
    // of the source region's volume
    domain_->source_regions_.volume(sr) += distance;

    // Tally valid position inside the source region (e.g., midpoint of
    // the ray) if not done already
    if (!domain_->source_regions_.position_recorded(sr)) {
      Position midpoint = r() + u() * (distance / 2.0);
      domain_->source_regions_.position(sr) = midpoint;
      domain_->source_regions_.position_recorded(sr) = 1;
    }

    // Release lock
    domain_->source_regions_.lock(sr).unlock();
  }
}

void RandomRay::attenuate_flux_linear_source(double distance, bool is_active)
{
  // Cast domain to LinearSourceDomain
  LinearSourceDomain* domain = dynamic_cast<LinearSourceDomain*>(domain_);
  if (!domain) {
    fatal_error("RandomRay::attenuate_flux_linear_source() called with "
                "non-LinearSourceDomain domain.");
  }

  // The number of geometric intersections is counted for reporting purposes
  n_event()++;

  // Determine source region index etc.
  int i_cell = lowest_coord().cell;

  // The source region is the spatial region index
  int64_t sr = domain_->source_region_offsets_[i_cell] + cell_instance();

  // The source element is the energy-specific region index
  int material = this->material();

  Position& centroid = domain_->source_regions_.centroid(sr);
  Position midpoint = r() + u() * (distance / 2.0);

  // Determine the local position of the midpoint and the ray origin
  // relative to the source region's centroid
  Position rm_local;
  Position r0_local;

  // In the first few iterations of the simulation, the source region
  // may not yet have had any ray crossings, in which case there will
  // be no estimate of its centroid. We detect this by checking if it has
  // any accumulated volume. If its volume is zero, just use the midpoint
  // of the ray as the region's centroid.
  if (domain_->source_regions_.volume_t(sr)) {
    rm_local = midpoint - centroid;
    r0_local = r() - centroid;
  } else {
    rm_local = {0.0, 0.0, 0.0};
    r0_local = -u() * 0.5 * distance;
  }
  double distance_2 = distance * distance;

  // Linear Source MOC incoming flux attenuation + source
  // contribution/attenuation equation
  for (int g = 0; g < negroups_; g++) {

    // Compute tau, the optical thickness of the ray segment
    double sigma_t = domain_->sigma_t_[material * negroups_ + g];
    double tau = sigma_t * distance;

    // If tau is very small, set it to zero to avoid numerical issues.
    // The following computations will still work with tau = 0.
    if (tau < 1.0e-8f) {
      tau = 0.0f;
    }

    // Compute linear source terms, spatial and directional (dir),
    // calculated from the source gradients dot product with local centroid
    // and direction, respectively.
    double spatial_source =
      domain_->source_regions_.source(sr, g) / sigma_t +
      rm_local.dot(domain_->source_regions_.source_gradients(sr, g) / sigma_t);
    double dir_source =
      u().dot(domain_->source_regions_.source_gradients(sr, g) / sigma_t);

    double gn = exponentialG(tau);
    double f1 = 1.0f - tau * gn;
    double f2 = (2.0f * gn - f1) * distance_2;
    double new_delta_psi = (angular_flux_[g] - spatial_source) * f1 * distance -
                          0.5 * dir_source * f2;

    double h1 = f1 - gn;
    double g1 = 0.5f - h1;
    double g2 = exponentialG2(tau);
    g1 = g1 * spatial_source;
    g2 = g2 * dir_source * distance * 0.5f;
    h1 = h1 * angular_flux_[g];
    h1 = (g1 + g2 + h1) * distance_2;
    spatial_source = spatial_source * distance + new_delta_psi;

    // Store contributions for this group into arrays, so that they can
    // be accumulated into the source region's estimates inside of the locked
    // region.
    delta_psi_[g] = new_delta_psi;
    delta_moments_[g] = r0_local * spatial_source + u() * h1;

    // Update the angular flux for this group
    angular_flux_[g] -= new_delta_psi * sigma_t;

    // If 2D mode is enabled, the z-component of the flux moments is forced
    // to zero
    if (source_shape_ == RandomRaySourceShape::LINEAR_XY) {
      delta_moments_[g].z = 0.0;
    }
  }

  // If ray is in the active phase (not in dead zone), make contributions to
  // source region bookkeeping
  if (is_active) {
    // Compute an estimate of the spatial moments matrix for the source
    // region based on parameters from this ray's crossing
    MomentMatrix moment_matrix_estimate;
    moment_matrix_estimate.compute_spatial_moments_matrix(
      rm_local, u(), distance);

    // Aquire lock for source region
    domain_->source_regions_.lock(sr).lock();

    // Accumulate deltas into the new estimate of source region flux for this
    // iteration
    for (int g = 0; g < negroups_; g++) {
      domain_->source_regions_.scalar_flux_new(sr, g) += delta_psi_[g];
      domain_->source_regions_.flux_moments_new(sr, g) += delta_moments_[g];
    }

    // Accumulate the volume (ray segment distance), centroid, and spatial
    // momement estimates into the running totals for the iteration for this
    // source region. The centroid and spatial momements estimates are scaled by
    // the ray segment length as part of length averaging of the estimates.
    domain_->source_regions_.volume(sr) += distance;
    domain_->source_regions_.centroid_iteration(sr) += midpoint * distance;
    moment_matrix_estimate *= distance;
    domain_->source_regions_.mom_matrix(sr) += moment_matrix_estimate;

    // Tally valid position inside the source region (e.g., midpoint of
    // the ray) if not done already
    if (!domain_->source_regions_.position_recorded(sr)) {
      domain_->source_regions_.position(sr) = midpoint;
      domain_->source_regions_.position_recorded(sr) = 1;
    }

    // Release lock
    domain_->source_regions_.lock(sr).unlock();
  }
}

void RandomRay::initialize_ray(uint64_t ray_id, FlatSourceDomain* domain)
{
  domain_ = domain;

  // Reset particle event counter
  n_event() = 0;

  is_active_ = (distance_inactive_ <= 0.0);

  wgt() = 1.0;

  // set identifier for particle
  id() = simulation::work_index[mpi::rank] + ray_id;

  // set random number seed
  int64_t particle_seed =
    (simulation::current_batch - 1) * settings::n_particles + id();
  init_particle_seeds(particle_seed, seeds());
  stream() = STREAM_TRACKING;

  // Sample from ray source distribution
  SourceSite site {ray_source_->sample(current_seed())};
  site.E = lower_bound_index(
    data::mg.rev_energy_bins_.begin(), data::mg.rev_energy_bins_.end(), site.E);
  site.E = negroups_ - site.E - 1.;
  this->from_source(&site);

  // Locate ray
  if (lowest_coord().cell == C_NONE) {
    if (!exhaustive_find_cell(*this)) {
      this->mark_as_lost(
        "Could not find the cell containing particle " + std::to_string(id()));
    }

    // Set birth cell attribute
    if (cell_born() == C_NONE)
      cell_born() = lowest_coord().cell;
  }

  // Initialize ray's starting angular flux to starting location's isotropic
  // source
  int i_cell = lowest_coord().cell;
  int64_t sr = domain_->source_region_offsets_[i_cell] + cell_instance();

  for (int g = 0; g < negroups_; g++) {
    double sigma_t = domain_->sigma_t_[domain_->source_regions_.material(sr) * negroups_ + g];
    angular_flux_[g] = domain_->source_regions_.source(sr, g) / sigma_t;
  }

  if (settings::run_mode == RunMode::TIME_DEPENDENT) {
    for (int g = 0; g < negroups_; g++) {
      double sigma_t_td = domain_->sigma_t_td_[domain_->source_regions_.material(sr) * negroups_ + g];
      angular_flux_td_[g] = domain_->source_regions_.source_td(sr, g) / sigma_t_td;
    }
    if (RandomRay::time_method_ == RandomRayTimeMethod::SDP) {
      for (int g = 0; g < negroups_; g++) {
        double sigma_t_td =
          domain_
            ->sigma_t_td_[domain_->source_regions_.material(sr) * negroups_ +
                          g];
        double source_derivative =
          domain_->source_regions_.source_time_derivative(sr, g);
        double flux_derivative_2 =
          domain_->source_regions_.scalar_flux_time_derivative_2(sr, g);
        double T1 = (source_derivative - flux_derivative_2);
        angular_flux_td_prime_[g] = T1 / sigma_t_td;
      }
    }
  }
}

} // namespace openmc
