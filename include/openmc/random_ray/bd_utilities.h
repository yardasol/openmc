#ifndef OPENMC_RANDOM_RAY_BD_UTILITIES_H
#define OPENMC_RANDOM_RAY_BD_UTILITIES_H

#include <algorithm>
#include <map>

#include "openmc/error.h"
#include "openmc/vector.h"

namespace openmc {

//----------------------------------------------------------------------------
// Helper Variables
// Coefficients come from Table 3 in Fornberg (1988)
// DOI: 10.1090/S0025-5718-1988-0935077-0
// Note that the signs are flipped compared to the citation, as the author was
// formulating weights for a forward difference
const std::map<int, vector<double>> bd_coefficients_first_order_ = {
  {1, {1.0, -1.0}}, {2, {1.5, -2.0, 0.5}},
  {3, {1.833333333333333, -3.0, 1.5, -0.333333333333333}},
  {4, {2.083333333333333, -4.0, 3.0, -1.333333333333333, 0.25}},
  {5, {2.283333333333333, -5.0, 5.0, -3.333333333333333, 1.25, -0.2}},
  {6, {2.45, -6.0, 7.5, -6.666666666666667, 3.75, -1.2, 0.166666666666667}}};

// Coefficients come from Table 3 in Fornberg (1988)
// DOI: 10.1090/S0025-5718-1988-0935077-0
const std::map<int, vector<double>> bd_coefficients_second_order_ = {
  {1, {1.0, -2.0, 1.0}}, {2, {2.0, -5.0, 4, -1}},
  {3, {2.916666666666667, -8.666666666666667, 9.5, -4.666666666666667, 0.916666666666667}},
  {4, {3.75, -12.833333333333333, 17.833333333333333, -13.0, 5.083333333333333, -0.833333333333333}},
  {5, {4.511111111111111, -17.4, 29.25, -28.222222222222222, 16.5, -5.4, 0.761111111111111}},
  {6, {5.211111111111111, -22.3, 43.95, -52.722222222222222, 41.0, -20.1, 5.661111111111111, -0.7}}};

// bd vector funtions
template<typename T>
T bd_time_derivative(int index, vector<T>* bd_vector, int bd_order, double dt,
  int offset, int derivative_order = 1)
{
  vector<double> bd_coeffs;
  int n_bd_terms;
  double time_factor;
  if (derivative_order == 1) {
    bd_coeffs = bd_coefficients_first_order_.at(bd_order);
    time_factor = 1 / dt;
    n_bd_terms = bd_order + 1;
  } else if (derivative_order == 2) {
    bd_coeffs = bd_coefficients_second_order_.at(bd_order);
    n_bd_terms = bd_order + 2;
    time_factor = 1 / (dt * dt);
  } else {
    fatal_error("Only first or second order bd derivatives are allowed.");
  }
  T bd_derivative = 0.0;
  for (int i = 0; i < n_bd_terms; i++) {
    double coeff = bd_coeffs[i];
    T x = (*bd_vector)[index + i * offset]; // n_source_elements_ in most cases
    bd_derivative += coeff * x * time_factor;
  }
  return bd_derivative;
}

// Update the first n elements of bd_vector with
// all n elements of  new_solution.
// Optionally scale rotate bd_vector so the last n
// elements are moved to the first n elements
// before updating.
// Optionally scale the new solutions by factor
template<typename T>
void update_bd_vector(
  vector<T>* bd_vector, vector<T>& new_solution, bool increment, T factor = 1)
{
  int n = new_solution.size();
  // Move the oldest solution to the front of the vector
  if (increment)
    rotate(bd_vector->rbegin(), bd_vector->rbegin() + n, bd_vector->rend());
  // Replace the oldest solution with the new solution
  for (int i = 0; i < n; i++)
    (*bd_vector)[i] = new_solution[i] * factor;
}

// Take RHS derivative to solve for the current timestep
template<typename T>
T rhs_backwards_difference(vector<T>* bd_vector, int64_t vector_size, int idx,
  const vector<double>& bd_coeffs, double dt)
{
  int bd_order = bd_coeffs.size() - 1;
  T rhs_bd = 0.0;
  for (int j = 1; j <= bd_order; j++)
    rhs_bd += bd_coeffs[j] * (*bd_vector)[idx + j * vector_size];
  rhs_bd /= dt;
  return rhs_bd;
}

} // namespace openmc

#endif // OPENMC_RANDOM_RAY_BD_UTILITIES_H
