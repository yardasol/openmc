#include "openmc/random_ray/bd_utilities.h"
#include "openmc/random_ray/random_ray_simulation.h"
#include "openmc/vector.h"
#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

using namespace openmc;

TEST_CASE("Test bd_time_derivative")
{
  vector<vector<double>> ref_derivative_first_order {{11.41, 3.9}, {11.98, 4.0},
    {12.0, 4.0}, {12.0, 4.0}, {12.0, 4.0}, {12.0, 4.0}};

  vector<vector<double>> ref_derivative_second_order {{11.4, 2.0},
    {12.0 , 2.0}, {12.0, 2.0}, {12.0,  2.0},
    {12.0, 2.0}, {12.0, 2.0}};

  int offset = 2;
  double dt = 0.1;
  // Two functions t^2 and t^3 across x = 2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3
  vector<double> test_bd_vector {8.0, 4.0, 6.859, 3.61, 5.832, 3.24, 4.913, 2.89,
    4.096, 2.56, 3.375, 2.25, 2.744, 1.96, 2.197, 1.69};

  vector<double> test_derivative {0.0, 0.0};
  // These both start to fail at o = 3, perhaps it's a type issue? There are
  // doubles in the function def...
  for (int i = 0; i < 6; i++) {
    int o = i + 1;
    double d0 = bd_time_derivative<double>(0, &test_bd_vector, o, dt, offset);
    double d1 = bd_time_derivative<double>(1, &test_bd_vector, o, dt, offset);
    test_derivative[0] = d0;
    test_derivative[1] = d1;
    REQUIRE_THAT(
      test_derivative, Catch::Matchers::Approx(ref_derivative_first_order[i]));
  }
  for (int i = 0; i < 6; i++) {
    int o = i + 1;
    double d0 = bd_time_derivative<double>(0, &test_bd_vector, o, dt, offset, 2);
    double d1 = bd_time_derivative<double>(1, &test_bd_vector, o, dt, offset, 2);
    test_derivative[0] = d0;
    test_derivative[1] = d1;
    REQUIRE_THAT(
      test_derivative, Catch::Matchers::Approx(ref_derivative_second_order[i]));
  }
}

TEST_CASE("Test update_bd_vector")
{
  vector<int> ref_vector_no_increment {1, 2, 3, 4, 5, 6};
  vector<int> ref_vector_increment {1, 2, 1, 2, 3, 4};

  vector<int> test_vector {0, 0, 3, 4, 5, 6};
  vector<int> new_solution {1, 2};

  update_bd_vector(&test_vector, new_solution, false);
  REQUIRE_THAT(test_vector, Catch::Matchers::Equals(ref_vector_no_increment));

  update_bd_vector(&test_vector, new_solution, true);
  REQUIRE_THAT(test_vector, Catch::Matchers::Equals(ref_vector_increment));
}

TEST_CASE("Test rhs_backwards_difference")
{

  const vector<double> bd_coeffs1 = {1.0, -1.0};
  const vector<double> bd_coeffs2 = {1.5, -2.0, 0.5};

  vector<double> bd_vector = {1.0, 0.9, 2.0, 1.9, 2.5, 2.4};

  int64_t vector_size = 2;
  int idx = 0;
  double dt = 1.0;

  double ref_rhs_bd1 = -2.0;
  double ref_rhs_bd2 = -2.75;

  double test_rhs_bd1 =
    rhs_backwards_difference(&bd_vector, vector_size, idx, bd_coeffs1, dt);
  double test_rhs_bd2 =
    rhs_backwards_difference(&bd_vector, vector_size, idx, bd_coeffs2, dt);

  REQUIRE(ref_rhs_bd1 == test_rhs_bd1);
  REQUIRE(ref_rhs_bd2 == test_rhs_bd2);
}

TEST_CASE("Test initialize_bd_vectors")
{
  vector<double> ref_scalar_flux_bd = {0.3, 0.4, 0.0, 0.0, 0.0, 0.0};
  // std::vector<double> ref_source_bd = {0.1, 0.2, 0.0, 0.0};
  vector<double> ref_precursors_bd = {1.0, 2.0, 3.0, 0.0, 0.0, 0.0};

  int bd_order_max = 1;
  int64_t n_source_elements = 2;
  int64_t n_delay_elements = 3;

  vector<double> criticality_scalar_flux = {0.3, 0.4};
  // std::vector<double> criticality_source = {0.1, 0.2};
  vector<double> criticality_precursors = {1.0, 2.0, 3.0};

  vector<double> scalar_flux_bd;
  // std::vector<double> source_bd;
  vector<double> precursors_bd;

  initialize_bd_vectors(n_source_elements, n_delay_elements, bd_order_max,
    &scalar_flux_bd, &precursors_bd, &criticality_scalar_flux,
    &criticality_precursors);
  REQUIRE_THAT(ref_scalar_flux_bd, Catch::Matchers::Equals(scalar_flux_bd));
  // REQUIRE_THAT(ref_source_bd, Catch::Matchers::Equals(source_bd));
  REQUIRE_THAT(ref_precursors_bd, Catch::Matchers::Equals(precursors_bd));
}

// This test gets stuck for some reason :/
TEST_CASE("Test increment_bd_vectors")
{
  vector<double> ref_scalar_flux_bd = {0.0, 0.0, 0.3, 0.4, 0.0, 0.0};
  // std::vector<double> ref_source_bd = {0.0, 0.0, 0.1, 0.2};
  vector<double> ref_precursors_bd = {0.0, 0.0, 0.0, 5.0, 6.0, 7.0};

  int64_t n_source_elements = 2;
  int64_t n_delay_elements = 3;

  vector<double> test_scalar_flux_bd = {0.3, 0.4, 0.0, 0.0, 1.0, 1.0};
  // std::vector<double> test_source_bd = {0.1, 0.2, 1.0, 1.0};
  vector<double> test_precursors_bd = {5.0, 6.0, 7.0, 1.0, 1.0, 1.0};

  increment_bd_vectors(n_source_elements, n_delay_elements,
    &test_scalar_flux_bd, &test_precursors_bd);
  REQUIRE_THAT(
    ref_scalar_flux_bd, Catch::Matchers::Equals(test_scalar_flux_bd));
  // REQUIRE_THAT(ref_source_bd, Catch::Matchers::Equals(test_source_bd));
  REQUIRE_THAT(ref_precursors_bd, Catch::Matchers::Equals(test_precursors_bd));
}
