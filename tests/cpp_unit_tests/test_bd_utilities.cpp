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

  int vector_size = 2;
  double dt = 0.1;
  // Two functions t^2 and t^3 across x = 2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3
  vector<double> test_bd_vector {8.0, 4.0, 6.859, 3.61, 5.832, 3.24, 4.913, 2.89,
    4.096, 2.56, 3.375, 2.25, 2.744, 1.96, 2.197, 1.69};

  vector<double> test_derivative {0.0, 0.0};
  for (int i = 0; i < 6; i++) {
    int o = i + 1;
    double d0 = bd_time_derivative<double>(test_bd_vector, vector_size, 0, o, dt);
    double d1 = bd_time_derivative<double>(test_bd_vector, vector_size, 1, o, dt);
    test_derivative[0] = d0;
    test_derivative[1] = d1;
    REQUIRE_THAT(
      test_derivative, Catch::Matchers::Approx(ref_derivative_first_order[i]));
  }
  for (int i = 0; i < 6; i++) {
    int o = i + 1;
    double d0 = bd_time_derivative<double>(test_bd_vector, vector_size, 0, o, dt, 2);
    double d1 = bd_time_derivative<double>(test_bd_vector, vector_size, 1, o, dt, 2);
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
  vector<vector<double>> ref_rhs_bd_first_order {{-68.59, -36.1},
    {-108.02, -56.0}, {-134.66666666666663, -69.33333333333331}, 
    {-154.66666666666666, -79.33333333333331},
    {-170.66666666666666, -87.33333333333331}, {-184.0, -94.0}};

  vector<vector<double>> ref_rhs_bd_second_order {{-788.5999999999999, -397.9999999999999},
    {-1588.0, -797.9999999999999},
    {-2321.3333333333344, -1164.6666666666667},
    {-2988.0, -1497.9999999999989},
    {-3596.88888888889, -1802.4444444444416},
    {-4156.888888888892, -2082.444444444443}};

  int vector_size = 2;
  double dt = 0.1;
  // Two functions t^2 and t^3 across x = 2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3
  vector<double> test_bd_vector {8.0, 4.0, 6.859, 3.61, 5.832, 3.24, 4.913, 2.89,
    4.096, 2.56, 3.375, 2.25, 2.744, 1.96, 2.197, 1.69};

  vector<double> test_rhs_bd {0.0, 0.0};
  for (int i = 0; i < 6; i++) {
    int o = i + 1;
    double d0 = rhs_backwards_difference<double>(test_bd_vector, vector_size, 0, o, dt);
    double d1 = rhs_backwards_difference<double>(test_bd_vector, vector_size, 1, o, dt);
    test_rhs_bd[0] = d0;
    test_rhs_bd[1] = d1;
    REQUIRE_THAT(
      test_rhs_bd, Catch::Matchers::Approx(ref_rhs_bd_first_order[i]));
  }
  for (int i = 0; i < 6; i++) {
    int o = i + 1;
    double rhs_d0 = rhs_backwards_difference<double>(test_bd_vector, vector_size, 0, o, dt, 2);
    double rhs_d1 = rhs_backwards_difference<double>(test_bd_vector, vector_size, 1, o, dt, 2);
    test_rhs_bd[0] = rhs_d0;
    test_rhs_bd[1] = rhs_d1;
    REQUIRE_THAT(
      test_rhs_bd, Catch::Matchers::Approx(ref_rhs_bd_second_order[i]));
  }
}

TEST_CASE("Test fill_bd_vector")
{
  vector<double> ref_scalar_flux_bd = {0.3, 0.4, 0.3, 0.4, 0.0, 0.0};
  vector<double> ref_precursors_bd = {1.0, 2.0, 3.0, 0.0, 0.0, 0.0};

  int bd_order = 1;
  int64_t n_source_elements = 2;
  int64_t n_delay_elements = 3;

  vector<double> scalar_flux_bd = {0.3, 0.4};
  vector<double> precursors_bd = {1.0, 2.0, 3.0};

  fill_bd_vector(n_source_elements, bd_order + 2, scalar_flux_bd);
  fill_bd_vector(n_delay_elements, bd_order + 1, precursors_bd);

  REQUIRE_THAT(ref_scalar_flux_bd, Catch::Matchers::Equals(scalar_flux_bd));
  REQUIRE_THAT(ref_precursors_bd, Catch::Matchers::Equals(precursors_bd));
}

TEST_CASE("Test increment_bd_vectors")
{
  vector<double> ref_scalar_flux_bd = {0.0, 0.0, 0.3, 0.4, 0.0, 0.0};
  vector<double> ref_precursors_bd = {0.0, 0.0, 0.0, 5.0, 6.0, 7.0};

  int64_t n_source_elements = 2;
  int64_t n_delay_elements = 3;

  vector<double> test_scalar_flux_bd = {0.3, 0.4, 0.0, 0.0, 1.0, 1.0};
  vector<double> test_precursors_bd = {5.0, 6.0, 7.0, 1.0, 1.0, 1.0};

  increment_bd_vector(n_source_elements, &test_scalar_flux_bd);
  increment_bd_vector(n_delay_elements, &test_precursors_bd);

  REQUIRE_THAT(
    ref_scalar_flux_bd, Catch::Matchers::Equals(test_scalar_flux_bd));
  REQUIRE_THAT(ref_precursors_bd, Catch::Matchers::Equals(test_precursors_bd));
}
