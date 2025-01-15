#include "openmc/random_ray/flat_source_domain.h"
#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

using namespace openmc;

TEST_CASE("Test bdf_time_derivative")
{
  std::vector<std::vector<float>> ref_derivative_first_order {
    {11.41, 3.9},
    {11.98, 4.0},
    {12.0, 4.0},
    {12.0, 4.0},
    {12.0, 4.0},
    {12.0, 4.0}};

  std::vector<std::vector<float>> ref_derivative_second_order {
    {11.39993, 2.00003},
    {11.99964, 2.00005},
    {11.99975, 2.00017},
    {11.99875, 1.99981},
    {11.99922, 1.99948},
    {11.99955, 2.00114}};


  int offset = 2;
  double dt = 0.1;
  // Two functions t^2 and t^3 across x = 2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3
  std::vector<float> test_bdf_vector {
    8.0 , 4.0 ,
    6.859, 3.61,
    5.832, 3.24,
    4.913, 2.89,
    4.096, 2.56,
    3.375, 2.25,
    2.744, 1.96,
    2.197, 1.69};

  std::vector<float> test_derivative {0.0, 0.0};
  // These both start to fail at o = 3, perhaps it's a type issue? There are
  // doubles in the function def...
  for (int i = 0; i < 6; i++) {
    int o = i + 1;
    float d0 = bdf_time_derivative<float>(0, &test_bdf_vector, o, dt, offset);
    float d1 = bdf_time_derivative<float>(1, &test_bdf_vector, o, dt, offset);
    test_derivative[0] = d0;
    test_derivative[1] = d1;
    REQUIRE_THAT(test_derivative, Catch::Matchers::Approx(ref_derivative_first_order[i]));
  }    
  for (int i = 0; i < 6; i++) {
    int o = i + 1;
    float d0 = bdf_time_derivative<float>(0, &test_bdf_vector, o, dt, offset, 2);
    float d1 = bdf_time_derivative<float>(1, &test_bdf_vector, o, dt, offset, 2);
    test_derivative[0] = d0;
    test_derivative[1] = d1;
    REQUIRE_THAT(test_derivative, Catch::Matchers::Approx(ref_derivative_second_order[i]));
  }
}

TEST_CASE("Test update_bdf_vector")
{
  std::vector<int> ref_vector_no_increment {1, 2, 3, 4, 5, 6};
  std::vector<int> ref_vector_increment {1, 2, 1, 2, 3, 4};

  std::vector<int> test_vector {0, 0, 3, 4, 5, 6};
  std::vector<int> new_solution {1, 2};

  update_bdf_vector(&test_vector, &new_solution, false);
  REQUIRE_THAT(test_vector, Catch::Matchers::Equals(ref_vector_no_increment));

  update_bdf_vector(&test_vector, &new_solution, true);
  REQUIRE_THAT(test_vector, Catch::Matchers::Equals(ref_vector_increment));
}
