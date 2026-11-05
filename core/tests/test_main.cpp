#include "test_framework.hpp"
#include "test_suites.hpp"

int main() {
  run_bitboard_tests();
  run_free_cell_tests();
  run_rng_tests();
  run_thread_pool_tests();
  run_view_tests();
  run_engine_tests();
  return test::summarize();
}
