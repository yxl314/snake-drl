#include "stacklog/stacklog.hpp"
#include "test_framework.hpp"
#include "test_suites.hpp"

// 测试程序负责初始化日志：库本身从不 init，这样消费者（训练/评估程序、
// Python 绑定）可以自己决定日志去哪。
int main() {
  stacklog::config log_config;
  log_config.dir = "test-logs";
  log_config.base_name = "snake_core_tests";
  log_config.min_level = stacklog::level::warn;
  stacklog::init(log_config);

  run_bitboard_tests();
  run_free_cell_tests();
  run_rng_tests();
  run_thread_pool_tests();
  run_view_tests();
  run_engine_tests();

  const int status = test::summarize();
  stacklog::shutdown();
  return status;
}
