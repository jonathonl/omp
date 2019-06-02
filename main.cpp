
#include "omp.hpp"
#include <iostream>
#include <cassert>

int main()
{
  std::vector<double> arr(257, 0.0);
  std::mutex named_section;
  std::size_t total = 0;

  omp::internal::thread_pool2 pool(8);
  omp::parallel_for_exp(omp::static_schedule(), arr.begin(), arr.end(), [&total, &named_section](double& element, const omp::iteration_context& ctx)
  {
    element = (ctx.index + 1);

    omp::critical(named_section, [&total, element, &ctx]()
    {
      ++total;
    });

    {
      std::lock_guard<std::mutex> critical(named_section);
      // lock_guard is usually a better alternative to omp::critical.
    }

    omp::critical([]()
    {

    });
  }, pool); // 8);

  omp::parallel_for(omp::dynamic_schedule(), omp::sequence_iterator(-2), omp::sequence_iterator(5), [&total, &named_section](int& element, const omp::iteration_context& ctx)
  {
    std::lock_guard<std::mutex> critical(named_section);
    ++total;
  }, 3);

  std::cout << total << std::endl;
  assert(total == 264);

  unsigned num_threads = 8;
  omp::parallel([](std::size_t thread_idx)
  {

  }, num_threads);

  return total == 264 ? EXIT_SUCCESS : EXIT_FAILURE;
}