
#include "omp.hpp"
#include <iostream>
#include <cassert>

int main()
{
  std::vector<double> arr(257, 0.0);
  std::mutex named_section;
  std::size_t total = 0;

  omp::parallel_for(omp::static_schedule(), arr.begin(), arr.end(), [&total, &named_section](double& element, std::size_t index)
  {
    element = (index + 1);

    omp::critical(named_section, [&total, element]()
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
  }, 8);

  omp::parallel_for(omp::dynamic_schedule(), omp::sequence_iterator(-2), omp::sequence_iterator(5), [&total, &named_section](int& element, std::size_t index)
  {
    std::lock_guard<std::mutex> critical(named_section);
    ++total;
  }, 3);

  std::cout << total << std::endl;
  assert(total == 264);

  unsigned num_threads = 8;
  omp::parallel([]()
  {

  }, num_threads);

  return 0;
}