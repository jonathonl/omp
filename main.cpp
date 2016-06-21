
#include "omp.hpp"
#include <iostream>

int main()
{
  std::vector<double> arr(257, 0.0);
  std::mutex named_section;
  std::size_t total = 0;

  omp::parallel_for(omp::static_schedule(6), arr.begin(), arr.end(), [&total, &named_section](double& element, std::size_t index)
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

  std::cout << total << std::endl;

  unsigned num_threads = 8;
  omp::parallel([]()
  {

  }, num_threads);

  return 0;
}