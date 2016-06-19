
#include "omp.hpp"

namespace omp
{
  namespace internal
  {
    std::mutex global_mutex;
  }

  void parallel(const std::function<void()>& operation, unsigned thread_cnt)
  {
    internal::thread_pool pool(operation, thread_cnt);
    pool.wait();
  }
}

