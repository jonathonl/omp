
#include "omp.hpp"

namespace omp
{
  namespace internal
  {
    std::mutex global_mutex;
    const unsigned default_num_threads(std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
  }


  schedule::schedule(std::size_t chunk_size) :
    chunk_size_(chunk_size)
  {
  }

  std::size_t schedule::chunk_size() const
  {
    return chunk_size_;
  }

  dynamic_schedule::dynamic_schedule(std::size_t chunk_size) :
    schedule(chunk_size)
  {
  }

  static_schedule::static_schedule(std::size_t chunk_size) :
    schedule(chunk_size)
  {
  }

  void parallel(const std::function<void()>& operation, unsigned thread_cnt)
  {
    internal::thread_pool pool(operation, thread_cnt);
    pool.wait();
  }
}

