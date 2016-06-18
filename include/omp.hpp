
#ifndef OMP_OMP_HPP
#define OMP_OMP_HPP

#include <cstdint>
#include <functional>
#include <vector>
#include <thread>
#include <mutex>


namespace omp
{
  extern std::mutex global_mutex;

  template<typename Iter>
  class thread_pool
  {
  public:
    thread_pool(Iter begin, Iter end, const std::function<void(typename Iter::reference,std::size_t)>& fn, unsigned num_threads = 0) :
      fn_(fn),
      cur_(begin),
      end_(end),
      index_(0)
    {
      if (!num_threads)
        num_threads = std::thread::hardware_concurrency();

      if (!num_threads)
        num_threads = 1;

      threads_.reserve(num_threads);
      for (unsigned i = 0; i < num_threads; ++i)
        threads_.emplace_back(std::bind(&thread_pool::routine, this));
    }

    void wait()
    {
      for (auto it = threads_.begin(); it != threads_.end(); ++it)
        it->join();
    }

  private:
    std::function<void(typename Iter::reference, std::size_t)> fn_;
    std::vector<std::thread> threads_;
    Iter cur_;
    const Iter end_;
    std::size_t index_;
    std::mutex mtx_;

    void routine()
    {
      while (true)
      {
        std::unique_lock<std::mutex> lk(mtx_);
        std::size_t i = index_++;
        Iter it = cur_;
        if (cur_ != end_)
          ++cur_;
        lk.unlock();

        if (it == end_)
          break;
        else
          fn_(*it, i); //fn_ ? fn_(*it, i) : void();
      }


    }
  };

  template <typename Iterator>
  void parallel_for(Iterator begin, Iterator end, const std::function<void(typename Iterator::reference, std::size_t)>& operation)
  {
    thread_pool<Iterator> pool(begin, end, operation);
    pool.wait();
  }

  template <typename Handler>
  void critical(std::mutex& mtx, Handler fn)
  {
    std::lock_guard<std::mutex> lk(mtx);
    fn();
  }

  template <typename Handler>
  void critical(Handler fn)
  {
    std::lock_guard<std::mutex> lk(global_mutex);
    fn();
  }
};

#endif //OMP_OMP_HPP
