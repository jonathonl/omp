
#ifndef OMP_OMP_HPP
#define OMP_OMP_HPP

#include <cstdint>
#include <functional>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cassert>
#include <iterator>
#include <algorithm>
#include <iostream>

namespace omp
{


  struct iteration_context
  {
    std::size_t thread_index;
    std::size_t index;
  };

  namespace internal
  {
    extern std::mutex global_mutex;
    extern const unsigned default_num_threads;

    inline std::uint64_t ceil_divide(std::uint64_t x, std::uint64_t y)
    {
      return (x + y - 1) / y;
    }

    class thread_pool
    {
    public:
      thread_pool(const std::function<void(std::size_t thread_idx)>& fn, unsigned num_threads = 0) :
        num_threads_(num_threads ? num_threads : default_num_threads)
      {
        threads_.reserve(num_threads - 1);
        for (unsigned i = 0; i < (num_threads_ - 1); ++i)
          threads_.emplace_back(fn, i);
        fn(num_threads_ - 1);
        for (auto it = threads_.begin(); it != threads_.end(); ++it)
          it->join();
      }
    private:
      const std::size_t num_threads_;
      std::vector<std::thread> threads_;
    };

    class thread_pool2
    {
    private:
      enum class state { shutdown = 0, run, running, sleep };
      std::vector<std::thread> threads_;
      std::vector<state> states_;
      std::mutex mtx_;
      std::condition_variable cv_;
      std::function<void(std::size_t)> fn_;
      std::size_t sleeping_counter_;
    public:
      thread_pool2(std::size_t num_threads = 0) :
        states_(num_threads ? num_threads - 1 : default_num_threads - 1, state::sleep),
        sleeping_counter_(states_.size())
      {
        threads_.reserve(states_.size());
        for (std::size_t i = 0; i < states_.size(); ++i)
        {
          threads_.emplace_back(std::bind(&thread_pool2::routine, this, i));
        }
      }

      ~thread_pool2()
      {
        std::cerr << "Destroying thread pool ..." << std::endl;
        {
          std::unique_lock<std::mutex> lk(mtx_);
          std::fill(states_.begin(), states_.end(), state::shutdown);
        }

        cv_.notify_all();

        for (auto& t : threads_)
          t.join();
        std::cerr << "Done destroying thread pool ..." << std::endl;
      }

      std::size_t thread_count() const { return threads_.size() + 1; }

      void routine(std::size_t thread_idx)
      {
        while (true)
        {
          {
            std::unique_lock<std::mutex> lk(mtx_);
            if (states_[thread_idx] == state::shutdown)
              break;
            if (states_[thread_idx] == state::running)
            {
              states_[thread_idx] = state::sleep;
              ++sleeping_counter_;
              cv_.notify_all();
            }
            cv_.wait(lk, [this, thread_idx] { return states_[thread_idx] != state::sleep; });
            if (states_[thread_idx] == state::shutdown)
              break;
            states_[thread_idx] = state::running;
          }

          if (fn_)
          {
            fn_(thread_idx);
          }
        }
      }

      //template <typename Fn>
      void operator()(std::function<void(std::size_t)>&& fn)
      {
        fn_ = std::move(fn);

        {
          std::unique_lock<std::mutex> lk(mtx_);
          std::fill(states_.begin(), states_.end(), state::run);
          sleeping_counter_ = 0;
        }
        cv_.notify_all();

        if (fn_)
        {
          fn_(states_.size());
        }

        {
          // Wait for child threads to complete.
          std::unique_lock<std::mutex> lk(mtx_);
          cv_.wait(lk, [this] { return sleeping_counter_ == states_.size(); }); // std::count(states_.begin(), states_.end(), state::sleep) == states_.size(); });
        }

        fn_ = nullptr;
      }
    };

    template<typename Iter>
    class dynamic_iterator_thread_pool
    {
    public:
      dynamic_iterator_thread_pool(std::size_t chunk_size, Iter begin, Iter end, const std::function<void(typename Iter::reference, const iteration_context&)>& fn, unsigned num_threads) :
        fn_(fn),
        cur_(begin),
        end_(end),
        index_(0),
        chunk_size_(chunk_size ? chunk_size : 1),
        num_threads_(num_threads ? num_threads : default_num_threads)
      {
        threads_.reserve(num_threads_ - 1);
        for (unsigned i = 0; i < (num_threads_ - 1); ++i)
          threads_.emplace_back(std::bind(&dynamic_iterator_thread_pool::routine, this, i));
        this->routine(num_threads_ - 1);

        for (auto it = threads_.begin(); it != threads_.end(); ++it)
          it->join();
      }

    private:
      std::function<void(typename Iter::reference, const omp::iteration_context&)> fn_;
      std::vector<std::thread> threads_;
      Iter cur_;
      const Iter end_;
      std::size_t index_;
      std::mutex mtx_;
      const std::size_t chunk_size_;
      const std::size_t num_threads_;

      void routine(std::size_t thread_index)
      {
        bool done = false;
        while (!done)
        {
          std::vector<Iter> chunk(chunk_size_);

          std::unique_lock<std::mutex> lk(mtx_);
          std::size_t index = index_;
          for (std::size_t chunk_offset = 0; chunk_offset < chunk.size(); ++chunk_offset)
          {
            ++index_;
            chunk[chunk_offset] = cur_;
            if (cur_ != end_)
              ++cur_;
          }
          lk.unlock();

          for (std::size_t chunk_offset = 0; chunk_offset < chunk.size(); ++chunk_offset)
          {
            if (chunk[chunk_offset] == end_)
            {
              done = true;
            }
            else
            {
              fn_(*chunk[chunk_offset], {thread_index, index + chunk_offset}); //fn_ ? fn_(*it, i) : void();
            }
          }
        }
      }
    };

    template<typename Iter>
    class static_iterator_thread_pool
    {
    public:
      static_iterator_thread_pool(std::size_t chunk_size, Iter begin, Iter end, const std::function<void(typename Iter::reference,const iteration_context&)>& fn, unsigned num_threads = 0) :
        fn_(fn),
        num_threads_(num_threads ? num_threads : default_num_threads),
        beg_(begin),
        end_(end),
        total_elements_(std::distance(beg_, end_)),
        chunk_size_(chunk_size ? chunk_size : static_cast<std::size_t>(total_elements_) / num_threads_)
      {
        threads_.reserve(num_threads_ - 1);
        for (unsigned i = 0; i < (num_threads_ - 1); ++i)
          threads_.emplace_back(std::bind(&static_iterator_thread_pool::routine, this, i));
        this->routine(num_threads_ - 1);

        for (auto it = threads_.begin(); it != threads_.end(); ++it)
          it->join();
      }
    private:
      std::function<void(typename Iter::reference, const omp::iteration_context&)> fn_;
      const std::size_t num_threads_;
      std::vector<std::thread> threads_;
      const Iter beg_;
      const Iter end_;
      long total_elements_;
      const std::size_t chunk_size_;
    public:
      void routine(std::size_t thread_index)
      {
        auto cur = beg_;

        std::advance(cur, thread_index * chunk_size_);
        for (std::size_t index = (thread_index * chunk_size_); index < total_elements_; index += (chunk_size_ * num_threads_ - chunk_size_), std::advance(cur, chunk_size_ * num_threads_ - chunk_size_))
        {
          for (std::size_t chunk_offset = 0; chunk_offset < chunk_size_ && index < total_elements_; ++chunk_offset)
          {
            assert(cur != end_);
            fn_(*cur, {thread_index,index}); //fn_ ? fn_(*it, i) : void();
            ++cur;
            ++index;
          }
        }
      }
    };

    template<typename Iter>
    class static_iterator_functor
    {
    public:
      static_iterator_functor(std::size_t chunk_size, Iter begin, Iter end, const std::function<void(typename Iter::reference,const iteration_context&)>& fn, unsigned num_threads) :
        fn_(fn),
        num_threads_(num_threads ? num_threads : default_num_threads),
        beg_(begin),
        end_(end),
        total_elements_(std::distance(beg_, end_)),
        chunk_size_(chunk_size ? chunk_size : ceil_divide(total_elements_, num_threads_))
      {
        //assert(chunk_size_ > 0);
//        threads_.reserve(num_threads_ - 1);
//        for (unsigned i = 0; i < (num_threads_ - 1); ++i)
//          threads_.emplace_back(std::bind(&static_iterator_thread_pool::routine, this, i));
//        this->routine(num_threads_ - 1);
//
//        for (auto it = threads_.begin(); it != threads_.end(); ++it)
//          it->join();
      }
    private:
      std::function<void(typename Iter::reference, const omp::iteration_context&)> fn_;
      const std::size_t num_threads_;
      const Iter beg_;
      const Iter end_;
      std::int64_t total_elements_;
      const std::int64_t chunk_size_;
    public:
      void operator()(std::size_t thread_index)
      {
        if (total_elements_ > 0)
        {
          auto cur = beg_;

          std::size_t index = (thread_index * chunk_size_);
          if (index >= total_elements_)
            return;

          std::advance(cur, thread_index * chunk_size_);
          for ( ; index < total_elements_; )
          {
            std::size_t end_off = index + chunk_size_;
            for (; index < end_off && index < total_elements_; ++index,++cur)
            {
              assert(cur != end_);
              fn_(*cur, {thread_index, index}); //fn_ ? fn_(*it, i) : void();
            }

            index += (chunk_size_ * num_threads_ - chunk_size_);
            if (index >= total_elements_)
              break;
            std::advance(cur, chunk_size_ * num_threads_ - chunk_size_);
          }
        }
      }
    };
  }

  class sequence_iterator
  {
  public:
    typedef sequence_iterator self_type;
    typedef int difference_type;
    typedef int value_type;
    typedef value_type& reference;
    typedef value_type* pointer;
    typedef std::random_access_iterator_tag iterator_category;

    sequence_iterator() : val_(0) {}
    sequence_iterator(value_type val) :
      val_(val)
    {

    }


    //reference          operator [] (difference_type);
    bool operator < (const self_type& other) { return val_ < other.val_; }
    bool operator > (const self_type& other) { return val_ > other.val_; }
    bool operator <= (const self_type& other) { return val_ <= other.val_; }
    bool operator >= (const self_type& other) { return val_ >= other.val_; }

    self_type operator++()
    {
      self_type ret = *this;
      ++val_;
      return ret;
    }

    self_type operator--()
    {
      self_type ret = *this;
      --val_;
      return ret;
    }

    self_type&      operator += (difference_type i)  { val_ += i; return *this; }
    self_type&      operator -= (difference_type i)  { val_ -= i; return *this; }
    self_type       operator +  (difference_type i) { return self_type(val_ + i); }
    self_type       operator -  (difference_type i) { return self_type(val_ - i); }
    difference_type operator - (const self_type& other) { return val_ - other.val_; }

    void operator++(int) { ++val_; }
    void operator--(int) { --val_; }
    reference operator*() { return val_; }
    pointer operator->() { return &val_; }
    bool operator==(const self_type& rhs) const { return (val_ == rhs.val_); }
    bool operator!=(const self_type& rhs) const { return (val_ != rhs.val_); }
  private:
    value_type val_;
  };

  class schedule
  {
  public:
    schedule(std::size_t chunk_size);
    std::size_t chunk_size() const;
  protected:
    std::size_t chunk_size_;
  };

  class dynamic_schedule : public schedule
  {
  public:
    dynamic_schedule(std::size_t chunk_size = 0);
  };

  class static_schedule : public schedule
  {
  public:
    static_schedule(std::size_t chunk_size = 0);
  };

  void parallel(const std::function<void(std::size_t)>& operation, unsigned thread_cnt = 0);

  template <typename Iterator>
  void parallel_for(const dynamic_schedule& sched, Iterator begin, Iterator end, const std::function<void(typename Iterator::reference, const iteration_context&)>& operation, unsigned thread_cnt = 0)
  {
    internal::dynamic_iterator_thread_pool<Iterator> pool(sched.chunk_size(), begin, end, operation, thread_cnt);
  }

  template <typename Iterator>
  void parallel_for(const static_schedule& sched, Iterator begin, Iterator end, const std::function<void(typename Iterator::reference, const iteration_context&)>& operation, unsigned thread_cnt = 0)
  {
    internal::static_iterator_thread_pool<Iterator> pool(sched.chunk_size(), begin, end, operation, thread_cnt);
  }

  template <typename Iterator>
  void parallel_for_exp(const static_schedule& sched, Iterator begin, Iterator end, const std::function<void(typename Iterator::reference, const iteration_context&)>& operation, internal::thread_pool2& tp)
  {

    tp(internal::static_iterator_functor<Iterator>(sched.chunk_size(), begin, end, operation, tp.thread_count())); //std::bind(&internal::static_iterator_functor<Iterator>::routine, &static_fn, std::placeholders::_1));
  }

  template <typename Iterator>
  void parallel_for(Iterator begin, Iterator end, const std::function<void(typename Iterator::reference, const iteration_context&)>& operation, unsigned thread_cnt = 0)
  {
    parallel_for(static_schedule(), begin, end, operation, thread_cnt);
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
    std::lock_guard<std::mutex> lk(internal::global_mutex);
    fn();
  }
}

#endif //OMP_OMP_HPP
