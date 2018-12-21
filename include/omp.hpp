
#ifndef OMP_OMP_HPP
#define OMP_OMP_HPP

#include <cstdint>
#include <functional>
#include <thread>
#include <vector>
#include <mutex>
#include <cassert>
#include <iterator>


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
