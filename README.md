# omp
A parallel programming library that mimics OpenMP syntax.

## Parallel For Example
```c++
std::vector<double> arr(256, 0.0);
std::mutex named_section;

omp::parallel_for(arr.begin(), arr.end(), [&named_section](double& element, std::size_t index)
{
  element = (index + 1);

  omp::critical(named_section, []()
  {

  });

  {
    std::lock_guard<std::mutex> critical(named_section);
    // lock_guard is usually a better alternative to omp::critical.
  }

  omp::critical([]()
  {

  });
});
```

## Parallel Example
```c++
unsigned num_threads = 8;
omp::parallel([]()
{
  
}, num_threads);
```