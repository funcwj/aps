// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#ifndef CSRC_UTILS_TIMER_H_
#define CSRC_UTILS_TIMER_H_

#include <sys/time.h>

namespace aps {
const int32_t SEC_TO_USEC = 1000 * 1000;

class Timer {
 public:
  Timer() { Reset(); }

  void Reset() { gettimeofday(&start_, NULL); }

  double Elapsed() const {
    struct timeval stop;
    gettimeofday(&stop, NULL);
    int64_t beg = start_.tv_sec * SEC_TO_USEC + start_.tv_usec,
            end = stop.tv_sec * SEC_TO_USEC + stop.tv_usec;
    return static_cast<double>(end - beg) / SEC_TO_USEC;
  }

 private:
  struct timeval start_;
};

}  // namespace aps

#endif  // CSRC_UTILS_TIMER_H_
