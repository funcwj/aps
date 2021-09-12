// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#ifndef CSRC_UTILS_LOG_H_
#define CSRC_UTILS_LOG_H_

#include <algorithm>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace aps {

class Logger {
 public:
  Logger(const std::string &type, const char *func, const char *file,
         size_t line)
      : type_(type), func_(func), file_(BaseName(file)), line_(line) {}

  ~Logger() {
    std::string msg = oss_.str();
    while (true) {
      std::string::iterator p = std::find(msg.begin(), msg.end(), '\n');
      if (p == msg.end()) break;
      msg.erase(p);
    }
    Log(msg);
  }

  void Log(const std::string &msg) {
    std::ostringstream prefix;
    prefix << Date() << " - " << type_ << " (" << func_ << "(...):" << file_
           << ":" << line_ << ")";
    std::cerr << prefix.str().c_str() << " " << msg.c_str() << std::endl;
    if (type_ == "ASSERT" || type_ == "FAIL") abort();
  }

  std::ostream &Stream() { return oss_; }

 private:
  std::ostringstream oss_;

  std::string type_;  // FAIL, INFO, WARN, ASSERT
  const char *func_;
  const char *file_;
  size_t line_;

  const char *BaseName(const char *path) {
    int pos = strlen(path) - 1;
    while (pos >= 0) {
      if (path[pos] == '/') break;
      pos--;
    }
    return path + pos + 1;
  }

  const std::string Date() {
    std::ostringstream time_format;
    time_t timer = time(0);
    tm now;
    localtime_r(&timer, &now);
    time_format << 1900 + now.tm_year << "/" << std::setw(2)
                << std::setfill('0') << 1 + now.tm_mon << "/" << std::setw(2)
                << std::setfill('0') << now.tm_mday << " " << std::setw(2)
                << std::setfill('0') << now.tm_hour << ":" << std::setw(2)
                << std::setfill('0') << now.tm_min << ":" << std::setw(2)
                << std::setfill('0') << now.tm_sec;
    return time_format.str();
  }
};

#define LOG_WARN Logger("WARN", __FUNCTION__, __FILE__, __LINE__).Stream()
#define LOG_INFO Logger("INFO", __FUNCTION__, __FILE__, __LINE__).Stream()
#define LOG_FAIL Logger("FAIL", __FUNCTION__, __FILE__, __LINE__).Stream()

#define ASSERT(cond)                                              \
  do {                                                            \
    if (cond)                                                     \
      (void)0;                                                    \
    else                                                          \
      Logger("ASSERT", __FUNCTION__, __FILE__, __LINE__).Stream() \
          << "Assert '" << #cond << "' failed!";                  \
  } while (0)

}  // namespace aps

#endif  // CSRC_UTILS_LOG_H_
