// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
// Reference to Kaldi's parse-options.{h,cc}

#ifndef CSRC_UTILS_ARGS_H_
#define CSRC_UTILS_ARGS_H_

#include <map>
#include <string>
#include <vector>
#include <algorithm>

#include "utils/log.h"
#include "utils/math.h"

namespace aps {

struct ArgumentAttr {
  std::string name_;
  std::string help_;
  bool required_;

  ArgumentAttr(const std::string &name, const std::string &help, bool required)
      : name_{name}, help_(help), required_(required) {}
};

class ArgParser {
 public:
  explicit ArgParser(const std::string &description);

  void ReadCommandArgs(int argc, char const *argv[]);

  void PrintUsage();

  template <typename T>
  void AddArgument(const std::string &name, T *value,
                   const std::string &help, bool required = false);

 private:
  bool help_;
  std::string description_;
  std::string command_name_;
  int32_t required_args_num_;
  size_t max_arg_name_len_;

  // record value of the arguments
  std::map<std::string, int32_t *> int32_args_;
  std::map<std::string, float *> float_args_;
  std::map<std::string, bool *> bool_args_;
  std::map<std::string, std::string *> string_args_;
  // record help/name strings & required flags
  std::vector<ArgumentAttr> attr_args_;

  std::string AddArgumentImpl(const std::string &name, const std::string &help,
                              bool required, float *value);
  std::string AddArgumentImpl(const std::string &name, const std::string &help,
                              bool required, int32_t *value);
  std::string AddArgumentImpl(const std::string &name, const std::string &help,
                              bool required, std::string *value);
  std::string AddArgumentImpl(const std::string &name, const std::string &help,
                              bool required, bool *value);

  std::string NormalizeArgName(const std::string &name);

  bool SetArgument(const std::string &key, const std::string &value);
};

}  // namespace aps

#endif  // CSRC_UTILS_ARGS_H_
