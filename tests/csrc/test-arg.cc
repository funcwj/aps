// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "utils/args.h"


int main(int argc, char const *argv[]) {
  using namespace aps;
  const std::string description = "Command for testing of the utils/args.h";
  ArgParser parser(description);

  int32_t required_int32 = 0;
  float required_float = 0;
  bool optional_bool = false;
  std::string optional_string = "";

  parser.AddArgument("required-int32", &required_int32,
                     "Required value for int32", true);
  parser.AddArgument("required-float", &required_float,
                     "Required value for float", true);
  parser.AddArgument("optional-bool", &optional_bool,
                     "Optional value for float", false);
  parser.AddArgument("optional-string", &optional_string,
                     "Optional value for string", false);

  parser.ReadCommandArgs(argc, argv);
  LOG_INFO << "required-int32: " << required_int32;
  LOG_INFO << "required-float: " << required_float;
  LOG_INFO << "optional-bool: " << optional_bool;
  LOG_INFO << "optional-string: " << optional_string;
  return 0;
}
