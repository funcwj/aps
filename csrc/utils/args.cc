// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "utils/args.h"

namespace aps {

ArgParser::ArgParser(const std::string &description)
    : description_(description),
      help_(false),
      command_name_(""),
      required_args_num_(0),
      max_arg_name_len_(0) {
  AddArgument("help", &help_, "Show the help message and exit", false);
}

void ArgParser::ReadCommandArgs(int argc, char const *argv[]) {
  const char *c = strrchr(argv[0], '/');
  command_name_ = c == NULL ? argv[0] : c + 1;
  int32_t i = 1;
  // processing --key name
  std::string key, value;
  while (i < argc) {
    if (i % 2 == 1) {
      if (std::strncmp(argv[i], "--", 2) == 0) {
        key = std::string(argv[i] + 2);
        if (key.size() == 0)
          LOG_FAIL << "Found invalid optional argument: \"--\"";
        // found --help
        if (key == "help") {
          PrintUsage();
          exit(0);
        }
      } else {
        // end for optional arguments list
        break;
      }
    } else {
      if (std::strncmp(argv[i], "--", 2) == 0)
        LOG_FAIL << "No value set for optional argument: --" << key;
      value = std::string(argv[i]);
      if (!SetArgument(key, value)) {
        PrintUsage();
        LOG_FAIL << "No defined optional argument: --" << key;
      }
    }
    i += 1;
  }
  if (required_args_num_ != argc - i) {
    PrintUsage();
    LOG_FAIL << "Missing required argument for command " << command_name_;
  }
  while (i < argc) {
    for (ArgumentAttr &attr : attr_args_) {
      if (!attr.required_) continue;
      value = std::string(argv[i]);
      if (!SetArgument(attr.name_, value)) {
        PrintUsage();
        LOG_FAIL << "No defined optional argument: --" << key;
      }
      i += 1;
    }
  }
}

bool ArgParser::SetArgument(const std::string &key, const std::string &value) {
  if (bool_args_.find(key) != bool_args_.end()) {
    if (value != "true" && value != "false")
      LOG_FAIL << "Invalid value for --" << key << ": " << value
               << ", required true/false";
    *(bool_args_[key]) = value == "true" ? true : false;
  } else if (int32_args_.find(key) != int32_args_.end()) {
    int32_t i;
    if (!StringToInt32(value, &i))
      LOG_FAIL << "Invalid value for --" << key << ": " << value
               << ", required int32";
    *(int32_args_[key]) = i;
  } else if (float_args_.find(key) != float_args_.end()) {
    // is float
    float f;
    if (!StringToFloat(value, &f))
      LOG_FAIL << "Invalid value for --" << key << ": " << value
               << ", required float";
    *(float_args_[key]) = f;
  } else if (string_args_.find(key) != string_args_.end()) {
    *(string_args_[key]) = value;
  } else {
    return false;
  }
  return true;
}

void ArgParser::PrintUsage() {
  std::cerr << "Usage: " << command_name_ << " <optional arguments> ";
  for (ArgumentAttr &attr : attr_args_) {
    if (attr.required_) std::cerr << "<" << attr.name_ << "> ";
  }
  std::cerr << "\n" << std::endl;
  std::cerr << description_ << std::endl;
  std::cerr << "\nRequired arguments:" << std::endl;
  for (ArgumentAttr &attr : attr_args_) {
    if (attr.required_) {
      std::cerr << "  " << std::setw(max_arg_name_len_ + 2) << std::left
                << attr.name_ << " : " << attr.help_ << std::endl;
    }
  }
  std::cerr << "\nOptional arguments:" << std::endl;
  for (ArgumentAttr &attr : attr_args_) {
    if (!attr.required_) {
      std::cerr << "  --" << std::setw(max_arg_name_len_) << std::left
                << attr.name_ << " : " << attr.help_ << std::endl;
    }
  }
  std::cerr << std::endl;
}

std::string ArgParser::NormalizeArgName(const std::string &name) {
  std::string out;
  for (const char &c : name) {
    if (c == '_')
      out += '-';
    else
      out += std::tolower(c);
  }
  ASSERT(out.length());
  return out;
}

std::string ArgParser::AddArgumentImpl(const std::string &name,
                                       const std::string &help, bool required,
                                       float *value) {
  float_args_[name] = value;
  if (required) {
    std::ostringstream ss;
    ss << help << " (float, default = " << *value << ")";
    return ss.str();
  } else {
    return help + " (float)";
  }
}

std::string ArgParser::AddArgumentImpl(const std::string &name,
                                       const std::string &help, bool required,
                                       int32_t *value) {
  int32_args_[name] = value;
  if (required) {
    std::ostringstream ss;
    ss << help << " (int32, default = " << *value << ")";
    return ss.str();
  } else {
    return help + " (int32)";
  }
}

std::string ArgParser::AddArgumentImpl(const std::string &name,
                                       const std::string &help, bool required,
                                       bool *value) {
  bool_args_[name] = value;
  if (required)
    return help + " (bool)";
  else
    return help + " (bool, default = " + ((*value) ? "true" : "false") + ")";
}

std::string ArgParser::AddArgumentImpl(const std::string &name,
                                       const std::string &help, bool required,
                                       std::string *value) {
  string_args_[name] = value;
  if (required)
    return help + " (string)";
  else
    return help + " (string, default = \"" + *value + "\")";
}

template <typename T>
void ArgParser::AddArgument(const std::string &name, T *ptr,
                            const std::string &help, bool required) {
  if (required) required_args_num_ += 1;
  max_arg_name_len_ = std::max(max_arg_name_len_, name.size());
  attr_args_.push_back(ArgumentAttr(
      name, AddArgumentImpl(NormalizeArgName(name), help, required, ptr),
      required));
}

template void ArgParser::AddArgument(const std::string &name, int32_t *ptr,
                                     const std::string &help, bool required);
template void ArgParser::AddArgument(const std::string &name, float *ptr,
                                     const std::string &help, bool required);
template void ArgParser::AddArgument(const std::string &name, bool *ptr,
                                     const std::string &help, bool required);
template void ArgParser::AddArgument(const std::string &name, std::string *ptr,
                                     const std::string &help, bool required);

}  // namespace aps
