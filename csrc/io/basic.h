// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#ifndef CSRC_IO_BASIC_H
#define CSRC_IO_BASIC_H

#include <fstream>
#include <iostream>

#include "utils/log.h"

class BinaryInput {
 public:
  BinaryInput() {}
  BinaryInput(const std::string &filename) : filename_(filename) {
    Open(filename);
  }

  ~BinaryInput() { Close(); }

  void Open(const std::string &filename) {
    is_.open(filename.c_str(), std::ios::in | std::ios::binary);
    if (!is_.is_open()) LOG_FAIL << "Open " << filename << " failed";
  }

  void Close() {
    if (is_.is_open()) is_.close();
  }

  std::istream &Stream() { return is_; }

 private:
  std::string filename_;
  std::ifstream is_;
};

class BinaryOutput {
 public:
  BinaryOutput() {}
  BinaryOutput(const std::string &filename) : filename_(filename) {
    Open(filename_);
  }


  ~BinaryOutput() { Close(); }

  void Open(const std::string &filename) {
    os_.open(filename.c_str(), std::ios::out | std::ios::binary);
    if (!os_.is_open()) LOG_FAIL << "Open " << filename << " failed";
  }

  void Close() {
    if (os_.is_open()) os_.close();
  }

  std::ostream &Stream() { return os_; }

 private:
  std::string filename_;
  std::ofstream os_;
};

void WriteBinary(std::ostream &os, const char *ptr, uint32_t num_bytes);

void ReadBinary(std::istream &is, char *ptr, uint32_t num_bytes);

void Seek(std::istream &is, int64_t off, std::ios_base::seekdir way);

void Seek(std::ostream &os, int64_t off, std::ios_base::seekdir way);

template <class T>
void ReadBinaryBasicType(std::istream &is, T *t);

template <class T>
void WriteBinaryBasicType(std::ostream &os, T t);

#endif
