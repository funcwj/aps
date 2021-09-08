// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#ifndef CSRC_UTILS_WAV_H_
#define CSRC_UTILS_WAV_H_

#include <algorithm>
#include <iostream>
#include <string>

#include "utils/io.h"
#include "utils/math.h"

namespace aps {

struct ChunkHeader {
  char id[4];
  uint32_t size;
};

struct RiffAndFmtHeader {
  ChunkHeader riff;       // "RIFF" + ...
  char format[4];         // "WAVE"
  ChunkHeader fmt;        // "fmt " + ...
  uint16_t audio_format;  // 1
  uint16_t num_channels;  // 1/2/..
  uint32_t sample_rate;   // 8000/16000
  uint32_t byte_rate;     // sample_rate * num_channel * sizeof(T)
  uint16_t block_align;   // num_channels * sizeof(T)
  uint16_t bit_width;     // sizeof(T) * 8
};

struct WavHeader {
  RiffAndFmtHeader riff_and_fmt;
  ChunkHeader data;  // "data" + ...
};

class WavReader {
 public:
  explicit WavReader(const std::string &filename);

  int NumChannels() const { return header_.riff_and_fmt.num_channels; }
  int SampleRate() const { return header_.riff_and_fmt.sample_rate; }
  size_t NumSamples() const { return num_samples_; }
  std::string Info() const;

  size_t Read(float *data_ptr, size_t num_samples, bool normalized = true);
  bool Done() { return read_samples_ >= num_samples_; }
  void Close() { is_.Close(); }
  void Reset();

 private:
  WavHeader header_;
  BinaryInput is_;
  size_t num_samples_;
  size_t read_samples_;
};

class WavWriter {
 public:
  explicit WavWriter(const std::string &filename, int sample_rate,
                     int num_channels, int bit_width);

  int NumChannels() const { return header_.riff_and_fmt.num_channels; }
  int SampleRate() const { return header_.riff_and_fmt.sample_rate; }
  size_t NumSamples() const { return write_samples_; }
  std::string Info() const;

  size_t Write(float *data_ptr, size_t num_samples, bool norm = true);
  void Close() {
    Flush();
    os_.Close();
  }
  void Reset();

 private:
  WavHeader header_;
  BinaryOutput os_;
  size_t write_samples_;

  void Flush();
};

}  // namespace aps

#endif  // CSRC_UTILS_WAV_H_
