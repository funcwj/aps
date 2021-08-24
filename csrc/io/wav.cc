// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "io/wav.h"

bool CheckRiffHeader(const WavHeader &header) {
  const RiffAndFmtHeader *riff_and_fmt = &(header.riff_and_fmt);
  const ChunkHeader *data = &(header.data);
  if (strncmp(reinterpret_cast<const char *>(riff_and_fmt->riff.id), "RIFF",
              4) != 0)
    return false;
  if (strncmp(reinterpret_cast<const char *>(riff_and_fmt->format), "WAVE",
              4) != 0)
    return false;
  if (strncmp(reinterpret_cast<const char *>(riff_and_fmt->fmt.id), "fmt ",
              4) != 0)
    return false;
  if (strncmp(reinterpret_cast<const char *>(data->id), "data", 4) != 0)
    return false;
  if (riff_and_fmt->sample_rate * riff_and_fmt->num_channels *
          riff_and_fmt->bit_width !=
      riff_and_fmt->byte_rate * 8)
    return false;
  if (riff_and_fmt->num_channels * riff_and_fmt->bit_width !=
      riff_and_fmt->block_align * 8)
    return false;
  uint32_t num_samples = data->size * 8 / riff_and_fmt->bit_width;
  if (num_samples % riff_and_fmt->num_channels != 0) return false;
  if (riff_and_fmt->riff.size <
      data->size + sizeof(WavHeader) - sizeof(ChunkHeader))
    return false;
  return true;
}

bool ReadWavHeader(std::istream &is, WavHeader *header) {
  RiffAndFmtHeader *riff_and_fmt = &(header->riff_and_fmt);
  ReadBinary(is, reinterpret_cast<char *>(riff_and_fmt),
             sizeof(RiffAndFmtHeader));
  if (riff_and_fmt->audio_format != 1)
    LOG_FAIL << "Now only support audio_format == 1, but get "
             << riff_and_fmt->audio_format;
  // Skip other parameters between format part and data part
  int64_t skip_bytes = riff_and_fmt->fmt.size - 16;
  if (skip_bytes < 0) LOG_FAIL << "Size of FmtSubChunk < 16";
  if (skip_bytes > 0) Seek(is, skip_bytes, std::ios::cur);
  ChunkHeader *data = &(header->data);
  ReadBinary(is, reinterpret_cast<char *>(data), sizeof(ChunkHeader));
  if (header->data.size == 0) LOG_FAIL << "Get data size == 0";
  if (!CheckRiffHeader(*header)) return false;
  return true;
}

WavReader::WavReader(const std::string &filename) {
  is_.Open(filename);
  if (!ReadWavHeader(is_.Stream(), &header_))
    LOG_FAIL << filename << ": Get bad wave header ...";
  // consider multi-channel audio cases
  num_samples_ =
      header_.data.size * 8 /
      (header_.riff_and_fmt.bit_width * header_.riff_and_fmt.num_channels);
  read_samples_ = 0;
}

size_t WavReader::Read(float *data_ptr, size_t num_samples, bool norm) {
  size_t start = read_samples_, offset = 0;
  while (read_samples_ < start + num_samples) {
    for (size_t c = 0; c < header_.riff_and_fmt.num_channels; c++) {
      offset = c * num_samples + read_samples_ - start;
      switch (header_.riff_and_fmt.bit_width) {
        case 8: {
          int8_t s;
          float denorm = norm ? MAX_INT8 : 1.0f;
          ReadBinaryBasicType(is_.Stream(), &s);
          data_ptr[offset] = static_cast<float>(s) / denorm;
          break;
        }
        case 16: {
          int16_t s;
          float denorm = norm ? MAX_INT16 : 1.0f;
          ReadBinaryBasicType(is_.Stream(), &s);
          data_ptr[offset] = static_cast<float>(s) / denorm;
          break;
        }
        case 32: {
          int32_t s;
          float denorm = norm ? MAX_INT32 : 1.0f;
          ReadBinaryBasicType(is_.Stream(), &s);
          data_ptr[offset] = static_cast<float>(s) / denorm;
          break;
        }
        default:
          LOG_FAIL << "Unsupported bit width ("
                   << header_.riff_and_fmt.bit_width << ")";
      }
    }
    read_samples_ += 1;
    if (read_samples_ >= num_samples_) break;
  }
  return read_samples_ - start;
}

std::string WavReader::Info() const {
  std::ostringstream os;
  os << "num_channels=" << NumChannels()
     << ", bit_width=" << header_.riff_and_fmt.bit_width / 8
     << ", sample_rate=" << SampleRate() << ", num_samples=" << NumSamples();
  return os.str();
}

void WavReader::Reset() {
  Seek(is_.Stream(), sizeof(WavHeader) + header_.riff_and_fmt.fmt.size - 16,
       std::ios_base::beg);
}

WavWriter::WavWriter(const std::string &filename, int sample_rate,
                     int num_channels, int bit_width) {
  header_ = {'R',
             'I',
             'F',
             'F',
             0,  // placehold
             'W',
             'A',
             'V',
             'E',
             'f',
             'm',
             't',
             ' ',
             16,
             1,
             static_cast<uint16_t>(num_channels),
             static_cast<uint32_t>(sample_rate),
             static_cast<uint32_t>(sample_rate * bit_width * num_channels),
             static_cast<uint16_t>(bit_width * num_channels),
             static_cast<uint16_t>(bit_width * 8),
             'd',
             'a',
             't',
             'a',
             0};
  os_.Open(filename);
  WriteBinary(os_.Stream(), reinterpret_cast<char *>(&header_),
              sizeof(header_));
  write_samples_ = 0;
}

size_t WavWriter::Write(float *data_ptr, size_t num_samples, bool norm) {
  size_t offset = 0;
  for (size_t n = 0; n < num_samples; n++) {
    for (size_t c = 0; c < header_.riff_and_fmt.num_channels; c++) {
      offset = c * num_samples + n;
      switch (header_.riff_and_fmt.bit_width) {
        case 8: {
          float scale = norm ? MAX_INT8 : 1.0f;
          int8_t s = static_cast<int8_t>(data_ptr[offset] * scale);
          WriteBinaryBasicType(os_.Stream(), s);
          break;
        }
        case 16: {
          float scale = norm ? MAX_INT16 : 1.0f;
          int16_t s = static_cast<int16_t>(data_ptr[offset] * scale);
          WriteBinaryBasicType(os_.Stream(), s);
          break;
        }
        case 32: {
          float scale = norm ? MAX_INT32 : 1.0f;
          int32_t s = static_cast<int32_t>(data_ptr[offset] * scale);
          WriteBinaryBasicType(os_.Stream(), s);
          break;
        }
        default:
          LOG_FAIL << "Unsupported bit width ("
                   << header_.riff_and_fmt.bit_width << ")";
      }
    }
  }
  write_samples_ += num_samples;
  return num_samples;
}

void WavWriter::Flush() {
  int num_bytes =
      header_.riff_and_fmt.bit_width * NumChannels() * NumSamples() / 8;
  header_.riff_and_fmt.riff.size =
      num_bytes + sizeof(WavHeader) - sizeof(ChunkHeader);
  header_.data.size = num_bytes;
  Seek(os_.Stream(), 0, std::ios_base::beg);
  WriteBinary(os_.Stream(), reinterpret_cast<char *>(&header_),
              sizeof(WavHeader));
}

void WavWriter::Reset() {
  Seek(os_.Stream(), sizeof(WavHeader), std::ios_base::beg);
}

std::string WavWriter::Info() const {
  std::ostringstream os;
  os << "num_channels=" << NumChannels()
     << ", bit_width=" << header_.riff_and_fmt.bit_width / 8
     << ", sample_rate=" << SampleRate() << ", num_samples=" << NumSamples();
  return os.str();
}
