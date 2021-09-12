// Copyright 2018 Jian Wu
// License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

#include "utils/io.h"

namespace aps {

void WriteBinary(std::ostream &os, const char *ptr, uint32_t num_bytes) {
  os.write(ptr, num_bytes);
  ASSERT(!os.fail() && "WriteBinary Failed");
}

void ReadBinary(std::istream &is, char *ptr, uint32_t num_bytes) {
  is.read(ptr, num_bytes);
  ASSERT(!is.fail() && "ReadBinary Failed");
}

void Seek(std::istream &is, int64_t off, std::ios_base::seekdir way) {
  is.seekg(off, way);
}

void Seek(std::ostream &os, int64_t off, std::ios_base::seekdir way) {
  os.seekp(off, way);
}

template <class T>
void ReadBinaryBasicType(std::istream &is, T *t) {
  ReadBinary(is, reinterpret_cast<char *>(t), sizeof(*t));
  ASSERT(!is.fail() && "ReadBinaryBasicType Failed");
}

template <class T>
void WriteBinaryBasicType(std::ostream &os, T t) {
  WriteBinary(os, reinterpret_cast<const char *>(&t), sizeof(t));
  ASSERT(!os.fail() && "WriteBinaryBasicType Failed");
}

template void ReadBinaryBasicType<int8_t>(std::istream &is, int8_t *t);
template void ReadBinaryBasicType<int16_t>(std::istream &is, int16_t *t);
template void ReadBinaryBasicType<int32_t>(std::istream &is, int32_t *t);

template void WriteBinaryBasicType<int8_t>(std::ostream &os, int8_t t);
template void WriteBinaryBasicType<int16_t>(std::ostream &os, int16_t t);
template void WriteBinaryBasicType<int32_t>(std::ostream &os, int32_t t);

}  // namespace aps
