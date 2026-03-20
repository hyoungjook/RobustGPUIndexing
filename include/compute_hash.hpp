/*
 *   Copyright 2025 Hyoungjoo Kim, Carnegie Mellon University
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#pragma once
#define _CG_ABI_EXPERIMENTAL  // enable experimental CGs API

#include <stdint.h>
#include <macros.hpp>
#include <utils.hpp>

namespace utils {

constexpr uint32_t PRIME0 = 0x9e3779b1;
constexpr uint32_t PRIME1 = 0x01000193;
constexpr uint32_t PRIME2 = 0xfffffffb;

DEVICE_QUALIFIER uint32_t finalize(uint32_t x) {
  // murmur3 finalizer
  x ^= x >> 16;
  x *= 0x85ebca6b;
  x ^= x >> 13;
  x *= 0xc2b2ae35;
  x ^= x >> 16;
  return x;
}

template <uint32_t prime0, typename tile_type>
DEVICE_QUALIFIER uint32_t compute_hash(const uint32_t* input, uint32_t length, const tile_type& tile) {
  // parallel polynomial rolling hash
  static constexpr uint32_t tile_size = tile_type::size();
  static constexpr uint32_t prime0_mult = utils::constexpr_pow(prime0, tile_size);
  // 1. exponent = [1, p, p^2, ..., p^31]; parallel prefix product
  uint32_t exponent = (tile.thread_rank() == 0) ? 1 : prime0;
  for (uint32_t offset = 1; offset < tile_size; offset <<= 1) {
    auto up_exponent = tile.shfl_up(exponent, offset);
    if (tile.thread_rank() >= offset) {
      exponent *= up_exponent;
    }
  }
  // 2. compute per-lane value
  const auto original_length = length;
  uint32_t hash = 0;
  while (true) {
    if (tile.thread_rank() < length) {
      auto slice = input[tile.thread_rank()];
      hash += exponent * slice;
    }
    if (length <= tile_size) { break; }
    input += tile_size;
    length -= tile_size;
    exponent *= prime0_mult;
  }
  // 3. reduce sum
  for (uint32_t offset = (tile_size / 2); offset != 0; offset >>= 1) {
    hash += tile.shfl_down(hash, offset);
  }
  hash = ((hash * prime0) + original_length) * prime0;
  // 4. finalize
  hash = finalize(hash);
  return tile.shfl(hash, 0);
}

template <uint32_t prime0>
DEVICE_QUALIFIER uint32_t compute_hash_slice(uint32_t slice) {
  // if key_length == 1, hash = finalize(((key * p) + 1) * p)
  uint32_t hash = ((slice * prime0) + 1) * prime0;
  return finalize(hash);
}

template <uint32_t prime0, uint32_t prime1, typename tile_type>
DEVICE_QUALIFIER uint2 compute_hashx2(const uint32_t* input, uint32_t length, const tile_type& tile) {
  // do compute_hash() for two different primes
  static constexpr uint32_t tile_size = tile_type::size();
  static constexpr uint32_t prime0_mult = utils::constexpr_pow(prime0, tile_size);
  static constexpr uint32_t prime1_mult = utils::constexpr_pow(prime1, tile_size);
  // 1. exponent = [1, p, p^2, ..., p^31]; parallel prefix product
  uint32_t exponent0 = (tile.thread_rank() == 0) ? 1 : prime0;
  uint32_t exponent1 = (tile.thread_rank() == 0) ? 1 : prime1;
  for (uint32_t offset = 1; offset < tile_size; offset <<= 1) {
    auto up_exponent0 = tile.shfl_up(exponent0, offset);
    auto up_exponent1 = tile.shfl_up(exponent1, offset);
    if (tile.thread_rank() >= offset) {
      exponent0 *= up_exponent0;
      exponent1 *= up_exponent1;
    }
  }
  // 2. compute per-lane value
  const auto original_length = length;
  uint32_t hash = 0, hash1 = 0;
  while (true) {
    if (tile.thread_rank() < length) {
      auto slice = input[tile.thread_rank()];
      hash += exponent0 * slice;
      hash1 += exponent1 * slice;
    }
    if (length <= tile_size) { break; }
    input += tile_size;
    length -= tile_size;
    exponent0 *= prime0_mult;
    exponent1 *= prime1_mult;
  }
  // 3. reduce sum
  for (uint32_t offset = (tile_size / 2); offset != 0; offset >>= 1) {
    hash += tile.shfl_down(hash, offset);
    hash1 += tile.shfl_up(hash1, offset);
  }
  hash = ((hash * prime0) + original_length) * prime0;
  hash1 = ((hash1 * prime1) + original_length) * prime1;
  if (tile.thread_rank() == tile_size - 1) { hash = hash1; }
  // 4. finalize
  hash = finalize(hash);
  return make_uint2(tile.shfl(hash, 0), tile.shfl(hash, tile_size - 1));
}

template <uint32_t prime0, uint32_t prime1, typename tile_type>
DEVICE_QUALIFIER uint2 compute_hashx2_slice(uint32_t slice, const tile_type& tile) {
  // if key_length == 1, hash = finalize(((key * p) + 1) * p)
  uint2 hash = make_uint2(((slice * prime0) + 1) * prime0,
                          ((slice * prime1) + 1) * prime1);
  if (tile.thread_rank() == 1) { hash.x = hash.y; }
  hash.x = finalize(hash.x);
  return make_uint2(tile.shfl(hash.x, 0), tile.shfl(hash.x, 1));
}

template <uint32_t prime0, bool use_first_slice, typename suffix_type, typename tile_type>
DEVICE_QUALIFIER uint32_t compute_hash_suffix(const suffix_type& suffix,
                                              uint32_t first_slice,
                                              const tile_type& tile) {
  // compute polynomial
  uint32_t hash = suffix.template compute_polynomial<prime0>();
  if constexpr (use_first_slice) {
    hash = (hash * prime0) + first_slice;
  }
  static constexpr uint32_t suffix_offset = use_first_slice ? 1 : 0;
  uint32_t length = suffix.get_key_length() + suffix_offset;
  hash = ((hash * prime0) + length) * prime0;
  // finalize
  return finalize(hash);
}

template <uint32_t prime0, uint32_t prime1, bool use_first_slice, typename suffix_type, typename tile_type>
static DEVICE_QUALIFIER uint2 compute_hashx2_suffix(const suffix_type& suffix,
                                                    uint32_t first_slice,
                                                    const tile_type& tile) {
  // compute polynomial
  uint2 hash = suffix.template compute_polynomialx2<prime0, prime1>();
  if constexpr (use_first_slice) {
    hash.x = (hash.x * prime0) + first_slice;
    hash.y = (hash.y * prime1) + first_slice;
  }
  static constexpr uint32_t suffix_offset = use_first_slice ? 1 : 0;
  uint32_t length = suffix.get_key_length() + suffix_offset;
  hash.x = ((hash.x * prime0) + length) * prime0;
  hash.y = ((hash.y * prime1) + length) * prime1;
  // finalize
  if (tile.thread_rank() == 1) { hash.x = hash.y; }
  hash.x = finalize(hash.x);
  return make_uint2(tile.shfl(hash.x, 0), tile.shfl(hash.x, 1));
}

} // namespace utils
