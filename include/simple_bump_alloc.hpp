/*
 *   Copyright 2022 The Regents of the University of California, Davis
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
#include <thrust/device_vector.h>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <macros.hpp>
#include <memory>
#include <type_traits>
#include <host_allocators.hpp>

template <uint32_t slab_size = 128, std::size_t max_bytes = 6UL * 1024 * 1024 * 1024>
struct simple_bump_allocator {
  using size_type = uint32_t;
  using pointer_type = size_type;
  static constexpr uint32_t slab_size_ = slab_size;
  static constexpr std::size_t max_count_ = max_bytes / slab_size_;
  static constexpr std::size_t total_bytes_ = max_count_ * slab_size_;
  struct device_instance_type {
    void* pool_;
    size_type* slab_count_;
  };

  simple_bump_allocator() {
    cuda_try(cudaMalloc(&pool_, total_bytes_));
    cuda_try(cudaMalloc(&slab_count_, sizeof(size_type)));
    cuda_try(cudaMemset(slab_count_, 0x00, sizeof(size_type)));
  }
  ~simple_bump_allocator() {
    cuda_try(cudaFree(pool_));
    cuda_try(cudaFree(slab_count_));
  }
  simple_bump_allocator(const simple_bump_allocator& other) = delete;
  simple_bump_allocator& operator=(const simple_bump_allocator& other) = delete;

  device_instance_type get_device_instance() const { return device_instance_type{pool_, slab_count_}; }

private:
  void* pool_;
  size_type* slab_count_;
};

template <uint32_t slab_size, std::size_t max_bytes>
struct device_allocator_context<simple_bump_allocator<slab_size, max_bytes>> {
  using host_alloc_type = simple_bump_allocator<slab_size, max_bytes>;
  using device_instance_type = typename host_alloc_type::device_instance_type;
  using size_type = typename host_alloc_type::size_type;
  using pointer_type = typename host_alloc_type::pointer_type;

  template <typename tile_type>
  DEVICE_QUALIFIER device_allocator_context(const device_instance_type& alloc, const tile_type& tile)
      : alloc_(alloc) {}

  template <typename tile_type>
  DEVICE_QUALIFIER pointer_type allocate(const tile_type& tile) {
    pointer_type new_slab_index;
    if (tile.thread_rank() == 0) {
      new_slab_index = atomicAdd(alloc_.slab_count_, 1);
      cuda_assert(new_slab_index != max_count_);
    }
    return tile.shfl(new_slab_index, 0);
  }

  DEVICE_QUALIFIER void deallocate(pointer_type p) noexcept {}

  DEVICE_QUALIFIER void* address(pointer_type p) const {
    return reinterpret_cast<void*>(reinterpret_cast<slab_type*>(alloc_.pool_) + p);
  }

private:
  static constexpr uint32_t slab_size_ = host_alloc_type::slab_size_;
  static constexpr std::size_t max_count_ = host_alloc_type::max_count_;
  static_assert(max_count_ <= std::numeric_limits<pointer_type>::max());
  struct slab_type { uint8_t _[slab_size_]; };

  const device_instance_type& alloc_;
};
