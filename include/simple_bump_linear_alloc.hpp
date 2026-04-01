/*
 *   Copyright 2022 The Regents of the University of California, Davis
 *   Copyright 2026 Hyoungjoo Kim, Carnegie Mellon University
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
#include <device_context.hpp>

template <uint32_t slab_size = 128>
struct simple_bump_linear_allocator {
  using size_type = uint32_t;
  using pointer_type = size_type;
  static constexpr uint32_t slab_size_ = slab_size;
  struct two_counts {
    size_type slab_count_;    // grows upward from the bottom
    size_type linear_count_;  // grows downward from the top
  };
  struct device_instance_type {
    void* pool_;
    two_counts* count_;
    pointer_type max_count_;
  };

  simple_bump_linear_allocator(float pool_ratio = 0.9f) {
    auto meminfo = utils::compute_device_memory_usage();
    auto max_bytes = static_cast<std::size_t>(static_cast<double>(meminfo.total_bytes) * pool_ratio);
    max_count_ = static_cast<pointer_type>(max_bytes / slab_size_);
    if ((max_bytes / slab_size_) >= std::numeric_limits<pointer_type>::max()) {
      std::cerr << "simple_bump_allocator: pointer exceeds uint32 limit" << std::endl;
      abort();
    }
    auto total_bytes = static_cast<std::size_t>(max_count_) * slab_size_;
    cuda_try(cudaMalloc(&pool_, total_bytes));
    cuda_try(cudaMalloc(&count_, sizeof(two_counts)));
    cuda_try(cudaMemset(count_, 0x00, sizeof(two_counts)));
  }
  ~simple_bump_linear_allocator() {
    cuda_try(cudaFree(pool_));
    cuda_try(cudaFree(count_));
  }
  simple_bump_linear_allocator(const simple_bump_linear_allocator& other) = delete;
  simple_bump_linear_allocator& operator=(const simple_bump_linear_allocator& other) = delete;

  device_instance_type get_device_instance() const {
    return device_instance_type{pool_, count_, max_count_};
  }

  void print_stats() const {
    two_counts h_count;
    cuda_try(cudaMemcpy(&h_count, count_, sizeof(two_counts), cudaMemcpyDeviceToHost));
    std::cout << "simple_bump_linear_allocator(" << slab_size_ << "B slabs): "
        << h_count.slab_count_ << "/" << max_count_ << " slabs + " << h_count.linear_count_ << " linear allocated " 
        << "(" << (float)(((std::size_t)h_count.slab_count_) * slab_size_ + 
                          ((std::size_t)h_count.linear_count_) * sizeof(size_type)) /
                         (((std::size_t)max_count_) * slab_size_) * 100.0f << "%)" << std::endl;
  }

private:
  void* pool_;
  two_counts* count_;
  pointer_type max_count_;
};

template <uint32_t slab_size>
struct device_allocator_context<simple_bump_linear_allocator<slab_size>> {
  using host_alloc_type = simple_bump_linear_allocator<slab_size>;
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
      new_slab_index = atomicAdd(&alloc_.count_->slab_count_, 1);
      assert(check_counters(new_slab_index + 1, alloc_.count_->linear_count_));
    }
    return tile.shfl(new_slab_index, 0);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER void deallocate_coop(pointer_type p, const tile_type& tile) noexcept {}
  DEVICE_QUALIFIER uint32_t deallocate_perlane(pointer_type p) noexcept { return 0; }
  template <typename tile_type>
  DEVICE_QUALIFIER void deallocate_perlane_finish(uint32_t sum, const tile_type& tile) noexcept {}

  DEVICE_QUALIFIER void* address(pointer_type p) const {
    return reinterpret_cast<void*>(reinterpret_cast<slab_type*>(alloc_.pool_) + p);
  }

  DEVICE_QUALIFIER void* get_linear() const {
    return reinterpret_cast<void*>(reinterpret_cast<slab_type*>(alloc_.pool_) + alloc_.max_count_);
  }

  template <typename tile_type>
  DEVICE_QUALIFIER size_type reallocate_linear(size_type size, const tile_type& tile) {
    if (tile.thread_rank() == 0) {
      atomicExch(&alloc_.count_->linear_count_, size);
      assert(check_counters(alloc_.count_->slab_count_, size));
    }
    return size;
  }

private:
  static constexpr uint32_t slab_size_ = host_alloc_type::slab_size_;
  static constexpr uint32_t linear_elems_per_slab_ = slab_size_ / sizeof(size_type);
  struct slab_type { uint8_t _[slab_size_]; };

  const device_instance_type& alloc_;

  DEVICE_QUALIFIER bool check_counters(size_type slab_count, size_type linear_count) {
    return (slab_count + ((linear_count + linear_elems_per_slab_ - 1) / linear_elems_per_slab_) <= alloc_.max_count_);
  }
};
