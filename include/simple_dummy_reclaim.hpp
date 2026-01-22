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
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <host_allocators.hpp>
#include <macros.hpp>
#include <memory>
#include <memory_utils.hpp>

struct simple_dummy_reclaimer {
  using pointer_type = uint32_t;
  struct device_instance_type {};
  simple_dummy_reclaimer() {}
  ~simple_dummy_reclaimer() {}
  simple_dummy_reclaimer(const simple_dummy_reclaimer& other) = delete;
  simple_dummy_reclaimer& operator=(const simple_dummy_reclaimer& other) = delete;

  device_instance_type get_device_instance() const { return device_instance_type{}; }

  static constexpr uint32_t block_size_ = 128;
};

template <>
struct device_reclaimer_context<simple_dummy_reclaimer> {
  using host_reclaim_type = simple_dummy_reclaimer;
  using device_instance_type = typename host_reclaim_type::device_instance_type;
  using pointer_type = typename host_reclaim_type::pointer_type;
  static constexpr uint32_t block_size_ = host_reclaim_type::block_size_;
  __host__ __device__ static constexpr uint32_t required_shmem_size() { return 0; }

  template <typename tile_type>
  DEVICE_QUALIFIER device_reclaimer_context(const device_instance_type& reclaimer,
                                            uint32_t* shmem_buffer,
                                            uint32_t num_active_blocks,
                                            const tile_type& block_wide_tile) noexcept {}

  template <typename tile_type>
  DEVICE_QUALIFIER void retire(const pointer_type& address, const tile_type& tile) noexcept {}

  template <typename block_type, typename allocator_type>
  DEVICE_QUALIFIER void begin_critical_section(const block_type& block, allocator_type& allocator) noexcept {}

  template <typename block_type>
  DEVICE_QUALIFIER void end_critical_section(const block_type& block) noexcept {}

  template <typename block_type, typename tile_type, typename allocator_type>
  DEVICE_QUALIFIER void drain_all(const block_type& block, const tile_type& tile, allocator_type& allocator) noexcept {}
};
