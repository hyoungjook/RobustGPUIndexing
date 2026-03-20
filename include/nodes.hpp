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

#include <type_traits>
#include <masstree_node_warp.hpp>
#include <masstree_node_subwarp.hpp>
#include <hashtable_node_warp.hpp>
#include <hashtable_node_subwarp.hpp>
#include <suffix_node_warp.hpp>
#include <suffix_node_subwarp.hpp>

template <typename tile_type, typename allocator_type>
using masstree_node = std::conditional_t<tile_type::size() == 32,
                                         masstree_node_warp<tile_type, allocator_type>,
                                         masstree_node_subwarp<tile_type, allocator_type>>;

template <typename tile_type, typename allocator_type>
using hashtable_node = std::conditional_t<tile_type::size() == 32,
                                          hashtable_node_warp<tile_type, allocator_type>,
                                          hashtable_node_subwarp<tile_type, allocator_type>>;

template <typename tile_type, typename allocator_type>
using suffix_node = std::conditional_t<tile_type::size() == 32,
                                       suffix_node_warp<tile_type, allocator_type>,
                                       suffix_node_subwarp<tile_type, allocator_type>>;
