/*
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

#include <utility>
#include <stdint.h>

// runtime value to template parameter converter
namespace adapter_util {

template <class F, class... Chosen>
decltype(auto) dispatch_bool(bool b, F&& f, Chosen... chosen) {
    if (b) {
        return std::forward<F>(f)(chosen..., std::true_type{});
    } else {
        return std::forward<F>(f)(chosen..., std::false_type{});
    }
}

template <uint32_t First, uint32_t... Rest, class F, class... Chosen>
decltype(auto) dispatch_uint32(uint32_t x, F&& f, Chosen... chosen) {
    using ReturnT =
        decltype(std::forward<F>(f)(chosen..., std::integral_constant<uint32_t, First>{}));
    bool matched = false;
    if constexpr (std::is_void_v<ReturnT>) {
        auto try_one = [&](auto c) {
            if (!matched && x == decltype(c)::value) {
                matched = true;
                std::forward<F>(f)(chosen..., c);
            }
        };
        try_one(std::integral_constant<uint32_t, First>{});
        (try_one(std::integral_constant<uint32_t, Rest>{}), ...);
        if (!matched) {
            throw std::runtime_error("dispatch_uint32: unexpected runtime integer");
        }
    } else {
        std::optional<ReturnT> result;
        auto try_one = [&](auto c) {
            if (!matched && x == decltype(c)::value) {
                matched = true;
                result.emplace(std::forward<F>(f)(chosen..., c));
            }
        };
        try_one(std::integral_constant<uint32_t, First>{});
        (try_one(std::integral_constant<uint32_t, Rest>{}), ...);
        if (!matched) {
            throw std::runtime_error("dispatch_uint32: unexpected runtime integer");
        }
        return std::move(*result);
    }
}

} // namespace adapter_util

