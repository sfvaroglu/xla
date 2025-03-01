/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/codegen/kernel_spec.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla {

KernelSpec::KernelSpec(absl::string_view name, se::ThreadDim thread_dim,
                       Buffers argument_buffers, Buffers result_buffers,
                       absl::flat_hash_set<int64_t> invariant_arguments,
                       std::optional<size_t> scratch_bytes)
    : KernelSpec(name, se::ClusterDim(), se::BlockDim(), thread_dim,
                 std::move(argument_buffers), std::move(result_buffers),
                 std::move(invariant_arguments), std::move(scratch_bytes)) {}

KernelSpec::KernelSpec(absl::string_view name, se::ClusterDim cluster_dim,
                       se::BlockDim block_dim, se::ThreadDim thread_dim,
                       Buffers argument_buffers, Buffers result_buffers,
                       absl::flat_hash_set<int64_t> invariant_arguments,
                       std::optional<size_t> scratch_bytes)
    : name_(name),
      cluster_dim_(cluster_dim),
      block_dim_(block_dim),
      thread_dim_(thread_dim),
      argument_buffers_(std::move(argument_buffers)),
      result_buffers_(std::move(result_buffers)),
      invariant_arguments_(std::move(invariant_arguments)),
      scratch_bytes_(scratch_bytes) {}

}  // namespace xla
