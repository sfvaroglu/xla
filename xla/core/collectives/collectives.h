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

#ifndef XLA_CORE_COLLECTIVES_COLLECTIVES_H_
#define XLA_CORE_COLLECTIVES_COLLECTIVES_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"

namespace xla {

// Collectives is a base class for host-initiated collective operations in XLA.
//
// Host-initiated collective operations are collective operations that are
// initiated by the host runtime, i.e. in XLA:GPU the default collectives
// implementation uses NCCL and Thunks initiate collective operations of the
// runtime-managed streams.
//
// IMPORTANT: XLA also supports device-initiated collective operations, which
// are collective operations for communication between device kernels. In
// XLA:GPU device-initiated collective operations are implemented using NVSHMEM.
class Collectives {
 public:
  virtual ~Collectives();

  // A base class for the device that the collectives are running on, i.e. in
  // XLA:GPU this is the GPU device (StreamExecutor).
  class Device {
   public:
    virtual ~Device() = default;
  };

  // A collective device together with its rank in the collective clique.
  struct DeviceRank {
    DeviceRank(Device* device, RankId rank) : device(device), rank(rank) {}

    Device* device;
    RankId rank;
  };

  // A base class for the communicator configuration.
  class Config {
   public:
    virtual ~Config() = default;
  };

  // Creates a unique CliqueId.
  virtual absl::StatusOr<CliqueId> CreateUniqueCliqueId() const = 0;

  // Creates communicators for given clique key and id.
  virtual absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(const CliqueKey& clique_key,
                      const std::optional<CliqueIds>& clique_ids,
                      absl::Span<const DeviceRank> ranks,
                      const Config& config) = 0;

  // Creates communicators by splitting `comms`.
  virtual absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  SplitCommunicators(absl::Span<const Communicator* const> comms, int32_t color,
                     absl::Span<const RankId> keys, const Config& config) = 0;

  // Collectives instance can be ephemeral and used only for a small number of
  // XLA program executions. XLA backends that rely on the collectives instances
  // as a part of the cache key can be notified when the collectives instance
  // is destroyed, so that they can invalidate the cache entries.
  //
  // After the on-destroy callback is invoked, XLA backends must not use any
  // of the communicators created by the collectives instance.
  //
  // It is an XLA client responsibility (i.e. Pathways) to guarantee that
  // collectives instance stays alive until all the XLA program executions that
  // use it are finished.
  void AddOnDestroyCallback(absl::AnyInvocable<void()> callback);

 protected:
  Collectives() = default;
  Collectives(Collectives&&) = default;
  Collectives& operator=(Collectives&&) = default;

  // Notifies all registered callbacks that the collectives instance is
  // about to be destroyed.
  //
  // IMPORTANT: Because callbacks are invoked from the base class destructor,
  // they will be called after the derived class is destroyed. If it is
  // important to call callbacks before the derived class is destroyed, the
  // derived class should call it explicitly in its own destructor.
  void NotifyOnDestroyCallbacks();

 private:
  std::vector<absl::AnyInvocable<void()>> on_destroy_callbacks_;
};

}  // namespace xla

#endif  // XLA_CORE_COLLECTIVES_COLLECTIVES_H_
