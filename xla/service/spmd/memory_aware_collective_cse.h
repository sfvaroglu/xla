#ifndef XLA_SERVICE_MEMORY_AWARE_COLLECTIVE_CSE_H_
#define XLA_SERVICE_MEMORY_AWARE_COLLECTIVE_CSE_H_

#include <cstdint>
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

class MemoryAwareCollectiveCSE : public HloModulePass {
 public:
  explicit MemoryAwareCollectiveCSE(bool for_replicas)
      : for_replicas_(for_replicas) {}

  ~MemoryAwareCollectiveCSE() override = default;

  absl::string_view name() const override {
    return "memory-aware-collective-cse";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  bool for_replicas_;
};

int64_t GetBufferSizeBytes(const HloInstruction* inst);

bool ShouldCSECollectives(const HloInstruction* earlier_coll,
                          const HloInstruction* later_coll, int64_t distance,
                          int64_t buffer_size);

}  // namespace xla

#endif
