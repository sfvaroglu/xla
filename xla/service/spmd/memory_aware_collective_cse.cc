#include "xla/service/spmd/memory_aware_collective_cse.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {

constexpr int64_t kVeryCloseDistance = 5;
constexpr int64_t kSmallBufferBytes = 10 << 20;
constexpr int64_t kTinyBufferBytes = 1 << 20;
constexpr int64_t kMediumDistance = 100;
constexpr int64_t kLargeDistance = 1000;

bool IsAddingOnlyDegenerateDimensions(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kReshape) {
    return false;
  }
  return ShapeUtil::ElementsIn(inst->shape()) ==
         ShapeUtil::ElementsIn(inst->operand(0)->shape());
}

HloInstruction* PassthroughDegenerateAddingReshapes(HloInstruction* hlo) {
  while (IsAddingOnlyDegenerateDimensions(hlo)) {
    hlo = hlo->mutable_operand(0);
  }
  return hlo;
}

}  // namespace

int64_t GetBufferSizeBytes(const HloInstruction* inst) {
  return ShapeUtil::ByteSizeOf(inst->shape());
}

bool ShouldCSECollectives(const HloInstruction* earlier_coll,
                          const HloInstruction* later_coll, int64_t distance,
                          int64_t buffer_size) {
  VLOG(3) << "Considering CSE: " << earlier_coll->name() << " -> "
          << later_coll->name() << " (distance=" << distance
          << ", size=" << buffer_size << " bytes)";

  if (distance <= kVeryCloseDistance) {
    VLOG(2) << "CSE approved: very close distance (" << distance << ")";
    return true;
  }

  if (buffer_size < kTinyBufferBytes && distance <= kLargeDistance) {
    VLOG(2) << "CSE approved: tiny buffer (" << buffer_size << " bytes) "
            << "with acceptable distance (" << distance << ")";
    return true;
  }

  if (buffer_size < kSmallBufferBytes && distance <= kMediumDistance) {
    VLOG(2) << "CSE approved: small buffer (" << buffer_size << " bytes) "
            << "with moderate distance (" << distance << ")";
    return true;
  }

  if (distance <= kVeryCloseDistance * 2) {
    VLOG(2) << "CSE approved: large buffer (" << buffer_size << " bytes) "
            << "but still very close (" << distance << ")";
    return true;
  }

  VLOG(2) << "CSE rejected: buffer too large (" << buffer_size << " bytes) "
          << "or distance too far (" << distance << ")";
  return false;
}

namespace {

absl::StatusOr<bool> RunOnComputation(HloComputation* comp, bool for_replicas) {
  VLOG(2) << "Running memory-aware collective CSE on " << comp->name();

  bool changed = false;

  absl::flat_hash_map<const HloInstruction*, int64_t> height;
  auto ordered_hlos = comp->MakeInstructionPostOrder();

  for (auto it = ordered_hlos.rbegin(); it != ordered_hlos.rend(); ++it) {
    auto hlo = *it;
    int64_t h = 0;
    for (auto user : hlo->users()) {
      h = std::max(h, height[user] + 1);
    }
    height[hlo] = h;
  }

  auto lowest_user_height = [&](const HloInstruction* hlo) {
    int64_t lowest = height[hlo];
    for (auto user : hlo->users()) {
      lowest = std::min(lowest, height[user]);
    }
    return lowest;
  };

  absl::flat_hash_map<const HloInstruction*, std::vector<HloInstruction*>>
      operand_to_collective;

  for (HloInstruction* hlo : ordered_hlos) {
    // Only consider all-gather, all-reduce, and reduce-scatter with channel IDs
    if (hlo->opcode() != HloOpcode::kAllGather &&
        hlo->opcode() != HloOpcode::kAllReduce &&
        hlo->opcode() != HloOpcode::kReduceScatter) {
      continue;
    }
    if (!DynCast<HloChannelInstruction>(hlo)) {
      continue;
    }

    auto& earlier_colls =
        operand_to_collective[PassthroughDegenerateAddingReshapes(
            hlo->mutable_operand(0))];
    bool found = false;
    int64_t hlo_height = height[hlo];

    for (HloInstruction* earlier_coll : earlier_colls) {
      if (!ShapeUtil::Equal(earlier_coll->shape(), hlo->shape())) {
        continue;
      }

      HloInstruction* hlo_operand = hlo->mutable_operand(0);
      TF_RETURN_IF_ERROR(
          hlo->ReplaceOperandWith(0, earlier_coll->mutable_operand(0)));

      if (!earlier_coll->IdenticalIgnoringChannelIdValues(*hlo)) {
        TF_RETURN_IF_ERROR(hlo->ReplaceOperandWith(0, hlo_operand));
        continue;
      }

      int64_t distance = lowest_user_height(earlier_coll) - hlo_height;
      int64_t buffer_size = GetBufferSizeBytes(earlier_coll);

      if (!ShouldCSECollectives(earlier_coll, hlo, distance, buffer_size)) {
        VLOG(1) << "Skipping CSE due to memory concerns: "
                << earlier_coll->name() << " -> " << hlo->name()
                << " (distance=" << distance
                << ", buffer_size=" << (buffer_size >> 20) << "MB)";
        TF_RETURN_IF_ERROR(hlo->ReplaceOperandWith(0, hlo_operand));
        continue;
      }

      found = true;
      changed = true;
      VLOG(1) << "Applying CSE: " << hlo->ToString() << " -> "
              << earlier_coll->ToString() << " (distance=" << distance
              << ", buffer=" << (buffer_size >> 20) << "MB)";
      TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(earlier_coll));
      break;
    }

    if (!found) {
      earlier_colls.push_back(hlo);
    }
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> MemoryAwareCollectiveCSE::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running memory-aware collective CSE on module: "
          << module->name();

  bool changed = false;
  for (auto comp : module->computations(execution_threads)) {
    VLOG(2) << "Processing computation: " << comp->name();
    TF_ASSIGN_OR_RETURN(auto comp_changed,
                        RunOnComputation(comp, for_replicas_));
    changed |= comp_changed;
    VLOG(2) << "Computation processed, changed=" << comp_changed;
  }

  if (changed) {
    VLOG(1) << "Memory-aware collective CSE made changes to " << module->name();
  }

  VLOG(1) << "Memory-aware collective CSE completed";
  return changed;
}

}  // namespace xla
