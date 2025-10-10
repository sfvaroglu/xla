#include "xla/service/spmd/memory_aware_collective_cse.h"

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class MemoryAwareCollectiveCSETest : public HloHardwareIndependentTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_module));
    HloPassPipeline pipeline("memory-aware-collective-cse");
    pipeline.AddPass<MemoryAwareCollectiveCSE>(false);
    TF_RETURN_IF_ERROR(pipeline.Run(module.get()).status());
    return absl::StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  int64_t CountAllGathers(const HloModule* module) {
    return absl::c_count_if(module->entry_computation()->instructions(),
                            [](const HloInstruction* inst) {
                              return inst->opcode() == HloOpcode::kAllGather &&
                                     !inst->users().empty();
                            });
  }

  int64_t CountAllReduces(const HloModule* module) {
    return absl::c_count_if(module->entry_computation()->instructions(),
                            [](const HloInstruction* inst) {
                              return inst->opcode() == HloOpcode::kAllReduce &&
                                     !inst->users().empty();
                            });
  }
};

TEST_F(MemoryAwareCollectiveCSETest, VeryCloseDistanceAlwaysCSE) {
  absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  param0 = s32[1,8]{1,0} parameter(0)
  ag1 = s32[2,8]{1,0} all-gather(param0), replica_groups={{0,1}}, dimensions={0},
    channel_id=0, use_global_device_ids=true
  ag2 = s32[2,8]{1,0} all-gather(param0), replica_groups={{0,1}}, dimensions={0},
    channel_id=1, use_global_device_ids=true
  ROOT tuple = (s32[2,8]{1,0}, s32[2,8]{1,0}) tuple(ag1, ag2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string));

  EXPECT_EQ(CountAllGathers(module.get()), 1)
      << "Very close distance (<=5 instructions) should always CSE";
}

TEST_F(MemoryAwareCollectiveCSETest, TinyBufferLargeDistance) {
  absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  p0 = f32[128,128] parameter(0)
  ag1 = f32[128,512] all-gather(p0), dimensions={1}, channel_id=1,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  
  a1 = f32[128,512] add(ag1, ag1)
  a2 = f32[128,512] add(a1, a1)
  a3 = f32[128,512] add(a2, a2)
  a4 = f32[128,512] add(a3, a3)
  a5 = f32[128,512] add(a4, a4)
  a6 = f32[128,512] add(a5, a5)
  a7 = f32[128,512] add(a6, a6)
  a8 = f32[128,512] add(a7, a7)
  a9 = f32[128,512] add(a8, a8)
  a10 = f32[128,512] add(a9, a9)
  
  ag2 = f32[128,512] all-gather(p0), dimensions={1}, channel_id=2,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  ROOT result = f32[128,512] add(a10, ag2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string));

  EXPECT_EQ(CountAllGathers(module.get()), 1)
      << "Tiny buffer (256KB < 1MB) should CSE even with large distance";
}

TEST_F(MemoryAwareCollectiveCSETest, SmallBufferModerateDistance) {
  absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  p0 = f32[512,512] parameter(0)
  ag1 = f32[512,2048] all-gather(p0), dimensions={1}, channel_id=1,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  
  a1 = f32[512,2048] add(ag1, ag1)
  a2 = f32[512,2048] add(a1, a1)
  a3 = f32[512,2048] add(a2, a2)
  a4 = f32[512,2048] add(a3, a3)
  a5 = f32[512,2048] add(a4, a4)
  
  ag2 = f32[512,2048] all-gather(p0), dimensions={1}, channel_id=2,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  ROOT result = f32[512,2048] add(a5, ag2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string));

  EXPECT_EQ(CountAllGathers(module.get()), 1)
      << "Small buffer (4MB < 10MB) should CSE with moderate distance (<100)";
}

TEST_F(MemoryAwareCollectiveCSETest, LargeBufferLargeDistanceNoCSE) {
  absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  p0 = f32[4096,4096] parameter(0)
  ag1 = f32[4096,16384] all-gather(p0), dimensions={1}, channel_id=1,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  
  a1 = f32[4096,16384] add(ag1, ag1)
  a2 = f32[4096,16384] add(a1, a1)
  a3 = f32[4096,16384] add(a2, a2)
  a4 = f32[4096,16384] add(a3, a3)
  a5 = f32[4096,16384] add(a4, a4)
  a6 = f32[4096,16384] add(a5, a5)
  a7 = f32[4096,16384] add(a6, a6)
  a8 = f32[4096,16384] add(a7, a7)
  a9 = f32[4096,16384] add(a8, a8)
  a10 = f32[4096,16384] add(a9, a9)
  a11 = f32[4096,16384] add(a10, a10)
  a12 = f32[4096,16384] add(a11, a11)
  
  ag2 = f32[4096,16384] all-gather(p0), dimensions={1}, channel_id=2,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  ROOT result = f32[4096,16384] add(a12, ag2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string));

  EXPECT_EQ(CountAllGathers(module.get()), 2)
      << "Large buffer (256MB > 10MB) with large distance should NOT CSE";
}

TEST_F(MemoryAwareCollectiveCSETest, MultipleCSEOpportunities) {
  absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  p0 = f32[256,256] parameter(0)
  p1 = f32[256,256] parameter(1)
  
  ag1a = f32[256,1024] all-gather(p0), dimensions={1}, channel_id=1,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  ag1b = f32[256,1024] all-gather(p1), dimensions={1}, channel_id=2,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  
  add1 = f32[256,1024] add(ag1a, ag1b)
  add2 = f32[256,1024] add(add1, add1)
  
  ag2a = f32[256,1024] all-gather(p0), dimensions={1}, channel_id=3,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  ag2b = f32[256,1024] all-gather(p1), dimensions={1}, channel_id=4,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  
  ROOT result = f32[256,1024] add(add2, add(ag2a, ag2b))
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string));

  EXPECT_EQ(CountAllGathers(module.get()), 2)
      << "Should CSE both pairs: (ag1a, ag2a) and (ag1b, ag2b)";
}

TEST_F(MemoryAwareCollectiveCSETest, AllReduceCSE) {
  absl::string_view hlo_string = R"(
HloModule test

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[256,256] parameter(0)
  
  ar1 = f32[256,256] all-reduce(p0), channel_id=1,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}, to_apply=add
  
  a1 = f32[256,256] add(ar1, ar1)
  a2 = f32[256,256] add(a1, a1)
  
  ar2 = f32[256,256] all-reduce(p0), channel_id=2,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}, to_apply=add
  
  ROOT result = f32[256,256] add(a2, ar2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string));

  EXPECT_EQ(CountAllReduces(module.get()), 1)
      << "Memory-aware CSE should also apply to all-reduce operations";
}

TEST_F(MemoryAwareCollectiveCSETest, DifferentShapesNoCSE) {
  absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  p0 = f32[256,256] parameter(0)
  
  ag1 = f32[256,1024] all-gather(p0), dimensions={1}, channel_id=1,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  
  a1 = f32[256,1024] add(ag1, ag1)
  
  ag2 = f32[256,512] all-gather(p0), dimensions={1}, channel_id=2,
    use_global_device_ids=true, replica_groups={{0,1}}
  
  ROOT result = tuple(ag1, ag2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string));

  EXPECT_EQ(CountAllGathers(module.get()), 2)
      << "Collectives with different output shapes should not CSE";
}

TEST_F(MemoryAwareCollectiveCSETest, AdjacentCollectivesAlwaysCSE) {
  absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  p0 = f32[8192,8192] parameter(0)
  
  ag1 = f32[8192,32768] all-gather(p0), dimensions={1}, channel_id=1,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  ag2 = f32[8192,32768] all-gather(p0), dimensions={1}, channel_id=2,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  
  ROOT result = f32[8192,32768] add(ag1, ag2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string));

  EXPECT_EQ(CountAllGathers(module.get()), 1)
      << "Adjacent collectives should always CSE regardless of buffer size "
         "(1GB)";
}

TEST_F(MemoryAwareCollectiveCSETest, RealWorldPattern_TransformerLayer) {
  absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  tokens = f32[1024,4096] parameter(0)
  
  ag_qkv = f32[1024,16384] all-gather(tokens), dimensions={1}, channel_id=1,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  
  matmul1 = f32[1024,16384] multiply(ag_qkv, ag_qkv)
  softmax = f32[1024,16384] tanh(matmul1)
  
  ag_output = f32[1024,16384] all-gather(tokens), dimensions={1}, channel_id=2,
    use_global_device_ids=true, replica_groups={{0,1,2,3}}
  
  ROOT result = f32[1024,16384] add(softmax, ag_output)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, RunPass(hlo_string));

  EXPECT_EQ(CountAllGathers(module.get()), 1)
      << "Real-world transformer pattern should CSE duplicate all-gathers";
}

}  // namespace
}  // namespace xla
