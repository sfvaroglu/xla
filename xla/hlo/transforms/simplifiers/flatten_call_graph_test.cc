/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"

#include <cstdint>
#include <memory>
#include <string>

#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal_util.h"
#include "xla/service/call_graph.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class FlattenCallGraphTest : public HloHardwareIndependentTestBase {
 protected:
  // Build and return a trivial computation taking and returning a scalar.
  std::unique_ptr<HloComputation> MakeScalarComputation() {
    HloComputation::Builder builder(TestName() + ".ScalarComputation");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    builder.AddInstruction(
        HloInstruction::CreateUnary(kScalarShape, HloOpcode::kNegate, param0));
    return builder.Build();
  }

  // Build and return a computation which takes a scalar and maps (kMap) the
  // given computation to the value 'callsites' number of times.
  std::unique_ptr<HloComputation> MakeMappingComputation(
      HloComputation* map_computation, int64_t callsites) {
    HloComputation::Builder builder(TestName() + ".MappingComputation");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* last_value = param0;
    for (int64_t i = 0; i < callsites; ++i) {
      last_value = builder.AddInstruction(HloInstruction::CreateMap(
          kScalarShape, {last_value}, map_computation));
    }
    return builder.Build();
  }

  // Build and return a computation which takes a scalar and calls (kCall) the
  // given computation with value 'callsites' number of times.
  std::unique_ptr<HloComputation> MakeCallingComputation(
      HloComputation* callee_computation, int64_t callsites,
      const std::string& suffix = ".CallingComputation") {
    HloComputation::Builder builder(TestName() + suffix);
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* last_value = param0;
    for (int64_t i = 0; i < callsites; ++i) {
      last_value = builder.AddInstruction(HloInstruction::CreateCall(
          kScalarShape, {last_value}, callee_computation));
    }
    return builder.Build();
  }

  // Build and return a computation which takes a scalar and returns a PRED
  // value.
  std::unique_ptr<HloComputation> MakeConditionComputation() {
    HloComputation::Builder builder(TestName() + ".ConditionComputation");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* zero = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
    builder.AddInstruction(
        HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), param0,
                                      zero, ComparisonDirection::kGt));
    return builder.Build();
  }

  absl::StatusOr<bool> RunFlattenCallGraph(HloModule* module) {
    FlattenCallGraph flatten;
    TF_ASSIGN_OR_RETURN(bool result, flatten.Run(module));
    return result;
  }

  const Shape kScalarShape = ShapeUtil::MakeShape(F32, {});
};

TEST_F(FlattenCallGraphTest, ComplexGraph) {
  // Test a call graph of a module with several computation called in various
  // contexts. The call graph looks like:
  //
  //      entry
  //      /  |
  //     a   |
  //   / | \ |
  //  b  |  cond
  //   \ |
  //    c
  //
  // Calls are made via kCall, kWhile, and kMap instructions.
  auto module = CreateNewVerifiedModule();
  HloComputation* cond_computation =
      module->AddEmbeddedComputation(MakeConditionComputation());
  HloComputation* c_computation =
      module->AddEmbeddedComputation(MakeScalarComputation());
  HloComputation* b_computation = module->AddEmbeddedComputation(
      MakeMappingComputation(c_computation, /*callsites=*/1));

  HloComputation* a_computation;
  {
    HloComputation::Builder builder(TestName() + ".a");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    HloInstruction* call = builder.AddInstruction(
        HloInstruction::CreateCall(kScalarShape, {param0}, c_computation));
    builder.AddInstruction(HloInstruction::CreateWhile(
        kScalarShape, cond_computation, b_computation, call));
    a_computation = module->AddEmbeddedComputation(builder.Build());
  }

  HloComputation* entry_computation;
  {
    HloComputation::Builder builder(TestName() + ".entry");
    HloInstruction* param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, kScalarShape, "param0"));
    builder.AddInstruction(HloInstruction::CreateWhile(
        kScalarShape, cond_computation, a_computation, param0));
    entry_computation = module->AddEntryComputation(builder.Build());
  }

  {
    TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
    EXPECT_TRUE(result);
    std::unique_ptr<CallGraph> flat_call_graph = CallGraph::Build(module.get());
    const CallGraphNode& c_node = flat_call_graph->GetNode(c_computation);
    EXPECT_EQ(1, c_node.caller_callsites().size());
  }
}

// Test corner case of a computation used as a body and a loop condition.
TEST_F(FlattenCallGraphTest, SharedWhileConditionAndBody) {
  auto module = CreateNewVerifiedModule();
  HloComputation* cond_computation;
  {
    HloComputation::Builder builder(TestName() + ".cond");
    HloInstruction* param0 =
        builder.AddInstruction(HloInstruction::CreateParameter(
            0, ShapeUtil::MakeShape(PRED, {}), "param0"));
    HloInstruction* false_constant = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
    builder.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), param0, false_constant,
        ComparisonDirection::kEq));
    cond_computation = module->AddEmbeddedComputation(builder.Build());
  }

  HloComputation* entry_computation;
  HloInstruction* while_op;
  {
    HloComputation::Builder builder(TestName() + ".entry");
    HloInstruction* false_constant = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
    while_op = builder.AddInstruction(HloInstruction::CreateWhile(
        ShapeUtil::MakeShape(PRED, {}), cond_computation, cond_computation,
        false_constant));
    entry_computation = module->AddEntryComputation(builder.Build());
  }

  {
    std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
    const CallGraphNode& cond_node = call_graph->GetNode(cond_computation);
    EXPECT_EQ(2, cond_node.caller_callsites().size());
  }

  {
    TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
    EXPECT_TRUE(result);
    EXPECT_NE(while_op->while_body(), while_op->while_condition());
    std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
    const CallGraphNode& cond_node = call_graph->GetNode(cond_computation);
    EXPECT_EQ(0, cond_node.caller_callsites().size());
  }
}

// Test flattening of a nested calling computations.
//
//   Entry
//    / \
//    \ /
//     B
//    / \
//    \ /
//     C
//
TEST_F(FlattenCallGraphTest, FlattenCalls) {
  auto module = CreateNewVerifiedModule();
  HloComputation* c_computation =
      module->AddEmbeddedComputation(MakeScalarComputation());

  HloComputation* b_computation = module->AddEmbeddedComputation(
      MakeCallingComputation(c_computation, /*callsites=*/2, ".B"));

  module->AddEntryComputation(
      MakeCallingComputation(b_computation, /*callsites=*/2, ".Entry"));

  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  EXPECT_EQ(7, module->computation_count());

  const CallGraphNode& c_node = call_graph->GetNode(c_computation);
  EXPECT_EQ(1, c_node.caller_callsites().size());

  const CallGraphNode& b_node = call_graph->GetNode(b_computation);
  EXPECT_EQ(1, b_node.caller_callsites().size());
}

TEST_F(FlattenCallGraphTest, FlattenCallsInConditional) {
  auto module = CreateNewVerifiedModule();
  HloComputation* sub_computation =
      module->AddEmbeddedComputation(MakeScalarComputation());

  // Create entry computation, which is a conditional that has the same
  // computation in the true and false branch.
  HloComputation::Builder builder(TestName());
  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(56.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(12.0f)));
  auto cond = builder.AddInstruction(HloInstruction::CreateConditional(
      kScalarShape, pred, constant1, sub_computation, constant2,
      sub_computation));
  module->AddEntryComputation(builder.Build());
  EXPECT_EQ(2, module->computation_count());

  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  // The true and false computations must now be different.
  EXPECT_EQ(4, module->computation_count());
  EXPECT_NE(cond->branch_computation(0), cond->branch_computation(1));

  //  The original computation is no longer used.
  const CallGraphNode& sub_node = call_graph->GetNode(sub_computation);
  EXPECT_EQ(0, sub_node.caller_callsites().size());
}

TEST_F(FlattenCallGraphTest, AsyncCall) {
  std::string hlo_string = R"(
HloModule AsyncCall

%called_computation (param_0: f32[4096], param_1: f32[4096]) -> f32[4096] {
  %param_0 = f32[4096]{0} parameter(0)
  %param_1 = f32[4096]{0} parameter(1)
  ROOT %result.1 = f32[4096]{0} add(f32[4096]{0} %param_0, f32[4096]{0} %param_1)
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %b = f32[4096]{0} parameter(1)
  %call-start.0 = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) call-start(f32[4096]{0} %a, f32[4096]{0} %b), to_apply=%called_computation
  %call-done.0 = f32[4096]{0} call-done(((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) %call-start.0)
  %call-start.1 = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) call-start(f32[4096]{0} %call-done.0, f32[4096]{0} %b), to_apply=%called_computation
  %call-done.1 = f32[4096]{0} call-done(((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) %call-start.1)
  ROOT %add_1 = f32[4096]{0} add(f32[4096]{0} %a, f32[4096]{0} %call-done.1)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);

  // We expect the entry computation, two async_wrapped computations and two
  // called_computation computations.
  EXPECT_EQ(5, module->computation_count());

  EXPECT_EQ(FindInstruction(module.get(), "call-start.0")
                ->async_wrapped_computation(),
            FindInstruction(module.get(), "call-done.0")
                ->async_wrapped_computation());
  EXPECT_EQ(FindInstruction(module.get(), "call-start.1")
                ->async_wrapped_computation(),
            FindInstruction(module.get(), "call-done.1")
                ->async_wrapped_computation());
  EXPECT_NE(FindInstruction(module.get(), "call-start.0")
                ->async_wrapped_computation(),
            FindInstruction(module.get(), "call-start.1")
                ->async_wrapped_computation());
  EXPECT_NE(FindInstruction(module.get(), "call-start.0")
                ->async_wrapped_instruction()
                ->called_computations()[0],
            FindInstruction(module.get(), "call-start.1")
                ->async_wrapped_instruction()
                ->called_computations()[0]);
}

TEST_F(FlattenCallGraphTest, WhileInCall) {
  std::string hlo_string = R"(
HloModule WhileInCall

  %while_cond {
    %p.cond = (f32[4096]{0}) parameter(0)
    ROOT %eq = pred[] constant(false)
  }

  %while_body {
    ROOT %p = (f32[4096]{0}) parameter(0)
  }

  %called_computation(arg: f32[4096]) -> f32[4096] {
    %arg = f32[4096]{0} parameter(0)
    %while_init = (f32[4096]{0}) tuple(%arg)
    %while = (f32[4096]{0}) while(%while_init), condition=%while_cond, body=%while_body
    ROOT %get-tuple-element = f32[4096]{0} get-tuple-element(%while), index=0
  }

  ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
    %a = f32[4096]{0} parameter(0)
    %b = f32[4096]{0} parameter(1)
    %call0 = f32[4096]{0} call(%a), to_apply=%called_computation
    %call1 = f32[4096]{0} call(%b), to_apply=%called_computation
    ROOT %multiply = f32[4096]{0} multiply(%call0, %call1)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_EQ(module->computation_count(), 4);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);
  EXPECT_EQ(module->computation_count(), 7);
}

TEST_F(FlattenCallGraphTest, CallInWhileInCall) {
  std::string hlo_string = R"(
HloModule CallInWhileInCall

  %called_computation_internal(arg: f32[4096]) -> f32[4096] {
    ROOT %arg.internal = f32[4096]{0} parameter(0)
  }

  %while_cond {
    %p.cond = (f32[4096]{0}) parameter(0)
    ROOT %eq = pred[] constant(false)
  }

  %while_body {
    %p.body = (f32[4096]{0}) parameter(0)
    %gte = f32[4096]{0} get-tuple-element(%p.body), index=0
    %call.internal = f32[4096]{0} call(%gte), to_apply=%called_computation_internal
    ROOT %tuple.body = (f32[4096]{0}) tuple(%call.internal)
  }

  %called_computation_external(arg: f32[4096]) -> f32[4096] {
    %arg.external = f32[4096]{0} parameter(0)
    %while.init = (f32[4096]{0}) tuple(%arg.external)
    %while = (f32[4096]{0}) while(%while.init), condition=%while_cond, body=%while_body
    ROOT %get-tuple-element = f32[4096]{0} get-tuple-element(%while), index=0
  }

  ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
    %a = f32[4096]{0} parameter(0)
    %b = f32[4096]{0} parameter(1)
    %call0 = f32[4096]{0} call(%a), to_apply=%called_computation_external
    %call1 = f32[4096]{0} call(%b), to_apply=%called_computation_external
    ROOT %multiply = f32[4096]{0} multiply(%call0, %call1)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_EQ(module->computation_count(), 5);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  EXPECT_TRUE(result);
  EXPECT_EQ(module->computation_count(), 9);
}

TEST_F(FlattenCallGraphTest, SortInCall) {
  std::string hlo_string = R"(
HloModule SortInCall

  %compare {
    %p.0.lhs = f32[] parameter(0)
    %p.0.rhs = f32[] parameter(1)
    ROOT %lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
  }

  %called_computation(arg: f32[4096]) -> f32[4096] {
    %x = f32[4096]{0} parameter(0)
    ROOT %sort =f32[4096]{0} sort(%x), dimensions={0}, to_apply=%compare
  }

  ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
    %a = f32[4096]{0} parameter(0)
    %b = f32[4096]{0} parameter(1)
    %call0 = f32[4096]{0} call(%a), to_apply=%called_computation
    %call1 = f32[4096]{0} call(%b), to_apply=%called_computation
    ROOT %multiply = f32[4096]{0} multiply(%call0, %call1)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_EQ(module->computation_count(), 3);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  ASSERT_EQ(module->computation_count(), 5);
  EXPECT_TRUE(result);

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  for (const CallGraphNode& node : call_graph->nodes()) {
    EXPECT_LE(node.caller_callsites().size(), 1);
  }
}

TEST_F(FlattenCallGraphTest, CallInSortInCall) {
  std::string hlo_string = R"(
HloModule CallInSortInCall

  %compare.impl {
    %p.0.lhs = f32[] parameter(0)
    %p.0.rhs = f32[] parameter(1)
    ROOT %lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
  }

  %compare {
    %p.0.lhs = f32[] parameter(0)
    %p.0.rhs = f32[] parameter(1)
    ROOT %lt = pred[] call(%p.0.lhs, %p.0.rhs), to_apply=%compare.impl
  }

  %called_computation(arg: f32[4096]) -> f32[4096] {
    %x = f32[4096]{0} parameter(0)
    ROOT %sort =f32[4096]{0} sort(%x), dimensions={0}, to_apply=%compare
  }

  ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
    %a = f32[4096]{0} parameter(0)
    %b = f32[4096]{0} parameter(1)
    %call0 = f32[4096]{0} call(%a), to_apply=%called_computation
    %call1 = f32[4096]{0} call(%b), to_apply=%called_computation
    ROOT %multiply = f32[4096]{0} multiply(%call0, %call1)
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_EQ(module->computation_count(), 4);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunFlattenCallGraph(module.get()));
  ASSERT_EQ(module->computation_count(), 7);
  EXPECT_TRUE(result);

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module.get());
  for (const CallGraphNode& node : call_graph->nodes()) {
    EXPECT_LE(node.caller_callsites().size(), 1);
  }
}

}  // namespace
}  // namespace xla
