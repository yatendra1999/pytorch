#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <c10/core/CPUAllocator.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/variant.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/static/ProcessedNodeInputs.h>

#ifdef FBCODE_CAFFE2
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#endif

namespace torch {
namespace jit {

#ifdef FBCODE_CAFFE2
template <typename Key, typename Value>
using FastMap = folly::F14FastMap<Key, Value>;
template <typename Key>
using FastSet = folly::F14FastSet<Key>;
#else
template <typename Key, typename Value>
using FastMap = std::unordered_map<Key, Value>;
template <typename Key>
using FastSet = std::unordered_set<Key>;
#endif

TORCH_API bool canEnableStaticRuntime(
    const std::shared_ptr<torch::jit::Graph>& graph);

TORCH_API std::string dumpValueSet(
    const FastSet<const Value*>& value_set,
    const char* set_name = "");

// Group values used by `graph` into three categories:
//
// - output_aliases:
//     values that are either outputs or contain aliases of outputs
// - external_aliases:
//     values that are inputs, constants, or their aliases.
//     The output aliases that end up here are as a result of aliasDb failing to
//     recognize them as outputs due to collection object (e.g., Tuple) aliasing
//     inputs.
// Values that dont't show up in output_aliases or external_aliases are created
// and consumed within the graph.
class ValueGroup {
 public:
  explicit ValueGroup() = default;
  void init(const std::shared_ptr<torch::jit::Graph>& graph, AliasDb& db);

  bool isExternalAlias(const Value* value) const {
    return external_aliases_.find(value) != external_aliases_.end();
  }

  bool isOutputAlias(const Value* value) const {
    return output_aliases_.find(value) != output_aliases_.end();
  }

  bool isAlwaysAlive(const Value* value) const {
    return isExternalAlias(value) || isOutputAlias(value);
  }

  std::string toString() const {
    return c10::str(
        dumpValueSet(output_aliases_, "ValueGroup::output_aliases_"),
        "\n",
        dumpValueSet(external_aliases_, "ValueGroup::external_aliases_"));
  }

 private:
  FastSet<const Value*> output_aliases_;
  FastSet<const Value*> external_aliases_;
};

struct TORCH_API StaticModuleOptions {
  // to batch allocate (deallocate) tensor storage for all non-escaping
  // temporary tensors
  bool cleanup_activations{true};
  // enabling out variant allows Static Runtime to do memory planning
  bool enable_out_variant{true};
  // to reuse tensor storage for tensors whose live-range do not overlap to
  // reduce memory footprint (enable_out_variant must be true)
  bool optimize_memory{true};
  // to batch allocate tensor storage for output tensors of the
  // graph, where storage is deallocated outside static runtime
  // (enable_out_variant must be true)
  bool manage_output_tensors{false};
};

/// The static runime supports two execution modes.
///
/// Mode 1: single-threaded with no parallelism except for intra-op parallelism
/// For this mode, you can do either:
/// @code
///   // m is a TorchScript module
///   auto module = StaticModule(m, opts);
///   auto output = module(args, kwargs);
/// @endcode
///
/// or
///
/// @code
///   // g is the TorchScript graph
///   auto module = StaticModule(g, opts);
///   auto output = module(args, kwargs);
/// @endcode
///
/// Mode 2: similar to data parallelism, run the same model for different inputs
/// on different threads at the same time.
/// You should have one StaticModule per model, and one StaticRuntime instance
/// per running thread. To avoiding creating StaticRuntimes on the fly, use a
/// synchronized stack (i.e. boost::lockfree::stack) to cache all the
/// StaticRuntime instances in your code.
/// @code
///   // initialization
///   auto module = std::make_shared<StaticModule>(m, opts);
///
///   // 128 is good for most cases. Pick a number that works for you
///   boost::lockfree::stack<std::shared_ptr<StaticRuntime>,
///     boost::lockfree::fixed_sized<true>> pool(128);
///
///   // inference
///   std::shared_ptr<StaticRuntime> runtime = nullptr;
///   pool.pop(runtime);
///   if (!runtime) {
///     // holds a reference to the underlying module
///     // but does its own memory management
///     runtime = std::make_shared<StaticRuntime>(*module);
///   }
///   auto output = runtime(args, kwargs);
///   pool.push(runtime);
/// @endcode
///

class MemoryPlanner;
class ProcessedNode;
class StaticRuntime;
class TORCH_API StaticModule {
 public:
  explicit StaticModule(
      std::shared_ptr<torch::jit::Graph> g,
      const StaticModuleOptions& opts = StaticModuleOptions());

  explicit StaticModule(
      const torch::jit::Module& m,
      bool is_frozen = false,
      const StaticModuleOptions& opts = StaticModuleOptions());

  typedef enum {
    CONSTANT_VALUE = -2, // VALUE nodes defined by prim::Constant
    INPUT_VALUE = -1, // VALUE nodes representing graph inputs
  } VALUE_KIND;

 private:
  explicit StaticModule(
      std::pair<std::shared_ptr<torch::jit::Graph>, c10::optional<Module>>
          graph_and_module,
      const StaticModuleOptions& opts);

  // for <kind, idx>
  //   if kind == CONSTANT_VALUE: map to constants_[idx]
  //   if kind == INPUT_VALUE: map to inputs_[idx]
  //   otherwise: map to nodes_[kind].outputs()[idx]
  using DefInfo = std::pair<int, int>;

 public:
  c10::IValue operator()(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);
  c10::IValue operator()(
      std::vector<c10::IValue>&& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);

  const Graph& graph() const {
    return *graph_;
  }

  const Module& module() const {
    DCHECK(module_.has_value());
    return *module_;
  }

  const StaticModuleOptions& opts() const;

  const ValueGroup& valueGroup() const {
    return value_group_;
  }

  size_t num_inputs() const;
  size_t num_outputs() const;

  C10_NODISCARD const std::vector<uint16_t>& output_indices() const {
    return output_indices_;
  }

  const std::vector<IValue>& constants() const {
    return constants_;
  }

 private:
  friend class StaticRuntime;

  // Our nodes don't have their inputs & outputs initialized; don't
  // let anybody but StaticRuntime and tests get them.
  const std::vector<ProcessedNode>& nodes() const {
    return nodes_;
  }

 public:
  auto num_nodes() const {
    return nodes_.size();
  }

  C10_NODISCARDNode* findNodeWithKindForTesting(const std::string& kind) const;

  bool is_optimizable_container_type(const Node* n) const {
    auto it = node_is_optimizable_container_type_.find(n);
    return it != node_is_optimizable_container_type_.end();
  }

  const c10::optional<c10::FunctionSchema>& schema() const {
    return schema_;
  }

  const FastMap<const Value*, std::vector<const Value*>>&
  values_share_same_storage() const {
    return value_to_same_storage_values_;
  }

  const ValueGroup& value_group() const {
    return value_group_;
  }

  bool first_input_is_self() const {
    return module_.has_value();
  }

  StaticRuntime& runtime();

 private:
  StaticModuleOptions opts_;
  std::shared_ptr<torch::jit::Graph> graph_;
  c10::optional<torch::jit::Module> module_;
  c10::optional<c10::FunctionSchema> schema_;
  std::unique_ptr<StaticRuntime> cached_runtime_;

  // Bookkeeping for creating new StaticRuntime instances
  // IValue table (defined by prim::Constant nodes)
  std::vector<IValue> constants_;
  // The nodes we need to run
  std::vector<ProcessedNode> nodes_;
  // Indices of graph outputs in the single values array.
  std::vector<uint16_t> output_indices_;

  ValueGroup value_group_;

  // map a value to the set of values that may share the same storage with it
  FastMap<const Value*, std::vector<const Value*>>
      value_to_same_storage_values_;

  FastSet<const Node*> node_is_optimizable_container_type_;
};

class TORCH_API StaticRuntime {
 public:
  explicit StaticRuntime(const StaticModule& sm);
  StaticRuntime(StaticRuntime&&) = delete;
  StaticRuntime& operator=(StaticRuntime&&) = delete;
  ~StaticRuntime();

  C10_DISABLE_COPY_AND_ASSIGN(StaticRuntime);

  c10::IValue operator()(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);
  c10::IValue operator()(
      std::vector<c10::IValue>&& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);

  void benchmark(
      const std::vector<std::vector<c10::IValue>>& args_list,
      const std::vector<std::unordered_map<std::string, c10::IValue>>&
          kwargs_list,
      const int warmup_runs,
      const int main_runs,
      bool print_per_node_time = false,
      bool generate_ai_pep_output = false);

  struct IndividualMetrics {
    float setup_time{0.0};
    float memory_alloc_time{0.0};
    float memory_dealloc_time{0.0};
    float output_dealloc_time{0.0};
    float first_iter_time{0.0};
    float total_time{0.0};
    size_t out_nodes_count{0};
    size_t total_nodes_count{0};
    std::vector<float> time_per_node;
    std::unordered_map<std::string, float> time_per_node_type;
    std::unordered_map<std::string, float> percent_per_node_type;
    std::unordered_map<std::string, int> instances_per_node_type;
    std::unordered_set<std::string> out_nodes;
    std::unordered_set<std::string> native_nodes;
  };

  IndividualMetrics benchmark_individual_ops(
      const std::vector<std::vector<c10::IValue>>& args_list,
      const std::vector<std::unordered_map<std::string, c10::IValue>>&
          kwargs_list,
      const int warmup_runs,
      const int main_runs);

  // Input is readwrite
  IValue& Input(uint32_t i) {
    DCHECK_LT(i, static_module_.num_inputs());
    DCHECK_LT(i, values_.size());
    return values_[i];
  }

  // Output is readonly. The writing process happens inside ProcessedNodes
  C10_NODISCARD const IValue& Output(uint32_t i) const {
    DCHECK(i < outputs_.size());
    return *outputs_[i];
  }

  const std::vector<IValue*> outputs() const {
    return outputs_;
  }

  const std::vector<ProcessedNode>& nodes() const {
    return nodes_;
  }

  std::vector<ProcessedNode>& nodes() {
    return nodes_;
  }

  const Graph& graph() const {
    return static_module_.graph();
  }

  const MemoryPlanner* get_memory_planner() const {
    return planner_.get();
  }

  void check_for_memory_leak(bool output_returned = true);

  bool is_optimizable_container_type(Node* n) const {
    return static_module_.is_optimizable_container_type(n);
  }

  // WARNING: Deallocate managed output tensors.  A client receiving Static
  // Runtime-managed Tensors needs to be very careful to call
  // `StaticRuntime::deallocateOutputTensors` after all references of output
  // Tensors are gone.
  void deallocateOutputTensors();

  bool checkOutputTensorMemoryLeaks();

  bool isManagedOutputTensor(const IValue& ivalue);

 private:
  template <typename IValueList>
  c10::IValue run_impl(
      IValueList&& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);

  // helper method for copying input args/kwargs into inputs_
  void set_inputs(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);
  void set_inputs(
      std::vector<c10::IValue>&& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);

  void verify_and_correct_memory_overlap(ProcessedNode& n);

  // clean up owning refs of input IValues
  void clean_up_input_ivalues() {
    for (const auto idx : c10::irange(static_module_.num_inputs())) {
      values_[idx] = IValue();
    }
  }

  IValue move_outputs_to_tuple(uint32_t num_outputs);

  void create_memory_planner();

  float benchmark_model(
      const std::vector<std::vector<c10::IValue>>& args_list,
      const std::vector<std::unordered_map<std::string, c10::IValue>>&
          kwargs_list,
      const int warmup_runs,
      const int main_runs);

  void display_nodes(
      const std::vector<c10::IValue>& args,
      const std::unordered_map<std::string, c10::IValue>& kwargs);

  // Memory planning is only enabled if sm->opts().cleanup_activations is true.
  // Otherwise, the memory used by activations is cached inside the static
  // runtime.
  const StaticModule& static_module_;
  std::unique_ptr<MemoryPlanner> planner_;
  // first static_module_.num_inputs() slots are inputs, next
  // static_module_.constants().size() slots are a copy of
  // static_module_.constants(), rest are regular values in the
  // graph. ProcessedNodes reference their inputs and outputs with
  // offsets into this array, which saves memory.
  std::vector<IValue> values_;
  std::vector<IValue*> outputs_;
  std::vector<ProcessedNode> nodes_;
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_API ProcessedNode {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  ProcessedNode() = default;
  // ProcessedNodes are created within StaticModule and then
  // associated with a shared values array using set_values() when
  // they are copied into a StaticRuntime.
  ProcessedNode(Node* n, ProcessedNodeInputs inputs, uint16_t outputs_offset, bool enable_out_variant, bool check_memory_overlap);

  ProcessedNode(const ProcessedNode&) = default;
  ProcessedNode& operator=(const ProcessedNode&) = default;

  // These should be noexcept, but some Android build is failing
  // saying the noexcept specification doesn't match the calculated
  // one. Maybe c10::variant is throwing it off?
  ProcessedNode(ProcessedNode&&) = default;
  ProcessedNode& operator=(ProcessedNode&&) = default;

  void run();

  Node* node() const {
    return node_;
  }

  // Input is readonly
  C10_NODISCARD const IValue& Input(uint32_t i) const {
    return values_[inputs_[i]];
  }

  // Output is readwrite
  IValue& Output(uint32_t i) {
    DCHECK(i < num_outputs_);
    return values_[outputs_offset_ + i];
  }

  C10_NODISCARD const IValue& Output(uint32_t i) const {
    DCHECK(i < num_outputs_);
    return values_[outputs_offset_ + i];
  }

  C10_NODISCARD c10::ArrayRef<const IValue> outputs() const {
    return c10::ArrayRef<const IValue>(values_ + outputs_offset_, num_outputs_);
  }

  C10_NODISCARD auto num_outputs() const {
    return num_outputs_;
  }

  C10_NODISCARD uint16_t num_inputs() const {
    return inputs_.size();
  }

  std::vector<IValue> clone_inputs() const;

  bool has_out_variant() const {
    return fn_.kind == FunctionKind::kOutVariant;
  }

  bool has_native() const {
    return fn_.kind == FunctionKind::kNativeFunction;
  }

#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  const char* get_op_name() const {
    return op_name_;
  }
#endif

  bool verify_no_memory_overlap() const;

  bool check_outputs_for_memory_overlap() const {
    return fn_.check_memory_overlap_;
  }

  void set_outputs_memory_overlap_detected() {
    fn_.overlap_detected_ = true;
  }

  bool outputs_memory_overlap_detected() {
    return fn_.overlap_detected_;
  }

  void verify_and_correct_memory_overlap();

  void set_values(IValue* values) {
    DCHECK(values_ == nullptr);
    values_ = values;
  }

  C10_NODISCARD uint16_t output_ivalue_index(uint16_t i) const {
    return outputs_offset_ + i;
  }
 private:
  C10_NODISCARD bool verify_outputs_dont_overlap_each_other() const;

  C10_NODISCARD bool verify_inputs_dont_overlap_outputs() const;

  Node* node_;
  enum class FunctionKind : uint8_t {
    kOutVariant,
    kNativeFunction,
    kInterpreterFallback,
  };
  struct Function {
    std::function<void(ProcessedNode*)> f;
    FunctionKind kind = FunctionKind::kOutVariant;
    bool check_memory_overlap_{false};
    bool overlap_detected_{false};
  };
  Function fn_;
  ProcessedNodeInputs inputs_;
  uint16_t outputs_offset_;
  uint16_t num_outputs_;
  IValue* values_ = nullptr; // unowned
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  const char* op_name_;
#endif
};

} // namespace jit
} // namespace torch
