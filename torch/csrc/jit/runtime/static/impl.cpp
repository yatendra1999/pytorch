#include <torch/csrc/jit/runtime/static/impl.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/core/interned_strings.h>
#include <ATen/record_function.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/irange.h>
#include <caffe2/core/scope_guard.h>
#include <caffe2/core/timer.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/eliminate_no_ops.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/variadic_ops.h>
#include <torch/csrc/jit/runtime/static/memory_planner.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/runtime/static/passes.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <iterator>
#include <sstream>
#include <stdexcept>

#ifdef FBCODE_CAFFE2
#include <folly/dynamic.h>
#include <folly/json.h>
#endif

namespace torch {
namespace jit {

// A manually curated set of ops that are disallowed in static runtime.
// These are rarely-used ops. Disallowing them typically eliminates
// corner cases in graph optimizations, allowing for more aggressive
// optimizations and better performance.
bool isUnsupportedOp(const NodeKind& kind) {
  return kind == aten::__is__ || kind == aten::__isnot__;
}

// graph must be frozen or canEnableStaticRuntime would return false if there's
// any prim::CallMethod op left in the graph
bool canEnableStaticRuntime(const std::shared_ptr<torch::jit::Graph>& graph) {
  // check for sub-blocks
  bool can_support = true;
  bool has_blocks = false;
  for (auto* node : graph->block()->nodes()) {
    if (node->blocks().size() > 0) {
      has_blocks = true;
      VLOG(1) << "Found nested sub-blocks in graph at node: "
              << PrintNode(node);
    }
    const auto kind = node->kind();
    if (kind == prim::Constant) {
      continue;
    }
    // check if can get op from Node
    const Operator* op = node->maybeOperator();
    if (isUnsupportedOp(kind) || (!op && !nativeOpIsRegistered(kind))) {
      can_support = false;
      LOG(WARNING) << "Found unsupported op: " << kind.toQualString();
    }
  }
  if (has_blocks) {
    LOG(WARNING)
        << "Found nested sub-block in graph. Static Runtime doesn't support nested sub-blocks.";
    can_support = false;
  }
  return can_support;
}

std::string dumpValueSet(
    const FastSet<const Value*>& value_set,
    const char* set_name) {
  std::ostringstream oss;
  oss << set_name << ": {";
  for (const auto* val : value_set) {
    oss << "%" << val->debugName() << ", ";
  }
  oss << "}";
  return oss.str();
}

namespace {

void OptimizeGraph(
    std::shared_ptr<torch::jit::Graph>& graph,
    const StaticModuleOptions& opts) {
  GRAPH_DUMP("Before optimizations: ", graph);
  Inline(*graph);
  ConstantPropagation(graph);
  Canonicalize(graph);
  ConstantPropagation(graph);
  RemoveTensorMutation(graph);
  ConstantPropagation(graph);
  EliminateDeadCode(graph);
  FuseInferenceOpsForSparseNN(graph);
  UseVariadicCat(graph);
  UseVariadicStack(graph);
  EliminateTrivialEquallySplit(graph);

  if (opts.enable_out_variant) {
    UseVariadicOp(
        graph,
        fromQualString("fb::sigrid_transforms_torch_bind"),
        fromQualString("fb::variadic_sigrid_transforms_torch_bind"));
    FuseSignLog1P(graph);

    // TODO: we can avoid this guard by moving operations
    // to exposed folders.
#ifdef FBCODE_CAFFE2
    ReplaceWithCopy(graph);
    FuseListUnpack(graph);
    EnableStaticRuntimeLayerNorm(graph);
#endif
  }

  ConstantPropagation(graph);
  RemoveImmutableInputDictLookups(graph);
  UseVariadicTupleUnpack(graph);
  UseVariadicGroupedAccessor(graph);
  EliminateNoOps(
      graph, /* custom_ops */ {fromQualString("fb::scale_gradient")});
  GRAPH_DUMP("Final graph after optimizations: ", graph);
}

// remove unused input 0 from graph
bool RemoveSelfFromGraphInput(std::shared_ptr<torch::jit::Graph>& graph) {
  if (graph->inputs().at(0)->type()->is_module()) {
    if (graph->inputs().at(0)->hasUses()) {
      return false;
    }
    graph->eraseInput(0);
  }
  return true;
}

// remove "self" from function schema
c10::FunctionSchema RemoveSelfFromSchema(const c10::FunctionSchema& s) {
  TORCH_CHECK(s.arguments().size() >= 1 && s.arguments()[0].name() == "self");
  std::vector<Argument> args({s.arguments().begin() + 1, s.arguments().end()});
  return s.cloneWithArguments(args);
}

std::vector<Value*> valueVecFromFastSet(const FastSet<const Value*>& s) {
  std::vector<Value*> result;
  result.reserve(s.size());
  for (auto* v : s) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    result.emplace_back(const_cast<Value*>(v));
  }
  return result;
}

bool mayContainAlias(AliasDb& db, const Value* a, const Value* b) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return db.mayContainAlias(const_cast<Value*>(a), const_cast<Value*>(b));
}

bool mayContainAlias(
    AliasDb& db,
    const FastSet<const Value*>& a,
    const FastSet<const Value*>& b) {
  return db.mayContainAlias(valueVecFromFastSet(a), valueVecFromFastSet(b));
}

//  Map each value to all values that are alive at the same time.
using LivenessMap = FastMap<const Value*, FastSet<const Value*>>;

template <typename Map>
std::string dumpMapFromValuesToListsOrSetsOfOtherValues(const Map& map) {
  std::ostringstream oss;
  oss << "{";
  for (const auto& p : map) {
    oss << "{%" << p.first->debugName() << ": {";
    for (const auto* val : p.second) {
      oss << "%" << val->debugName() << ", ";
    }
    oss << "}},\n";
  }
  oss << "}";
  return oss.str();
}

std::string dumpLivenessMap(const LivenessMap& liveness_map) {
  return dumpMapFromValuesToListsOrSetsOfOtherValues(liveness_map);
};

//  The algorithm does a traversal of the execution graph
//  while keeping track of the live values.
LivenessMap GetLivenessMap(
    const std::shared_ptr<torch::jit::Graph>& graph,
    const ValueGroup& value_group,
    AliasDb& db) {
  // map a Value to a set of Values that overlap live-ranges with the Value's
  FastMap<const Value*, FastSet<const Value*>> liveness_map;

  // map Values to its creation order in graph (Note: only traverse top-level
  // nodes such that nodes under control-flows are represented by top-level
  // block nodes)
  std::vector<const Value*> values_in_creation_order;
  FastMap<const Value*, size_t> values_to_idx_in_creation_order;
  for (const auto* node : graph->nodes()) {
    values_to_idx_in_creation_order.reserve(
        values_to_idx_in_creation_order.size() + node->outputs().size());
    for (const auto* v : node->outputs()) {
      values_to_idx_in_creation_order.emplace(
          v, values_in_creation_order.size());
      values_in_creation_order.emplace_back(v);
    }
  }

  // presence of a Value in live_values_use_chain means the Value alive
  // Value mapped to set of Nodes that may use the Value (i.e., use-chain of
  // Value)
  FastMap<const Value*, FastSet<const Node*>> live_values_use_chain;
  // Node mapped to set of Values that the Node may use (i.e., def-chain of node
  // inputs)
  FastMap<const Node*, FastSet<const Value*>> live_nodes_def_chain;

  // add v to the current liveness_map
  std::function<void(const Value* v)> add_live_value_fn = [&](const Value* v) {
    if (liveness_map.count(v)) {
      return;
    }

    auto& v_live_set = liveness_map[v] = {};

    v_live_set.reserve(live_values_use_chain.size());
    for (const auto& live_v : live_values_use_chain) {
      v_live_set.insert(live_v.first);
      liveness_map[live_v.first].insert(v);
    }

    // only add values to the live set if they
    // have deps, otherwise they die immediately
    if (v->uses().size()) {
      live_values_use_chain[v] = FastSet<const Node*>(v->uses().size());
      // record the relationship between v (Value) and its uses (Node)
      for (const auto& u : v->uses()) {
        const auto* node = u.user;
        live_values_use_chain[v].insert(node);
        live_nodes_def_chain[node].insert(v);
      }
    }

    // FIXME(penguin): the following alias refinement seems to assume
    // that `v` refers to a new  tensor created by the node that defines
    // v, thus other Values "before" the node that defines `v` cannot
    // possibly be aliased to `v`.
    // TODO(penguin): Is it a limitation of TS alias analysis
    // so that we need to do such refinement? If so, better improve
    // alias analysis so that we dont need this special handling here
    //
    // Refine aliases of v by include only those created after v
    std::vector<const Value*> refined_aliases;
    auto idx = values_to_idx_in_creation_order[v];
    for (; idx < values_in_creation_order.size(); ++idx) {
      auto* alias_v = values_in_creation_order[idx];
      if (mayContainAlias(db, v, alias_v)) {
        refined_aliases.emplace_back(alias_v);
      }
    }
    // for all the values in the alias set,
    // we set them "alive"
    for (auto* aliased_v : refined_aliases) {
      GRAPH_DEBUG(
          "aliased_v: %",
          aliased_v->debugName(),
          " (for %",
          v->debugName(),
          ")");
      add_live_value_fn(aliased_v);
    }
  };

  auto remove_dead_values = [&](const Node* node) {
    auto find = live_nodes_def_chain.find(node);
    if (find != live_nodes_def_chain.end()) {
      for (const auto* v : find->second) {
        live_values_use_chain[v].erase(node);
        if (!live_values_use_chain[v].size()) {
          // v is now dead
          GRAPH_DEBUG(
              "%",
              v->debugName(),
              " is now dead after ",
              node->output(0)->debugName())
          live_values_use_chain.erase(v);
        }
      }
    }
  };

  for (const auto* node : graph->nodes()) {
    for (const auto* v : node->outputs()) {
      if (!value_group.isAlwaysAlive(v)) {
        add_live_value_fn(v);
      }
    }

    remove_dead_values(node);
  }
  GRAPH_DEBUG("LivenessMap: ", dumpLivenessMap(liveness_map));

  for (const auto& v : live_values_use_chain) {
    TORCH_CHECK(
        value_group.isAlwaysAlive(v.first),
        v.first->debugName(),
        "is not in the value_group.isAlwaysAlive group");
  }

  auto insert_all_pairs_in_liveness_map =
      [&](at::ArrayRef<const Value*> values) {
        for (size_t i = 0; !values.empty() && i < values.size() - 1; ++i) {
          auto value_it = liveness_map.find(values[i]);
          if (value_it == liveness_map.end()) {
            continue;
          }
          for (size_t j = i + 1; j < values.size(); ++j) {
            auto value2_it = liveness_map.find(values[j]);
            if (value2_it != liveness_map.end()) {
              value_it->second.insert(values[j]);
              value2_it->second.insert(values[i]);
            }
          }
        }
      };

  for (const auto* node : graph->nodes()) {
    auto inputs = node->inputs();
    auto outputs = node->outputs();
    for (const auto* input : inputs) {
      for (const auto* output : outputs) {
        auto input_it = liveness_map.find(input);
        if (input_it == liveness_map.end()) {
          continue;
        }
        auto output_it = liveness_map.find(output);
        if (output_it == liveness_map.end()) {
          continue;
        }
        input_it->second.insert(output);
        output_it->second.insert(input);
      }
    }

    // All inputs should be alive at the same time.
    insert_all_pairs_in_liveness_map(inputs);

    // All outputs should be alive at the same time.
    insert_all_pairs_in_liveness_map(outputs);
  };

  return liveness_map;
};

// Collect the set of Values that are candidates for memory planning:
//   - Values that are used in in-place operators (i.e., _out variants), and
//   - excluding those that are either inputs or outputs of
//     non in-place operators
//
// Returns
//   first: Values that are candidates for memory planning
//   second: A deterministc order of all values
std::pair<std::vector<const Value*>, std::vector<const Value*>>
GetMemoryPlanningCandidates(
    const std::shared_ptr<torch::jit::Graph>& graph,
    const FastMap<Node*, bool>& node_has_out_variant) {
  // for determinism
  FastSet<const Value*> seen_values;
  std::vector<const Value*> all_values;
  FastSet<const Value*> can_reuse;
  // values used by unsupported ops (as either inputs or outputs)
  // these need to be removed from "can_reuse" after analyzing all nodes
  FastSet<const Value*> cannot_reuse;
  for (auto* n : graph->nodes()) {
    bool can_reuse_inputs_outputs =
        canReuseInputsOutputs(n, node_has_out_variant);
    for (const auto* v : n->inputs()) {
      if (!seen_values.count(v)) {
        all_values.emplace_back(v);
        seen_values.insert(v);
      }
      if (can_reuse_inputs_outputs) {
        can_reuse.insert(v);
      } else {
        cannot_reuse.insert(v);
      }
    }
    for (const auto* v : n->outputs()) {
      all_values.emplace_back(v);
      seen_values.insert(v);
      if (can_reuse_inputs_outputs) {
        can_reuse.insert(v);
      } else {
        cannot_reuse.insert(v);
      }
    }
  }
  for (const auto* v : cannot_reuse) {
    can_reuse.erase(v);
  }
  // find a deterministic order
  std::vector<const Value*> optimizable;
  for (const auto* v : all_values) {
    if (can_reuse.count(v)) {
      optimizable.emplace_back(v);
      can_reuse.erase(v);
    }
  }
  return std::make_pair(optimizable, all_values);
}

// Equipped with a liveness map we can allocate memory to
// ivalues, reusing memory along the way. However, we are
// constrained by the set of optimizable_values
// (inputs/outputs of out variants). Inputs/outputs of view ops
// can't be reused.
//
// Algorithm:
// # clusters of values sharing the same memory
// # are called "value_to_same_storage_values" in the implementation
// # inserting into a cluster denotes sharing memory.
//
// clusters = {}
// for all v in optimzable_values:
//   for all cluster in clusters: # can we insert into cluster?
//     for all live_v in live_during(v):
//        if cluster.contains(live_v):
//          skip to next custer
//     cluster.add(v)
//     skip to next v
//   if no cluster found:
//     clusters.add(cluster{v})
//
//
// NB: This is a deterministic implementation, which makes it easier to tune
// and debug.
FastMap<const Value*, std::vector<const Value*>> GenerateSameStorageValues(
    const LivenessMap& alive_during,
    const ValueGroup& value_group,
    const std::pair<std::vector<const Value*>, std::vector<const Value*>>&
        optimizable,
    AliasDb& db) {
  const auto& optimizable_values = optimizable.first;
  const auto& all_values = optimizable.second;

  // map Value* to a set Value* that can share the same storage with it
  FastMap<const Value*, std::vector<const Value*>> same_storage_values;

  // make new_v and old_v map to the same storage (i.e., add to each other's
  // same_storage_values set)
  auto share_storage_fn = [&](const Value* new_v, const Value* old_v) {
    if (new_v == old_v) {
      return;
    }
    DCHECK(same_storage_values.count(old_v));
    FastSet<const Value*> seen;
    std::vector<const Value*> values;
    for (auto* v : same_storage_values.at(old_v)) {
      if (seen.count(v)) {
        continue;
      }
      seen.insert(v);
      values.emplace_back(v);
    }
    for (auto* v : same_storage_values.at(new_v)) {
      if (seen.count(v)) {
        continue;
      }
      seen.insert(v);
      values.emplace_back(v);
    }
    for (const auto* v : values) {
      same_storage_values[v] = values;
    }
  };

  // initialize with known same_storage_values (aliasing values)
  for (const auto* v : all_values) {
    if (!same_storage_values.count(v)) {
      same_storage_values[v] = {v};
    }
    // NOTE: if we had AliasDb::mustAlias, we could do the following:
    // // skip always alive values (alias inputs/outputs/weights)
    // if (value_group.isAlwaysAlive(v)) {
    //   continue;
    // }
    // for (const auto& p : same_storage_values) {
    //   if (db.mustAlias(p.first, v)) {
    //     share_storage_fn(v, p.first);
    //   }
    // }
    // It also wouldn't matter because ops always create new Tensor
    // objects as aliases; there is no point in trying to reuse their
    // storage.
  }

  // to preserve determinism
  std::vector<const Value*> seen;

  auto compute_liveset_fn = [&alive_during, &same_storage_values](
                                FastSet<const Value*>& live, const Value* v) {
    for (const auto* sv : same_storage_values.at(v)) {
      const auto& l = alive_during.count(sv) ? alive_during.at(sv)
                                             : FastSet<const Value*>{};
      live.insert(l.begin(), l.end());
    }
  };

  // check if same_storage_values[s] intersects with live
  auto intersect_fn = [&same_storage_values](
                          FastSet<const Value*>& live, const Value* s) {
    bool intersect = false;
    for (const auto* v : same_storage_values.at(s)) {
      if (live.count(v)) {
        intersect = true;
        break;
      }
    }
    return intersect;
  };

  for (const auto* v : optimizable_values) {
    if (value_group.isAlwaysAlive(v)) {
      continue;
    }
    // get values that are live during the lifetime of v
    FastSet<const Value*> live;
    compute_liveset_fn(live, v);
    for (const auto* s : seen) {
      // if live(same_storage_values[v]) and same_storage_values[s]
      // do not overlap, then s and v can share the same storage
      if (!intersect_fn(live, s) && !value_group.isAlwaysAlive(s)) {
        share_storage_fn(v, s);
        // since s is added to same_storage_values[v], live needs
        // to be recomputed, so bail out here
        break;
      }
    }
    seen.emplace_back(v);
  }

  GRAPH_DEBUG(
      "same_storage_values: ",
      dumpMapFromValuesToListsOrSetsOfOtherValues(same_storage_values));

  return same_storage_values;
}

void PrepareGraphForStaticModule(
    std::shared_ptr<torch::jit::Graph> graph,
    const StaticModuleOptions& opts) {
  TORCH_CHECK(canEnableStaticRuntime(graph));
  OptimizeGraph(graph, opts);
}

std::pair<std::shared_ptr<Graph>, c10::optional<Module>> PrepareForStaticModule(
    const torch::jit::Module& m,
    bool is_frozen,
    const StaticModuleOptions& opts) {
  VLOG(1) << "StaticModuleOptions: cleanup_activations "
          << opts.cleanup_activations << ", enable_out_variant "
          << opts.enable_out_variant << ", optimize_memory "
          << opts.optimize_memory << ", manage_output_tensors "
          << opts.manage_output_tensors;

  Module module = m.copy();
  if (!is_frozen) {
    module.eval();
    module = freeze_module(module);
  }

  Method method = module.get_method("forward");
  auto graph = module.get_method("forward").graph();

  PrepareGraphForStaticModule(graph, opts);

  return std::make_pair(graph, module);
}

std::pair<std::shared_ptr<Graph>, c10::optional<Module>> PrepareForStaticModule(
    std::shared_ptr<torch::jit::Graph> graph,
    const StaticModuleOptions& opts) {
  PrepareGraphForStaticModule(graph, opts);
  return std::make_pair(graph, c10::nullopt);
}

} // namespace

void ValueGroup::init(
    const std::shared_ptr<torch::jit::Graph>& graph,
    AliasDb& db) {
  external_aliases_.clear();
  output_aliases_.clear();
  // Build `input_or_constant_aliases` as we look through nodes forwardly from
  // the graph's inputs and add aliases of the inputs being created by the
  // nodes.
  external_aliases_.insert(graph->inputs().begin(), graph->inputs().end());
  for (const auto* node : graph->nodes()) {
    if (node->kind() == prim::Constant) {
      for (const auto* output : node->outputs()) {
        external_aliases_.insert(output);
      }
    }
  }
  for (const auto* node : graph->nodes()) {
    if (node->kind() == prim::Constant) {
      // Constants are already in `input_or_constant_aliases`.
      continue;
    }
    for (const auto* v : node->outputs()) {
      if (mayContainAlias(db, {v}, external_aliases_)) {
        external_aliases_.insert(v);
      }
    }
  }

  // Build `output_aliases` as we look through nodes reversely so that we can
  // start from the output values, and follow the flows backwardly from there.
  output_aliases_.insert(graph->outputs().begin(), graph->outputs().end());
  for (const auto* node : graph->nodes().reverse()) {
    if (node->kind() == prim::Constant) {
      // Constants cannot create any aliases.
      continue;
    }
    for (const auto* v : node->outputs()) {
      // Add values that can aliase input/constant values. Note some output
      // aliases may end up in this category via collection objects (e.g.,
      // Tuple).
      if (mayContainAlias(db, {v}, external_aliases_)) {
        external_aliases_.insert(v);
        continue;
      }
      if (mayContainAlias(db, {v}, output_aliases_)) {
        output_aliases_.insert(v);
      }
    }
  }
}

bool containTensorsOnly(at::ArrayRef<Value*> values) {
  // return true only if all outputs are tensors
  return std::all_of(values.begin(), values.end(), [](const Value* value) {
    return value->type()->castRaw<TensorType>() != nullptr;
  });
}

StaticModule::StaticModule(
    std::shared_ptr<torch::jit::Graph> g,
    const StaticModuleOptions& opts)
    : StaticModule(PrepareForStaticModule(g->copy(), opts), opts) {}

StaticModule::StaticModule(
    const torch::jit::Module& m,
    bool is_frozen,
    const StaticModuleOptions& opts)
    : StaticModule(PrepareForStaticModule(m, is_frozen, opts), opts) {}

StaticModule::StaticModule(
    std::pair<std::shared_ptr<torch::jit::Graph>, c10::optional<Module>>
        graph_and_module,
    const StaticModuleOptions& opts)
    : opts_(opts),
      graph_(std::move(graph_and_module.first)),
      module_(std::move(graph_and_module.second)) {
  // check opt flags
  if (opts.manage_output_tensors) {
    TORCH_CHECK(
        opts_.enable_out_variant,
        "When manage_output_tensors is true, enable_out_variant must be set to true");
  }
  if (opts_.optimize_memory) {
    TORCH_CHECK(
        opts_.enable_out_variant,
        "When optimize_memory is true, enable_out_variant must be set to true");
  }

  // handle schema
  if (module_.has_value()) {
    Method method = module_->get_method("forward");
    if (RemoveSelfFromGraphInput(graph_)) {
      schema_ = RemoveSelfFromSchema(method.function().getSchema());
      module_ = c10::nullopt;
    } else {
      schema_ = method.function().getSchema();
    }
  }

  // map Value* to its SSA definition IR
  FastMap<Value*, DefInfo> value_to_ssa_def;

  // N inputs map to the first N entries in storage
  for (const auto i : c10::irange(graph_->inputs().size())) {
    Value* input = graph_->inputs()[i];
    value_to_ssa_def[input] = std::make_pair(INPUT_VALUE, i);
  }

  // NB: before optimizing the order of execution, ensure that the
  // memory optimization pass (LivenessMap) is
  // aware of the new order!

  // Fill constants first, so we have a std::vector<IValue> we can reference
  // later
  for (Node* node : graph_->nodes()) {
    if (node->kind() != prim::Constant) {
      continue;
    }
    auto* v = node->output();
    TORCH_CHECK(v->type()->kind() != FunctionType::Kind);
    constants_.emplace_back(toIValue(v).value());
  }
  {
    // construct SSA definition for constant nodes
    int i = 0;
    for (Node* node : graph_->nodes()) {
      if (node->kind() != prim::Constant) {
        continue;
      }
      auto* v = node->output();
      value_to_ssa_def[v] = std::make_pair(CONSTANT_VALUE, i++);
    }
  }

  AliasDb alias_db(
      graph_, /*isFrozen=*/false, /*enablePreciseTupleContainerAnalysis=*/true);
  GRAPH_DEBUG("AliasDb: ", alias_db.toString());

  // construct SSA definition for non-constant nodes
  int node_idx = 0;
  FastMap<Node*, bool> node_has_out_variant;

  const auto inputs_index_offset = 0;
  const auto constants_index_offset = inputs_index_offset + num_inputs();
  const auto values_index_offset =
      constants_index_offset + constants().size();

  // Map node_idx to index offset in values_. Can't reserve space
  // because we don't know how many non-constant nodes there are yet.
  std::vector<uint32_t> node_output_idx_map;
  uint32_t node_outputs_seen_so_far = 0;
  for (Node* node : graph_->nodes()) {
    if (node->kind() == prim::Constant) {
      continue;
    }
    // Assign memory for the outputs
    const auto outputs_offset_for_node =
      node_outputs_seen_so_far + values_index_offset;
    TORCH_CHECK(outputs_offset_for_node < (1 << 16), "outputs offset in values table", outputs_offset_for_node, " would overflow 2-byte index storage");
    node_output_idx_map.push_back(outputs_offset_for_node);
    node_outputs_seen_so_far += node->outputs().size();
  }

  for (Node* node : graph_->nodes()) {
    if (node->kind() == prim::Constant) {
      continue;
    }
    ProcessedNodeInputs input_indices(node->inputs().size());
    std::vector<DefInfo> input_ssa_defs;
    for (const auto input_idx : c10::irange(node->inputs().size())) {
      Value *const input  = node->inputs()[input_idx];
      int inner_node_idx = 0;
      int out_idx = 0;
      std::tie(inner_node_idx, out_idx) = value_to_ssa_def.at(input);
      unsigned int input_ivalue_idx = 0;
      if (inner_node_idx == StaticModule::INPUT_VALUE) {
        input_ivalue_idx = out_idx + inputs_index_offset;
      } else if (inner_node_idx == StaticModule::CONSTANT_VALUE) {
        input_ivalue_idx = out_idx + constants_index_offset;
      } else {
        DCHECK_GE(inner_node_idx, 0);
        const auto global_value_idx = node_output_idx_map[inner_node_idx] + out_idx;
        if (inner_node_idx < node_output_idx_map.size() - 1) {
          DCHECK_LT(global_value_idx, node_output_idx_map[inner_node_idx + 1]);
        } else {
          DCHECK_LT(global_value_idx, constants_index_offset + node_outputs_seen_so_far);
        }
        input_ivalue_idx = global_value_idx;
      }
      TORCH_CHECK(input_ivalue_idx < (1 << 16),
                  "input index in values table ", input_ivalue_idx,
                  " would overflow 2-byte index storage");
      input_indices[input_idx] = input_ivalue_idx;
    }

    // create a new ProcessedNode
    // see [Check and correct bad schema alias info at runtime]
    bool check_outputs_for_overlap =
        !alias_db.mayContainAlias(node->inputs(), node->outputs()) &&
        containTensorsOnly(node->outputs());
    nodes_.emplace_back(
        node,
        std::move(input_indices),
        node_output_idx_map[node_idx],
        opts.enable_out_variant,
        check_outputs_for_overlap);

    node_has_out_variant.emplace(node, nodes_.back().has_out_variant());
    for (const auto i : c10::irange(node->outputs().size())) {
      value_to_ssa_def[node->outputs()[i]] = std::make_pair(node_idx, i);
    }
    node_idx++;
  }
  for (auto& pnode : nodes_) {
    if (pnode.num_outputs() == 1 &&
        isOptimizableContainerType(pnode.node(), node_has_out_variant)) {
      node_is_optimizable_container_type_.emplace(pnode.node());
    }
  }
  output_indices_.reserve(graph_->outputs().size());
  for (auto output : graph_->outputs()) {
    int node_idx = 0;
    int out_idx = 0;
    std::tie(node_idx, out_idx) = value_to_ssa_def[output];
    uint32_t output_index = 0;
    if (node_idx == StaticModule::INPUT_VALUE) {
      output_index = out_idx + inputs_index_offset;
    } else if (node_idx == StaticModule::CONSTANT_VALUE) {
      output_index = constants_index_offset + out_idx;
    } else {
      output_index = nodes_[node_idx].output_ivalue_index(out_idx);
    }
    TORCH_CHECK(
        output_index < (1 << 16),
        "output index ",  output_index, " would overflow 2-byte index storage");
    output_indices_.emplace_back(output_index);
  }

  // Prepare for memory planning
  value_group_.init(graph_, alias_db);
  GRAPH_DEBUG(value_group_.toString());

  if (opts_.optimize_memory) {
    auto lm = GetLivenessMap(graph_, value_group_, alias_db);
    auto values = GetMemoryPlanningCandidates(graph_, node_has_out_variant);
    value_to_same_storage_values_ =
        GenerateSameStorageValues(lm, value_group_, values, alias_db);
  }
}

const StaticModuleOptions& StaticModule::opts() const {
  return opts_;
}

size_t StaticModule::num_outputs() const {
  return graph_->outputs().size();
}

size_t StaticModule::num_inputs() const {
  return graph_->inputs().size();
}

StaticRuntime& StaticModule::runtime() {
  if (!cached_runtime_) {
    cached_runtime_ = std::make_unique<StaticRuntime>(*this);
  }
  return *cached_runtime_;
}

Node* StaticModule::findNodeWithKindForTesting(const std::string& kind) const {
  for (auto& pnode : nodes()) {
    if (pnode.node()->kind().toQualString() == kind) {
      return pnode.node();
    }
  }
  return nullptr;
}

c10::IValue StaticModule::operator()(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  return runtime()(args, kwargs);
}

c10::IValue StaticModule::operator()(
    std::vector<c10::IValue>&& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  return runtime()(std::move(args), kwargs);
}

StaticRuntime::StaticRuntime(const StaticModule& sm) : static_module_(sm), nodes_(sm.nodes()) {
  const auto total_num_node_outputs = std::accumulate(nodes_.begin(), nodes_.end(), 0, [](uint32_t sum, const ProcessedNode &pnode) {
    return sum + pnode.num_outputs();
  });
  values_.resize(
      sm.num_inputs() + sm.constants().size() + total_num_node_outputs);
  const auto inputs_index_offset = 0;
  const auto constants_index_offset = inputs_index_offset + sm.num_inputs();
  const auto constants_begin_it = values_.begin() + constants_index_offset;
  const auto constants_end_it = constants_begin_it + sm.constants().size();
  std::copy(sm.constants().begin(), sm.constants().end(), constants_begin_it);

  for (const auto idx : c10::irange(sm.nodes().size())) {
    auto& n = nodes_[idx];
    n.set_values(values_.data());
  }

  // TODO: can we convert outputs_ to store indices?
  for (auto index : sm.output_indices()) {
    outputs_.emplace_back(&values_[index]);
  }
}

StaticRuntime::~StaticRuntime() = default;

void StaticRuntime::set_inputs(
    const std::vector<IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  if (!kwargs.empty()) {
    // This is not ideal
    TORCH_CHECK(
        static_module_.schema(),
        "Schema is not available. Consider creating the Static Runtime "
        "with StaticModule(const torch::jit::Module& m) instead.");
    std::vector<c10::IValue> stack;
    stack.reserve(static_module_.num_inputs());
    if (static_module_.first_input_is_self()) {
      stack.emplace_back(static_module_.module()._ivalue());
    }
    stack.insert(stack.end(), args.begin(), args.end());

    static_module_.schema()->checkAndNormalizeInputs(stack, kwargs);
    DCHECK_EQ(static_module_.num_inputs(), stack.size());
    for (const auto i : c10::irange(stack.size())) {
      Input(i) = std::move(stack[i]);
    }
  } else {
    if (static_module_.first_input_is_self()) {
      Input(0) = static_module_.module()._ivalue();
      DCHECK_EQ(static_module_.num_inputs(), args.size() + 1);
      for (const auto i : c10::irange(args.size())) {
        Input(i + 1) = args[i];
      }
    } else {
      DCHECK_EQ(static_module_.num_inputs(), args.size());
      for (const auto i : c10::irange(args.size())) {
        Input(i) = args[i];
      }
    }
  }
}

void StaticRuntime::set_inputs(
    std::vector<IValue>&& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  if (!kwargs.empty()) {
    // This is not ideal
    TORCH_CHECK(
        static_module_.schema(),
        "Schema is not available. Consider creating the Static Runtime "
        "with StaticModule(const torch::jit::Module& m) instead.");
    std::vector<c10::IValue> stack;
    stack.reserve(static_module_.num_inputs());
    if (static_module_.first_input_is_self()) {
      stack.emplace_back(static_module_.module()._ivalue());
    }
    stack.insert(
        stack.end(),
        std::make_move_iterator(args.begin()),
        std::make_move_iterator(args.end()));

    static_module_.schema()->checkAndNormalizeInputs(stack, kwargs);
    DCHECK_EQ(static_module_.num_inputs(), stack.size());
    for (const auto i : c10::irange(stack.size())) {
      Input(i) = std::move(stack[i]);
    }
  } else {
    if (static_module_.first_input_is_self()) {
      Input(0) = static_module_.module()._ivalue();
      DCHECK_EQ(static_module_.num_inputs(), args.size() + 1);
      for (const auto i : c10::irange(args.size())) {
        Input(i + 1) = std::move(args[i]);
      }
    } else {
      DCHECK_EQ(static_module_.num_inputs(), args.size());
      for (const auto i : c10::irange(args.size())) {
        Input(i) = std::move(args[i]);
      }
    }
  }
}

void StaticRuntime::create_memory_planner() {
  if (!planner_) {
    planner_ = std::make_unique<MemoryPlanner>(
        this,
        static_module_.values_share_same_storage(),
        static_module_.value_group(),
        static_module_.opts().enable_out_variant,
        static_module_.opts().manage_output_tensors);
  }
}

c10::IValue StaticRuntime::move_outputs_to_tuple(uint32_t num_outputs) {
  bool should_move[num_outputs];
  for (const auto i : c10::irange(num_outputs)) {
    // REVIEW: is this actually safe or does trying to manage an
    // output tensor indicate deeper problems? what was supposed to
    // stop it?
    should_move[i] = !isManagedOutputTensor(*outputs_[i]);
  }
#define TORCH_SR_MOVE_IF_POSSIBLE(idx)                      \
  (should_move[(idx)] ? IValue(std::move(*outputs_[(idx)])) \
                      : IValue(*outputs_[(idx)]))
  switch (num_outputs) {
    case 1:
      return c10::ivalue::Tuple::create(TORCH_SR_MOVE_IF_POSSIBLE(0));
    case 2:
      return c10::ivalue::Tuple::create(
          TORCH_SR_MOVE_IF_POSSIBLE(0), TORCH_SR_MOVE_IF_POSSIBLE(1));
    case 3:
      return c10::ivalue::Tuple::create(
          TORCH_SR_MOVE_IF_POSSIBLE(0),
          TORCH_SR_MOVE_IF_POSSIBLE(1),
          TORCH_SR_MOVE_IF_POSSIBLE(2));
    default: {
      std::vector<c10::IValue> outputs;
      outputs.reserve(num_outputs);
      for (const auto i : c10::irange(num_outputs)) {
        // use move here. Otherwise, clean up outputs_[i] explicitly
        outputs.emplace_back(TORCH_SR_MOVE_IF_POSSIBLE(i));
      }
      return c10::ivalue::Tuple::create(std::move(outputs));
    }
  }
#undef TORCH_SR_MOVE_IF_POSSIBLE
}

/// [Check and correct bad schema alias info at runtime]
/// Static runtime relies on the operator schema's alias info to be correct for
/// memory planning. Because it's hard to enforce the alias info to be correct,
/// we need to do runtime detection for accidental aliases that do not comply
/// with the schema. Only aliases of managed tensors are problematic. To avoid
/// runtime crashes, we can add runtime detection and force the op to comply
/// with its schema by cloning the alias. Because all managed tensors' data_ptrs
/// are part of the internal buffer that the MemoryPlanner allocates, we can
/// check aliases by checking the memory overlap with this internal buffer. But
/// a tensor's storage can be resized during inferenceso we need another way to
/// handle the resized case.
///
/// There are two ways for incorrect schema to break memory planning. Let's look
/// at two examples:
///
/// Example 1:
/// @code
///   def forward(x):
///     a = x + x
///     b = bad_op(a)  # b ends up aliasing a incorrectly
///     return (b)
/// @endcode
/// bad_op: its schema says it returns a new Tensor, but it actually returns an
/// alias. In this case, the memory planner would recognize `a` as a managed
/// tensor and clean up its memory before returning `b`. But `b` is actually an
/// alias of `a`, when `a`'s data_ptr get reset, `b`'s data_ptr gets reset too.
///
/// Example 2:
/// @code
///   def forward(x):
///     a = x + x
///     a2 = bad_op(a) # a2 ends up alias a incorrectly
///     b = a + a
///     c = b * b # c shares storage with a
///     d = c + 2 # d shares storage with b
///     e = a2 * a2
///     return (d, e)
/// @endcode
/// With the memory reuse algorithm, `c` could end up sharing storage with `a`,
/// but because of bad_op, `a2` now aliases `a`. `c` overwrites `a` and
/// therefore `a2`, leading to the wrong results. We solve this problem with two
/// steps. Note this doesn't happen with the current memory reuse algorithm
/// because of the way it's implemented. Things could change with a different
/// implementation.
///
/// Step 1, annotate the ProcessedNodes with a flag `check_memory_overlap_` set
/// to true if its outputs do not alias its inputs as indicated by the AliasDb
/// and all of its outputs are Tensors. Then at runtime, we check that the
/// nodes' output tensors do not overlap with the internal buffer that the
/// MemoryPlanner allocates. For latency concerns, we only run this check for
/// fallback ops. The schemas of native ops and out variants are vetted and
/// enforced with static runtime unit tests. For the first iteration, we do a
/// full memory overlap check with
/// ProcessedNode::verify_and_correct_memory_overlap() because the internal
/// buffer doesn't exist yet.
///
/// Step 2, if a managed tensor gets resized during inference, it gets a new
/// data_ptr which is not from the buffer. We can tackle this corner case by
/// delaying the deallocation of the managed tensors to after the outputs are no
/// longer used (essentially merging the internal/output buffers into one).
/// Before the merging is implemented, we add another flag `overlap_detected_`
/// to flag any node with overlap detected in Step 1 and do a full memory
/// overlap check if the fast check (by checking memory overlap with internal
/// buffer) fails. There is still a corner case that fails with the added flag.
/// If a resize is triggered at the same time as the op creating an alias at the
/// same time, the current checks would fail to detect the alias.
///
/// There is another case of failure that step 2 can prevent. With
/// StaticModule::opts().cleanup_activations = false, the returned Static
/// Runtime instance in the instance pool can be re-entered while an unintended
/// output tensor's alias is still being used by the client (in the multi-threaded
/// setting). This can only be prevented by delaying the deallocation and
/// returning the Static Runtime instance after the client is done with the
/// outputs.

void StaticRuntime::verify_and_correct_memory_overlap(ProcessedNode& n) {
  // The slow check can be removed once the internal/output buffers are merged
  if (C10_UNLIKELY(n.check_outputs_for_memory_overlap())) {
    if (C10_UNLIKELY(!planner_ && static_module_.opts().cleanup_activations)) {
      // slow check, for first iter only with cleanup_activations = true
      n.verify_and_correct_memory_overlap();
    } else if (planner_) {
      bool overlap_detected_with_fast_check = false;
      for (size_t i = 0; i < n.outputs().size(); i++) {
        at::Tensor& t = n.Output(i).toTensor();
        if (planner_->overlapWithInternalBuffer(t.data_ptr())) {
          VLOG(1) << "Detected alias for node: " << PrintNode(n.node());
          n.Output(i) = at::native::clone(t, c10::nullopt);
          // set flag if overlap detected
          overlap_detected_with_fast_check = true;
          n.set_outputs_memory_overlap_detected();
        }
      }
      if (n.outputs_memory_overlap_detected() &&
          !overlap_detected_with_fast_check) {
        // slow check. Only run when the fast check fails.
        n.verify_and_correct_memory_overlap();
      }
    }
  }
}

template <typename IValueList>
c10::IValue StaticRuntime::run_impl(
    IValueList&& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  // We assume inference workloads, so we do not need
  // autograd. Enabling this is a significant win on dispatcher
  // overhead because it saves a round of dispatch for at least some
  // functions, such as resize_ and resize_as_.
  c10::InferenceMode mode;

  if (planner_) {
    DCHECK(
        !static_module_.opts().manage_output_tensors ||
        checkOutputTensorMemoryLeaks());
    planner_->allocate();
  }

  set_inputs(std::forward<IValueList>(args), kwargs);

  // NB: before optimizing the order of execution, ensure that the
  // memory optimization pass (LivenessMap) is
  // aware of the new order!
  for (auto& n : nodes_) {
    // LOG(INFO) << "Running node: " << PrintNode(n.node());
    n.run();
    // Check for incorrect schema alias info.
    verify_and_correct_memory_overlap(n);
  }

  if (static_module_.opts().cleanup_activations) {
    // MemoryPlanner is created after the first invocation of `run()`. This is
    // done intentionally because MemoryPlanner uses `Tensor` sizes of the
    // previous `run()` for memory planning of subsequent runs
    create_memory_planner();
    planner_->deallocate();
    // clean up owning refs of input tensors
    clean_up_input_ivalues();
  }

  // no need to keep references of outputs in static runtime anymore
  if (static_module_.num_outputs() > 1) {
    return move_outputs_to_tuple(static_module_.num_outputs());
  }
#ifndef NDEBUG
  check_for_memory_leak(false);
#endif
  // REVIEW: same safety question as above
  if (C10_UNLIKELY(isManagedOutputTensor(*outputs_[0]))) {
    return *outputs_[0];
  } else {
    // use move here. Otherwise, clean up outputs_[0] explicitly
    return std::move(*outputs_[0]);
  }
}

c10::IValue StaticRuntime::operator()(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  return run_impl(args, kwargs);
}

c10::IValue StaticRuntime::operator()(
    std::vector<c10::IValue>&& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  return run_impl(std::move(args), kwargs);
}

namespace {

std::string generate_latency_json(const std::string& label, double millis) {
#ifdef FBCODE_CAFFE2
  folly::dynamic json = folly::dynamic::object();
  json["type"] = label;
  json["metric"] = "latency";
  json["unit"] = "ms";
  json["value"] = millis;
  return "PyTorchObserver " + folly::toJson(json);
#else
  return "";
#endif
}

} // namespace

void StaticRuntime::benchmark(
    const std::vector<std::vector<c10::IValue>>& args_list,
    const std::vector<std::unordered_map<std::string, c10::IValue>>&
        kwargs_list,
    const int warmup_runs,
    const int main_runs,
    bool print_per_node_time,
    bool generate_ai_pep_output) {
  TORCH_CHECK(
      kwargs_list.size() == 0 || args_list.size() == kwargs_list.size());
  std::cout << "Input size: " << args_list.size() << std::endl;
  if (args_list.size() == 0) {
    return;
  }
  float time_per_iter =
      benchmark_model(args_list, kwargs_list, warmup_runs, main_runs);
  std::cout << "Static runtime ms per iter: " << time_per_iter
            << ". Iters per second: " << 1000.0 / time_per_iter << std::endl;

  IndividualMetrics results =
      benchmark_individual_ops(args_list, kwargs_list, warmup_runs, main_runs);

  if (print_per_node_time) {
    for (const auto i : c10::irange(nodes_.size())) {
      const Node* node = nodes_[i].node();
      std::cout << "Node #" << i << ": " << results.time_per_node[i]
                << " ms/iter, ";
      node->print(std::cout, 0, nullptr, false);
    }
  }

  std::vector<std::pair<std::string, double>> time_per_node_type_vec{
      results.time_per_node_type.begin(), results.time_per_node_type.end()};
  std::sort(
      time_per_node_type_vec.begin(),
      time_per_node_type_vec.end(),
      [](auto& left, auto& right) { return left.second > right.second; });

  std::cout << "Time per node type:" << std::endl;
  for (const auto& p : time_per_node_type_vec) {
    const std::string& kind = p.first;
    const double ms = p.second;
    std::cout << std::setw(15) << ms << " ms. " << std::setw(10)
              << results.percent_per_node_type[kind] << "%. " << kind << " ("
              << results.instances_per_node_type[kind] << " nodes";
    if (results.out_nodes.count(kind)) {
      std::cout << ", out variant)" << std::endl;
    } else if (results.native_nodes.count(kind)) {
      std::cout << ", native)" << std::endl;
    } else {
      std::cout << ")" << std::endl;
    }

    if (generate_ai_pep_output) {
      LOG(INFO) << generate_latency_json(kind, ms);
    }
  }
  if (generate_ai_pep_output) {
    LOG(INFO) << generate_latency_json(
        "static_runtime_first_iter", results.first_iter_time);
  }
  std::cout << std::setw(15) << results.total_time << " ms. in Total"
            << std::endl;
  std::cout << "StaticRuntime setup time: " << results.setup_time << " ms"
            << std::endl;
  std::cout << "Memory allocation time: " << results.memory_alloc_time
            << " ms\n";
  std::cout << "Memory deallocation time: " << results.memory_dealloc_time
            << " ms" << std::endl;
  std::cout << "Outputs deallocation time: " << results.output_dealloc_time
            << " ms" << std::endl;
  std::cout << "First iter time: " << results.first_iter_time << " ms"
            << std::endl;
  std::cout << "Number of operators: " << nodes_.size() << std::endl;

  if (planner_) {
    std::cout << "Total number of managed tensors: "
              << planner_->total_num_managed_tensors() << std::endl;
    std::cout << "Total number of managed output tensors: "
              << planner_->total_num_managed_output_tensors() << std::endl;
    std::cout << "Total number of unmanaged values: "
              << planner_->total_num_unmanaged() << std::endl;
    std::cout << "Total memory managed: " << planner_->total_managed()
              << " bytes" << std::endl;
    if (static_module_.opts().optimize_memory) {
      std::cout << "Total number of reused tensors: "
                << planner_->total_reused_tensors() << std::endl;
    }
    std::cout << "Total number of 'out' variant nodes/total number of nodes: "
              << results.out_nodes_count << "/" << results.total_nodes_count
              << " ("
              << 100.0 * (results.out_nodes_count) /
            static_cast<float>(results.total_nodes_count)
              << "%)" << std::endl;
  }
  check_for_memory_leak();

#ifndef NDEBUG
  std::unordered_map<std::string, c10::IValue> empty_kwargs;
  display_nodes(
      args_list[0], kwargs_list.size() > 0 ? kwargs_list[0] : empty_kwargs);
#endif
}

float StaticRuntime::benchmark_model(
    const std::vector<std::vector<c10::IValue>>& args_list,
    const std::vector<std::unordered_map<std::string, c10::IValue>>&
        kwargs_list,
    const int warmup_runs,
    const int main_runs) {
  TORCH_CHECK(warmup_runs >= 0 && main_runs >= 1);
  TORCH_CHECK(
      kwargs_list.size() == 0 || args_list.size() == kwargs_list.size());

  const bool is_kwargs_empty = kwargs_list.size() == 0;
  const std::unordered_map<std::string, c10::IValue> empty_kwargs;
  bool manage_output_tensors = static_module_.opts().manage_output_tensors;
  for (const auto i : c10::irange(warmup_runs)) {
    (void)i; // Suppress unused variable warning
    for (const auto j : c10::irange(args_list.size())) {
      operator()(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);
      if (manage_output_tensors) {
        deallocateOutputTensors();
      }
    }
  }
  caffe2::Timer timer;
  for (const auto i : c10::irange(main_runs)) {
    (void)i; // Suppress unused variable warning
    for (const auto j : c10::irange(args_list.size())) {
      operator()(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);
      if (manage_output_tensors) {
        deallocateOutputTensors();
      }
    }
  }
  float millis = timer.MilliSeconds();
  return millis / (static_cast<float>(main_runs) * args_list.size());
}

bool display_ivalue(const IValue& iv) {
  if (iv.isTensor()) {
    std::cout << "Tensor " << iv.toTensor().toString() << " {";
    for (const auto i : c10::irange(iv.toTensor().sizes().size())) {
      std::cout << iv.toTensor().sizes()[i];
      if (iv.toTensor().sizes().size() > i + 1) {
        std::cout << ", ";
      }
    }
    std::cout << "}\n";
    return true;
  } else if (iv.isTensorList()) {
    std::cout << "TensorList {" << iv.toTensorList().size() << "}\n";
    return true;
  } else if (iv.isGenericDict()) {
    std::cout << "Dict {" << iv.toGenericDict().size() << "}\n";
    return true;
  } else if (iv.isTuple()) {
    std::cout << "Tuple {" << iv.toTupleRef().elements().size() << "}\n";
    return true;
  } else if (iv.isInt()) {
    std::cout << "int {" << iv.toInt() << "}\n";
    return true;
  } else if (iv.isBool()) {
    std::cout << "bool {" << iv.toBool() << "}\n";
    return true;
  } else if (iv.isDouble()) {
    std::cout << "double {" << iv.toDouble() << "}\n";
    return true;
  }
  return false;
}

void display_pnode_info(const ProcessedNode& pnode) {
  pnode.node()->print(std::cout, 0, nullptr, false);
  for (const auto i : c10::irange(pnode.num_inputs())) {
    std::cout << "\ti" << i << ": ";
    if (!display_ivalue(pnode.Input(i))) {
      std::cout << *(pnode.node()->inputs()[i]->type()) << '\n';
    }
  }
  const auto outputs = pnode.outputs();
  for (const auto i : c10::irange(outputs.size())) {
    std::cout << "\to" << i << ": ";
    if (!display_ivalue(outputs[i])) {
      std::cout << *(pnode.node()->outputs()[i]->type()) << '\n';
    }
  }
}

void StaticRuntime::display_nodes(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  c10::InferenceMode mode;
  if (planner_) {
    planner_->allocate();
  }
  set_inputs(args, kwargs);

  for (auto& node : nodes_) {
    node.run();
    display_pnode_info(node);
  }

  if (static_module_.opts().cleanup_activations) {
    // MemoryPlanner is created after the first invocation of `run()`. This is
    // done intentionally because MemoryPlanner uses `Tensor` sizes of the
    // previous `run()` for memory planning of subsequent runs
    create_memory_planner();
    planner_->deallocate();
    // clean up owning refs of input tensors
    clean_up_input_ivalues();
  }
}

StaticRuntime::IndividualMetrics StaticRuntime::benchmark_individual_ops(
    const std::vector<std::vector<c10::IValue>>& args_list,
    const std::vector<std::unordered_map<std::string, c10::IValue>>&
        kwargs_list,
    const int warmup_runs,
    const int main_runs) {
  TORCH_CHECK(
      kwargs_list.size() == 0 || args_list.size() == kwargs_list.size());
  TORCH_CHECK(warmup_runs >= 1 && main_runs >= 1);
  if (args_list.size() == 0) {
    return {};
  }

  const bool is_kwargs_empty = kwargs_list.size() == 0;
  const std::unordered_map<std::string, c10::IValue> empty_kwargs;
  bool manage_output_tensors = static_module_.opts().manage_output_tensors;
  // See comment on above use of InferenceMode for
  // explanation.
  c10::InferenceMode mode;

  IndividualMetrics results;
  results.time_per_node.resize(nodes_.size(), 0);

  // setup time
  caffe2::Timer timer;

  set_inputs(args_list[0], is_kwargs_empty ? empty_kwargs : kwargs_list[0]);

  results.setup_time = timer.MilliSeconds();

  // The first iteration profiles each node's output Tensors' sizes and
  // initializes the memory planner with the profile information. Folllowing
  // iterations just use the already established memory planning.
  timer.Start();
  operator()(args_list[0], is_kwargs_empty ? empty_kwargs : kwargs_list[0]);
  if (manage_output_tensors) {
    deallocateOutputTensors();
  }
  results.first_iter_time = timer.MilliSeconds();

  // warmup runs
  for (const auto i : c10::irange(warmup_runs - 1)) {
    (void)i; // Suppress unused variable warning
    for (const auto j : c10::irange(args_list.size())) {
      operator()(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);
      if (manage_output_tensors) {
        deallocateOutputTensors();
      }
    }
  }

  // main runs
  for (const auto i : c10::irange(main_runs)) {
    (void)i; // Suppress unused variable warning

    for (const auto j : c10::irange(args_list.size())) {
      set_inputs(args_list[j], is_kwargs_empty ? empty_kwargs : kwargs_list[j]);

      timer.Start();
      if (planner_) {
        planner_->allocate();
      }
      float millis = timer.MilliSeconds();
      results.memory_alloc_time += millis;

      for (const auto k : c10::irange(nodes_.size())) {
        timer.Start();
        nodes_[k].run();
        millis = timer.MilliSeconds();
        results.time_per_node[k] += millis;
      }
      timer.Start();
      if (static_module_.opts().cleanup_activations) {
        create_memory_planner();
        planner_->deallocate();
        // clean up owning refs of input tensors
        clean_up_input_ivalues();
      }
      if (manage_output_tensors) {
        deallocateOutputTensors();
      }
      millis = timer.MilliSeconds();
      results.memory_dealloc_time += millis;

      timer.Start();
      // no need to keep references of outputs in static runtime anymore
      c10::IValue output;
      if (static_module_.num_outputs() > 1) {
        output = move_outputs_to_tuple(static_module_.num_outputs());
      }

#ifndef NDEBUG
      check_for_memory_leak(false);
#endif

      // use move here. Otherwise, clean up outputs_[0] explicitly
      output = std::move(*outputs_[0]);
      // release outputs explicitly to measure the time it takes
      output = IValue();
      millis = timer.MilliSeconds();
      results.output_dealloc_time += millis;
    }
  }

  // post processing
  const float num_total_iters =
      (static_cast<float>(main_runs) * args_list.size());
  for (const auto i : c10::irange(nodes_.size())) {
    const Node* node = nodes_[i].node();
    std::string kind = std::string(node->kind().toQualString());
    results.time_per_node[i] /= num_total_iters;
    results.time_per_node_type[kind] += results.time_per_node[i];
    results.instances_per_node_type[kind]++;
    if (nodes_[i].has_out_variant()) {
      results.out_nodes.insert(kind);
      results.out_nodes_count++;
    } else if (nodes_[i].has_native()) {
      results.native_nodes.insert(kind);
    }
    results.total_time += results.time_per_node[i];
  }
  results.total_nodes_count = nodes_.size();
  results.memory_alloc_time /= num_total_iters;
  results.memory_dealloc_time /= num_total_iters;
  results.output_dealloc_time /= num_total_iters;
  for (const auto& p : results.time_per_node_type) {
    const std::string& kind = p.first;
    results.percent_per_node_type[kind] = p.second / results.total_time * 100;
  }
  return results;
}

void StaticRuntime::check_for_memory_leak(bool output_returned) {
  if (!static_module_.opts().cleanup_activations) {
    return;
  }

  // check for inputs
  for (const auto i : c10::irange(static_module_.num_inputs())) {
    TORCH_CHECK(values_[i].isNone(), "Input ", i, " was not cleaned up");
  }
  FastSet<const IValue*> output_ivalues(outputs_.begin(), outputs_.end());
  for (const auto n : c10::irange(nodes_.size())) {
    auto& pnode = nodes_[n];
    for (const auto i : c10::irange(pnode.num_outputs())) {
      const IValue* ival = &pnode.Output(i);
      const Value* val = pnode.node()->output(i);
      // subtlety: isManagedOutputTensorValue may give a false
      // negative here if an output is an alias of this value, so
      // check the actual tensor!
      if (planner_ &&
          (planner_->isManagedOutputTensor(*ival) ||
           planner_->isManagedOutputTensorValue(val))) {
        // `ival` contains a managed output tensor that the runtime doesn't
        // reclaim at the end of an iteration, but the client does so
        // by explicitly calling `StaticRuntime::deallocateOutputTensors`.
        continue;
      }
      const std::string error_msg = "Output " + c10::to_string(i) + ", %" +
          val->debugName() + " of node " + c10::to_string(n) +
          " was not cleaned up";
      if (output_ivalues.count(ival) == 0) {
        // check for intermediates
        if (!ival->isNone()) {
          TORCH_CHECK(
              ival->isTensor() ||
                  static_module_.is_optimizable_container_type(pnode.node()),
              error_msg);
          if (ival->isTensor()) {
            const auto& t = ival->toTensor();
            if (t.defined()) {
              auto* storage_impl = t.storage().unsafeGetStorageImpl();
              TORCH_CHECK(
                  storage_impl->data() == nullptr ||
                      (planner_ &&
                       planner_->isManagedStorageImpl(storage_impl)),
                  error_msg);
            }
          }
        }
      } else {
        // check for outputs
        if (output_returned) {
          TORCH_CHECK(ival->isNone(), error_msg);
        }
      }
    }
  }
  VLOG(1) << "Finished checking for memory leak";
}

void StaticRuntime::deallocateOutputTensors() {
  if (!static_module_.opts().manage_output_tensors) {
    TORCH_CHECK(
        !planner_ || planner_->numOutputBufferBytes() == 0,
        "manage_output_tensors is disabled, but output tensor buffer is not empty.");
    return;
  }
  if (planner_) {
    planner_->deallocateOutputTensors();
    DCHECK(checkOutputTensorMemoryLeaks());
  }
}

bool StaticRuntime::checkOutputTensorMemoryLeaks() {
  if (!static_module_.opts().manage_output_tensors || !planner_) {
    return true;
  }
  for (const auto n : c10::irange(nodes_.size())) {
    auto& pnode = nodes_[n];
    for (const auto i : c10::irange(pnode.num_outputs())) {
      const IValue* ival = &pnode.Output(i);
      const Value* val = pnode.node()->output(i);
      if (!planner_->isManagedOutputTensorValue(val)) {
        continue;
      }
      const auto& t = ival->toTensor();
      if (t.defined()) {
        auto* storage_impl = t.storage().unsafeGetStorageImpl();
        const std::string error_msg = "Output " + c10::to_string(i) + ", %" +
            val->debugName() + " of node " + c10::to_string(n) +
            " was not cleaned up";
        TORCH_CHECK(storage_impl->data() == nullptr, error_msg);
      }
    }
  }
  VLOG(1) << "Finished checking for memory leak from output tensors";
  return true;
}

bool StaticRuntime::isManagedOutputTensor(const IValue& ivalue) {
  return planner_ && planner_->isManagedOutputTensor(ivalue);
}

ProcessedNode::ProcessedNode(
    Node* node,
    ProcessedNodeInputs inputs,
    uint16_t outputs_offset,
    bool enable_out_variant,
    bool check_memory_overlap)
    : node_(node),
      inputs_(std::move(inputs)),
      outputs_offset_(outputs_offset)
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
      ,
      op_name_(node->kind().toQualString())
#endif
{
  TORCH_CHECK(
      node->outputs().size() < (1 << (sizeof(num_outputs_) * 8)),
      node->outputs().size(),
      " outputs to ProcessedNode ",
      node->kind().toQualString(),
      " is too many to use 2-byte indexing");
  num_outputs_ = node->outputs().size();

  if (enable_out_variant) {
    std::function<void(ProcessedNode*)> f = getOutOfPlaceOperation(node);
    if (f) {
      fn_ = {f, FunctionKind::kOutVariant};
      VLOG(1) << "Switch to out variant for node: " << PrintNode(node);
      return;
    }
  }
  {
    std::function<void(ProcessedNode*)> f = getNativeOperation(node);
    if (f) {
      fn_ = {f, FunctionKind::kNativeFunction};
      VLOG(1) << "Switch to native impl for node: " << PrintNode(node);
      return;
    }
  }
  {
    const Operator& op = node->getOperator();
    std::function<void(ProcessedNode*)> f =
        [node_op = op.getOperation(node)](ProcessedNode* pnode) mutable {
          std::vector<IValue> stack;
          Node* node = pnode->node_;
          const size_t size = node->inputs().size();
          stack.reserve(size + (hasVarArgs(node) ? 1 : 0));
          for (const auto i : c10::irange(size)) {
            stack.emplace_back(pnode->Input(i));
          }
          // Need to store the number of inputs in stack for variadic ops.
          if (hasVarArgs(node)) {
            stack.emplace_back(static_cast<int>(size));
          }

          node_op(stack);

          DCHECK_EQ(stack.size(), node->outputs().size());
          for (const auto i : c10::irange(node->outputs().size())) {
            pnode->Output(i) = std::move(stack[i]);
          }
        };
    fn_ = {f, FunctionKind::kInterpreterFallback, check_memory_overlap};
    VLOG(1) << "Fallback interpreter for node: " << PrintNode(node);
  }
}

std::vector<IValue> ProcessedNode::clone_inputs() const {
  std::vector<IValue> result;
  result.reserve(inputs_.size());
  for (const auto idx : c10::irange(num_inputs())) {
    result.emplace_back(Input(idx));
  }
  return result;
}

void ProcessedNode::run() {
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  bool pre_sampled = false;
  if (C10_UNLIKELY(at::shouldRunRecordFunction(&pre_sampled))) {
    at::RecordFunction guard(at::RecordScope::FUNCTION, pre_sampled);
    if (guard.isActive()) {
      if (guard.needsInputs()) {
        guard.before(get_op_name(), clone_inputs());
      } else {
        guard.before(get_op_name());
      }
    }
    fn_.f(this);
  } else {
    fn_.f(this);
  }
#else
  fn_.f(this);
#endif
#ifndef NDEBUG
  verify_no_memory_overlap();
#endif
}

static bool checkNoMemoryOverlap(const at::Tensor& a, const at::Tensor& b) {
  at::MemOverlapStatus status = at::get_overlap_status(a, b);
  if (status == at::MemOverlapStatus::FULL ||
      status == at::MemOverlapStatus::PARTIAL) {
    return false;
  }
  if (status == at::MemOverlapStatus::TOO_HARD) {
    LOG(WARNING) << "Detected TOO_HARD memory overlap status";
  }
  return true;
}

bool ProcessedNode::verify_no_memory_overlap() const {
  return verify_outputs_dont_overlap_each_other() &&
      verify_inputs_dont_overlap_outputs();
}

bool ProcessedNode::verify_outputs_dont_overlap_each_other() const {
  for (const auto i : c10::irange(num_outputs_)) {
    if (!Output(i).isTensor()) {
      continue;
    }
    const auto& out0_t = Output(i).toTensor();
    for (const auto j : c10::irange(i + 1, num_outputs_)) {
      if (!Output(j).isTensor()) {
        continue;
      }
      const auto& out1_t = Output(j).toTensor();
      if (!checkNoMemoryOverlap(out0_t, out1_t)) {
        LOG(INFO) << "Node output " << i << " overlaps with output " << j
                  << ", " << PrintNode(node_);
        return false;
      }
    }
  }
  return true;
}

bool ProcessedNode::verify_inputs_dont_overlap_outputs() const {
  auto schema = node()->maybeSchema();
  // skip memory overlap check for mutable ops with only one output
  if (!schema || (schema->is_mutable() && num_outputs_ == 1)) {
    return true;
  }
  for (const auto i : c10::irange(inputs_.size())) {
    const IValue* in = &Input(i);
    if (!in->isTensor()) {
      continue;
    }
    const auto& in_t = in->toTensor();
    for (const auto j : c10::irange(num_outputs_)) {
      const IValue& out = Output(j);
      if (!out.isTensor()) {
        continue;
      }
      const auto& out_t = out.toTensor();
      if (!checkNoMemoryOverlap(in_t, out_t)) {
        LOG(INFO) << "Node input " << i << " overlaps with output " << j << ", "
                  << PrintNode(node_);
        LOG(INFO) << *schema;
        return false;
      }
    }
  }
  return true;
}

void ProcessedNode::verify_and_correct_memory_overlap() {
  for (const auto i : c10::irange(inputs_.size())) {
    const IValue& in = Input(i);
    if (!in.isTensor()) {
      continue;
    }
    const auto& in_t = in.toTensor();
    for (const auto j : c10::irange(num_outputs_)) {
      const auto& out_t = Output(j).toTensor();
      if (!checkNoMemoryOverlap(in_t, out_t)) {
        VLOG(1) << "Detected alias for node: " << PrintNode(node());
        Output(i) = at::native::clone(out_t, c10::nullopt);
        set_outputs_memory_overlap_detected();
      }
    }
  }
}

} // namespace jit
} // namespace torch
