#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"

#include <algorithm>

#include <queue>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

struct ChainResult {
	Value value;
	Inst* write = nullptr;
};

ChainResult SearchChain(Value value, uint32_t lane, uint32_t wave_size) {
	for (;;) {
		value      = value.Resolve();
		auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::WriteLane) {
			return {value};
		}
		const auto selector = inst->Arg(2).Resolve();
		if (!selector.IsImmediate() || selector.GetType() != Type::U32) {
			return {value};
		}
		if (selector.U32() % wave_size == lane) {
			return {value, inst};
		}
		value = inst->Arg(0);
	}
}

bool IsPossibleToEliminate(Value source, uint32_t lane, uint32_t wave_size) {
	std::queue<Value>         queue;
	std::unordered_set<Inst*> visited;
	queue.push(source);

	while (!queue.empty()) {
		const auto chain = SearchChain(queue.front(), lane, wave_size);
		queue.pop();
		if (chain.write != nullptr) {
			continue;
		}
		auto* inst = chain.value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::Phi || inst->NumArgs() == 0) {
			return false;
		}
		if (!visited.insert(inst).second) {
			continue;
		}
		for (size_t index = inst->NumArgs(); index-- > 0;) {
			queue.push(inst->Arg(index));
		}
	}
	return true;
}

using PhiMap = std::unordered_map<Inst*, Inst*>;

Value GetRealValue(PhiMap& phi_map, Value source, uint32_t lane, uint32_t wave_size) {
	const auto chain = SearchChain(source, lane, wave_size);
	if (chain.write != nullptr) {
		return chain.write->Arg(1);
	}

	auto* inst = chain.value.ResolveInstruction();
	EXIT_IF(inst->GetOpcode() != ValueOpcode::Phi);
	const auto [entry, is_new] = phi_map.try_emplace(inst);
	if (!is_new) {
		return Value(entry->second);
	}

	auto* block           = inst->Parent();
	auto  insertion_point = std::find_if(block->begin(), block->end(),
	                                     [&](const Inst& candidate) { return &candidate == inst; });
	EXIT_IF(insertion_point == block->end());
	auto& phi = *block->PrependNewInst(insertion_point, ValueOpcode::Phi);
	phi.SetFlags(Type::U32);
	entry->second = &phi;

	std::vector<Value> arguments;
	arguments.reserve(inst->NumArgs());
	for (size_t index = 0; index < inst->NumArgs(); index++) {
		arguments.push_back(GetRealValue(phi_map, inst->Arg(index), lane, wave_size));
	}
	const auto first = arguments.front().Resolve();
	if (std::ranges::all_of(arguments,
	                        [&](Value argument) { return argument.Resolve() == first; })) {
		phi.ReplaceUsesWith(first);
	} else {
		for (size_t index = 0; index < arguments.size(); index++) {
			phi.AddPhiOperand(inst->PhiBlock(index), arguments[index]);
		}
	}
	return Value(&phi);
}

} // namespace

ReadLaneStats EliminateReadLane(Program& program, uint32_t wave_size) {
	ReadLaneStats stats;
	if (wave_size != 32u && wave_size != 64u) {
		return stats;
	}

	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::ReadLane) {
				continue;
			}
			const auto selector = inst.Arg(1).Resolve();
			if (!selector.IsImmediate() || selector.GetType() != Type::U32) {
				continue;
			}

			const auto lane  = selector.U32() % wave_size;
			const auto chain = SearchChain(inst.Arg(0), lane, wave_size);
			if (chain.write != nullptr) {
				inst.ReplaceUsesWith(chain.write->Arg(1));
				stats.rewritten_reads++;
				continue;
			}
			auto* producer = chain.value.TryInstruction();
			if (producer == nullptr || producer->GetOpcode() != ValueOpcode::Phi ||
			    !IsPossibleToEliminate(chain.value, lane, wave_size)) {
				continue;
			}

			PhiMap phi_map;
			inst.ReplaceUsesWith(GetRealValue(phi_map, chain.value, lane, wave_size));
			stats.rewritten_reads++;
		}
	}
	return stats;
}

namespace {

Block::iterator PositionOf(Inst* inst) {
	auto* block = inst->Parent();
	auto  it    = std::find_if(block->begin(), block->end(),
	                           [&](const Inst& candidate) { return &candidate == inst; });
	EXIT_IF(it == block->end());
	return it;
}

// What lane `lane` of a register holds, when every lane holds the same values except where
// V_WRITELANE put a scalar into one lane. A compiler keeps scalars it has no register for in the
// lanes of a vector register, so the value must come back exactly: through the writes to other
// lanes, and through the merges that masked writes and joining paths leave behind.
Value LaneOfUniformRegister(std::unordered_map<Inst*, Value>& known, Value source, uint32_t lane) {
	source     = source.Resolve();
	auto* inst = source.TryInstruction();
	if (inst == nullptr) {
		return source;
	}
	if (const auto found = known.find(inst); found != known.end()) {
		return found->second;
	}
	switch (inst->GetOpcode()) {
		case ValueOpcode::WriteLane: {
			const auto selector = inst->Arg(2).Resolve();
			const auto below    = LaneOfUniformRegister(known, inst->Arg(0), lane);
			Value      result;
			if (selector.IsImmediate()) {
				result = selector.U32() == lane ? inst->Arg(1) : below;
			} else {
				auto*      block = inst->Parent();
				const auto at    = PositionOf(inst);
				const auto hit =
				    block->PrependNewInst(at, ValueOpcode::IEqual32, {selector, Value(lane)});
				result = Value(&*block->PrependNewInst(at, ValueOpcode::SelectU32,
				                                       {Value(&*hit), inst->Arg(1), below}));
			}
			known.insert_or_assign(inst, result);
			return result;
		}
		case ValueOpcode::SelectU32: {
			const auto taken   = LaneOfUniformRegister(known, inst->Arg(1), lane);
			const auto skipped = LaneOfUniformRegister(known, inst->Arg(2), lane);
			Value      result  = taken;
			if (taken.Resolve() == inst->Arg(1).Resolve() &&
			    skipped.Resolve() == inst->Arg(2).Resolve()) {
				// No lane write below it: an ordinary value, the same in every lane.
				result = source;
			} else if (taken.Resolve() != skipped.Resolve()) {
				result = Value(&*inst->Parent()->PrependNewInst(
				    PositionOf(inst), ValueOpcode::SelectU32, {inst->Arg(0), taken, skipped}));
			}
			known.insert_or_assign(inst, result);
			return result;
		}
		case ValueOpcode::Phi: {
			if (inst->NumArgs() == 0) {
				return source;
			}
			// The new phi is known before its operands are, so that a loop ends.
			auto& phi = *inst->Parent()->PrependNewInst(PositionOf(inst), ValueOpcode::Phi);
			phi.SetFlags(Type::U32);
			known.insert_or_assign(inst, Value(&phi));
			for (size_t index = 0; index < inst->NumArgs(); index++) {
				phi.AddPhiOperand(inst->PhiBlock(index),
				                  LaneOfUniformRegister(known, inst->Arg(index), lane));
			}
			return Value(&phi);
		}
		default: return source;
	}
}

} // namespace

uint32_t LowerLaneOpsToSingleLane(Program& program) {
	uint32_t replaced = 0;
	// Per lane that is read: what each register value holds in it.
	std::unordered_map<uint32_t, std::unordered_map<Inst*, Value>> known;
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
			switch (inst.GetOpcode()) {
				case ValueOpcode::ReadLane: {
					const auto selector = inst.Arg(1).Resolve();
					if (selector.IsImmediate() && selector.GetType() == Type::U32) {
						const auto lane = selector.U32() % program.wave_size;
						inst.ReplaceUsesWith(LaneOfUniformRegister(known[lane], inst.Arg(0), lane));
					} else {
						inst.ReplaceUsesWith(inst.Arg(0));
					}
					replaced++;
					break;
				}
				case ValueOpcode::ReadFirstLane:
					inst.ReplaceUsesWith(inst.Arg(0));
					replaced++;
					break;
				case ValueOpcode::LaneId:
					inst.ReplaceUsesWith(Value(0u));
					replaced++;
					break;
				case ValueOpcode::Ballot: {
					const auto all = block->PrependNewInst(
					    it, ValueOpcode::SelectU32, {inst.Arg(0), Value(0xffffffffu), Value(0u)});
					const auto word = Value(&*all);
					const auto mask = block->PrependNewInst(
					    it, ValueOpcode::CompositeConstructU32x4, {word, word, word, word});
					inst.ReplaceUsesWith(Value(&*mask));
					replaced++;
					break;
				}
				default: break;
			}
		}
	}
	// What is left of a register with lane writes is used as this invocation's own lane, which
	// is lane 0. After the reads above, so that they still see the writes.
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
			if (inst.GetOpcode() != ValueOpcode::WriteLane) {
				continue;
			}
			const auto selector = inst.Arg(2).Resolve();
			if (selector.IsImmediate()) {
				inst.ReplaceUsesWith(selector.U32() % program.wave_size == 0u ? inst.Arg(1)
				                                                              : inst.Arg(0));
			} else {
				const auto hit =
				    block->PrependNewInst(it, ValueOpcode::IEqual32, {selector, Value(0u)});
				const auto own = block->PrependNewInst(it, ValueOpcode::SelectU32,
				                                       {Value(&*hit), inst.Arg(1), inst.Arg(0)});
				inst.ReplaceUsesWith(Value(&*own));
			}
			replaced++;
		}
	}
	return replaced;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
