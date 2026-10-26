#ifndef EMULATOR_INCLUDE_EMULATOR_FAKE_EA_TDF_H_
#define EMULATOR_INCLUDE_EMULATOR_FAKE_EA_TDF_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// TDF, the tagged data format of the payloads of Blaze.
namespace Libs::FakeEa {

enum TdfType : uint8_t { Int, Str, Bin, Struct, List, Map, Union, Variable, ObjType, ObjId, Float, Time };

inline constexpr uint32_t Tag(const char (&name)[5]) {
	uint32_t value = 0;
	for (int i = 0; i < 4; i++) value = (value << 6) | ((static_cast<uint32_t>(name[i]) - 0x20u) & 0x3fu);
	return value;
}

struct Reader {
	const uint8_t* data = nullptr;
	size_t         size = 0;
	size_t         at   = 0;
	bool           ok   = true;

	uint8_t Byte() {
		if (at >= size) {
			ok = false;
			return 0;
		}
		return data[at++];
	}
	int64_t Var() {
		const uint8_t first = Byte();
		int64_t       value = first & 0x3f;
		const bool    neg   = (first & 0x40) != 0;
		unsigned      shift = 6;
		for (bool more = (first & 0x80) != 0; more && ok;) {
			const uint8_t next = Byte();
			if (shift > 62) {
				ok = false;
				break;
			}
			value |= static_cast<int64_t>(static_cast<uint64_t>(next & 0x7f) << shift);
			shift += 7;
			more = (next & 0x80) != 0;
		}
		return neg ? -value : value;
	}
	uint32_t TagValue() {
		uint32_t value = 0;
		for (int i = 0; i < 3; i++) value = (value << 8) | Byte();
		return value;
	}
	void Skip(size_t count) {
		if (count > size - at) {
			ok = false;
			at = size;
		} else {
			at += count;
		}
	}
	void SkipValue(uint8_t type, int depth = 0) {
		if (depth > 16) {
			ok = false;
			return;
		}
		switch (type) {
			case Int:
			case Time: Var(); break;
			case Str:
			case Bin: Skip(static_cast<size_t>(std::max<int64_t>(Var(), 0))); break;
			case Struct:
				while (ok && at < size && data[at] != 0) {
					TagValue();
					const uint8_t member = Byte();
					SkipValue(member, depth + 1);
				}
				Byte();
				break;
			case List: {
				const uint8_t item  = Byte();
				const auto    count = Var();
				for (int64_t i = 0; i < count && ok; i++) SkipValue(item, depth + 1);
				break;
			}
			case Map: {
				const uint8_t key   = Byte();
				const uint8_t value = Byte();
				const auto    count = Var();
				for (int64_t i = 0; i < count && ok; i++) {
					SkipValue(key, depth + 1);
					SkipValue(value, depth + 1);
				}
				break;
			}
			case Union:
				if (Byte() != 0x7f) {
					TagValue();
					SkipValue(Byte(), depth + 1);
				}
				break;
			case ObjType: Var(); Var(); break;
			case ObjId: Var(); Var(); Var(); break;
			case Float: Skip(4); break;
			default: ok = false; break;
		}
	}
};

inline void AppendVar(std::vector<uint8_t>& out, int64_t value) {
	const bool neg = value < 0;
	auto       v   = static_cast<uint64_t>(neg ? -value : value);
	uint8_t    first = static_cast<uint8_t>(v & 0x3f);
	v >>= 6;
	if (v != 0) first |= 0x80;
	if (neg) first |= 0x40;
	out.push_back(first);
	while (v != 0) {
		uint8_t next = static_cast<uint8_t>(v & 0x7f);
		v >>= 7;
		if (v != 0) next |= 0x80;
		out.push_back(next);
	}
}

// The payload as text, one member for each line, for the trace. A payload the decoder does not know
// ends with the bytes in hex.
std::string DumpTdf(const uint8_t* data, size_t size);

// Replaces, in place, the text of every map entry with a text key and a text value, when the key
// has "secret", "key", "token" or "password" in it, by filler of the same length. Returns how many.
// A reply recorded from EA's server goes through this before it is kept.
size_t BlankSecrets(std::vector<uint8_t>& payload);

} // namespace Libs::FakeEa

#endif // EMULATOR_INCLUDE_EMULATOR_FAKE_EA_TDF_H_
