#ifndef EMULATOR_SRC_LIBS_AUDIOAMBISONICS_H_
#define EMULATOR_SRC_LIBS_AUDIOAMBISONICS_H_

#include <cstddef>
#include <cstdint>
#include <span>

namespace Libs::Audio::Ambisonics {

// A game can hand its 3D sound to the system as an ambisonic sound field: one mono stream for
// each spherical-harmonic channel, with the place of every sound encoded in how it is spread
// over the channels. The system decodes the field for the speakers that are there.
//
// Channel numbers are ACN: 0 is W (all directions), 1 is Y (left), 2 is Z (up), 3 is X (front).
// The levels are taken as SN3D, where W carries a sound at full level and the other three at
// most at full level. This is an assumption: no title was seen that tells the two common
// conventions apart.

// One channel of the field for one block of frames.
struct Channel {
	uint32_t     acn     = 0;
	const float* samples = nullptr;
	float        gain    = 1.0f;
};

// Decodes the field to interleaved stereo: `out` gets 2 * `frames` samples.
// Each output is a virtual cardioid microphone that points to its side: left = (W + Y) / 2,
// right = (W - Y) / 2. A sound on the left is in the left output only, a sound in front or
// behind is in both at half level. Stereo has no height and no front and back, so Z, X and the
// channels of higher orders, which only sharpen directions, are not used.
inline void DecodeToStereo(std::span<const Channel> channels, size_t frames, float* out) {
	for (size_t i = 0; i < frames * 2; i++) {
		out[i] = 0.0f;
	}
	for (const auto& channel: channels) {
		if (channel.samples == nullptr || channel.acn > 1) {
			continue;
		}
		const float left  = 0.5f * channel.gain;
		const float right = channel.acn == 0 ? left : -left;
		for (size_t i = 0; i < frames; i++) {
			out[i * 2] += left * channel.samples[i];
			out[i * 2 + 1] += right * channel.samples[i];
		}
	}
}

} // namespace Libs::Audio::Ambisonics

#endif // EMULATOR_SRC_LIBS_AUDIOAMBISONICS_H_
