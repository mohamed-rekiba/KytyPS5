#ifndef EMULATOR_INCLUDE_EMULATOR_FAKE_EA_UPSTREAM_H_
#define EMULATOR_INCLUDE_EMULATOR_FAKE_EA_UPSTREAM_H_

#include <cstdint>
#include <memory>
#include <vector>

// A connection to EA's real Blaze server, for the calls the built-in server does not answer yet
// (KYTY_FAKE_EA_UPSTREAM=1). It asks EA's server picker for a server, as FC 27 does, and does not
// check the certificate of the server.
namespace Libs::FakeEa {

class Upstream {
public:
	Upstream();
	~Upstream();
	Upstream(const Upstream&)            = delete;
	Upstream& operator=(const Upstream&) = delete;

	// False: no server was found, or the connection failed. Call once.
	bool Connect();

	// Sends a request frame (header and payload) and reads the reply frame into `reply`.
	bool Exchange(const std::vector<uint8_t>& request, std::vector<uint8_t>& reply);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::FakeEa

#endif // EMULATOR_INCLUDE_EMULATOR_FAKE_EA_UPSTREAM_H_
