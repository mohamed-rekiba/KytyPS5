#!/usr/bin/env bash
# The unit tests that CI builds and runs, in one list.
#
#   unit-tests.sh targets [extra...]   the CMake targets: each name with _tests
#   unit-tests.sh filter [extra...]    the ctest -R filter: ^(name|name|...)$
#   unit-tests.sh check-windows FILE   fails when the Windows lines of the workflow FILE do
#                                      not name exactly these tests (Windows runs cmd, which
#                                      cannot call this script, so it keeps its own copy)
#
# An extra name is a test that only some platforms build (memory_tracker, page_manager).
set -euo pipefail

names=(
	archive_file
	audio_out2_port
	avplayer_file
	buffer_chunk
	cache_collection
	cache_folder
	controller_settings
	depth_snapshot_plan
	device_suitability
	fence_retirement
	file_lock
	host_lowering
	image_readback
	ime_dialog
	lru_cache
	metadata_fill
	pad_haptics
	pad_input
	pipeline_use
	program_list
	queue_commits
	readback_plan
	resource_materialization
	resource_tracking
	scalar_provenance
	shader_cfg
	shader_vertex_metadata
	soft_float64
	submit_plan
	trophy_system
	video_out_resolution
	virtual_memory_allocation
	write_heat
)

mode="${1:-}"
shift || true

case "$mode" in
	targets)
		all=("${names[@]}" "$@")
		printf '%s_tests\n' "${all[@]}" | paste -sd ' ' -
		;;
	filter)
		all=("${names[@]}" "$@")
		printf '^(%s)$\n' "$(printf '%s\n' "${all[@]}" | paste -sd '|' -)"
		;;
	check-windows)
		workflow="$1"
		expected="$(printf '%s\n' "${names[@]}" | sort)"
		filter="$(grep -o 'ctest --test-dir _Build/windows .*-R "^([^)]*)' "$workflow" |
			sed 's/.*"^(//; s/)$//' | tr '|' '\n' | sort)"
		targets="$(sed -n '/cmake --build _Build\/windows/,/--parallel/p' "$workflow" |
			grep -o '[a-z0-9_]*_tests' | sed 's/_tests$//' |
			grep -vx 'shader_recompiler_compute' | sort)"
		status=0
		if [ "$filter" != "$expected" ]; then
			echo "Windows ctest filter differs from .github/scripts/unit-tests.sh:" >&2
			diff <(echo "$expected") <(echo "$filter") >&2 || true
			status=1
		fi
		if [ "$targets" != "$expected" ]; then
			echo "Windows build targets differ from .github/scripts/unit-tests.sh:" >&2
			diff <(echo "$expected") <(echo "$targets") >&2 || true
			status=1
		fi
		exit "$status"
		;;
	*)
		echo "usage: $0 targets|filter [extra...] | check-windows FILE" >&2
		exit 2
		;;
esac
