#!/usr/bin/env bash
# Source before running histopathology notebooks/scripts on macOS:
#   source scripts/env.sh
#
# Homebrew installs libopenslide to /opt/homebrew/lib; Python's openslide-python
# needs this on the dynamic linker search path.

if [[ -d /opt/homebrew/lib ]]; then
  export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"
fi
