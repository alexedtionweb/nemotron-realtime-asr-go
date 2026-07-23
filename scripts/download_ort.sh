#!/usr/bin/env bash
# Download and install the ONNX Runtime shared library used by nemotron-realtime-asr-go.
#
# Usage:
#   ./scripts/download_ort.sh              # installs to /usr/local/lib (default)
#   ./scripts/download_ort.sh ./libs       # installs to a local directory (no sudo)
#
# After a local install, point the server at the library:
#   go run examples/server.go -ort-lib ./libs/libonnxruntime.so ...
set -euo pipefail

# Must match the C API version required by yalue/onnxruntime_go (ORT_API_VERSION
# in onnxruntime_c_api.h). v1.31.0 of that wrapper requires API version 26,
# which is shipped by ORT 1.26.0.
ORT_VERSION="1.26.0"
INSTALL_DIR="${1:-/usr/local/lib}"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}" in
  Linux)
    case "${ARCH}" in
      x86_64)  PLATFORM="linux-x64"   ;;
      aarch64) PLATFORM="linux-aarch64" ;;
      *) echo "Unsupported Linux arch: ${ARCH}"; exit 1 ;;
    esac
    LIB_NAME="libonnxruntime.so"
    ;;
  Darwin)
    case "${ARCH}" in
      x86_64) PLATFORM="osx-x86_64"  ;;
      arm64)  PLATFORM="osx-arm64"   ;;
      *) echo "Unsupported macOS arch: ${ARCH}"; exit 1 ;;
    esac
    LIB_NAME="libonnxruntime.dylib"
    ;;
  *)
    echo "Unsupported OS: ${OS}"
    exit 1
    ;;
esac

TARBALL="onnxruntime-${PLATFORM}-${ORT_VERSION}.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${TARBALL}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "Downloading ORT ${ORT_VERSION} for ${PLATFORM}..."
curl -fsSL --progress-bar "${URL}" -o "${TMP}/${TARBALL}"

echo "Extracting..."
tar -xzf "${TMP}/${TARBALL}" -C "${TMP}"

# The extracted layout is: onnxruntime-<platform>-<ver>/lib/
SRC_LIB_DIR="$(find "${TMP}" -type d -name lib | head -1)"
if [[ -z "${SRC_LIB_DIR}" ]] || ! ls "${SRC_LIB_DIR}/${LIB_NAME}"* >/dev/null 2>&1; then
  echo "Could not locate ${LIB_NAME} inside the archive"
  exit 1
fi

# Decide whether we need sudo for the destination
mkdir -p "${INSTALL_DIR}"
if [[ "${INSTALL_DIR}" == /usr/* || "${INSTALL_DIR}" == /opt/* ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

# Copy the lib directory contents preserving symlinks (-P).
# Plain 'cp' dereferences symlinks, turning the versioned .so.1 symlink into
# a regular file, which makes ldconfig emit "is not a symbolic link" warnings.
${SUDO} cp -P "${SRC_LIB_DIR}"/${LIB_NAME}* "${INSTALL_DIR}/"

# Refresh the dynamic linker cache on Linux system installs
if [[ "${OS}" == "Linux" && -n "${SUDO}" ]]; then
  sudo ldconfig
fi

echo ""
echo "Installed to: ${INSTALL_DIR}/${LIB_NAME}"
echo ""
echo "Run the example server:"
if [[ "${INSTALL_DIR}" == /usr/local/lib ]]; then
  echo "  go run examples/server.go -model-dir /path/to/model"
else
  echo "  go run examples/server.go -ort-lib ${INSTALL_DIR}/${LIB_NAME} -model-dir /path/to/model"
fi
