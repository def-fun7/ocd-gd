#!/usr/bin/env bash
set -euo pipefail

echo "========================================================="
echo "🪐 AGAMA Cross-Platform Prerequisites & Setup"
echo "========================================================="

# 1. Detect Operating System Environment
OS_TYPE="$(uname -s)"
case "${OS_TYPE}" in
    Linux*)     OS='Linux';;
    Darwin*)    OS='macOS';;
    CYGWIN*|MINGW*|MSYS*) OS='Windows_Bash';;
    *)          OS='Unknown';;
esac

echo "Detected OS: ${OS}"

# ---------------------------------------------------------
# Step 1: Handle OS-Specific Toolchains & Include Paths
# ---------------------------------------------------------

if [ "$OS" = "macOS" ]; then
  echo "🍺 Setting up macOS environment via Homebrew..."

  if ! command -v brew &> /dev/null; then
    echo "❌ Error: Homebrew is required on macOS. Install from https://brew.sh/"
    exit 1
  fi

  # Auto-install libraries if missing
  brew install gsl openblas eigen gmp make gcc pkg-config || true

  BREW_PREFIX="$(brew --prefix)"

  # Set C/C++ Header search paths explicitly for Eigen, GSL, and OpenBLAS
  export CFLAGS="-I${BREW_PREFIX}/include -I${BREW_PREFIX}/include/eigen3 ${CFLAGS:-}"
  export CPPFLAGS="-I${BREW_PREFIX}/include -I${BREW_PREFIX}/include/eigen3 ${CPPFLAGS:-}"
  export CPLUS_INCLUDE_PATH="${BREW_PREFIX}/include:${BREW_PREFIX}/include/eigen3:${CPLUS_INCLUDE_PATH:-}"
  export C_INCLUDE_PATH="${BREW_PREFIX}/include:${C_INCLUDE_PATH:-}"
  export LDFLAGS="-L${BREW_PREFIX}/lib ${LDFLAGS:-}"
  export PKG_CONFIG_PATH="${BREW_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

elif [ "$OS" = "Windows_Bash" ]; then
  echo "🪟 Setting up Windows environment (Git Bash / MSYS2)..."

  # Windows requires MSVC or MinGW toolchain
  if command -v winget &> /dev/null; then
    echo "📦 Checking Windows package manager (winget)..."
    # Note: On Windows, standard C dependencies are typically compiled with Visual Studio or MSYS2
    winget install --id Kitware.CMake -e --accept-source-agreements || true
  fi

  # Configure environment variables for MSVC or MinGW builds
  export UV_CONFIG_SETTING_BUILD_OPTION="--yes"

elif [ "$OS" = "Linux" ]; then
  echo "🐧 Setting up Linux prerequisites..."

  if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential gcc g++ make libgsl-dev libopenblas-dev libeigen-dev libgmp-dev pkg-config
  elif command -v dnf &> /dev/null; then
    sudo dnf install -y gcc gcc-c++ make gsl-devel openblas-devel eigen3-devel gmp-devel pkgconf-pkg-config
  elif command -v pacman &> /dev/null; then
    sudo pacman -S --needed --noconfirm base-devel gsl openblas eigen gmp pkgconf
  fi
fi

# ---------------------------------------------------------
# Step 2: Validate Compiler & Make Availability
# ---------------------------------------------------------
echo "🔍 Validating build tools..."

if command -v make &> /dev/null; then
  MAKE_CMD="make"
elif command -v gmake &> /dev/null; then
  MAKE_CMD="gmake"
else
  echo "⚠️ Warning: Neither 'make' nor 'gmake' was found in PATH."
fi

# ---------------------------------------------------------
# Step 3: Run Non-Interactive `uv sync`
# ---------------------------------------------------------
echo "📦 Syncing environment & compiling AGAMA..."

# Pass non-interactive option to setup.py via uv
uv sync --config-settings="--build-option=--yes" --all-groups

echo "========================================================="
echo "✅ Environment setup complete!"
echo "========================================================="