# Building ur_rtde from Source - Complete Guide

This guide walks you through building and installing `ur_rtde` from source on macOS using conda.

## Prerequisites

- macOS (tested on macOS with ARM64/Apple Silicon)
- Conda/Miniconda installed
- Xcode Command Line Tools installed (`xcode-select --install`)

## Step-by-Step Instructions

### 1. Activate Your Conda Environment

```bash
conda activate ur-test
```

If you don't have an environment yet, create one:
```bash
conda create --name ur-test python=3.10
conda activate ur-test
```

### 2. Install Build Dependencies via Conda

Install CMake, Boost, Git, and C++ compiler toolchain:

```bash
conda install -y -c conda-forge cmake boost-cpp git cxx-compiler
```

This installs:
- `cmake` - Build system
- `boost-cpp` - C++ Boost libraries (required by ur_rtde)
- `git` - Version control (for cloning)
- `cxx-compiler` - C++ compiler toolchain with standard library headers

### 3. Install Python Build Dependencies

```bash
pip install setuptools_scm pybind11
```

These are required by the build system:
- `setuptools_scm` - Version management
- `pybind11` - Python-C++ bindings framework

### 4. Clone the Repository

```bash
cd /path/to/your/workspace
git clone https://gitlab.com/sdurobotics/ur_rtde.git ur_rtde_source
cd ur_rtde_source
```

### 5. Initialize Git Submodules

The repository uses submodules (specifically pybind11):

```bash
git submodule update --init --recursive
```

### 6. Fix setup.py for macOS Compatibility

The default `setup.py` needs modifications to work with newer CMake versions and macOS SDK paths. Edit `setup.py`:

**Location**: Around line 77-83 in `setup.py`

**Find this section:**
```python
cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

env = os.environ.copy()
env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                      self.distribution.get_version())
```

**Replace with:**
```python
cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
# Fix CMake policy version issue with newer CMake versions
cmake_args += ['-DCMAKE_POLICY_VERSION_MINIMUM=3.5']

# Add C++ standard library include paths for macOS
if platform.system() == "Darwin":
    import subprocess
    try:
        sdk_path = subprocess.check_output(['xcrun', '--show-sdk-path']).decode().strip()
        cxx_flags = f'-isysroot {sdk_path} -I{sdk_path}/usr/include/c++/v1'
        cmake_args += [f'-DCMAKE_CXX_FLAGS={cxx_flags}']
    except:
        pass

env = os.environ.copy()
env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                      self.distribution.get_version())
```

**What these changes do:**
1. **CMake Policy Fix**: `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` tells CMake to use policies compatible with version 3.5+, which is needed for newer CMake versions (4.0+) that removed compatibility with older policy versions.

2. **macOS SDK Paths**: On macOS, the C++ standard library headers are in a specific location within the SDK. The code:
   - Gets the SDK path using `xcrun --show-sdk-path`
   - Adds `-isysroot` flag to tell the compiler where the SDK is
   - Adds the include path for C++ standard library headers (`/usr/include/c++/v1`)

### 7. Build and Install

From the `ur_rtde_source` directory:

```bash
pip install . --no-build-isolation
```

**Why `--no-build-isolation`?**
- This ensures the build uses your conda environment's packages (CMake, Boost, etc.) rather than trying to install them in an isolated build environment.

### 8. Verify Installation

Test that the package installed correctly:

```bash
python -c "import rtde_control; import rtde_receive; print('ur_rtde installed successfully!')"
```

You should see: `ur_rtde installed successfully!`

## Troubleshooting

### Issue: "fatal error: 'string' file not found"

**Cause**: The compiler can't find C++ standard library headers.

**Solution**: Make sure you:
1. Installed `cxx-compiler` via conda
2. Applied the macOS SDK path fix in `setup.py`
3. Have Xcode Command Line Tools installed (`xcode-select --install`)

### Issue: "CMake policy version" errors

**Cause**: Newer CMake versions (4.0+) removed compatibility with older policy versions.

**Solution**: The `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` flag in setup.py fixes this.

### Issue: "Could not find Boost"

**Cause**: Boost libraries not found.

**Solution**: Make sure you installed `boost-cpp` via conda:
```bash
conda install -y -c conda-forge boost-cpp
```

### Issue: Build fails with "git submodule" errors

**Cause**: Submodules not initialized.

**Solution**: Run:
```bash
git submodule update --init --recursive
```

## Alternative: Install Pre-built Package

If building from source is problematic, you can install from PyPI (though it may not have pre-built wheels for your platform):

```bash
pip install ur-rtde
```

However, building from source gives you:
- Latest version from GitLab
- Ability to modify the code
- Better compatibility with your specific system

## Summary of Key Points

1. **Use conda for system dependencies** (CMake, Boost, compiler)
2. **Use pip for Python dependencies** (setuptools_scm, pybind11)
3. **Fix setup.py** for macOS compatibility (CMake policy + SDK paths)
4. **Use `--no-build-isolation`** to ensure conda packages are used
5. **Initialize submodules** before building

## File Locations

After successful installation:
- **Source code**: `/path/to/ur_rtde_source/`
- **Installed package**: In your conda environment's `site-packages` directory
- **Modified setup.py**: Keep your changes if you need to rebuild

## Rebuilding After Changes

If you modify the source code and want to rebuild:

```bash
cd ur_rtde_source
pip install . --no-build-isolation --force-reinstall
```

The `--force-reinstall` flag ensures a clean rebuild.

