#!/bin/bash

for cmd in git python3 pip; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: $cmd is not installed."
        exit 1
    fi
done

# Install Python libraries
pip install -r py/requirements.txt

# Install Spack 1.2
cd ~/.local
git clone --depth=2 --branch=releases/v1.2 https://github.com/spack/spack.git
echo "source ~/spack/share/spack/setup-env.sh" >> ~/.bashrc

spack env create sycl
spack env activate sycl

spack add gcc@11.4.0
spack add gcc@12.3.0
spack add llvm@18.1.8 +cuda targets=x86,nvptx,amdgpu ~lldb
spack add cuda@12.8.0
spack add hipsycl@24.10.0

spack concretize
spack install -j$(nproc)

spack compiler find $(spack location -i gcc@11.4.0)
spack compiler find $(spack location -i gcc@12.3.0)

# Install Julia 1.12.7
cd ~/.local
wget https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.7-linux-x86_64.tar.gz
tar xf julia-1.12.7-linux-x86_64.tar.gz
rm julia-1.12.7-linux-x86_64.tar.gz
echo 'export PATH="$HOME/.local/julia-1.12.7/bin:$PATH"' >> ~/.bashrc 

cd ~
source ~/.bashrc
echo "Installation complete"
