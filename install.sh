#!/bin/bash

for cmd in git python3 pip; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: $cmd is not installed."
        exit 1
    fi
done

# Install Python libraries
pip install -r py/requirements.txt

# Install Spack
cd ~/.local
git clone --depth=2 --branch=releases/v1.2 https://github.com/spack/spack.git
echo "source ~/spack/share/spack/setup-env.sh" >> ~/.bashrc
# spack external find cuda
# spack compiler find
# spack compilers
# spack install hipsycl +cuda ^llvm
# spack load hipsycl
# spack install hipsycl +cuda
# spack add hipsycl +cuda

# Install Julia 1.12.7
cd ~/.local
wget https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.7-linux-x86_64.tar.gz
tar xf julia-1.12.7-linux-x86_64.tar.gz
rm julia-1.12.7-linux-x86_64.tar.gz
echo 'export PATH="$HOME/.local/julia-1.12.7/bin:$PATH"' >> ~/.bashrc 

cd ~
source ~/.bashrc
echo "Installation complete"
