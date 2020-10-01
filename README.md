# ctc-decoder
requirements:

kenlm

to build kenlm :

pip install packaging

pip install torch

pip install numpy

apt-get install zlibc zlib1g zlib1g-dev libeigen3-dev bzip2 liblzma-dev libboost-all-dev

> wget http://kheafield.com/code/kenlm.tar.gz

> tar -xvzf kenlm.tar.gz

add set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") to the CmakeList.txt in the kenlm root folder

> cd kenlm

> mkdir -p build && cd build

> cmake ..

> make -j 4

After installation, do not forget to export the PATH, as :

> export KENLM_ROOT_DIR=/home/kenlm2

Compiling

Use cmake,

    mkdir -p build

    cd build

    cmake ..

    make -j 4

To install:

    cd bindings/python

    pip install -e .

To run examples:

    python examples/decoder_example.py ../../test
