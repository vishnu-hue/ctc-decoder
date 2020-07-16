# ctc-decoder
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
