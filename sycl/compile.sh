#!/bin/bash

acpp Solver.cpp SyclMethods.cpp -O2 -I. -I$(spack location -i hipsycl)/include/AdaptiveCpp/sycl -lnvidia-ml -o sycl
