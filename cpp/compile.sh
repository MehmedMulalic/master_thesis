#!/bin/bash

acpp solver.cpp sycl_methods.cpp -O2 -I. -I$(spack location -i hipsycl)/include/AdaptiveCpp/sycl -lnvidia-ml -o sycl
