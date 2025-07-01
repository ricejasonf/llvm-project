//===- Nbdl.h - Nbdl binding functions for HeavyScheme ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file declares values and functions for the (nbdl comp) scheme library
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_NBDL_H
#define LLVM_HEAVY_NBDL_H

#define HEAVY_NBDL_LIB                _HEAVYL4SnbdlL4Scomp
#define HEAVY_NBDL_LIB_(NAME)         _HEAVYL4SnbdlL4Scomp ## NAME
// (import (nbdl comp))
#define HEAVY_NBDL_LIB_STR            "_HEAVYL4SnbdlL4Scomp"
#define HEAVY_NBDL_LOAD_MODULE        HEAVY_NBDL_LIB_(_load_module)
#define HEAVY_NBDL_INIT               HEAVY_NBDL_LIB_(_init)

namespace heavy {
class Context;
}

extern "C" {
// initialize the module for run-time independent of the compiler
void HEAVY_NBDL_LOAD_MODULE(heavy::Context& Context);
void HEAVY_NBDL_INIT(heavy::Context& Context);
}

#endif  // LLVM_HEAVY_NBDL_H
