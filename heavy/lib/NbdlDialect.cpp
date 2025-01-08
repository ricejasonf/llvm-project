//===--- NbdlDialect.cpp - Nbdl Dialect --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations for Nbdl's MLIR Dialect
//
//===----------------------------------------------------------------------===//

#include <nbdl_gen/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <llvm/ADT/TypeSwitch.h>

// Include generated source files.

#include "nbdl_gen/NbdlDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "nbdl_gen/NbdlTypes.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "nbdl_gen/NbdlAttrs.cpp.inc"
#define GET_OP_CLASSES
#include "nbdl_gen/NbdlOps.cpp.inc"

void nbdl_gen::NbdlDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "nbdl_gen/NbdlTypes.cpp.inc"
    >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "nbdl_gen/NbdlAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "nbdl_gen/NbdlOps.cpp.inc"
      >();

}
