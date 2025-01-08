//===- Dialect.h - Classes for representing declarations --------*- C++ -*-===//
//
// Released under the Apache License v2.0
// See https://llvm.org/LICENSE.txt
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Declares the mlir Dialect for Nbdl.
//
//===----------------------------------------------------------------------===//

#ifndef NBDL_DIALECT_H
#define NBDL_DIALECT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the generated header files

#include "nbdl_gen/NbdlDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "nbdl_gen/NbdlTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "nbdl_gen/NbdlAttrs.h.inc"

#define GET_OP_CLASSES
#include "nbdl_gen/NbdlOps.h.inc"

#endif
