//===--- Dialect.cpp - HeavyScheme MLIR Dialect Registration --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations for HeavyScheme's MLIR Dialect
//
//===----------------------------------------------------------------------===//

#include "heavy/Dialect.h"
#include "heavy/HeavyScheme.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir::heavy_mlir;

Dialect::Dialect(mlir::MLIRContext* Ctx) : mlir::Dialect("heavy", Ctx) {
  addTypes<HeavyValue>();
  addAttributes<HeavyValueAttr>();

  addOperations<
#define GET_OP_LIST
#include "heavy/Ops.cpp.inc"
    >();
}

void Dialect::printAttribute(
                        mlir::Attribute Attr,
                        mlir::DialectAsmPrinter& P) const {
  // All attributes are HeavyValueAttr
  heavy::Value* V = Attr.cast<HeavyValueAttr>().getValue();
  heavy::write(P.getStream(), V);
}
void Dialect::printType(mlir::Type Type,
                        mlir::DialectAsmPrinter& P) const {
  // All types are HeavyValue
  P.getStream() << "HeavyValue";
}

heavy::Value* HeavyValueAttr::getValue() const {
  return getImpl()->Val;
}

void ApplyOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                    mlir::Value Operator, ArrayRef<mlir::Value> Operands) {
  ApplyOp::build(B, OpState, B.getType<HeavyValue>(), Operator, Operands);
}

void BindingOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                      mlir::Value Input) {
  BindingOp::build(B, OpState, B.getType<HeavyValue>(), Input);
}

void DefineOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                     mlir::Value Binding) {
  DefineOp::build(B, OpState, B.getType<HeavyValue>(), Binding);
}

void LiteralOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                      heavy::Value* V) {
  // create a HeavyValueAttr from heavy::Value*
  LiteralOp::build(B, OpState, B.getType<HeavyValue>(),
                   HeavyValueAttr::get(B.getContext(), V));
}

using namespace mlir;
#define GET_OP_CLASSES
#include "heavy/Ops.cpp.inc"
