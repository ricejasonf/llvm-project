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
  addTypes<HeavyLambda, HeavyRest, HeavyValue>();
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
                    mlir::Value Operator,
                    llvm::ArrayRef<mlir::Value> Operands,
                    bool IsTailPos) {
  ApplyOp::build(B, OpState, B.getType<HeavyValue>(), Operator, Operands,
                 B.getBoolAttr(IsTailPos));
}

void BindingOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                      mlir::Value Input, bool IsTopLevel) {
  BindingOp::build(B, OpState, B.getType<HeavyValue>(), Input, IsTopLevel);
}

void BuiltinOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                      heavy::Builtin* Builtin) {
  // TODO eventually we need to have a "symbol" to the externally
  //      linked function
  BuiltinOp::build(B, OpState, B.getType<HeavyValue>(),
      HeavyValueAttr::get(B.getContext(), Builtin));
}

void IfOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                 mlir::Value Input) {
  IfOp::build(B, OpState, B.getType<HeavyValue>(), Input);
}

void LambdaOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                     llvm::StringRef Name,
                     uint32_t Arity, bool HasRestParam,
                     llvm::ArrayRef<mlir::Value> Captures) {
  mlir::Type HeavyValueTy = B.getType<HeavyValue>();

  // create the FunctionType
  llvm::SmallVector<mlir::Type, 16> Types{};
  if (Arity > 0) {
    for (unsigned i = 0; i < Arity - 1; i++) {
      Types.push_back(HeavyValueTy);
    }
    mlir::Type LastParamTy = HasRestParam ?
                                HeavyValueTy :
                                HeavyValueTy;
    Types.push_back(LastParamTy);
  }

  FunctionType FuncTy = B.getFunctionType(Types, HeavyValueTy);
  OpState.addRegion();
  LambdaOp::build(B, OpState, HeavyValueTy,
                  B.getStringAttr(Name),
                  TypeAttr::get(FuncTy),
                  B.getBoolAttr(HasRestParam),
                  Captures);
}

void LiteralOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                      heavy::Value* V) {
  // create a HeavyValueAttr from heavy::Value*
  LiteralOp::build(B, OpState, B.getType<HeavyValue>(),
                   HeavyValueAttr::get(B.getContext(), V));
}

void SetOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                     mlir::Value Binding, mlir::Value Input) {
  SetOp::build(B, OpState, B.getType<HeavyValue>(), Binding, Input);
}

void UndefinedOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState) {
  UndefinedOp::build(B, OpState, B.getType<HeavyValue>());
}

using namespace mlir;
#define GET_OP_CLASSES
#include "heavy/Ops.cpp.inc"
