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
#include "heavy/Context.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace heavy;

Dialect::Dialect(mlir::MLIRContext* Ctx)
  : mlir::Dialect("heavy", Ctx, mlir::TypeID::get<Dialect>()) {
  addTypes<HeavyValueTy>();
  addAttributes<HeavyValueAttr>();

  addTypes<HeavyRestTy>();
  addTypes<HeavyPairTy>();

  addTypes<HeavySyntaxTy>();
  addTypes<HeavyOpGenTy>();
  addTypes<HeavyMlirValueTy>();

  addOperations<
#define GET_OP_LIST
#include "heavy/Ops.cpp.inc"
    >();
}

void Dialect::printAttribute(
                        mlir::Attribute Attr,
                        mlir::DialectAsmPrinter& P) const {
  // All attributes are HeavyValueAttr
  heavy::Value V = Attr.cast<HeavyValueAttr>().getValue();
  heavy::write(P.getStream(), V);
}
void Dialect::printType(mlir::Type Type,
                        mlir::DialectAsmPrinter& P) const {
  char const* Name;
  if (Type.isa<HeavyValueTy>()) {
    Name = "value";
  } else if (Type.isa<HeavyRestTy>()) {
    Name = "rest";
  } else if (Type.isa<HeavyPairTy>()) {
    Name = "pair";
  } else if (Type.isa<HeavySyntaxTy>()) {
    Name = "syntax";
  } else if (Type.isa<HeavyOpGenTy>()) {
    Name = "OpGen&";
  } else if (Type.isa<HeavyMlirValueTy>()) {
    Name = "MlirValue";
  } else {
    llvm_unreachable("no other types in dialect");
  }

  P.getStream() << Name;
}

heavy::Value HeavyValueAttr::getValue() const {
  return getImpl()->Val;
}

void BindingOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                      mlir::Value Input) {
  BindingOp::build(B, OpState, B.getType<HeavyValueTy>(), Input);
}

void BuiltinOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                      heavy::Builtin* Builtin) {
  // TODO eventually we need to have a "symbol" to the externally
  //      linked function
  BuiltinOp::build(B, OpState, B.getType<HeavyValueTy>(),
      HeavyValueAttr::get(B.getContext(), Builtin));
}

void ConsOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                   mlir::Value X, mlir::Value Y) {
  mlir::Type HeavyValueT = B.getType<HeavyValueTy>();
  // TODO return type should be Pair
  ConsOp::build(B, OpState, HeavyValueT, X, Y);
}

void GlobalOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                     llvm::StringRef SymName) {
  mlir::Type HeavyValueT = B.getType<HeavyValueTy>();
  GlobalOp::build(B, OpState, HeavyValueT, SymName);
}

void IfOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                 mlir::Value Input) {
  IfOp::build(B, OpState, B.getType<HeavyValueTy>(), Input);
}

void LambdaOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                     llvm::StringRef Name,
                     llvm::ArrayRef<mlir::Value> Captures) {
  mlir::Type HeavyValueT = B.getType<HeavyValueTy>();

  LambdaOp::build(B, OpState, HeavyValueT,
                  Name, Captures);
}

void LiteralOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                      heavy::Value V) {
  // create a HeavyValueAttr from heavy::Value
  LiteralOp::build(B, OpState, B.getType<HeavyValueTy>(),
                   HeavyValueAttr::get(B.getContext(), V));
}

void LoadClosureOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                          mlir::Value Closure, uint32_t Index) {
  mlir::Type HeavyValueT = B.getType<HeavyValueTy>();
  LoadClosureOp::build(B, OpState, HeavyValueT, Closure,
                       B.getUI32IntegerAttr(Index));
}

void LoadGlobalOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                         llvm::StringRef SymName) {
  mlir::Type HeavyValueT = B.getType<HeavyValueTy>();
  LoadGlobalOp::build(B, OpState, HeavyValueT, SymName);
}

void MatchOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                    heavy::Value val, mlir::Value input) {
  assert((!isa<Pair, Vector>(val)) && "expected non-structural constant");
  auto ValAttr = HeavyValueAttr::get(B.getContext(), val);
  MatchOp::build(B, OpState, ValAttr, input);
}

void MatchPairOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                        mlir::Value input) {
  mlir::Type HeavyValueT = B.getType<HeavyValueTy>();
  MatchPairOp::build(B, OpState,
                     /*car=*/HeavyValueT,
                     /*cdr=*/HeavyValueT,
                     input);
}

void RenameOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                     mlir::Value Value) {
  // How awful is this?
  void* OpaquePtr = Value.getAsOpaquePointer();
  uint64_t OpaqueValue = reinterpret_cast<uint64_t>(OpaquePtr);
  bool IsSigned = false;
  mlir::IntegerType UI64 = B.getIntegerType(64, IsSigned);
  auto APVal = llvm::APInt(64, OpaqueValue, IsSigned);
  auto Attr = mlir::IntegerAttr::get(UI64, APVal);
  mlir::Type HeavyValueT = B.getType<HeavyValueTy>();
  RenameOp::build(B, OpState, HeavyValueT, Attr);
}

void SetOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                     mlir::Value Binding, mlir::Value Input) {
  SetOp::build(B, OpState, B.getType<HeavyValueTy>(), Binding, Input);
}

void SpliceOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                   mlir::Value X, mlir::Value Y) {
  mlir::Type HeavyValueT = B.getType<HeavyValueTy>();
  SpliceOp::build(B, OpState, HeavyValueT, X, Y);
}

void SyntaxClosureOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                            mlir::Value input) {
  mlir::Type HeavyValueT = B.getType<HeavyValueTy>();
  SyntaxClosureOp::build(B, OpState, HeavyValueT, input);
}

void SyntaxOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState) {
  mlir::Type HeavySyntaxT = B.getType<HeavySyntaxTy>();
  SyntaxOp::build(B, OpState, HeavySyntaxT);
}

void UndefinedOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState) {
  UndefinedOp::build(B, OpState, B.getType<HeavyValueTy>());
}

using namespace mlir;
#define GET_OP_CLASSES
#include "heavy/Ops.cpp.inc"
