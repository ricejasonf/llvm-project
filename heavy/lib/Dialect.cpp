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

#include "heavy/Context.h"
#include "heavy/Dialect.h"
#include "heavy/Lexer.h"
#include "heavy/Parser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include <string>

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

mlir::Attribute Dialect::parseAttribute(mlir::DialectAsmParser& P,
                                        mlir::Type type) const {
  // Parse HeavyValueAttr.
  std::string string;
  if (P.parseString(&string))
    return {};
  mlir::MLIRContext* MlirCtx = P.getContext();
  auto StringAttr = mlir::StringAttr::get(MlirCtx, string);
  return HeavyValueAttr::get(MlirCtx, StringAttr);
}

mlir::Type Dialect::parseType(mlir::DialectAsmParser& P) const {
  llvm::StringRef Name;
  if (mlir::failed(P.parseKeyword(&Name)))
    return nullptr;
  mlir::Builder B = P.getBuilder();
  if (Name == HeavyValueTy::getMnemonic())
    return B.getType<HeavyValueTy>();
  if (Name == HeavySyntaxTy::getMnemonic())
    return B.getType<HeavySyntaxTy>();
  if (Name == HeavyPairTy::getMnemonic())
    return B.getType<HeavyPairTy>();
  if (Name == HeavyOpGenTy::getMnemonic())
    return B.getType<HeavyOpGenTy>();
  if (Name == HeavyMlirValueTy::getMnemonic())
    return B.getType<HeavyMlirValueTy>();
  llvm_unreachable("unhandled type");
}

void Dialect::printAttribute(
                        mlir::Attribute Attr,
                        mlir::DialectAsmPrinter& P) const {
  // All attributes are HeavyValueAttr
  P.printAttribute(mlir::cast<HeavyValueAttr>(Attr).getExpr());
}

void Dialect::printType(mlir::Type Type,
                        mlir::DialectAsmPrinter& P) const {
  char const* Name;
  if (mlir::isa<HeavyValueTy>(Type)) {
    Name = "value";
  } else if (mlir::isa<HeavyRestTy>(Type)) {
    Name = "rest";
  } else if (mlir::isa<HeavyPairTy>(Type)) {
    Name = "pair";
  } else if (mlir::isa<HeavySyntaxTy>(Type)) {
    Name = "syntax";
  } else if (mlir::isa<HeavyOpGenTy>(Type)) {
    Name = "OpGen&";
  } else if (mlir::isa<HeavyMlirValueTy>(Type)) {
    Name = "MlirValue";
  } else {
    llvm_unreachable("no other types in dialect");
  }

  P.getStream() << Name;
}

void BindingOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                      mlir::Value Input) {
  BindingOp::build(B, OpState, B.getType<HeavyValueTy>(), Input);
}

#if 0
void BuiltinOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                      heavy::Builtin* Builtin) {
  // TODO eventually we need to have a "symbol" to the externally
  //      linked function
  BuiltinOp::build(B, OpState, B.getType<HeavyValueTy>(),
      HeavyValueAttr::get(B.getContext(), Builtin));
}
#endif

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

void ToVectorOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                            mlir::Value input) {
  mlir::Type HeavyValueT = B.getType<HeavyValueTy>();
  ToVectorOp::build(B, OpState, HeavyValueT, input);
}

void VectorOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                            llvm::ArrayRef<mlir::Value> args) {
  mlir::Type HeavyValueT = B.getType<HeavyValueTy>();
  VectorOp::build(B, OpState, HeavyValueT, args);
}

void UndefinedOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState) {
  UndefinedOp::build(B, OpState, B.getType<HeavyValueTy>());
}

void SourceLocOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState) {
  mlir::Type HeavyValueT = B.getType<HeavyValueTy>();
  SourceLocOp::build(B, OpState, HeavyValueT);
}

using namespace mlir;
#define GET_OP_CLASSES
#include "heavy/Ops.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "heavy/Attrs.cpp.inc"

namespace heavy {
// Manually define HeavyValueAttrStorage to cache the parsed
// scheme expression.
struct HeavyValueAttrStorage : public detail::HeavyValueAttrStorage {
  // Cache the result of parsing the expr.
  // This requires visitation by the garbage collector.
  heavy::Value Val = nullptr;

  HeavyValueAttrStorage(StringAttr expr)
    : detail::HeavyValueAttrStorage(std::move(expr))
  { }

  static HeavyValueAttrStorage *construct(mlir::AttributeStorageAllocator &allocator,
                                          KeyTy &&tblgenKey) {
    auto expr = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<HeavyValueAttrStorage>()) HeavyValueAttrStorage(std::move(expr));
  }
};
}  // namespace heavy

heavy::StringAttr HeavyValueAttr::getExpr() const {
  return getImpl()->expr;
}

heavy::Value& HeavyValueAttr::getCachedValue() {
  return getImpl()->Val;
}

heavy::Value HeavyValueAttr::getValue(heavy::Context& C) const {
  heavy::Value& Val = getImpl()->Val;
  if (!Val) {
    llvm::StringRef Expr = getImpl()->expr;
    heavy::Lexer Lexer(Expr);
    heavy::Parser Parser(Lexer, C);
    heavy::ValueResult ValueResult = Parser.Parse();
    Val = ValueResult.isUsable() ? ValueResult.get() :
                                   heavy::Undefined();
  }
  return Val;
}

HeavyValueAttr HeavyValueAttr::get(mlir::MLIRContext* Ctx,
                                   heavy::Value V) {
  // Write the value to a string.
  std::string string;
  llvm::raw_string_ostream stream(string);
  heavy::write(stream, V);
  mlir::StringAttr StringAttr = mlir::StringAttr::get(Ctx, string);
  HeavyValueAttr Attr = get(Ctx, StringAttr);
  Attr.getCachedValue() = V;
  return Attr;
}

