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
#include "llvm/ADT/TypeSwitch.h"
#include <string>

using namespace heavy;

#include "heavy/Dialect/HeavyDialect.cpp.inc"

void HeavyDialect::initialize() {
  //addTypes<HeavyOpGenType>();
  //addTypes<HeavyMlirValueType>();

  addTypes<
#define GET_TYPEDEF_LIST
#include "heavy/Dialect/HeavyTypes.cpp.inc"
    >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "heavy/Dialect/HeavyAttrs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "heavy/Dialect/HeavyOps.cpp.inc"
    >();
}

mlir::Attribute HeavyDialect::parseAttribute(mlir::DialectAsmParser& P,
                                             mlir::Type type) const {
  // Parse HeavyValueAttr.
  std::string string;
  if (P.parseString(&string))
    return {};
  mlir::MLIRContext* MlirCtx = P.getContext();
  auto StringAttr = mlir::StringAttr::get(MlirCtx, string);
  return HeavyValueAttr::get(MlirCtx, StringAttr);
}

void HeavyDialect::printAttribute(
                        mlir::Attribute Attr,
                        mlir::DialectAsmPrinter& P) const {
  // All attributes are HeavyValueAttr
  P.printAttribute(mlir::cast<HeavyValueAttr>(Attr).getExpr());
}

void BindingOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                      mlir::Value Input) {
  BindingOp::build(B, OpState, B.getType<HeavyBindingType>(), Input);
}

void ConsOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                   mlir::Value X, mlir::Value Y) {
  mlir::Type ResultType = B.getType<HeavyPairType>();
  ConsOp::build(B, OpState, ResultType, X, Y);
}

void IfOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                 mlir::Value Input) {
  IfOp::build(B, OpState, B.getType<HeavyValueType>(), Input);
}

void LambdaOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                     llvm::StringRef Name,
                     llvm::ArrayRef<mlir::Value> Captures) {
  mlir::Type ResultType = B.getType<HeavyProcedureType>();
  LambdaOp::build(B, OpState, ResultType, Name, Captures);
}

void LiteralOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                      heavy::Value V) {
  assert(!isa<SyntaxClosure>(V));
  // create a HeavyValueAttr from heavy::Value
  LiteralOp::build(B, OpState, B.getType<HeavyValueType>(),
                   HeavyValueAttr::get(B.getContext(), V));
}

void LoadRefOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                          mlir::Value ValueRefs, uint32_t Index) {
  mlir::Type ResultType = B.getType<HeavyValueType>();
  LoadRefOp::build(B, OpState, ResultType, ValueRefs,
                       B.getUI32IntegerAttr(Index));
}

void LoadGlobalOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                         llvm::StringRef SymName) {
  mlir::Type ResultType = B.getType<HeavyUnknownType>();
  LoadGlobalOp::build(B, OpState, ResultType, SymName);
}

void LoadModuleOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                         llvm::StringRef SymName) {
  LoadModuleOp::build(B, OpState, {}, SymName);
}

void MatchOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                    heavy::Value val, mlir::Value input) {
  assert((!isa<Pair, Vector>(val)) && "expected non-structural constant");
  auto ValAttr = HeavyValueAttr::get(B.getContext(), val);
  MatchOp::build(B, OpState, ValAttr, input);
}

void MatchPairOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                        mlir::Value input) {
  mlir::Type ResultType = B.getType<HeavyValueType>();
  MatchPairOp::build(B, OpState,
                     /*car=*/ResultType,
                     /*cdr=*/ResultType,
                     input);
}

void MatchTailOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                        uint32_t Length, mlir::Value Input) {
  mlir::Type ResultType = B.getType<HeavyValueType>();
  MatchTailOp::build(B, OpState, ResultType,
                     Length, Input);
}

void MatchArgsOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                        mlir::FunctionType FT, mlir::Value Input) {
  assert((!FT.getInputs().empty() &&
          isa<HeavyContextType>(FT.getInputs().front())) &&
    "expecting heavy-scheme function type");
  // Drop the context argument.
  auto ResultTypes = FT.getInputs().drop_front();
  MatchArgsOp::build(B, OpState, ResultTypes, Input);
}

void MatchVectorOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                          uint32_t Head, uint32_t Tail, mlir::Value Input) {
  llvm_unreachable("FIXME This is all very untested.");
  assert(Tail >= Head);
  unsigned NumResults = Tail - Head;
  mlir::Type ResultType = B.getType<HeavyValueType>();
  llvm::SmallVector<mlir::Type, 4> ResultTypes(NumResults, ResultType);
  MatchVectorOp::build(B, OpState, ResultTypes, Head, Tail, Input);
}

void SubpatternOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                        mlir::Value Input, mlir::Value Tail,
                        std::unique_ptr<mlir::Region>&& Body,
                        unsigned NumPacks) {
  mlir::Type ResultType = B.getType<HeavyValueType>();
  llvm::SmallVector<mlir::Type, 4> ResultTypes(NumPacks, ResultType);
  OpState.addOperands({Input, Tail});
  OpState.addRegion(std::move(Body));
  OpState.addTypes(std::move(ResultTypes));
}

void ExpandPacksOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                        mlir::Value Cdr, mlir::ValueRange Packs,
                        std::unique_ptr<mlir::Region>&& Body) {
  mlir::Type ResultType = B.getType<HeavyValueType>();
  OpState.addOperands(Cdr);
  OpState.addOperands(Packs);
  OpState.addRegion(std::move(Body));
  OpState.addTypes(ResultType);
}

void RenameOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                     llvm::StringRef Id, mlir::Value Capture) {
  mlir::Type ResultType = B.getType<HeavyBindingType>();
  RenameOp::build(B, OpState, ResultType, Id, Capture);
}

void RenameGlobalOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                     llvm::StringRef Id, llvm::StringRef Sym) {
  mlir::Type ResultType = B.getType<HeavyUnknownType>();
  RenameGlobalOp::build(B, OpState, ResultType, Id, Sym);
}

void SetOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                     mlir::Value Binding, mlir::Value Input) {
  SetOp::build(B, OpState, B.getType<HeavyValueType>(), Binding, Input);
}

void SpliceOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                   mlir::Value X, mlir::Value Y) {
  mlir::Type ResultType = B.getType<HeavyValueType>();
  SpliceOp::build(B, OpState, ResultType, X, Y);
}

void SyntaxClosureOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                            mlir::Value SourceVal, mlir::Value Input,
                            mlir::Value Env) {
  mlir::Type ResultType = B.getType<HeavyValueType>();
  SyntaxClosureOp::build(B, OpState, ResultType, SourceVal, Input, Env);
}

void SyntaxOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                    llvm::StringRef MangledName) {
  mlir::Type HeavySyntaxT = B.getType<HeavySyntaxType>();
  SyntaxOp::build(B, OpState, HeavySyntaxT, MangledName);
}

void ToVectorOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                            mlir::Value input) {
  mlir::Type ResultType = B.getType<HeavyVectorType>();
  ToVectorOp::build(B, OpState, ResultType, input);
}

void VectorOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                            llvm::ArrayRef<mlir::Value> args) {
  mlir::Type ResultType = B.getType<HeavyVectorType>();
  VectorOp::build(B, OpState, ResultType, args);
}

void RenameEnvOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                    mlir::Value Env, llvm::ArrayRef<mlir::Value> Bindings) {
  mlir::Type ResultType = B.getType<HeavyValueType>();
  RenameEnvOp::build(B, OpState, ResultType, Env, Bindings);
}

void UndefinedOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState) {
  UndefinedOp::build(B, OpState, B.getType<HeavyUndefinedType>());
}

void SourceLocOp::build(mlir::OpBuilder& B, mlir::OperationState& OpState,
                        mlir::Value Arg) {
  mlir::Type ResultType = B.getType<HeavyValueType>();
  SourceLocOp::build(B, OpState, ResultType, Arg);
}

using namespace mlir;
#define GET_OP_CLASSES
#include "heavy/Dialect/HeavyOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "heavy/Dialect/HeavyAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "heavy/Dialect/HeavyTypes.cpp.inc"

namespace heavy::detail {
// Manually define HeavyValueAttrStorage to cache the parsed
// scheme expression.
struct HeavyValueAttrStorage : mlir::AttributeStorage {
  using KeyTy = std::tuple<StringAttr>;
  HeavyValueAttrStorage(StringAttr expr) : expr(std::move(expr)) {}

  KeyTy getAsKey() const {
    return KeyTy(expr);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (expr == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static HeavyValueAttrStorage *construct(
      mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto expr = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<HeavyValueAttrStorage>())
               HeavyValueAttrStorage(std::move(expr));
  }

  StringAttr expr;

  // Cache the result of parsing the expr.
  // This requires visitation by the garbage collector.
  heavy::Value Val = nullptr;
};
}  // namespace heavy::detail

mlir::StringAttr HeavyValueAttr::getExpr() const {
  return getImpl()->expr;
}
heavy::Value& HeavyValueAttr::getCachedValue() {
  return getImpl()->Val;
}

heavy::Value HeavyValueAttr::getValue(heavy::Context& C) const {
  heavy::Value& Val = getImpl()->Val;
  if (!Val)
    Val = C.ParseLiteral(getImpl()->expr);
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

