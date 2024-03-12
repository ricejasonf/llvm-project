//===--- Heap.h - Classes for representing declarations ----*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file declares heavy::Heap.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_HEAP_H
#define LLVM_HEAVY_HEAP_H

#include "heavy/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Allocator.h"

namespace heavy {
using llvm::ArrayRef;
using llvm::StringRef;
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;
using AllocatorTy = llvm::BumpPtrAllocator;

// Implement CopyCollector in Heap.cpp.
class CopyCollector;

class Heap {
  AllocatorTy TrashHeap;  
  std::unique_ptr<CopyCollector> Collector;
  llvm::AllocatorBase<AllocatorTy>& getAllocator() { return TrashHeap; }

public:
  Heap();
  ~Heap();

  // Factory functions should only be concerned with
  // allocating the objects.
  Undefined   CreateUndefined() { return {}; }
  Bool        CreateBool(bool V) { return V; }
  Char        CreateChar(uint32_t V) { return V; }
  Int         CreateInt(int32_t x) { return Int(x); }
  Empty       CreateEmpty() { return {}; }
  BigInt*     CreateBigInt(llvm::APInt V) {
    return new (TrashHeap) BigInt(V);
  }
  BigInt*     CreateBigInt(int64_t X) {
    llvm::APInt Val(64, X, /*IsSigned=*/true);
    return CreateBigInt(Val);
  }
  Float*      CreateFloat(double Double) {
    return CreateFloat(llvm::APFloat(Double));
  }
  Float*      CreateFloat(llvm::APFloat V);
  Pair*       CreatePair(Value V1, Value V2) {
    return new (TrashHeap) Pair(V1, V2);
  }
  Pair*       CreatePair(Value V1) {
    return new (TrashHeap) Pair(V1, CreateEmpty());
  }
  PairWithSource* CreatePairWithSource(Value V1, Value V2,
                                       SourceLocation Loc) {
    return new (TrashHeap) PairWithSource(V1, V2, Loc);
  }
  Value       CreateList(llvm::ArrayRef<Value> Vs);
  String*     CreateString(unsigned Length, char InitChar);
  String*     CreateString(StringRef S);
  String*     CreateString(StringRef S1, StringRef S2);
  String*     CreateString(StringRef, StringRef, StringRef);
  String*     CreateString(StringRef, StringRef, StringRef, StringRef);
  Vector*     CreateVector(ArrayRef<Value> Xs);
  Vector*     CreateVector(unsigned N);
  ByteVector* CreateByteVector(llvm::ArrayRef<Value> Xs);
  EnvFrame*   CreateEnvFrame(llvm::ArrayRef<Symbol*> Names);

  template <typename F>
  Lambda* CreateLambda(F Fn, llvm::ArrayRef<heavy::Value> Captures) {
    OpaqueFn FnData = createOpaqueFn(Fn);
    void* Mem = Lambda::allocate(getAllocator(), FnData, Captures);
    Lambda* New = new (Mem) Lambda(FnData, Captures);

    return New;
  }

  template <typename F>
  Lambda* CreateLambda(F Fn) {
    return CreateLambda(Fn, {});
  }

  template <typename F>
  Syntax* CreateSyntax(F Fn) {
    auto FnData = createOpaqueFn(Fn);
    void* Mem = Syntax::allocate(getAllocator(), FnData);
    Syntax* New = new (Mem) Syntax(FnData);

    return New;
  }

  Builtin* CreateBuiltin(ValueFn Fn) {
    return new (TrashHeap) Builtin(Fn);
  }

  BuiltinSyntax* CreateBuiltinSyntax(SyntaxFn Fn) {
    return new (TrashHeap) BuiltinSyntax(Fn);
  }

  SyntaxClosure* CreateSyntaxClosure(SourceLocation Loc, Value EnvStack,
                                     Value Node) {
    return new (TrashHeap) SyntaxClosure(Loc, EnvStack, Node);
  }

  SourceValue* CreateSourceValue(SourceLocation Loc) {
    return new (TrashHeap) SourceValue(Loc);
  }

  Error* CreateError(SourceLocation Loc, Value Message, Value Irritants) {
    return new (TrashHeap) Error(Loc, Message, Irritants);
  }
  Error* CreateError(SourceLocation Loc, StringRef Str, Value Irritants) {
    return CreateError(Loc, CreateString(Str), Irritants);
  }

  ExternName* CreateExternName(SourceLocation Loc, String* Str) {
    return new (TrashHeap) ExternName(Str, Loc);
  }

  Exception* CreateException(Value V) {
    return new (TrashHeap) Exception(V);
  }

  Binding* CreateBinding(Symbol* S, Value V) {
    return new (TrashHeap) Binding(S, V);
  }

protected:
  // The Context object should canonicalize the String object
  // with an identifier table.
  Symbol* CreateSymbol(String* S, SourceLocation Loc) {
    return new (TrashHeap) Symbol(S, Loc);
  }

  // CreateImportSet - overloads for the allocation only.
  ImportSet* CreateImportSetImpl(Module* M) {
    return new (TrashHeap) ImportSet(M);
  }

  ImportSet* CreateImportSetImpl(ImportSet::ImportKind Kind, ImportSet* Parent,
                             Value Specifier) {
    return new (TrashHeap) ImportSet(Kind, Parent, Specifier);
  }
};
}  // namespace heavy

#endif
