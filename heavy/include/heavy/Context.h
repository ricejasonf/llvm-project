//===--- Context.h - Classes for representing declarations ----*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines heavy::Context.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_CONTEXT_H
#define LLVM_HEAVY_CONTEXT_H

#include "heavy/Dialect.h"
#include "heavy/EvaluationStack.h"
#include "heavy/HeavyScheme.h"
#include "heavy/Source.h"
#include "heavy/Value.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <utility>

namespace heavy {
using AllocatorTy = llvm::BumpPtrAllocator;
using llvm::ArrayRef;
using llvm::StringRef;
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;

class OpEval;
class OpGen;
class Context;


// The resulting Value of these functions
// may be invalidated on a call to garbage
// collection if it is not bound to a variable
// at top level scope
// (defined in OpGen.cpp)
Value eval(Context&, Value V, Value EnvStack = nullptr);
void write(llvm::raw_ostream&, Value);

Value opEval(OpEval&);

class OpEvalImpl;
struct OpEval {
  std::unique_ptr<OpEvalImpl> Impl;
  OpEval(Context& C);
  ~OpEval();
};

class Context : DialectRegisterer {
  friend class OpGen;
  friend class OpEvalImpl;
  friend class HeavyScheme;
  friend void* allocate(Context& C, size_t Size, size_t Alignment);
  AllocatorTy TrashHeap;

  std::deque<std::pair<Module, ModuleImportFn*>> Modules = {};
  llvm::StringMap<Module*> ModuleLookup = {};
  llvm::StringMap<String*> IdTable = {};
  // EnvStack
  //  - Should be at least one element on top of
  //    an Environment
  //  - Calls to procedures or eval will set the EnvStack
  //    and swap it back upon completion (via RAII)
  Module* SystemModule;
  Environment* SystemEnvironment;
  ValueFn HandleParseResult;
  Value EnvStack;
  EvaluationStack EvalStack;
  mlir::MLIRContext MlirContext;
  Value Loc = {}; // last know location for errors
  Value Err = nullptr;
  std::unordered_map<void*, Value> EmbeddedEnvs;
public:
  std::unique_ptr<heavy::OpGen> OpGen;
  heavy::OpEval OpEval;

  mlir::Operation* getModuleOp();
  void dumpModuleOp();
  void PushTopLevel(Value);

  // Returns Error for a Value when
  // the callers asserts its kind is invalid
  Value SetInvalidKind(Value V) {
    String* S = CreateString("invalid type ",
                             getKindName(V.getKind()));
    return SetError(S, V);
  }

  void PushMutableModule() {
    EnvStack = CreatePair(CreateModule(), EnvStack);
  }

  void RegisterModule(llvm::StringRef MangledName,
                      heavy::ModuleImportFn* Import) {
    CreateModule(MangledName, Import);
    //Module* M = CreateModule(MangledName, Import);
    // TEMP load the module is if called via (import {name})


  }

  // Import - Finds the Environment in EnvStack, adds the
  //          ImportSet to it, and checks for name collisions
  //          Returns true on Error
  bool Import(ImportSet*);

  void AddBuiltin(StringRef Str, ValueFn Fn);

  void AddBuiltinSyntax(StringRef Str, SyntaxFn Fn) {
    SystemModule->Insert(CreateBinding(CreateSymbol(Str),
                                       CreateBuiltinSyntax(Fn)));
  }

  static std::unique_ptr<Context> CreateEmbedded();

  Context();
  Context(ValueFn ParseResultHandler);
  ~Context();

  // Returns a Builtin from the SystemModule
  // for use within builtin syntaxes that wish
  // to defer to evaluation
  Builtin* GetBuiltin(StringRef Name);

  // Lookup
  //  - Takes a Symbol or nullptr
  //  - Returns a matching Binder or nullptr
  Value Lookup(Symbol* Name, Value Stack,
               Value NextStack = nullptr);
  Value Lookup(Symbol* Name) {
    return Lookup(Name, EnvStack);
  }
  Value Lookup(Value Name) {
    Symbol* S = dyn_cast<Symbol>(Name);
    if (!S) return nullptr;
    return Lookup(S);
  }

  // PushEnvFrame - Creates and pushes an EnvFrame to the
  //                current environment (EnvStack)
  EnvFrame* PushEnvFrame(llvm::ArrayRef<Symbol*> Names);
  void PopEnvFrame();
  void PushLocalBinding(Binding* B);

  // PushLambdaFormals - Checks formals, creates an EnvFrame,
  //                     and pushes it onto the EnvStack
  //                     Returns the pushed EnvFrame or nullptr
  EnvFrame* PushLambdaFormals(Value Formals, bool& HasRestParam);
private:
  bool CheckLambdaFormals(Value Formals,
                          llvm::SmallVectorImpl<Symbol*>& Names,
                          bool& HasRestParam);
public:

  // Check Error
  //  - Returns true if there is an error or exception
  //  - Builtins will have to check this to stop evaluation
  //    when errors occur
  bool CheckError() {
    return Err ? true : false;
  }

  void ClearError() { Err = nullptr; }

  Value SetError(Value E) {
    assert(isa<Error>(E) || isa<Exception>(E));
    Err = E;
    return CreateUndefined();
  }

  Value SetError(SourceLocation Loc, String* S, Value V) {
    return SetError(CreateError(Loc, S, CreatePair(V)));
  }

  Value SetError(String* S, Value V) {
    SourceLocation Loc = V.getSourceLocation();
    return SetError(CreateError(Loc, S, CreatePair(V)));
  }

  Value SetError(StringRef S, Value V) {
    return SetError(CreateString(S), V);
  }

  Value SetError(StringRef S) {
    return SetError(S, CreateUndefined());
  }

  Value SetError(SourceLocation Loc, StringRef S, Value V) {
    return SetError(Loc, CreateString(S), V);
  }

  Value setLoc(Value V) {
    SourceLocation L = V.getSourceLocation();
    if (L.isValid()) {
      Loc = L;
    }
    return V;
  }

  SourceLocation getErrorLocation() {
    SourceLocation L = Err.getSourceLocation();
    if (L.isValid()) return L;
    return Loc;
  }

  StringRef getErrorMessage() {
    assert(Err && "PrintError requires an error be set");
    if (Error* E = dyn_cast_or_null<Error>(Err)) {
      return E->getErrorMessage();
    } else {
      return "Unknown error (invalid error type)";
    }
  }

  Undefined   CreateUndefined() { return {}; }
  Bool        CreateBool(bool V) { return V; }
  Char        CreateChar(uint32_t V) { return V; }
  Int         CreateInt(int32_t x) { return Int(x); }
  Empty       CreateEmpty() { return {}; }
  BigInt*     CreateBigInt(llvm::APInt V);
  BigInt*     CreateBigInt(int64_t X) {
    llvm::APInt Val(64, X, /*IsSigned=*/true);
    return CreateBigInt(Val);
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
  String*     CreateString(StringRef S);
  String*     CreateString(StringRef S1, StringRef S2);
  String*     CreateString(StringRef, StringRef, StringRef);
  String*     CreateIdTableEntry(llvm::StringRef S);
  String*     CreateIdTableEntry(llvm::StringRef Prefix,
                                 llvm::StringRef S);
  Symbol*     CreateSymbol(StringRef S,
                           SourceLocation Loc = SourceLocation());

  Vector*     CreateVector(ArrayRef<Value> Xs);
  Vector*     CreateVector(unsigned N);
  Environment* CreateEnvironment(Value Stack) {
    return new (TrashHeap) Environment(Stack);
  }
  EnvFrame*   CreateEnvFrame(llvm::ArrayRef<Symbol*> Names);

  String* CreateMutableString(StringRef V) {
    String* New = CreateString(V);
    New->IsMutable = true;
    return New;
  }

  Vector* CreateMutableVector(llvm::ArrayRef<Value> Vs) {
    Vector* New = CreateVector(Vs);
    New->IsMutable = true;
    return New;
  }

  template <typename F>
  Lambda* CreateLambda(F Fn, llvm::ArrayRef<heavy::Value> Captures) {
    auto FnData = Lambda::createFunctionDataView(Fn);
    void* Mem = Lambda::allocate(TrashHeap, FnData, Captures);
    Lambda* New = new (Mem) Lambda(FnData, Captures);

    return New;
  }

  Builtin* CreateBuiltin(ValueFn Fn) {
    return new (TrashHeap) Builtin(Fn);
  }
  BuiltinSyntax* CreateBuiltinSyntax(SyntaxFn Fn) {
    return new (TrashHeap) BuiltinSyntax(Fn);
  }

  Error* CreateError(SourceLocation Loc, Value Message, Value Irritants) {
    return new (TrashHeap) Error(Loc, Message, Irritants);
  }
  Error* CreateError(SourceLocation Loc, StringRef Str, Value Irritants) {
    return CreateError(Loc, CreateString(Str), Irritants);
  }

  Exception* CreateException(Value V) {
    return new (TrashHeap) Exception(V);
  }

  // creates anonymous module
  // (usually for the current environment)
  Module* CreateModule(heavy::ModuleImportFn* Import = nullptr) {
    Modules.emplace_back(heavy::Module{}, Import);
    return &(Modules.back().first);
  }

  Module* CreateModule(llvm::StringRef MangledName,
                       heavy::ModuleImportFn* Import = nullptr) {
    Module* M = CreateModule();

    auto Result = ModuleLookup.try_emplace(MangledName, M);
    assert(Result.second && "module should be created only once");
    return M;
  }

  Binding* CreateBinding(Symbol* S, Value V) {
    return new (TrashHeap) Binding(S, V);
  }
  Binding* CreateBinding(Value V) {
    // TODO create Binding class with no symbol
    Symbol* S = CreateSymbol("NONAME");
    return new (TrashHeap) Binding(S, V);
  }

  Quote* CreateQuote(Value V) { return new (TrashHeap) Quote(V); }

  ImportSet* CreateImportSet(Value Spec);
  ImportSet* CreateImportSetExcept(Value Spec);
  ImportSet* CreateImportSetOnly(Value Spec);
  ImportSet* CreateImportSetRename(Value Spec);
  ImportSet* CreateImportSetLibrary(Module* Library);

  // These accessors help track the location
  // so it convenient to overwrite a variable
  Value car(Value V) { return setLoc(V).car(); }
  Value cdr(Value V) { return setLoc(V).cdr(); }
  Value cadr(Value V) { return setLoc(V).cadr(); }
  Value cddr(Value V) { return setLoc(V).cddr(); }
};

}

#endif
