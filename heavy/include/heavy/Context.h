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

#include "heavy/ContinuationStack.h"
#include "heavy/Heap.h"
#include "heavy/Source.h"
#include "heavy/Value.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
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

namespace mlir {
  class MLIRContext;
}

namespace heavy {
using llvm::ArrayRef;
using llvm::StringRef;
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;

class OpGen;
class Context;

void compile(Context&, Value V, Value Env, Value Handler);
void eval(Context&, Value V, Value Env);
void write(llvm::raw_ostream&, Value);

class OpEvalImpl;
void opEval(mlir::Operation*);
void invokeSyntaxOp(heavy::Context& C, mlir::Operation* Op,
                    heavy::Value Value);

class ContextLocalLookup {
  friend class ContextLocal;
  llvm::DenseMap<uintptr_t, Value> LookupTable;
};

class Context : public ContinuationStack<Context>,
                public ContextLocalLookup,
                public IdTable,
                protected Heap<Context>
{
  friend class OpGen;
  friend class OpEvalImpl;
  friend class HeavyScheme;
  friend class Heap<Context>;
  friend void* allocate(Context& C, size_t Size, size_t Alignment);
  friend void initModule(heavy::Context&, llvm::StringRef MangledName,
                         ModuleInitListTy InitList);
  friend void registerModuleVar(heavy::Context& C,
                                heavy::Module* M,
                                llvm::StringRef VarSymbol,
                                llvm::StringRef VarId,
                                Value Val);

  static constexpr size_t MiB = 1024 * 1024;

  llvm::StringMap<std::unique_ptr<Module>> Modules;
  llvm::DenseMap<void*, std::unique_ptr<Environment>> EmbeddedEnvs;
  llvm::DenseMap<String*, Value> KnownAddresses;

  // MaxMemHint
  //         - The threshold used to determine if a garbage
  //           collection run is needed. This value is not a
  //           hard limit and the limit is increased when a
  //           collection run yields a low return. The limit
  //           may also increase with the allocation of a large
  //           object.
  size_t MaxMemHint;

  // EnvStack
  //  - an improper list ending with an Environment
  //  - Calls to procedures or eval will set the EnvStack
  //    and swap it back upon completion (via RAII)
  Value EnvStack;

public: // Provide access in lib/Mlir bindings.
  std::unique_ptr<mlir::MLIRContext> MLIRContext;
private:
  SourceLocation Loc = {}; // last known location for errors
  Value Err = nullptr;
  Value ExceptionHandlers = heavy::Empty();
  mlir::Operation* ModuleOp = nullptr;

public:
  heavy::OpGen* OpGen = nullptr;
  heavy::OpEvalImpl* OpEval = nullptr;

  // SetErrorHandler - Set the bottom most exception handler to handle
  //                   hard errors including uncaught exceptions.
  void SetErrorHandler(Value Handler);
  void WithExceptionHandlers(Value NewHandlers, Value Thunk);
  void WithExceptionHandler(Value Handler, Value Thunk);
  void Raise(Value Obj);
  void RaiseError(String* Msg, llvm::ArrayRef<Value> IrrArgs);
  void RaiseError(llvm::StringRef Msg, llvm::ArrayRef<Value> IrrArgs = {}) {
    RaiseError(CreateString(Msg), IrrArgs);
  }

  void CollectGarbage();

  // WithEnv - Call a thunk with an environment and the ability to clean up
  //           the environment object if necessary.
  void WithEnv(std::unique_ptr<heavy::Environment> E, heavy::Environment* Env,
               Value Thunk);
  void WithEnv(std::unique_ptr<heavy::Environment> EnvPtr, Value Thunk) {
    heavy::Environment* Env = EnvPtr.get();
    WithEnv(std::move(EnvPtr), Env, Thunk);
  }

  // RunSync - Run in a nested loop. This breaks
  //           when the callee finishes or on
  //           exception so the parent loop can
  //           finish execution such as clean up.
  Value RunSync(Value Callee, Value Arg);

  mlir::Operation* getModuleOp();
  void dumpModuleOp();
  void verifyModule();
  void PushTopLevel(Value);

  // Return true on invalid kind
  bool CheckKind(ValueKind VK, Value V);
  bool CheckNumber(Value V);
  template <typename T>
  bool CheckKind(Value V) {
    return CheckKind(T::getKind(), V);
  }

  Value getEnvironment() {
    return EnvStack;
  }
  void setEnvironment(Value E) {
    assert((isa<Environment, Pair, Empty>(E)) &&
        "invalid environment specifier");
    EnvStack = E;
  }

  Environment* getTopLevelEnvironment();

  Module* RegisterModule(llvm::StringRef MangledName,
                         heavy::ModuleLoadNamesFn* LoadNames = nullptr);

  void AddKnownAddress(llvm::StringRef MangledName, heavy::Value Value);
  Value GetKnownValue(llvm::StringRef MangledName);

  // Import - Apply an ImportSet to an Environment checking
  //          for name collisions. Use the current environment
  //          by default.
  //          Return true on Error
  void Import(heavy::ImportSet* ImportSet);

  // CreateEnvironment - Create a non-garbage collected instance of
  //                     Environment. The unique_ptr may contain nullptr
  //                     if the import operation fails.
  std::unique_ptr<Environment> CreateEnvironment(heavy::ImportSet* ImportSet);

  // LoadModule - Idempotently load a library
  void LoadModule(Value Spec, bool IsFileLoaded = false);
  void PushModuleCleanup(llvm::StringRef MangledName, Value Fn);
  void IncludeModuleFile(heavy::SourceLocation Loc, heavy::String* Filename,
                         std::string ModuleMangledName);


  Context();
  ~Context();

  // Lookup
  //  - Takes a Symbol
  //  - Returns a matching Binder or nullptr
  EnvEntry Lookup(Symbol* Name, Value Stack);
  EnvEntry Lookup(Symbol* Name) {
    return Lookup(Name, EnvStack);
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

  void EmitStackSpaceError();

  void Apply(SourceLocation CallLoc, Value Callee, ValueRefs Args) {
    setLoc(CallLoc);
    ContinuationStack<Context>::Apply(Callee, Args);
  }
  using ContinuationStack<Context>::Apply;

  // Check Error
  //  - Returns true if there is an error or exception
  //  - Builtins will have to check this to stop evaluation
  //    when errors occur
  bool CheckError() {
    return Err ? true : false;
  }

  void ClearError() { Err = nullptr; }

  // SetError - Put the context into a hard error state. The stack
  //            should be cleared, and subsequent calls to eval
  //            should be a noop unless the error is cleared
  //            by the user.
  void SetError(Value E);
  void SetError(SourceLocation Loc, String* S, Value V) {
    return SetError(CreateError(Loc, S, CreatePair(V)));
  }
  void SetError(String* S, Value V) {
    SourceLocation Loc = V.getSourceLocation();
    return SetError(CreateError(Loc, S, CreatePair(V)));
  }
  void SetError(StringRef S, Value V) {
    return SetError(CreateString(S), V);
  }
  void SetError(StringRef S) {
    return SetError(S, CreateUndefined());
  }
  void SetError(SourceLocation Loc, StringRef S, Value V) {
    return SetError(Loc, CreateString(S), V);
  }

  void setLoc(SourceLocation L) {
    if (L.isValid()) {
      Loc = L;
    }
  }

  Value setLoc(Value V) {
    setLoc(V.getSourceLocation());
    return V;
  }

  SourceLocation getLoc() const { return Loc; }

  SourceLocation getErrorLocation() {
    assert(Err && "requires an error be set");
    SourceLocation L = Err.getSourceLocation();
    if (L.isValid()) return L;
    return Loc;
  }

  StringRef getErrorMessage() {
    assert(Err && "requires an error be set");
    if (Error* E = dyn_cast_or_null<Error>(Err)) {
      return E->getErrorMessage();
    } else {
      return "Unknown error (invalid error type)";
    }
  }

  Syntax* CreateSyntaxWithOp(mlir::Operation* SyntaxOp);

  Value RebuildLiteral(Value V);

  Heap<Context>& getAllocator() { return *this; }

  // Factory functions should only be concerned with
  // allocating the objects.
  Undefined   CreateUndefined() { return {}; }
  Bool        CreateBool(bool V) { return V; }
  Char        CreateChar(uint32_t V) { return V; }
  Int         CreateInt(int32_t x) { return Int(x); }
  Empty       CreateEmpty() { return {}; }
  BigInt*     CreateBigInt(llvm::APInt V) {
    return new (*this) BigInt(V);
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
    return new (*this) Pair(V1, V2);
  }
  Pair*       CreatePair(Value V1) {
    return new (*this) Pair(V1, CreateEmpty());
  }
  PairWithSource* CreatePairWithSource(Value V1, Value V2,
                                       SourceLocation Loc) {
    return new (*this) PairWithSource(V1, V2, Loc);
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
  OpaquePtr* CreateOpaquePtr(void* Ptr) {
    return new (getAllocator()) OpaquePtr(Ptr);
  }
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
    return new (*this) Builtin(Fn);
  }

  BuiltinSyntax* CreateBuiltinSyntax(SyntaxFn Fn) {
    return new (*this) BuiltinSyntax(Fn);
  }

  SyntaxClosure* CreateSyntaxClosure(SourceLocation Loc, Value Node) {
    return new (*this) SyntaxClosure(Loc, EnvStack, Node);
  }

  SourceValue* CreateSourceValue(SourceLocation Loc) {
    return new (*this) SourceValue(Loc);
  }

  Error* CreateError(SourceLocation Loc, Value Message, Value Irritants) {
    return new (*this) Error(Loc, Message, Irritants);
  }
  Error* CreateError(SourceLocation Loc, StringRef Str, Value Irritants) {
    return CreateError(Loc, CreateString(Str), Irritants);
  }

  ExternName* CreateExternName(SourceLocation Loc, String* Str) {
    return new (*this) ExternName(Str, Loc);
  }

  ExternName* CreateExternName(SourceLocation Loc, llvm::StringRef Name) {
    String* Str = CreateIdTableEntry(Name);
    return CreateExternName(Loc, Str);
  }

  Exception* CreateException(Value V) {
    return new (*this) Exception(V);
  }

  Binding* CreateBinding(Symbol* S, Value V) {
    return new (*this) Binding(S, V);
  }

  // CreateImportSet - Call CC with created ImportSet.
  void CreateImportSet(Value Spec);

  Binding* CreateBinding(Value V) {
    // TODO create Binding class with no symbol maybe?
    Symbol* S = CreateSymbol("NONAME");
    return CreateBinding(S, V);
  }


  Symbol* CreateSymbol(llvm::StringRef S,
                       SourceLocation Loc = SourceLocation()) {
    String* Str = CreateIdTableEntry(S);
    return new (*this) Symbol(Str, Loc);
  }

  // These accessors help track the location
  // so it is convenient to overwrite a variable
  Value car(Value V) { return setLoc(V).car(); }
  Value cdr(Value V) { return setLoc(V).cdr(); }
  Value cadr(Value V) { return setLoc(V).cadr(); }
  Value cddr(Value V) { return setLoc(V).cddr(); }
};

}  // end namespace heavy

#endif
