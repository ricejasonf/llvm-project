//===- HeavyScheme.h - Classes for representing declarations ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines HeavyScheme decalarations for values and evaluation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_HEAVY_SCHEME_H
#define LLVM_HEAVY_HEAVY_SCHEME_H

#include "heavy/Dialect.h"
#include "heavy/EvaluationStack.h"
#include "heavy/Source.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/TrailingObjects.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
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
class Value;
class Pair;
class Binding;
class Symbol;
using ValueRefs = llvm::ArrayRef<heavy::Value*>;
using ValueFn = heavy::Value* (*)(Context&, ValueRefs);
using SyntaxFn = mlir::Value (*)(OpGen&, Pair*);


// The resulting Value* of these functions
// may be invalidated on a call to garbage
// collection if it is not bound to a variable
// at top level scope
// (defined in OpGen.cpp)
Value* eval(Context&, Value* V, Value* EnvStack = nullptr);
void write(llvm::raw_ostream&, Value*);
void LoadSystemModule(Context&);

// Value - A result of an evaluation
class Value {
  friend class Context;
public:
  enum class Kind {
    Undefined = 0,
    Binding,
    Boolean,
    Builtin,
    BuiltinSyntax,
    Char,
    Empty,
    Error,
    Environment,
    EnvFrame,
    Exception,
    Float,
    ForwardRef,
    Integer,
    Module,
    Pair,
    PairWithSource,
    Lambda,
    Quote,
    String,
    Symbol,
    Syntax,
    Vector,
  };

private:
  Kind ValueKind;
  bool IsMutable = false;

protected:
  Value (Kind VK)
    : ValueKind(VK)
  { }

public:
  bool isMutable() const { return IsMutable; }
  Kind getKind() const { return ValueKind; }
  static StringRef getKindName(heavy::Value::Kind);
  StringRef getKindName() const { return getKindName(getKind()); }

  // not used
  bool isSyntax() const {
    return getKind() == Kind::Syntax ||
           getKind() == Kind::BuiltinSyntax;
  }

  bool isSymbol(StringRef);
  SourceLocation getSourceLocation();
  void dump();
};

// ValueWithSource
//  - A base class to supplement a Value with
//    source information and to make it immutable
//    (The source location may still be invalid)
class ValueWithSource {
  SourceLocation Loc;
public:
  ValueWithSource(SourceLocation L)
    : Loc(L)
  { }

  SourceLocation getSourceLocation() const {
    return Loc;
  }
};

// This type is for internal use only
// specifically for uninitialized bindings
// (didn't want to use NULL and have to
//  check that everywhere)
class Undefined : public Value {
public:
  Undefined()
    : Value(Kind::Undefined)
  { }

  static bool classof(Value const* V) { return V->getKind() == Kind::Undefined; }
};

class Empty : public Value {
public:
  Empty()
    : Value(Kind::Empty)
  { }

  static bool classof(Value const* V) { return V->getKind() == Kind::Empty; }
};

class Error: public Value,
             public ValueWithSource {
  Value* Message;
  Value* Irritants;
public:

  Error(SourceLocation L, Value* M, Value* I)
    : Value(Kind::Error)
    , ValueWithSource(L)
    , Message(M)
    , Irritants(I)
  { }

  StringRef getErrorMessage();

  static bool classof(Value const* V) { return V->getKind() == Kind::Error; }
};

// Environment
//  - Represents an Environment Specifier created with (environment ...)
//    or the default environment
//  - Stacks Modules the bottom of which is the SystemModule.
//  - Only the top module can be mutable
//  - Adding top level definitions that shadow parent environments
//    is forbidden
class Environment : public Value {
  friend class Context;

  Value* EnvStack;

public:
  Environment(Value* Stack)
    : Value(Kind::Environment)
    , EnvStack(Stack)
  { }

  //Binding* AddDefinition(Symbol* Name, ...
  static bool classof(Value const* V)
  { return V->getKind() == Kind::Environment; }
};

// EnvFrame - Represents a local scope that introduces variables
//          - This should be used exclusively at compile time
//            (unless we go the route of capturing entire scopes
//             to keep values alive)
class EnvFrame final
  : public Value,
    private llvm::TrailingObjects<EnvFrame, Binding*> {

  friend class llvm::TrailingObjects<EnvFrame, Binding*>;
  friend class Context;

  unsigned NumBindings;
  size_t numTrailingObjects(OverloadToken<Binding*> const) const {
    return NumBindings;
  }

  EnvFrame(unsigned NumBindings)
    : Value(Value::Kind::EnvFrame),
      NumBindings(NumBindings)
  { }

public:
  llvm::ArrayRef<Binding*> getBindings() const {
    return llvm::ArrayRef<Binding*>(
        getTrailingObjects<Binding*>(), NumBindings);
  }

  llvm::MutableArrayRef<Binding*> getBindings() {
    return llvm::MutableArrayRef<Binding*>(
        getTrailingObjects<Binding*>(), NumBindings);
  }

  static size_t sizeToAlloc(unsigned NumBindings) {
    return totalSizeToAlloc<Binding*>(NumBindings);
  }

  // Returns nullptr if not found
  Binding* Lookup(Symbol* Name) const;

  static bool classof(Value const* V) {
    return V->getKind() == Kind::EnvFrame;
  }
};

class Exception: public Value {
public:
  Value* Val;
  Exception(Value* Val)
    : Value(Kind::Exception)
    , Val(Val)
  { }

  static bool classof(Value const* V) {
    return V->getKind() == Kind::Exception;
  }
};

class Boolean : public Value {
  bool Val;
public:
  Boolean(bool V)
    : Value(Kind::Boolean)
    , Val(V)
  { }

  auto getVal() { return Val; }
  static bool classof(Value const* V) { return V->getKind() == Kind::Boolean; }
};

// Base class for Numeric types
class Float;
class Integer;
class Number : public Value {
protected:
  Number(Kind K)
    : Value(K)
  { }

public:
  static bool classof(Value const* V) {
    return V->getKind() == Kind::Integer ||
           V->getKind() == Kind::Float;
  }

  bool isExact() {
    return getKind() == Kind::Integer;
  }

  static bool isExactZero(heavy::Value*);
  static Value::Kind CommonKind(Number* X, Number* Y);
};

class Integer : public Number {
  friend class Number;
  friend class NumberOp;
  llvm::APInt Val;

public:
  Integer(llvm::APInt V)
    : Number(Kind::Integer)
    , Val(V)
  { }

  auto getVal() { return Val; }

  static bool classof(Value const* V) {
    return V->getKind() == Kind::Integer;
  }
};

class Float : public Number {
  friend class NumberOp;
  llvm::APFloat Val;

public:
  Float(llvm::APFloat V)
    : Number(Kind::Float)
    , Val(V)
  { }

  auto getVal() { return Val; }
  static bool classof(Value const* V) { return V->getKind() == Kind::Float; }
};

inline Value::Kind Number::CommonKind(Number* X, Number* Y) {
  if (isa<Float>(X) || isa<Float>(Y)) {
    return Value::Kind::Float;
  }
  return Value::Kind::Integer;
}
inline bool Number::isExactZero(Value* V) {
  if (V->getKind() == Kind::Integer) {
    return cast<Integer>(V)->Val == 0;
  }
  return false;
}

class Char : public Value {
  uint32_t Val;

public:
  Char(char V)
    : Value(Kind::Char)
    , Val(V)
  { }

  auto getVal() { return Val; }
  static bool classof(Value const* V) { return V->getKind() == Kind::Char; }
};

class Symbol : public Value,
               public ValueWithSource {
  StringRef Val;

public:
  Symbol(StringRef V, SourceLocation L = SourceLocation())
    : Value(Kind::Symbol)
    , ValueWithSource(L)
    , Val(V)
  { }

  using ValueWithSource::getSourceLocation;

  StringRef getVal() { return Val; }
  static bool classof(Value const* V) { return V->getKind() == Kind::Symbol; }

  bool equals(StringRef Str) const { return Val == Str; }
  bool equals(Symbol* S) const {
    return S->getVal() == Val;
  }
};

class String final
  : public Value,
    private llvm::TrailingObjects<String, char> {
  friend class Context;
  friend class llvm::TrailingObjects<String, char>;

  unsigned Len = 0;
  size_t numTrailingObjects(OverloadToken<char> const) const {
    return Len;
  }

public:
  String(StringRef S)
    : Value(Kind::String),
      Len(S.size())
  { 
    std::memcpy(getTrailingObjects<char>(), S.data(), S.size());
  }

  template <typename ...StringRefs>
  String(unsigned TotalLen, StringRefs ...Ss)
    : Value(Kind::String),
      Len(TotalLen)
  { 
    std::array<StringRef, sizeof...(Ss)> Arr = {Ss...};
    char* StrData = getTrailingObjects<char>();
    for (StringRef S : Arr) {
      std::memcpy(StrData, S.data(), S.size());
      StrData += S.size();
    }
  }
  
  static size_t sizeToAlloc(unsigned Length) {
    return totalSizeToAlloc<char>(Length);
  }

  llvm::StringRef getView() const {
    return llvm::StringRef(getTrailingObjects<char>(), Len);
  }

  static bool classof(Value const* V) { return V->getKind() == Kind::String; }
};

class Pair : public Value {
public:
  Value* Car;
  Value* Cdr;

  Pair(Value* First, Value* Second)
    : Value(Kind::Pair)
    , Car(First)
    , Cdr(Second)
  { }

  Pair(Kind K, Value* First, Value* Second)
    : Value(K)
    , Car(First)
    , Cdr(Second)
  { }

  static bool classof(Value const* V) {
    return V->getKind() == Kind::Pair ||
           V->getKind() == Kind::PairWithSource;
  }
};

class PairWithSource : public Pair,
                       public ValueWithSource {

public:
  PairWithSource(Value* First, Value* Second, SourceLocation L)
    : Pair(Kind::PairWithSource, First, Second)
    , ValueWithSource(L)
  { }

  // returns the character that opens the pair
  // ( | { | [
  char getBraceType() {
    // TODO
    return '(';
  }

  static bool classof(Value const* V) { return V->getKind() == Kind::PairWithSource; }
};

class Builtin : public Value {
public:
  ValueFn Fn;

  Builtin(ValueFn F)
    : Value(Kind::Builtin)
    , Fn(F)
  { }

  static bool classof(Value const* V) {
    return V->getKind() == Kind::Builtin;
  }
};

class BuiltinSyntax : public Value {
public:
  SyntaxFn Fn;

  BuiltinSyntax(SyntaxFn F)
    : Value(Kind::BuiltinSyntax)
    , Fn(F)
  { }

  static bool classof(Value const* V) {
    return V->getKind() == Kind::BuiltinSyntax;
  }
};

// A lambda object that has not been compiled
class Lambda final
  : public Value,
    private llvm::TrailingObjects<Lambda, Value*> {

  friend class llvm::TrailingObjects<Lambda, Value*>;

  mlir::Operation* Op;
  unsigned NumCaptures: 8;

  size_t numTrailingObjects(OverloadToken<Value*> const) const {
    return NumCaptures;
  }

public:
  Lambda(mlir::Operation* Op, unsigned NumCaptures)
    : Value(Kind::Lambda)
    , Op(Op)
    , NumCaptures(NumCaptures)
  { }

  static bool classof(Value const* V) {
    return V->getKind() == Kind::Lambda;
  }

  mlir::Operation* getOp() const { return Op; }

  llvm::ArrayRef<Value*> getCaptures() const {
    return llvm::ArrayRef<Value*>(
        getTrailingObjects<Value*>(), NumCaptures);
  }

  llvm::MutableArrayRef<Value*> getCaptures() {
    return llvm::MutableArrayRef<Value*>(
        getTrailingObjects<Value*>(), NumCaptures);
  }
};

class Quote : public Value {
public:
  Value* Val;
  Quote(Value* V)
    : Value(Kind::Quote)
    , Val(V)
  { }

  static bool classof(Value const* V) {
    return V->getKind() == Kind::Quote;
  }
};

class Syntax : public Value {
public:
  // TODO ???
  Value* Transformer;
  static bool classof(Value const* V) { return V->getKind() == Kind::Syntax; }
};

class Vector final
  : public Value,
    private llvm::TrailingObjects<Vector, Value*> {

  friend class Context;
  friend class llvm::TrailingObjects<Vector, Value*>;

  unsigned Len = 0;
  size_t numTrailingObjects(OverloadToken<Value*> const) const {
    return Len;
  }

  Vector(llvm::ArrayRef<Value*> Vs)
    : Value(Kind::Vector),
      Len(Vs.size())
  {
    std::memcpy(getTrailingObjects<Value*>(), Vs.data(),
                Len * sizeof(Value*));
  }

  Vector(Value* V, unsigned N)
    : Value(Kind::Vector),
      Len(N)
  {
    Value** Xs = getTrailingObjects<Value*>();
    for (unsigned i = 0; i < Len; ++i) {
      Xs[i] = V;
    }
  }

public:
  llvm::ArrayRef<Value*> getElements() const {
    return llvm::ArrayRef<Value*>(
        getTrailingObjects<Value*>(), Len);
  }

  llvm::MutableArrayRef<Value*> getElements() {
    return llvm::MutableArrayRef<Value*>(
        getTrailingObjects<Value*>(), Len);
  }
  static bool classof(Value const* V) { return V->getKind() == Kind::Vector; }

  static size_t sizeToAlloc(unsigned Length) {
    return totalSizeToAlloc<Value*>(Length);
  }

};

class Binding : public Value {
  friend class Context;
  Symbol* Name;

public:
  Value* Val;

  Binding(Symbol* N, Value* V)
    : Value(Kind::Binding)
    , Name(N)
    , Val(V)
  { }

  Symbol* getName() {
    return Name;
  }

  Value* getValue() {
    return Val;
  }

  Value* Lookup(Symbol* S) {
    if (Name->equals(S)) return Val;
    return nullptr;
  }

  static bool classof(Value const* V) { return V->getKind() == Kind::Binding; }
};

class Module : public Value {
  friend class Context;
  using MapTy = llvm::StringMap<Binding*, AllocatorTy&>;
  // TODO An IdentifierTable would probably be
  //      better than using the strings themselves
  //      as keys.
  MapTy Map;

public:
  Module(AllocatorTy& A)
    : Value(Kind::Module)
    , Map(A)
  { }

  Binding* Insert(Binding* B) {
    Map.insert(std::make_pair(B->getName()->getVal(), B));
    return B;
  }

  // Returns nullptr if not found
  Binding* Lookup(StringRef Str) {
    return Map.lookup(Str);
  }

  // Returns nullptr if not found
  Binding* Lookup(Symbol* Name) {
    return Lookup(Name->getVal());
  }

  static bool classof(Value const* V) { return V->getKind() == Kind::Module; }

  class Iterator : public llvm::iterator_facade_base<
                                              Iterator, 
                                              std::forward_iterator_tag,
                                              Binding*>
  {
    friend class Module;
    using ItrTy = typename MapTy::iterator;
    ItrTy Itr;
    Iterator(ItrTy I) : Itr(I) { }

  public:
    Iterator& operator=(Iterator const& R) { Itr = R.Itr; return *this; }
    bool operator==(Iterator const& R) const { return Itr == R.Itr; }
    Binding* const& operator*() const { return (*Itr).getValue(); }
    Binding*& operator*() { return (*Itr).getValue(); }
    Iterator& operator++() { ++Itr; return *this; }
  };

  Iterator begin() {
    return Iterator(Map.begin());
  }

  Iterator end() {
    return Iterator(Map.end());
  }
};

// ForwardRef - used for garbage collection
class ForwardRef : public Value {
public:
  Value* Val;

  ForwardRef(Value* V)
    : Value(Kind::ForwardRef)
  { }

  static bool classof(Value const* V) { return V->getKind() == Kind::ForwardRef; }
};

// isSymbol - For matching symbols in syntax builtins
inline bool Value::isSymbol(StringRef Str) {
  if (Symbol* S = dyn_cast<Symbol>(this)) {
    return S->equals(Str);
  }
  return false;
}

inline SourceLocation Value::getSourceLocation() {
  ValueWithSource* VS = nullptr;
  switch (getKind()) {
  case Kind::Error:
    VS = cast<Error>(this);
    break;
  case Kind::Symbol:
    VS = cast<Symbol>(this);
    break;
  case Kind::PairWithSource:
    VS = cast<PairWithSource>(this);
    break;
  default:
    return SourceLocation();
  }
  return VS->getSourceLocation();
}

class Context : DialectRegisterer {
  AllocatorTy TrashHeap;

  // "static" values
  Undefined Undefined_ = {};
  Empty     Empty_ = {};

  // EnvStack
  //  - Should be at least one element on top of
  //    an Environment
  //  - Calls to procedures or eval will set the EnvStack
  //    and swap it back upon completion (via RAII)
public:
  Module* SystemModule;
  Environment* SystemEnvironment;
  Value* EnvStack;
  std::unordered_map<void*, Value*> EmbeddedEnvs;
  EvaluationStack EvalStack;
  mlir::MLIRContext MlirContext;
  std::unique_ptr<heavy::OpGen> OpGen;
  std::unique_ptr<heavy::OpEval> OpEval;
  Value* Err = nullptr;
  bool IsTopLevel = true;

  template <typename T>
  T* CheckKind(heavy::Value* Val) {
    if (T* V = dyn_cast<T>(Val)) return V;

    // TODO get the Kind from T
    SetError("invalid type, expecting ???", Val);
    return nullptr;
  }

  // used by builtin functions
  bool CheckArity(unsigned Len, ValueRefs Args) {
    StackFrame* F = EvalStack.top();
    if (Args.size() == Len) return false;
    SetError(F->getCallLoc(), "invalid arity", F->getCallee());
  }

  void AddBuiltin(StringRef Str, ValueFn Fn);

  void AddBuiltinSyntax(StringRef Str, SyntaxFn Fn) {
    SystemModule->Insert(CreateBinding(CreateSymbol(Str),
                                       CreateBuiltinSyntax(Fn)));
  }

  static std::unique_ptr<Context> CreateEmbedded();

  Context();
  ~Context();

  // Returns a Builtin from the SystemModule
  // for use within builtin syntaxes that wish
  // to defer to evaluation
  Builtin* GetBuiltin(StringRef Name);

  unsigned GetIntWidth() const {
    return sizeof(int) * 8;
  }

  // Lookup
  //  - Takes a Symbol or nullptr
  //  - Returns a matching Binder or nullptr
  static Binding* Lookup(Symbol* Name,
                         Value* Stack,
                         Value* NextStack = nullptr);
  Binding* Lookup(Symbol* Name) {
    return Lookup(Name, EnvStack);
  }
  Binding* Lookup(Value* Name) {
    Symbol* S = dyn_cast<Symbol>(Name);
    if (!S) return nullptr;
    return Lookup(S);
  }

  // PushEnvFrame - Creates and pushes an EnvFrame to the
  //                current environment (EnvStack)
  EnvFrame* PushEnvFrame(llvm::ArrayRef<Symbol*> Names);
  void PopEnvFrame();

  Value* CreateGlobal(Symbol* S, Value* V, Value* OrigCall);

  // PushLambdaFormals - Checks formals, creates an EnvFrame,
  //                     and pushes it onto the EnvStack
  //                     Returns the pushed EnvFrame or nullptr
  EnvFrame* PushLambdaFormals(Value* Formals, bool& HasRestParam);
private:
  bool CheckLambdaFormals(Value* Formals,
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

  Value* SetError(Value* E) {
    assert(isa<Error>(E) || isa<Exception>(E));
    Err = E;
    return CreateUndefined();
  }

  Value* SetError(SourceLocation Loc, String* S, Value* V) {
    return SetError(CreateError(Loc, S, CreatePair(V)));
  }

  Value* SetError(String* S, Value* V) {
    SourceLocation Loc = V->getSourceLocation();
    return SetError(CreateError(Loc, S, CreatePair(V)));
  }

  Value* SetError(StringRef S, Value* V) {
    return SetError(CreateString(S), V);
  }

  Value* SetError(StringRef S) {
    return SetError(S, CreateUndefined());
  }

  Value* SetError(SourceLocation Loc, StringRef S, Value* V) {
    return SetError(Loc, CreateString(S), V);
  }

  SourceLocation getErrorLocation() {
    if (Err) return Err->getSourceLocation();
    return SourceLocation();
  }

  StringRef getErrorMessage() {
    assert(Err && "PrintError requires an error be set");
    if (Error* E = dyn_cast_or_null<Error>(Err)) {
      return E->getErrorMessage();
    } else {
      return "Unknown error (invalid error type)";
    }
  }

  Undefined*  CreateUndefined() { return &Undefined_; }
  Boolean*    CreateBoolean(bool V) { return new (TrashHeap) Boolean(V); }
  Char*       CreateChar(char V) { return new (TrashHeap) Char(V); }
  Empty*      CreateEmpty() { return &Empty_; }
  Integer*    CreateInteger(llvm::APInt V);
  Integer*    CreateInteger(int64_t X) {
    int BitWidth = GetIntWidth();
    llvm::APInt Val(BitWidth, X, /*IsSigned=*/true);
    return CreateInteger(Val);
  }
  Float*      CreateFloat(llvm::APFloat V);
  Pair*       CreatePair(Value* V1, Value* V2) {
    return new (TrashHeap) Pair(V1, V2);
  }
  Pair*       CreatePair(Value* V1) {
    return new (TrashHeap) Pair(V1, CreateEmpty());
  }
  PairWithSource* CreatePairWithSource(Value* V1, Value* V2,
                                       SourceLocation Loc) {
    return new (TrashHeap) PairWithSource(V1, V2, Loc);
  }
  String*     CreateString(StringRef S);
  String*     CreateString(StringRef S1, StringRef S2);
  String*     CreateString(StringRef, StringRef, StringRef);
  Symbol*     CreateSymbol(StringRef V,
                         SourceLocation Loc = SourceLocation()) {
    // FIXME uhh this should store a copy V's contents somewhere
    // TODO make a lookup table for symbols
    return new (TrashHeap) Symbol(V, Loc);
  }
  Vector*     CreateVector(ArrayRef<Value*> Xs);
  Vector*     CreateVector(unsigned N);
  Environment* CreateEnvironment(Value* Stack) {
    return new (TrashHeap) Environment(Stack);
  }
  EnvFrame*   CreateEnvFrame(llvm::ArrayRef<Symbol*> Names);

  String* CreateMutableString(StringRef V) {
    String* New = CreateString(V);
    New->IsMutable = true;
    return New;
  }

  Vector* CreateMutableVector(llvm::ArrayRef<Value*> Vs) {
    Vector* New = CreateVector(Vs);
    New->IsMutable = true;
    return New;
  }

  Builtin* CreateBuiltin(ValueFn Fn) {
    return new (TrashHeap) Builtin(Fn);
  }
  BuiltinSyntax* CreateBuiltinSyntax(SyntaxFn Fn) {
    return new (TrashHeap) BuiltinSyntax(Fn);
  }

  Error* CreateError(SourceLocation Loc, Value* Message, Value* Irritants) {
    return new (TrashHeap) Error(Loc, Message, Irritants);
  }
  Error* CreateError(SourceLocation Loc, StringRef Str, Value* Irritants) {
    return CreateError(Loc, CreateString(Str), Irritants);
  }

  Exception* CreateException(Value* V) {
    return new (TrashHeap) Exception(V);
  }

  Module* CreateModule() {
    return new (TrashHeap) Module(TrashHeap);
  }
  Binding* CreateBinding(Symbol* S, Value* V) {
    return new (TrashHeap) Binding(S, V);
  }
  Binding* CreateBinding(Value* V) {
    // TODO create Binding class with no symbol
    Symbol* S = CreateSymbol("NONAME");
    return new (TrashHeap) Binding(S, V);
  }

  Quote* CreateQuote(Value* V) { return new (TrashHeap) Quote(V); }
};

inline StringRef Error::getErrorMessage() {
  if (String* S = dyn_cast<String>(Message)) {
    return  S->getView();
  }
  return "Unknown error (invalid error message)";
}

inline Binding* EnvFrame::Lookup(Symbol* Name) const {
  // linear search
  for (Binding* B : getBindings()) {
    if (Name->equals(B->getName())) return B;
  }
  return nullptr;
}

// ValueVisitor
// This will be the base class for evaluation and printing
template <typename Derived, typename RetTy = void>
class ValueVisitor {
#define DISPATCH(NAME) \
  return getDerived().Visit ## NAME(static_cast<NAME*>(V), \
                                    std::forward<Args>(args)...)
#define VISIT_FN(NAME) \
  template <typename T, typename ...Args> \
  RetTy Visit ## NAME(T* V, Args&& ...args) { \
    return getDerived().VisitValue(V, std::forward<Args>(args)...); }

  Derived& getDerived() { return static_cast<Derived&>(*this); }
  Derived const& getDerived() const { return static_cast<Derived>(*this); }

protected:
  // Derived must implement VisitValue OR all of the
  // concrete visitors
  template <typename T>
  RetTy VisitValue(T* V) = delete;

  // The default implementations for visiting
  // nodes is to call Derived::VisitValue

  VISIT_FN(Undefined)
  VISIT_FN(Binding)
  VISIT_FN(Boolean)
  VISIT_FN(Builtin)
  VISIT_FN(BuiltinSyntax)
  VISIT_FN(Char)
  VISIT_FN(Empty)
  VISIT_FN(Error)
  VISIT_FN(Environment)
  VISIT_FN(EnvFrame)
  VISIT_FN(Exception)
  VISIT_FN(Float)
  VISIT_FN(ForwardRef)
  VISIT_FN(Integer)
  VISIT_FN(Module)
  VISIT_FN(Pair)
  // VISIT_FN(PairWithSource) **PairWithSource Implemented below**
  VISIT_FN(Lambda)
  VISIT_FN(Quote)
  VISIT_FN(String)
  VISIT_FN(Symbol)
  VISIT_FN(Syntax)
  VISIT_FN(Vector)

  template <typename ...Args>
  RetTy VisitPairWithSource(Pair* P, Args&& ...args) {
    return getDerived().VisitPair(P, std::forward<Args>(args)...);
  }

public:
  template <typename ...Args>
  RetTy Visit(Value* V, Args&& ...args) {
    switch (V->getKind()) {
    case Value::Kind::Undefined:      DISPATCH(Undefined);
    case Value::Kind::Binding:        DISPATCH(Binding);
    case Value::Kind::Boolean:        DISPATCH(Boolean);
    case Value::Kind::Builtin:        DISPATCH(Builtin);
    case Value::Kind::BuiltinSyntax:  DISPATCH(BuiltinSyntax);
    case Value::Kind::Char:           DISPATCH(Char);
    case Value::Kind::Empty:          DISPATCH(Empty);
    case Value::Kind::Error:          DISPATCH(Error);
    case Value::Kind::Environment:    DISPATCH(Environment);
    case Value::Kind::EnvFrame:       DISPATCH(EnvFrame);
    case Value::Kind::Exception:      DISPATCH(Exception);
    case Value::Kind::Float:          DISPATCH(Float);
    case Value::Kind::ForwardRef:     DISPATCH(ForwardRef);
    case Value::Kind::Integer:        DISPATCH(Integer);
    case Value::Kind::Module:         DISPATCH(Module);
    case Value::Kind::Pair:           DISPATCH(Pair);
    case Value::Kind::PairWithSource:    DISPATCH(PairWithSource);
    case Value::Kind::Lambda:      DISPATCH(Lambda);
    case Value::Kind::Quote:          DISPATCH(Quote);
    case Value::Kind::String:         DISPATCH(String);
    case Value::Kind::Symbol:         DISPATCH(Symbol);
    case Value::Kind::Syntax:         DISPATCH(Syntax);
    case Value::Kind::Vector:         DISPATCH(Vector);
    default:
      llvm_unreachable("Invalid Value Kind");
    }
  }

#undef DISPATCH
#undef VISIT_FN
};

#define GET_KIND_NAME_CASE(KIND) \
  case Value::Kind::KIND: return StringRef(#KIND, sizeof(#KIND));
inline StringRef Value::getKindName(heavy::Value::Kind Kind) {
  switch (Kind) {
  GET_KIND_NAME_CASE(Undefined)
  GET_KIND_NAME_CASE(Binding)
  GET_KIND_NAME_CASE(Boolean)
  GET_KIND_NAME_CASE(Builtin)
  GET_KIND_NAME_CASE(BuiltinSyntax)
  GET_KIND_NAME_CASE(Char)
  GET_KIND_NAME_CASE(Empty)
  GET_KIND_NAME_CASE(Error)
  GET_KIND_NAME_CASE(Environment)
  GET_KIND_NAME_CASE(EnvFrame)
  GET_KIND_NAME_CASE(Exception)
  GET_KIND_NAME_CASE(Float)
  GET_KIND_NAME_CASE(ForwardRef)
  GET_KIND_NAME_CASE(Integer)
  GET_KIND_NAME_CASE(Module)
  GET_KIND_NAME_CASE(Pair)
  GET_KIND_NAME_CASE(PairWithSource)
  GET_KIND_NAME_CASE(Lambda)
  GET_KIND_NAME_CASE(Quote)
  GET_KIND_NAME_CASE(String)
  GET_KIND_NAME_CASE(Symbol)
  GET_KIND_NAME_CASE(Syntax)
  GET_KIND_NAME_CASE(Vector)
  default:
    return StringRef("?????");
  }
}
#undef GET_KIND_NAME_CASE

} // namespace heavy

#endif // LLVM_CLANG_AST_HEAVY_SCHEME_H
