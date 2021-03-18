//===---- Value.h - Classes for representing declarations ----*- C++ ----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines HeavyScheme decalarations for heavy::Value.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_VALUE_H
#define LLVM_HEAVY_VALUE_H

#include "heavy/Source.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/ADT/PointerSumType.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TrailingObjects.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>


namespace heavy {
class Context;

// Value - A result of evaluation.
//         The type is deliberately left incomplete
//         so we can pass around Value* as an invalid
//         pointer.
class Value;

// ValuePtr - An instance of PointerSumType for holding
//            ValueBase*, int, or other small stuff in
//            a pointer sized object
class ValuePtr;

enum class ValueKind {
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
  LambdaIr,
  Quote,
  String,
  Symbol,
  Syntax,
  Vector,
};

class ValueBase {
public:

private:
  ValueKind VKind;
  bool IsMutable = false;

protected:
  ValueBase(ValueKind VK)
    : VKind(VK)
  { }

public:
  bool isMutable() const { return IsMutable; }
  ValueKind getKind() const { return VKind; }

  void dump();
};

// Value types that will be members of ValuePtr
struct Undefined {
  static bool classof(ValuePtr);
};

struct Empty {
  static bool classof(ValuePtr);
};

struct Int : llvm::PointerEmbeddedInt<int32_t> {
  using Base = llvm::PointerEmbeddedInt<int32_t>;
  static bool classof(ValuePtr);
};

}

namespace heavy { namespace detail {
template <typename T>
struct StatelessPointerTraitBase {
  static_assert(std::is_empty<T>::value, "must be empty type");
  static void* getAsVoidPointer(T) { return nullptr; }
  static T getFromVoidPointer(void* P) { return {}; }

  // all the bits??
  static constexpr int NumLowBitsAvailable =
      llvm::detail::ConstantLog2<alignof(void*)>::value;
};
}}

namespace llvm {
template <>
struct PointerLikeTypeTraits<heavy::Empty>
  : heavy::detail::StatelessPointerTraitBase<heavy::Empty>
{ };

template <>
struct PointerLikeTypeTraits<heavy::Undefined>
  : heavy::detail::StatelessPointerTraitBase<heavy::Undefined>
{ };

template <>
struct PointerLikeTypeTraits<heavy::Int>
  : PointerLikeTypeTraits<heavy::Int::Base>
{ };
}

namespace heavy {

struct ValueSumType {
  enum SumKind {
    ValueBase = 0,
    Undefined,
    Empty,
    Int,
    Operation,
  };

  using type = llvm::PointerSumType<SumKind,
    llvm::PointerSumTypeMember<ValueBase,  heavy::ValueBase*>,
    llvm::PointerSumTypeMember<Int,        heavy::Int>,
    llvm::PointerSumTypeMember<Empty,      heavy::Empty>,
    llvm::PointerSumTypeMember<Undefined,  heavy::Undefined>>;
    //llvm::PointerSumTypeMember<Operation,  mlir::Operation*>>;
};

using ValuePtrBase = typename ValueSumType::type;

struct ValuePtr : ValuePtrBase {
  /* implicit */
  ValuePtr(Value* V)
    : ValuePtrBase(reinterpret_cast<ValuePtrBase>(V))
  { }

  operator Value*() const {
    return reinterpret_cast<Value*>(this);
  }

  ValueKind getKind() {
    switch (getTag()) {
    case ValueSumType::ValueBase:
      return get<ValueSumType::Value>()->getKind();
    case ValueSumType::Undefined:
      return ValueKind::Undefined;
    case ValueSumType::Empty:
      return ValueKind::Empty;
    case ValueSumType::Int:
      return ValueKind::Int;
    case ValueSumType::Operation:
      return ValueKind::Operation;
    }
  }
};

inline static bool Undefined::classof(ValuePtr V) {
  return V.getTag() == ValueSumType::Undefined;
}

inline static bool Empty::classof(ValuePtr V) {
  return V.getTag() == ValueSumType::Empty;
}

inline static bool Int::classof(ValuePtr V) {
  return V.getTag() == ValueSumType::Int;
}

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

// Concrete Values

class Error: public ValueBase,
             public ValueWithSource {
  Value* Message;
  Value* Irritants;
public:

  Error(SourceLocation L, Value* M, Value* I)
    : ValueBase(ValueKind::Error)
    , ValueWithSource(L)
    , Message(M)
    , Irritants(I)
  { }

  llvm::StringRef getErrorMessage();

  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Error;
  }
};

// Environment
//  - Represents an Environment Specifier created with (environment ...)
//    or the default environment
//  - Stacks Modules the bottom of which is the SystemModule.
//  - Only the top module can be mutable
//  - Adding top level definitions that shadow parent environments
//    is forbidden
class Environment : public ValueBase {
  friend class Context;

  Value* EnvStack;

public:
  Environment(Value* Stack)
    : ValueBase(Kind::Environment)
    , EnvStack(Stack)
  { }

  //Binding* AddDefinition(Symbol* Name, ...
  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Environment;
  }
};

// EnvFrame - Represents a local scope that introduces variables
//          - This should be used exclusively at compile time
//            (unless we go the route of capturing entire scopes
//             to keep values alive)
class EnvFrame final
  : public ValueBase,
    private llvm::TrailingObjects<EnvFrame, Binding*> {

  friend class llvm::TrailingObjects<EnvFrame, Binding*>;
  friend class Context;

  unsigned NumBindings;
  size_t numTrailingObjects(OverloadToken<Binding*> const) const {
    return NumBindings;
  }

  EnvFrame(unsigned NumBindings)
    : ValueBase(ValueKind::EnvFrame),
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

  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::EnvFrame;
  }
};

class Exception: public ValueBase {
public:
  Value* Val;
  Exception(Value* Val)
    : ValueBase(Kind::Exception)
    , Val(Val)
  { }

  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Exception;
  }
};

class Boolean : public ValueBase {
  bool Val;
public:
  Boolean(bool V)
    : ValueBase(Kind::Boolean)
    , Val(V)
  { }

  auto getVal() { return Val; }
  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Boolean;
  }
};

// Base class for Numeric types (other than Int)
class Float;
class Number : public ValueBase {
protected:
  Number(Kind K)
    : ValueBase(K)
  { }

public:
  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::BigInt ||
           V.getKind() == ValueKind::Float;
  }

  static bool isExact(ValuePtr V) {
    switch(V.getKind()) {
    case ValueKind::BigInt:
    case ValueKind::Int:
      return true;
    default:
      return false;
    }
  }

  static bool isExactZero(heavy::Value*);
  static ValueKind CommonKind(Value* X, Value* Y);
};

// BigInt currently assumes 64 bits
class BigInt : public Number {
  friend class Number;
  friend class NumberOp;
  llvm::APInt Val;

public:
  BigInt(llvm::APInt V)
    : Number(Kind::BigInt)
    , Val(V)
  { }

  auto getVal() { return Val; }

  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::BigInt;
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
  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Float;
  }
};

inline ValueKind Number::CommonKind(ValuePtr X, ValuePtr Y) {
  ValueKind XK = X.getKind();
  ValueKind YK = Y.getKind();
  if (XK == ValueKind::Float ||
      YK == ValueKind::Float) {
    return ValueKind::Float;
  }

  if (XK == ValueKind::BigInt ||
      YK == ValueKind::BigInt) {
    return ValueKind::BigInt;
  }

  return ValueKind::Int;
}

inline bool Number::isExactZero(Value* V) {
  if (!Number::isExact(V)) return false;
  if (BigInt* I = dyn_cast<BigInt>(V)) {
    I->Val == 0;
  }
  return cast<Int>(V) == 0;
}

// TODO maybe Char could fit in ValuePtr?
class Char : public ValueBase {
  uint32_t Val;

public:
  Char(char V)
    : ValueBase(Kind::Char)
    , Val(V)
  { }

  auto getVal() { return Val; }
  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Char;
  }
};

class Symbol : public ValueBase,
               public ValueWithSource {
  llvm::StringRef Val;

public:
  Symbol(llvm::StringRef V, SourceLocation L = SourceLocation())
    : ValueBase(Kind::Symbol)
    , ValueWithSource(L)
    , Val(V)
  { }

  using ValueWithSource::getSourceLocation;

  llvm::StringRef getVal() { return Val; }
  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Symbol;
  }

  bool equals(llvm::StringRef Str) const { return Val == Str; }
  bool equals(Symbol* S) const {
    return S->getVal() == Val;
  }
};

class String final
  : public ValueBase,
    private llvm::TrailingObjects<String, char> {
  friend class Context;
  friend class llvm::TrailingObjects<String, char>;

  unsigned Len = 0;
  size_t numTrailingObjects(OverloadToken<char> const) const {
    return Len;
  }

public:
  String(llvm::StringRef S)
    : ValueBase(Kind::String),
      Len(S.size())
  {
    std::memcpy(getTrailingObjects<char>(), S.data(), S.size());
  }

  template <typename ...StringRefs>
  String(unsigned TotalLen, StringRefs ...Ss)
    : ValueBase(Kind::String),
      Len(TotalLen)
  {
    std::array<llvm::StringRef, sizeof...(Ss)> Arr = {Ss...};
    char* StrData = getTrailingObjects<char>();
    for (llvm::StringRef S : Arr) {
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

  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::String;
  }
};

class Pair : public ValueBase {
public:
  Value* Car;
  Value* Cdr;

  Pair(Value* First, Value* Second)
    : ValueBase(Kind::Pair)
    , Car(First)
    , Cdr(Second)
  { }

  Pair(Kind K, Value* First, Value* Second)
    : ValueBase(K)
    , Car(First)
    , Cdr(Second)
  { }

  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Pair ||
           V.getKind() == ValueKind::PairWithSource;
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

  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::PairWithSource;
  }
};

class Builtin : public ValueBase {
public:
  ValueFn Fn;

  Builtin(ValueFn F)
    : ValueBase(Kind::Builtin)
    , Fn(F)
  { }

  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Builtin;
  }
};

class BuiltinSyntax : public ValueBase {
public:
  SyntaxFn Fn;

  BuiltinSyntax(SyntaxFn F)
    : ValueBase(Kind::BuiltinSyntax)
    , Fn(F)
  { }

  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::BuiltinSyntax;
  }
};

class Lambda final
  : public ValueBase,
    private llvm::TrailingObjects<Lambda, Value*> {

  friend class llvm::TrailingObjects<Lambda, Value*>;

  ValueFn Fn;
  unsigned NumCaptures: 8;

  size_t numTrailingObjects(OverloadToken<Value*> const) const {
    return NumCaptures;
  }

public:
  Lambda(ValueFn Fn, unsigned NumCaptures)
    : ValueBase(Kind::Lambda)
    , Fn(Fn)
    , NumCaptures(NumCaptures)
  { }

  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Lambda;
  }

  ValueFn getFn() const { return Fn; }

  llvm::ArrayRef<Value*> getCaptures() const {
    return llvm::ArrayRef<Value*>(
        getTrailingObjects<Value*>(), NumCaptures);
  }

  llvm::MutableArrayRef<Value*> getCaptures() {
    return llvm::MutableArrayRef<Value*>(
        getTrailingObjects<Value*>(), NumCaptures);
  }
};

// A lambda object that has not been compiled
// for use with the tree walking evaluator (OpEval)
class LambdaIr final
  : public ValueBase,
    private llvm::TrailingObjects<LambdaIr, Value*> {

  friend class llvm::TrailingObjects<LambdaIr, Value*>;

  FuncOp Op;
  unsigned NumCaptures: 8;

  size_t numTrailingObjects(OverloadToken<Value*> const) const {
    return NumCaptures;
  }

public:
  LambdaIr(FuncOp Op, unsigned NumCaptures)
    : ValueBase(Kind::LambdaIr)
    , Op(Op)
    , NumCaptures(NumCaptures)
  { }

  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::LambdaIr;
  }

  FuncOp getOp() const { return Op; }
  mlir::Block& getBody() {
    return Op.getBody().front();
  }

  llvm::ArrayRef<Value*> getCaptures() const {
    return llvm::ArrayRef<Value*>(
        getTrailingObjects<Value*>(), NumCaptures);
  }

  llvm::MutableArrayRef<Value*> getCaptures() {
    return llvm::MutableArrayRef<Value*>(
        getTrailingObjects<Value*>(), NumCaptures);
  }

  static size_t sizeToAlloc(unsigned Length) {
    return totalSizeToAlloc<Value*>(Length);
  }
};

class Quote : public ValueBase {
public:
  Value* Val;
  Quote(Value* V)
    : ValueBase(Kind::Quote)
    , Val(V)
  { }

  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Quote;
  }
};

class Syntax : public ValueBase {
public:
  // TODO ???
  Value* Transformer;
  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Syntax;
  }
};

class Vector final
  : public ValueBase,
    private llvm::TrailingObjects<Vector, Value*> {

  friend class Context;
  friend class llvm::TrailingObjects<Vector, Value*>;

  unsigned Len = 0;
  size_t numTrailingObjects(OverloadToken<Value*> const) const {
    return Len;
  }

  Vector(llvm::ArrayRef<Value*> Vs)
    : ValueBase(Kind::Vector),
      Len(Vs.size())
  {
    std::memcpy(getTrailingObjects<Value*>(), Vs.data(),
                Len * sizeof(Value*));
  }

  Vector(Value* V, unsigned N)
    : ValueBase(Kind::Vector),
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
  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Vector;
  }

  static size_t sizeToAlloc(unsigned Length) {
    return totalSizeToAlloc<Value*>(Length);
  }

};

class Binding : public ValueBase {
  friend class Context;
  Symbol* Name;
  Value* Val;

public:

  Binding(Symbol* N, Value* V)
    : ValueBase(Kind::Binding)
    , Name(N)
    , Val(V)
  { }

  Symbol* getName() {
    return Name;
  }

  Value* getValue() {
    return Val;
  }

  void setValue(Value* V) {
    assert(!isa<Binding>(V) && "bindings may not nest bindings");
    Val = V;
  }

  Value* Lookup(Symbol* S) {
    if (Name->equals(S)) return this;
    return nullptr;
  }

  bool isSyntactic() const {
    return Val->getKind() == ValueKind::Syntax ||
           Val->getKind() == ValueKind::BuiltinSyntax;
  }


  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Binding;
  }
};

class Module : public ValueBase {
  friend class Context;
  using MapTy = llvm::StringMap<Binding*, AllocatorTy&>;
  // TODO An IdentifierTable would probably be
  //      better than using the strings themselves
  //      as keys.
  MapTy Map;

public:
  Module(AllocatorTy& A)
    : ValueBase(Kind::Module)
    , Map(A)
  { }

  Binding* Insert(Binding* B) {
    Map.insert(std::make_pair(B->getName()->getVal(), B));
    return B;
  }

  // Returns nullptr if not found
  Binding* Lookup(llvm::StringRef Str) {
    return Map.lookup(Str);
  }

  // Returns nullptr if not found
  Binding* Lookup(Symbol* Name) {
    return Lookup(Name->getVal());
  }

  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::Module;
  }

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
class ForwardRef : public ValueBase {
public:
  Value* Val;

  ForwardRef(Value* V)
    : ValueBase(Kind::ForwardRef)
  { }

  static bool classof(ValuePtr V) {
    return V.getKind() == ValueKind::ForwardRef;
  }
};

// isSymbol - For matching symbols in syntax builtins
inline bool isSymbol(ValuePtr V , llvm::StringRef Str) {
  if (Symbol* S = dyn_cast<Symbol>(V)) {
    return S->equals(Str);
  }
  return false;
}

inline SourceLocation getSourceLocation(ValuePtr V) {
  ValueWithSource* VS = nullptr;
  switch (V.getKind()) {
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

inline llvm::StringRef Error::getErrorMessage() {
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
  VISIT_FN(LambdaIr)
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
  RetTy Visit(ValuePtr V, Args&& ...args) {
    switch (V.getKind()) {
    case ValueKind::Undefined:      DISPATCH(Undefined);
    case ValueKind::Binding:        DISPATCH(Binding);
    case ValueKind::Boolean:        DISPATCH(Boolean);
    case ValueKind::Builtin:        DISPATCH(Builtin);
    case ValueKind::BuiltinSyntax:  DISPATCH(BuiltinSyntax);
    case ValueKind::Char:           DISPATCH(Char);
    case ValueKind::Empty:          DISPATCH(Empty);
    case ValueKind::Error:          DISPATCH(Error);
    case ValueKind::Environment:    DISPATCH(Environment);
    case ValueKind::EnvFrame:       DISPATCH(EnvFrame);
    case ValueKind::Exception:      DISPATCH(Exception);
    case ValueKind::Float:          DISPATCH(Float);
    case ValueKind::ForwardRef:     DISPATCH(ForwardRef);
    case ValueKind::Integer:        DISPATCH(Integer);
    case ValueKind::Module:         DISPATCH(Module);
    case ValueKind::Pair:           DISPATCH(Pair);
    case ValueKind::PairWithSource: DISPATCH(PairWithSource);
    case ValueKind::Lambda:         DISPATCH(Lambda);
    case ValueKind::LambdaIr:       DISPATCH(LambdaIr);
    case ValueKind::Quote:          DISPATCH(Quote);
    case ValueKind::String:         DISPATCH(String);
    case ValueKind::Symbol:         DISPATCH(Symbol);
    case ValueKind::Syntax:         DISPATCH(Syntax);
    case ValueKind::Vector:         DISPATCH(Vector);
    default:
      llvm_unreachable("Invalid Value Kind");
    }
  }

#undef DISPATCH
#undef VISIT_FN
};

#define GET_KIND_NAME_CASE(KIND) \
  case ValueKind::KIND: return llvm::StringRef(#KIND, sizeof(#KIND));
inline llvm::StringRef getKindName(heavy::ValueKind Kind) {
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
  GET_KIND_NAME_CASE(LambdaIr)
  GET_KIND_NAME_CASE(Quote)
  GET_KIND_NAME_CASE(String)
  GET_KIND_NAME_CASE(Symbol)
  GET_KIND_NAME_CASE(Syntax)
  GET_KIND_NAME_CASE(Vector)
  default:
    return llvm::StringRef("?????");
  }
}
#undef GET_KIND_NAME_CASE

}

namespace llvm {
  // TODO implement isa_impl for heavy::Value* and Operation
}

#endif
