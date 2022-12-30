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
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/ADT/PointerSumType.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/TrailingObjects.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace mlir {
  class Value;
  class Operation;
}

namespace heavy {
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;
using llvm::isa_and_nonnull;
using mlir::Operation;
struct ContArg { ContArg() = delete; };

// ExternValue - Stores a concrete value in an aligned storage
//               with a Value that points to it. Via StandardLayout,
//               we can have a single symbol point to the value with a
//               way of accessing its storage for initialization.
//               This should be useful for hand coding Scheme
//               modules in C++
template <size_t StorageLen, size_t Alignment = alignof(void*)>
struct ExternValue;

template <typename DerivedT>
void* allocate(llvm::AllocatorBase<DerivedT>& Allocator,
               size_t Size, size_t Alignment) {
  return static_cast<DerivedT&>(Allocator).Allocate(Size, Alignment);
}

template <size_t MaxSize, size_t MaxAlignment>
void* allocate(ExternValue<MaxSize, MaxAlignment>& Val,
//void* allocate(std::aligned_storage_t<MaxSize, MaxAlignment>& Storage,
               size_t Size, size_t Alignment) {
  assert(Size <= MaxSize && "allocation out of bounds for storage");
  assert(MaxAlignment % Alignment == 0 && "improper alignment for storage");
  return &(Val.Storage);
}

// Value - A result of evaluation. This derives from
//         an llvm::PointerSumType to point to values
//         on the heap as well as store small values
//         like integers in an object the size of a pointer.
class Value;

// other forward decls
class OpGen;
class Context;
class Pair;
class Binding;
class Symbol;
class Environment;
class Module;
class ImportSet;
class EnvFrame;
using ValueRefs   = llvm::MutableArrayRef<heavy::Value>;
using ValueFnTy   = void (Context&, ValueRefs);
using ValueFn     = ValueFnTy*;
using SyntaxFn    = mlir::Value (*)(OpGen&, Pair*);
using TransformFn = Value (*)(Context&, Pair*);
using OpaqueFnPtrTy = void(*)(void*, Context&, ValueRefs);
struct OpaqueFn {
  OpaqueFnPtrTy CallFn;
  llvm::StringRef Storage;
};
template <typename F>
static OpaqueFn createOpaqueFn(F& Fn);

enum class ValueKind {
  Undefined = 0,
  BigInt,
  Binding,
  Bool,
  Builtin,
  BuiltinSyntax,
  Char,
  ContArg,
  Empty,
  EnvFrame,
  Environment,
  Error,
  Exception,
  ExternName,
  Float,
  ForwardRef,
  Int,
  ImportSet,
  Lambda,
  Module,
  Operation,
  Pair,
  PairWithSource,
  Quote,
  String,
  Symbol,
  Syntax,
  SyntaxClosure,
  Vector,
};

class alignas(void*) ValueBase {
  friend class Context;
  ValueKind VKind;
  bool IsMutable = false;

protected:
  ValueBase(ValueKind VK)
    : VKind(VK)
  { }

public:
  bool isMutable() const { return IsMutable; }
  ValueKind getKind() const { return VKind; }
  SourceLocation getSourceLocation();
  void dump();
};

// Value types that will be members of Value
struct Undefined {
  static bool classof(Value);
  static ValueKind getKind() { return ValueKind::Undefined; }
};

struct Empty {
  static bool classof(Value);
  static ValueKind getKind() { return ValueKind::Empty; }
};

struct Int : llvm::PointerEmbeddedInt<int32_t> {
  using Base = llvm::PointerEmbeddedInt<int32_t>;
  using Value = heavy::Value;
  using Base::PointerEmbeddedInt;

  // why doesn't the above using-declaration work for this
  Int(Base B) : Base(B) { }

  static bool classof(Value);
  static ValueKind getKind() { return ValueKind::Int; }
};

struct Bool : llvm::PointerEmbeddedInt<bool> {
  using Base = llvm::PointerEmbeddedInt<bool>;
  using Value = heavy::Value;
  using Base::PointerEmbeddedInt;

  Bool(Base B) : Base(B) { }

  static bool classof(Value);
  static ValueKind getKind() { return ValueKind::Bool; }
};

struct Char : llvm::PointerEmbeddedInt<uint32_t> {
  using Base = llvm::PointerEmbeddedInt<uint32_t>;
  using Value = heavy::Value;
  using Base::PointerEmbeddedInt;

  Char(Base B) : Base(B) { }

  static bool classof(Value);
  static ValueKind getKind() { return ValueKind::Char; }
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

template <>
struct PointerLikeTypeTraits<heavy::Bool>
  : PointerLikeTypeTraits<heavy::Bool::Base>
{ };

template <>
struct PointerLikeTypeTraits<heavy::Char>
  : PointerLikeTypeTraits<heavy::Char::Base>
{ };
}

namespace heavy {

struct ValueSumType {
  enum SumKind {
    ValueBase = 0,
    Int,
    Bool,
    Char,
    Empty,
    Undefined,
    Operation,
    ContArg,
  };

  // mlir::Operation* is incomplete so we have to assume
  // its alignment and verify later (in Dialect.h)
  struct OperationTraits {
    static inline void *getAsVoidPointer(mlir::Operation* P) {
      return reinterpret_cast<void *>(P);
    }
    static inline mlir::Operation* getFromVoidPointer(void *P) {
      return reinterpret_cast<mlir::Operation*>(P);
    }

    static constexpr int NumLowBitsAvailable =
      llvm::detail::ConstantLog2<alignof(void*)>::value;
  };

  struct ContArgTraits {
    static inline void *getAsVoidPointer(heavy::ContArg* P) {
      return reinterpret_cast<void *>(P);
    }
    static inline heavy::ContArg* getFromVoidPointer(void *P) {
      return reinterpret_cast<heavy::ContArg*>(P);
    }

    static constexpr int NumLowBitsAvailable =
      llvm::detail::ConstantLog2<alignof(void*)>::value;
  };

  using type = llvm::PointerSumType<SumKind,
    llvm::PointerSumTypeMember<ValueBase,  heavy::ValueBase*>,
    llvm::PointerSumTypeMember<Int,        heavy::Int>,
    llvm::PointerSumTypeMember<Bool,       heavy::Bool>,
    llvm::PointerSumTypeMember<Char,       heavy::Char>,
    llvm::PointerSumTypeMember<Empty,      heavy::Empty>,
    llvm::PointerSumTypeMember<Undefined,  heavy::Undefined>,
    llvm::PointerSumTypeMember<Operation,  mlir::Operation*, OperationTraits>,
    llvm::PointerSumTypeMember<ContArg,    heavy::ContArg*, ContArgTraits>>;
};

using ValuePtrBase = typename ValueSumType::type;

class Value : ValuePtrBase {
public:
  Value() = default;

  Value(ValueBase* V)
    : ValuePtrBase(create<ValueSumType::ValueBase>(V))
  { }

  Value(std::nullptr_t)
    : ValuePtrBase(create<ValueSumType::ValueBase>(nullptr))
  { }

  Value(Undefined)
    : ValuePtrBase(create<ValueSumType::Undefined>({}))
  {
    assert(*this);
  }

  Value(Empty)
    : ValuePtrBase(create<ValueSumType::Empty>({}))
  {
    assert(*this);
  }

  Value(Int I)
    : ValuePtrBase(create<ValueSumType::Int>(I))
  {
    assert(*this);
  }

  Value(Bool B)
    : ValuePtrBase(create<ValueSumType::Bool>(B))
  {
    assert(*this);
  }

  Value(Char C)
    : ValuePtrBase(create<ValueSumType::Char>(C))
  {
    assert(*this);
  }

  Value(ContArg* C)
    : ValuePtrBase(create<ValueSumType::ContArg>(C))
  {
    assert(*this);
  }

  Value(mlir::Operation* Op)
    : ValuePtrBase(create<ValueSumType::Operation>(Op))
  { }

  using ValuePtrBase::getTag;
  using ValuePtrBase::is;
  using ValuePtrBase::get;
  using ValuePtrBase::getOpaqueValue;

  bool operator==(Value V) const {
    return getOpaqueValue() == V.getOpaqueValue();
  }
  bool operator!=(Value V) const {
    return getOpaqueValue() != V.getOpaqueValue();
  }


  // We want this to behave like a pointer so that embedded
  // values are considered true.
  explicit operator bool() const {
    switch (getTag()) {
      case ValueSumType::Int:
      case ValueSumType::Bool:
      case ValueSumType::Char:
      case ValueSumType::Empty:
      case ValueSumType::Undefined:
        return true;
      default:
        return ValuePtrBase::operator bool();
    }
  }

  ValueKind getKind() const {
    switch (getTag()) {
    case ValueSumType::ValueBase:
      return get<ValueSumType::ValueBase>()->getKind();
    case ValueSumType::Int:
      return ValueKind::Int;
    case ValueSumType::Bool:
      return ValueKind::Bool;
    case ValueSumType::Char:
      return ValueKind::Char;
    case ValueSumType::Empty:
      return ValueKind::Empty;
    case ValueSumType::Undefined:
      return ValueKind::Undefined;
    case ValueSumType::Operation:
      return ValueKind::Operation;
    case ValueSumType::ContArg:
      return ValueKind::ContArg;
    }
    llvm_unreachable("cannot get here");
  }

  bool isNumber() {
    return getTag() == ValueSumType::Int ||
           getKind() == ValueKind::Float;
  }

  bool isEmpty() {
    return getKind() == ValueKind::Empty;
  }

  bool isTrue() {
    // returns true for everything except
    // explicit #f (per r7rS)
    if (is<ValueSumType::Bool>())
     return get<ValueSumType::Bool>();
    return true;
  }

  SourceLocation getSourceLocation() {
    if (is<ValueSumType::ValueBase>()) {
      return get<ValueSumType::ValueBase>()
        ->getSourceLocation();
    }
    return SourceLocation();
  }

  // The car/cdr et al  return nullptr if any
  // value is invalid for that accessor
  Value car();
  Value cdr();
  Value cadr();
  Value cddr();

  void dump();
};

inline bool Undefined::classof(Value V) {
  return V.getTag() == ValueSumType::Undefined;
}

inline bool Empty::classof(Value V) {
  return V.getTag() == ValueSumType::Empty;
}

inline bool Int::classof(Value V) {
  return V.getTag() == ValueSumType::Int;
}

inline bool Bool::classof(Value V) {
  return V.getTag() == ValueSumType::Bool;
}

inline bool Char::classof(Value V) {
  return V.getTag() == ValueSumType::Char;
}

}

namespace llvm {
template <typename T>
struct isa_impl<T, ::heavy::Value> {
  static inline bool doit(::heavy::Value V) {
    assert(V && "value cannot be null");
    return T::classof(V);
  }
};

template <>
struct isa_impl<::mlir::Operation, ::heavy::Value> {
  static inline bool doit(::heavy::Value V) {
    return V.is<::heavy::ValueSumType::Operation>();
  }
};

template <>
struct isa_impl<::heavy::ContArg, ::heavy::Value> {
  static inline bool doit(::heavy::Value V) {
    return V.is<::heavy::ValueSumType::ContArg>();
  }
};

template <typename T>
struct cast_retty_impl<T, ::heavy::Value> {
  using ret_type = std::conditional_t<
    std::is_base_of<::heavy::ValueBase, T>::value, T*, T>;
};

template <>
struct cast_retty_impl<::mlir::Operation, ::heavy::Value> {
  using ret_type = ::mlir::Operation*;
};

template <>
struct cast_retty_impl<::heavy::ContArg, ::heavy::Value> {
  using ret_type = ::heavy::ContArg*;
};

template <typename T>
struct cast_convert_val<T, ::heavy::Value,
                           ::heavy::Value> {
  static T* doit(::heavy::Value V) {
    static_assert(std::is_base_of<::heavy::ValueBase, T>::value,
      "should be converting to an instance of ValueBase here");
    return static_cast<T*>(V.get<heavy::ValueSumType::ValueBase>());
  }
};

template <>
struct cast_convert_val<mlir::Operation, ::heavy::Value,
                                         ::heavy::Value> {
  static mlir::Operation* doit(::heavy::Value V) {
    return V.get<heavy::ValueSumType::Operation>();
  }
};

template <>
struct cast_convert_val<::heavy::ContArg, ::heavy::Value,
                                          ::heavy::Value> {
  static ::heavy::ContArg* doit(::heavy::Value V) {
    return V.get<heavy::ValueSumType::ContArg>();
  }
};

template <>
struct cast_convert_val<::heavy::Int, ::heavy::Value,
                                      ::heavy::Value> {
  static auto doit(::heavy::Value V) {
    return V.get<heavy::ValueSumType::Int>();
  }
};

template <>
struct cast_convert_val<::heavy::Char, ::heavy::Value,
                                      ::heavy::Value> {
  static auto doit(::heavy::Value V) {
    return V.get<heavy::ValueSumType::Char>();
  }
};

template <>
struct cast_convert_val<::heavy::Bool, ::heavy::Value,
                                       ::heavy::Value> {
  static auto doit(::heavy::Value V) {
    return V.get<heavy::ValueSumType::Bool>();
  }
};

template <>
struct cast_convert_val<::heavy::Undefined, ::heavy::Value,
                                            ::heavy::Value> {
  static auto doit(::heavy::Value V) {
    return ::heavy::Undefined{};
  }
};
template <>
struct cast_convert_val<::heavy::Empty, ::heavy::Value,
                                        ::heavy::Value> {
  static auto doit(::heavy::Value V) {
    return ::heavy::Empty{};
  }
};


// overloading these gets around annoying
// const& situations that don't allow
// non-simplified types
template <typename T>
inline auto cast(::heavy::Value V) {
  // gets around annoying const& situations
  // that don't allow non-simplified types
  assert(isa<T>(V) && "value must be of compatible type");
  return cast_convert_val<T, ::heavy::Value,
                             ::heavy::Value>::doit(V);
}

template <typename T>
inline T* dyn_cast(::heavy::Value V) {
  // gets around annoying const& situations
  // that don't allow non-simplified types
  assert(V && "value cannot be null");
  return isa<T>(V) ? cast<T>(V) : nullptr;
}

template <typename T>
inline T* dyn_cast_or_null(::heavy::Value V) {
  return (V && isa<T>(V)) ? cast<T>(V) : nullptr;
}

template <typename T>
inline T* cast_or_null(::heavy::Value V) {
  if (!V) return nullptr;
  return cast<T>(V);
}

template <>
struct DenseMapInfo<::heavy::Value> {
  static ::heavy::Value getEmptyKey() { 
    return DenseMapInfo<::heavy::ValueBase*>::getEmptyKey();
  }
  static ::heavy::Value getTombstoneKey() {
    return DenseMapInfo<::heavy::ValueBase*>::getTombstoneKey();
  }

  static unsigned getHashValue(::heavy::Value Arg) {
    uintptr_t OpaqueValue = Arg.getOpaqueValue();
    return DenseMapInfo<uintptr_t>::getHashValue(OpaqueValue);
  }

  static bool isEqual(::heavy::Value LHS, ::heavy::Value RHS) {
    return LHS == RHS;
  }
};

}

namespace heavy {

template <typename To>
using cast_ty = typename llvm::cast_retty<To, Value>::ret_type;

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
  Value Message;
  Value Irritants;
public:

  Error(SourceLocation L, Value M, Value I)
    : ValueBase(ValueKind::Error)
    , ValueWithSource(L)
    , Message(M)
    , Irritants(I)
  { }

  llvm::StringRef getErrorMessage();
  Value getMessage() { return Message; }
  Value getIrritants() { return Irritants; }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::Error;
  }
  static ValueKind getKind() { return ValueKind::Error; }
};

class Exception: public ValueBase {
public:
  Value Val;
  Exception(Value Val)
    : ValueBase(ValueKind::Exception)
    , Val(Val)
  { }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::Exception;
  }
  static ValueKind getKind() { return ValueKind::Exception; }
};

// Base class for Numeric types (other than Int)
class Float;
// TODO Deprecate Number and break out the
//      static function predicates
class Number : public ValueBase {
protected:
  Number(ValueKind K)
    : ValueBase(K)
  { }

public:
  static bool classof(Value V) {
    return V.getKind() == ValueKind::BigInt ||
           V.getKind() == ValueKind::Float;
  }

  static bool isExact(Value V) {
    switch(V.getKind()) {
    case ValueKind::BigInt:
    case ValueKind::Int:
      return true;
    default:
      return false;
    }
  }

  static bool isExactZero(heavy::Value);
  static ValueKind CommonKind(Value X, Value Y);
};

// BigInt currently assumes 64 bits
class BigInt : public Number {
  friend class Number;
  friend class NumberOp;
  llvm::APInt Val;

public:
  BigInt(llvm::APInt V)
    : Number(ValueKind::BigInt)
    , Val(V)
  { }

  auto getVal() { return Val; }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::BigInt;
  }
  static ValueKind getKind() { return ValueKind::BigInt; }
};

class Float : public ValueBase {
  friend class NumberOp;
  llvm::APFloat Val;

public:
  Float(llvm::APFloat V)
    : ValueBase(ValueKind::Float)
    , Val(V)
  { }

  auto getVal() { return Val; }
  static bool classof(Value V) {
    return V.getKind() == ValueKind::Float;
  }
  static ValueKind getKind() { return ValueKind::Float; }
};

inline ValueKind Number::CommonKind(Value X, Value Y) {
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

inline bool Number::isExactZero(Value V) {
  if (!Number::isExact(V)) return false;
  if (BigInt* I = dyn_cast<BigInt>(V)) {
    I->Val == 0;
  }
  return cast<Int>(V) == 0;
}

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
    : ValueBase(ValueKind::String),
      Len(S.size())
  {
    std::memcpy(getTrailingObjects<char>(), S.data(), S.size());
  }

  template <typename ...StringRefs>
  String(unsigned TotalLen, StringRefs ...Ss)
    : ValueBase(ValueKind::String),
      Len(TotalLen)
  {
    std::array<llvm::StringRef, sizeof...(Ss)> Arr = {Ss...};
    char* StrData = getTrailingObjects<char>();
    for (llvm::StringRef S : Arr) {
      std::memcpy(StrData, S.data(), S.size());
      StrData += S.size();
    }
  }

  static constexpr size_t sizeToAlloc(unsigned Length) {
    return totalSizeToAlloc<char>(Length);
  }

  llvm::StringRef getView() const {
    return llvm::StringRef(getTrailingObjects<char>(), Len);
  }

  bool equals(String* S) const {
    return getView().equals(S->getView());
  }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::String;
  }
  static ValueKind getKind() { return ValueKind::String; }
};

// Symbol - Is an identifier with source information
//          and a String which can be unique via an
//          identifier table.
class Symbol : public ValueBase,
               public ValueWithSource {
  String* Val;

public:
  Symbol(String* V, SourceLocation L = SourceLocation())
    : ValueBase(ValueKind::Symbol)
    , ValueWithSource(L)
    , Val(V)
  { }

  using ValueWithSource::getSourceLocation;

  // TODO Deprecate getVal to getView to be consistent.
  llvm::StringRef getVal() const { return Val->getView(); }
  llvm::StringRef getView() const { return Val->getView(); }
  String* getString() const { return Val; }

  bool equals(llvm::StringRef Str) const { return getVal() == Str; }
  bool equals(Symbol* S) const {
    // compare the String* since they are uniqued
    return Val == S->Val;
  }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::Symbol;
  }
  static ValueKind getKind() { return ValueKind::Symbol; }
};

// ExternName - Represent the name of a global variable in terms
//              of its linkage symbol. This is used for the
//              "effective renaming" of variables used in top
//              level syntax functions.
class ExternName : public ValueBase,
                   public ValueWithSource {
  String* Name;

  public:

  ExternName(String* V, SourceLocation L = SourceLocation())
    : ValueBase(ValueKind::ExternName)
    , ValueWithSource(L)
    , Name(V)
  { }

  using ValueWithSource::getSourceLocation;

  llvm::StringRef getView() const {
    return Name->getView();
  }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::ExternName;
  }
  static ValueKind getKind() { return ValueKind::ExternName; }
};


class Pair : public ValueBase {
public:
  Value Car;
  Value Cdr;

  Pair(Value First, Value Second)
    : ValueBase(ValueKind::Pair)
    , Car(First)
    , Cdr(Second)
  { }

  Pair(ValueKind K, Value First, Value Second)
    : ValueBase(K)
    , Car(First)
    , Cdr(Second)
  { }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::Pair ||
           V.getKind() == ValueKind::PairWithSource;
  }
  static ValueKind getKind() { return ValueKind::Pair; }
};

class PairWithSource : public Pair,
                       public ValueWithSource {

public:
  PairWithSource(Value First, Value Second, SourceLocation L)
    : Pair(ValueKind::PairWithSource, First, Second)
    , ValueWithSource(L)
  { }

  // returns the character that opens the pair
  // ( | \{ | [
  char getBraceType() {
    // TODO
    return '(';
  }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::PairWithSource;
  }
  static ValueKind getKind() { return ValueKind::PairWithSource; }
};

class Builtin : public ValueBase {
public:
  ValueFn Fn;

  Builtin(ValueFn F)
    : ValueBase(ValueKind::Builtin)
    , Fn(F)
  { }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::Builtin;
  }
  static ValueKind getKind() { return ValueKind::Builtin; }
};

class BuiltinSyntax : public ValueBase {
public:
  SyntaxFn Fn;

  BuiltinSyntax(SyntaxFn F)
    : ValueBase(ValueKind::BuiltinSyntax)
    , Fn(F)
  { }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::BuiltinSyntax;
  }
  static ValueKind getKind() { return ValueKind::BuiltinSyntax; }
};

class Lambda final
  : public ValueBase,
    private llvm::TrailingObjects<Lambda, Value, char> {

public:
  friend class llvm::TrailingObjects<Lambda, Value, char>;
  friend Context;

private:
  OpaqueFnPtrTy FnPtr;
  unsigned short NumCaptures;
  unsigned short StorageLen;

  size_t numTrailingObjects(OverloadToken<Value> const) const {
    return NumCaptures;
  }

  size_t numTrailingObjects(OverloadToken<char> const) const {
    return StorageLen;
  }

  void* getStoragePtr() { return getTrailingObjects<char>(); }

public:
  Lambda(OpaqueFn FnData, llvm::ArrayRef<Value> Captures)
    : ValueBase(ValueKind::Lambda),
      FnPtr(FnData.CallFn),
      NumCaptures(Captures.size()),
      StorageLen(FnData.Storage.size())
  {
    // Storage
    void const* OrigStorage = FnData.Storage.data();
    size_t StorageLen       = FnData.Storage.size();
    void* StoragePtr = getStoragePtr();
    std::memcpy(StoragePtr, OrigStorage, StorageLen);

    // Captures
    auto CapturesItr = Captures.begin();
    for (heavy::Value& V : getCaptures()) {
      V = *CapturesItr;
      ++CapturesItr;
    }
  }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::Lambda;
  }
  static ValueKind getKind() { return ValueKind::Lambda; }

  static size_t sizeToAlloc(OpaqueFn const& FnData,
                            unsigned short NumCaptures) {
    return totalSizeToAlloc<Value, char>(NumCaptures,
                                         FnData.Storage.size());
  }

  static constexpr size_t sizeToAlloc(unsigned short NumCaptures,
                                      size_t FnStorageLen) {
    return totalSizeToAlloc<Value, char>(NumCaptures,
                                         FnStorageLen);
  }

  template <typename Allocator>
  static void* allocate(Allocator& Alloc, OpaqueFn,
                        llvm::ArrayRef<heavy::Value> Captures);

  size_t getObjectSize() {
    return totalSizeToAlloc<Value, char>(NumCaptures, StorageLen);
  }

  void call(Context& C, ValueRefs Args) {
    return FnPtr(getStoragePtr(), C, Args);
  }

  Value getCapture(unsigned I) {
    return getCaptures()[I];
  }

  llvm::MutableArrayRef<Value> getCaptures() {
    return llvm::MutableArrayRef<Value>(
        getTrailingObjects<Value>(), NumCaptures);
  }
};

class Quote : public ValueBase {
public:
  Value Val;
  Quote(Value V)
    : ValueBase(ValueKind::Quote)
    , Val(V)
  { }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::Quote;
  }
  static ValueKind getKind() { return ValueKind::Quote; }
};

class Syntax final
  : public ValueBase,
    private llvm::TrailingObjects<Syntax, char> {
  friend class llvm::TrailingObjects<Syntax, char>;
  friend Context;

  OpaqueFnPtrTy FnPtr;
  unsigned StorageLen;

  void* getStoragePtr() { return getTrailingObjects<char>(); }

public:
  Syntax(OpaqueFn FnData)
    : ValueBase(ValueKind::Syntax),
      FnPtr(FnData.CallFn),
      StorageLen(FnData.Storage.size())
  {
    // Storage
    void const* OrigStorage = FnData.Storage.data();
    size_t StorageLen       = FnData.Storage.size();
    void* StoragePtr = getStoragePtr();
    std::memcpy(StoragePtr, OrigStorage, StorageLen);
  }

  void call(Context& C, ValueRefs Args) {
    return FnPtr(getStoragePtr(), C, Args);
  }

  static size_t sizeToAlloc(OpaqueFn const& FnData) {
    return totalSizeToAlloc<char>(FnData.Storage.size());
  }
  static constexpr size_t sizeToAlloc(size_t FnStorageLen) {
    return totalSizeToAlloc<char>(FnStorageLen);
  }

  template <typename Allocator>
  static void* allocate(Allocator& Alloc, OpaqueFn);

  static bool classof(Value V) {
    return V.getKind() == ValueKind::Syntax;
  }
  static ValueKind getKind() { return ValueKind::Syntax; }
};

// SyntaxClosure - Wraps an AST node with a captured environment
//                 to lookup names when compiling.
//                 Objects of this type are meant to be
//                 ephemeral and used only during syntax
//                 transformation.
class SyntaxClosure : public ValueBase {
public:
  Value Env;
  Value Node;

  SyntaxClosure(Value Env, Value Node)
    : ValueBase(ValueKind::SyntaxClosure),
      Env(Env),
      Node(Node)
  { }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::SyntaxClosure;
  }
  static ValueKind getKind() { return ValueKind::SyntaxClosure; }
};

class Vector final
  : public ValueBase,
    private llvm::TrailingObjects<Vector, Value> {

  friend class Context;
  friend class llvm::TrailingObjects<Vector, Value>;

  unsigned Len = 0;
  size_t numTrailingObjects(OverloadToken<Value> const) const {
    return Len;
  }

  Vector(llvm::ArrayRef<Value> Vs)
    : ValueBase(ValueKind::Vector),
      Len(Vs.size())
  {
    std::memcpy(getTrailingObjects<Value>(), Vs.data(),
                Len * sizeof(Value));
  }

  Vector(Value V, unsigned N)
    : ValueBase(ValueKind::Vector),
      Len(N)
  {
    Value* Xs = getTrailingObjects<Value>();
    for (unsigned i = 0; i < Len; ++i) {
      Xs[i] = V;
    }
  }

public:
  Value& get(unsigned I) {
    assert(I < Len && "invalid index for vector");
    return *(getTrailingObjects<Value>() + I);
  }

  llvm::ArrayRef<Value> getElements() const {
    return llvm::ArrayRef<Value>(
        getTrailingObjects<Value>(), Len);
  }

  llvm::MutableArrayRef<Value> getElements() {
    return llvm::MutableArrayRef<Value>(
        getTrailingObjects<Value>(), Len);
  }
  static bool classof(Value V) {
    return V.getKind() == ValueKind::Vector;
  }
  static ValueKind getKind() { return ValueKind::Vector; }

  static size_t sizeToAlloc(unsigned Length) {
    return totalSizeToAlloc<Value>(Length);
  }

};

// EnvEntry - Used to store lookup results for
struct EnvEntry {
  heavy::Value Value;
  String* MangledName = nullptr;

  operator bool() const { return bool(Value); }
};
using EnvBucket = std::pair<String*, EnvEntry>;

class Binding : public ValueBase {
  friend class Context;
  Symbol* Name;
  Value Val;

public:

  Binding(Symbol* N, Value V)
    : ValueBase(ValueKind::Binding)
    , Name(N)
    , Val(V)
  { }

  Symbol* getName() {
    return Name;
  }

  bool isNull() { return bool(Val); }

  Value getValue() {
    // A binding with a null value is used to represent
    // unbound symbols (particularly in pattern matching)
    assert(Val && "binding value should not be null");
    return Val;
  }

  void setValue(Value V) {
    assert(!isa<Binding>(V) && "bindings may not nest bindings");
    Val = V;
  }

  EnvEntry Lookup(Symbol* S) {
    assert(Val && "null binding should not be a part of lookup");
    if (Name->equals(S)) return EnvEntry{Value(this)};
    return {};
  }

  bool isSyntactic() {
    return Val.getKind() == ValueKind::Syntax ||
           Val.getKind() == ValueKind::BuiltinSyntax;
  }


  static bool classof(Value V) {
    return V.getKind() == ValueKind::Binding;
  }
  static ValueKind getKind() { return ValueKind::Binding; }
};

// ForwardRef - used for garbage collection
class ForwardRef : public ValueBase {
public:
  Value Val;

  ForwardRef(Value V)
    : ValueBase(ValueKind::ForwardRef)
  { }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::ForwardRef;
  }
  static ValueKind getKind() { return ValueKind::ForwardRef; }
};

// ModuleLoadNamesFn - customization point for dynamically initializing a
//                  module and loading its lookup table for the compiler
using ModuleLoadNamesFn = void(heavy::Context&);
// initModule - Creates a compile-time name/value lookup for importing modules
//              This should be called by the module's import function.
using ModuleInitListPairTy = std::pair<llvm::StringRef, heavy::Value>;
using ModuleInitListTy     = std::initializer_list<ModuleInitListPairTy>;
void initModule(heavy::Context&, llvm::StringRef MangledName,
                ModuleInitListTy InitList);
void registerModuleVar(heavy::Context& C,
                       heavy::Module* M,
                       llvm::StringRef VarSymbol,
                       llvm::StringRef VarId,
                       Value Val);

class Module : public ValueBase {
  friend class Context;
  using MapTy = llvm::DenseMap<String*, EnvEntry>;
  using MapIteratorTy  = typename MapTy::iterator;
  heavy::Context& Context; // for String lookup
  heavy::ModuleLoadNamesFn* LoadNamesFn; // for lazy importing
  // Store global cleanups. Requires garbage collection.
  heavy::Lambda* Cleanup = nullptr;
  MapTy Map;

public:
  Module(heavy::Context& C, heavy::ModuleLoadNamesFn* LoadNamesFn = nullptr)
    : ValueBase(ValueKind::Module),
      Context(C),
      LoadNamesFn(LoadNamesFn),
      Map()
  { }

  ~Module();
  void PushCleanup(heavy::Lambda* CleanupFn);

  heavy::Context& getContext() { return Context; }

  // LoadNames - Idempotently loads names into module
  void LoadNames() {
    if (!LoadNamesFn) return;
    LoadNamesFn(Context);
    LoadNamesFn = nullptr;
  }

  Binding* Insert(Binding* B) {
    Map.insert(EnvBucket(B->getName()->getString(), EnvEntry{B}));
    return B;
  }

  void Insert(EnvBucket Bucket) {
    Map.insert(Bucket);
  }

  // Returns EnvEntry() if not found
  EnvEntry Lookup(String* Str) {
    return Map.lookup(Str);
  }

  // Returns EnvEntry() if not found
  EnvEntry Lookup(Symbol* Name) {
    return Map.lookup(Name->getString());
  }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::Module;
  }
  static ValueKind getKind() { return ValueKind::Module; }

  class ValueIterator : public llvm::iterator_facade_base<
                                              ValueIterator,
                                              std::forward_iterator_tag,
                                              Value>
  {
    friend class Module;
    using ItrTy = typename MapTy::iterator;
    ItrTy Itr;
    ValueIterator(ItrTy I) : Itr(I) { }

  public:
    ValueIterator& operator=(ValueIterator const& R)
    { Itr = R.Itr; return *this; }
    bool operator==(ValueIterator const& R) const { return Itr == R.Itr; }
    Value const& operator*() const { return (*Itr).second.Value; }
    Value& operator*() { return (*Itr).second.Value; }
    ValueIterator& operator++() { ++Itr; return *this; }
  };

  ValueIterator values_begin() {
    return ValueIterator(Map.begin());
  }

  ValueIterator values_end() {
    return ValueIterator(Map.end());
  }

  using iterator = typename MapTy::iterator;
  // Used by ImportSet::Iterator
  auto begin() { return Map.begin(); }
  auto end() { return Map.end(); }
};

class ImportSet : public ValueBase {
public:
  enum class ImportKind {
    Library,
    Only,
    Except,
    Prefix,
    Rename,
  };

private:
  // Parent - Parent may be nullptr in the case of Library
  ImportSet* Parent = nullptr;
  // Specifier - Refers directly to a subset of the AST for the
  //             import set syntax. Its representation is specific
  //             to the ImportKind and documented in Lookup.
  heavy::Value Specifier;
  ImportKind Kind;

  // FilterName - used for iteration of Module members
  //              filtered by import sets
  String* FilterFromPairs(heavy::Context& C, String* S);
  String* FilterName(heavy::Context&, String*);
  EnvEntry LookupFromPairs(heavy::Context& C, Symbol* S);

  // recurse to the Library import set
  Module* getModule() {
    if (!Parent) return cast<Module>(Specifier);
    return Parent->getModule();
  }

public:
  ImportSet(ImportKind Kind, ImportSet* Parent, Value Specifier)
    : ValueBase(ValueKind::ImportSet),
      Parent(Parent),
      Specifier(Specifier),
      Kind(Kind)
  {
    assert((Specifier && Parent && Kind != ImportKind::Library)
      && "parent cannot be null unless it is a library");
  }

  ImportSet(Module* M)
    : ValueBase(ValueKind::ImportSet),
      Parent(nullptr),
      Specifier(M),
      Kind(ImportKind::Library)
  { }

  EnvEntry Lookup(heavy::Context& C, Symbol* S);

  static bool classof(Value V) {
    return V.getKind() == ValueKind::ImportSet;
  }
  static ValueKind getKind() { return ValueKind::ImportSet; }

  bool isInIdentiferList(String* S) {
    Value Current = Specifier;
    while (Pair* P = dyn_cast<Pair>(Current)) {
      String* S2 = cast<Symbol>(P->Car)->getString();
      if (S == S2) return true;
      Current = P->Cdr;
    }
    return false;
  }

  bool isInIdentiferList(Symbol* S) {
    return isInIdentiferList(S->getString());
  }

  // The String* member of EnvBucket will
  // be modified to reflect the possibly renamed
  // value or nullptr if it was filtered out.
  // (this should eliminate redundant calls
  // to filter name and having to store the end()
  // of the map)
  class Iterator : public llvm::iterator_facade_base<
                                              Iterator,
                                              std::forward_iterator_tag,
                                              EnvBucket>
  {
    friend class ImportSet;
    using ItrTy = typename Module::iterator;
    heavy::Context& Context; // for String lookup
    ImportSet& Filter;
    ItrTy Itr;

    Iterator(heavy::Context& C, ItrTy I, ImportSet& Filter)
      : Context(C),
        Filter(Filter),
        Itr(I)
    { }

    // returns the possibly renamed key
    // or nullptr if it is filtered
    String* getName() const {
      String* Orig = (*Itr).getFirst();
      return Filter.FilterName(Context, Orig);
    }

    EnvBucket getValue() const {
      return EnvBucket{getName(), (*Itr).getSecond()};
    }

  public:
    Iterator& operator=(Iterator const& R) {
      //Context = R.Context
      Filter = R.Filter;
      Itr = R.Itr;
      return *this;
    }

    bool operator==(Iterator const& R) const { return Itr == R.Itr; }
    EnvBucket operator*() const { return getValue(); }
    Iterator& operator++() { ++Itr; return *this; }
  };

  Iterator begin() {
    // recurse to get the Module to get the iterator
    Module* M = getModule();
    heavy::Context& C = M->getContext();
    return Iterator(C, M->begin(), *this);
  }

  Iterator end() {
    Module* M = getModule();
    heavy::Context& C = M->getContext();
    return Iterator(C, M->end(), *this);
  }
};

// Environment
//  - EnvMap is where we put all imported variables
//  - EnvStack allows shadowed underlying layers such as
//    core syntax or embedded environments
//  - Represents an Environment Specifier created with (environment ...)
//    or the default or embedded environments
//  - Adding top level definitions that shadow names in EnvMap
//    is forbidden
class Environment : public ValueBase {
  friend class Context;

private:
  using MapTy = llvm::DenseMap<String*, EnvEntry>;

  std::unique_ptr<heavy::OpGen> OpGen;
  Environment* Parent = nullptr;
  MapTy EnvMap;

public:
  // Implemented in Context.cpp
  Environment(Environment* Parent);
  Environment(heavy::Context& C, std::string ModulePrefix = {});
  ~Environment();

  heavy::OpGen* GetOpGen() {
    if (OpGen) return OpGen.get();
    return Parent->GetOpGen();
  }

  // Returns nullptr if not found
  EnvEntry Lookup(String* Str) {
    EnvEntry Result = EnvMap.lookup(Str);
    if (Result) return Result;
    if (Parent) return Parent->Lookup(Str);
    return {};
  }

  // Returns EnvEntry() if not found
  EnvEntry Lookup(Symbol* Name) {
    return Lookup(Name->getString());
  }

  static BuiltinSyntax* getImportSyntax();

  // ImportValue returns false if the name already exists
  bool ImportValue(EnvBucket X) {
    assert(X.first && "name should point to a string in identifier table");
    assert(X.second.MangledName && "import requires external name");
    return EnvMap.insert(X).second;
  }

  // Insert - Add a named mutable location. Overwriting
  //          with a new object is okay here if it is mutable
  void Insert(Binding* B, String* MangledName) {
    String* Name = B->getName()->getString();
    auto& Entry = EnvMap[Name];
    assert(!Entry.MangledName &&
        "insert may not modify immutable locations");
    Entry.Value = B;
    Entry.MangledName = MangledName;
  }

  // SetSyntax - Extend the syntactic environment.
  void SetSyntax(Symbol* Name, Syntax* S) {
    EnvMap[Name->getString()] = EnvEntry{Value(S)};
  }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::Environment;
  }
  static ValueKind getKind() { return ValueKind::Environment; }
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
  EnvEntry Lookup(Symbol* Name) const;

  static bool classof(Value V) {
    return V.getKind() == ValueKind::EnvFrame;
  }
  static ValueKind getKind() { return ValueKind::EnvFrame; }
};

inline EnvEntry EnvFrame::Lookup(Symbol* Name) const {
  // linear search
  for (Binding* B : getBindings()) {
    if (Name->equals(B->getName())) return EnvEntry{B};
  }
  return {};
}


// isSymbol - For matching symbols in syntax builtins
inline bool isSymbol(Value V , llvm::StringRef Str) {
  if (Symbol* S = dyn_cast<Symbol>(V)) {
    return S->equals(Str);
  }
  return false;
}

inline SourceLocation ValueBase::getSourceLocation() {
  Value Self(this);
  ValueWithSource* VS = nullptr;
  switch (getKind()) {
  case ValueKind::Error:
    VS = cast<Error>(Self);
    break;
  case ValueKind::Symbol:
    VS = cast<Symbol>(Self);
    break;
  case ValueKind::PairWithSource:
    VS = cast<PairWithSource>(Self);
    break;
  default:
    return SourceLocation();
  }
  return VS->getSourceLocation();
}

inline Value Value::car() {
  if (Pair* P = dyn_cast<Pair>(*this))
    return P->Car;
  return nullptr;
}

inline Value Value::cdr() {
  if (Pair* P = dyn_cast<Pair>(*this))
    return P->Cdr;
  return nullptr;
}

inline Value Value::cadr() {
  if (Pair* P = dyn_cast<Pair>(*this))
    if (Pair* P2 = dyn_cast<Pair>(P->Cdr))
      return P2->Car;
  return nullptr;
}

inline Value Value::cddr() {
  if (Pair* P = dyn_cast<Pair>(*this))
    if (Pair* P2 = dyn_cast<Pair>(P->Cdr))
      return P2->Cdr;
  return nullptr;
}

inline llvm::StringRef Error::getErrorMessage() {
  if (String* S = dyn_cast<String>(Message)) {
    return  S->getView();
  }
  return "Unknown error (invalid error message)";
}

#define GET_KIND_NAME_CASE(KIND) \
  case ValueKind::KIND: return llvm::StringRef(#KIND, sizeof(#KIND));
inline llvm::StringRef getKindName(heavy::ValueKind Kind) {
  switch (Kind) {
  GET_KIND_NAME_CASE(Undefined)
  GET_KIND_NAME_CASE(BigInt)
  GET_KIND_NAME_CASE(Binding)
  GET_KIND_NAME_CASE(Bool)
  GET_KIND_NAME_CASE(Builtin)
  GET_KIND_NAME_CASE(BuiltinSyntax)
  GET_KIND_NAME_CASE(Char)
  GET_KIND_NAME_CASE(Empty)
  GET_KIND_NAME_CASE(EnvFrame)
  GET_KIND_NAME_CASE(Environment)
  GET_KIND_NAME_CASE(Error)
  GET_KIND_NAME_CASE(Exception)
  GET_KIND_NAME_CASE(ExternName)
  GET_KIND_NAME_CASE(Float)
  GET_KIND_NAME_CASE(ForwardRef)
  GET_KIND_NAME_CASE(Int)
  GET_KIND_NAME_CASE(ImportSet)
  GET_KIND_NAME_CASE(Lambda)
  GET_KIND_NAME_CASE(Module)
  GET_KIND_NAME_CASE(Operation)
  GET_KIND_NAME_CASE(Pair)
  GET_KIND_NAME_CASE(PairWithSource)
  GET_KIND_NAME_CASE(Quote)
  GET_KIND_NAME_CASE(String)
  GET_KIND_NAME_CASE(Symbol)
  GET_KIND_NAME_CASE(Syntax)
  GET_KIND_NAME_CASE(SyntaxClosure)
  GET_KIND_NAME_CASE(Vector)
  default:
    return llvm::StringRef("?????");
  }
}
#undef GET_KIND_NAME_CASE

// implemented in Context.cpp
bool equal_slow(Value V1, Value V2);
bool eqv_slow(Value V1, Value V2);

inline bool eqv(Value V1, Value V2) {
  if (V1 == V2) return true;
  if (V1.getKind() != V2.getKind()) return false;
  return eqv_slow(V1, V2);
}

inline bool equal(Value V1, Value V2) {
  if (V1 == V2) return true;
  if (V1.getKind() != V2.getKind()) return false;
  return equal_slow(V1, V2);
}

template <typename F>
OpaqueFn createOpaqueFn(F& Fn) {
  // The way the llvm::TrailingObjects<Lambda, Value, char> works
  // is that the storage pointer (via char) has the same alignment as
  // its previous trailing objects' type `Value` which is pointer-like.
  // Types requiring greater alignments probably wouldn't work
  // (though I am not exactly sure what would happen)
  static_assert(alignof(F) <= alignof(Value),
    "function storage alignment is too large");

  static_assert(std::is_trivially_copyable<F>::value,
    "F must be trivially_copyable");
  using FuncTy = std::remove_const_t<F>;
  auto CallFn = [](void* Storage, Context& C, ValueRefs Values) {
    FuncTy& Func = *static_cast<FuncTy*>(Storage);
    // llvm::errs() << "Func: " << __PRETTY_FUNCTION__ << '\n';
    Func(C, Values);
  };
  llvm::StringRef Storage(reinterpret_cast<char const*>(&Fn), sizeof(Fn));
  return OpaqueFn{CallFn, Storage};
}

template <typename Allocator>
void* Lambda::allocate(Allocator& Alloc, OpaqueFn FnData,
                       llvm::ArrayRef<heavy::Value> Captures) {
  size_t size = Lambda::sizeToAlloc(FnData, Captures.size());
  return heavy::allocate(Alloc, size, alignof(Lambda));
}

template <typename Allocator>
void* Syntax::allocate(Allocator& Alloc, OpaqueFn FnData) {
  size_t size = Syntax::sizeToAlloc(FnData);
  return heavy::allocate(Alloc, size, alignof(Syntax));
}

template <size_t StorageLen, size_t Alignment>
struct ExternValue {
  static constexpr size_t storage_len = StorageLen;
  heavy::Value Value;
  std::aligned_storage_t<StorageLen, Alignment> Storage;

  ExternValue() = default;
  ExternValue(ExternValue const&) = delete;

  operator heavy::Value() { return Value; }
};

// FIXME Add storage size for type erased functions.
template <size_t CaptureCount, size_t FnStorageLen = sizeof(void*)>
struct ExternLambda : public ExternValue<
        Lambda::sizeToAlloc(CaptureCount, FnStorageLen),
        alignof(Lambda)> {
  // Take invocable object and allocates
  // it as a type-erased Scheme Lambda.
  // Must invoke with OpaqueFn
  template <typename F>
  void operator=(F f) {
    static_assert(Lambda::sizeToAlloc(CaptureCount, FnStorageLen) >=
                  Lambda::sizeToAlloc(CaptureCount, sizeof(f)),
        "storage must have sufficient size");
    auto FnData = createOpaqueFn(f);
    void* Mem = Lambda::allocate(*this, FnData, /*Captures=*/{});
    Lambda* New = new (Mem) Lambda(FnData, /*Captures=*/{});

    this->Value = New;
  }
};

struct ExternFunction : ExternValue<sizeof(Builtin)> {
  void operator=(heavy::ValueFn Fn) {
    void* Mem = heavy::allocate(*this, sizeof(Builtin),
                                alignof(Builtin));
    Builtin* New = new (Mem) Builtin(Fn);
    this->Value = New;
  }
};
struct ExternBuiltinSyntax : ExternValue<sizeof(BuiltinSyntax)> {
  void operator=(heavy::SyntaxFn Fn) {
    void* Mem = heavy::allocate(*this, sizeof(BuiltinSyntax),
                                alignof(BuiltinSyntax));
    BuiltinSyntax* New = new (Mem) BuiltinSyntax(Fn);
    this->Value = New;
  }
};
template <size_t FnStorageLen = sizeof(void*)>
struct ExternSyntax : ExternValue<sizeof(Syntax) + FnStorageLen> {
  // Take invocable object and allocates
  // it as a type-erased Scheme syntax function.
  // Must invoke with OpaqueFn
  template <typename F>
  void operator=(F f) {
    static_assert(Syntax::sizeToAlloc(FnStorageLen) >=
                  Syntax::sizeToAlloc(sizeof(f)),
        "storage must have sufficient size");
    auto FnData = createOpaqueFn(f);
    void* Mem = Syntax::allocate(*this, FnData);
    Syntax* New = new (Mem) Syntax(FnData);

    this->Value = New;
  }
};

// ExternString - Used for the symbol name of "import"
template <size_t Len>
struct ExternString : public ExternValue<String::sizeToAlloc(Len)> {
  void operator=(llvm::StringRef Str) {
    void* Mem = heavy::allocate(*this, String::sizeToAlloc(Len),
                                alignof(String));
    String* New = new (Mem) String(Len, Str);
    this->Value = New;
  }

  operator String*() {
    return cast<String>(this->Value);
  }
};
}

#endif
