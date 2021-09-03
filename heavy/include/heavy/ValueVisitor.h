//===---- ValueVisitor.h - Classes for representing declarations ----*- C++ ----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines heavy::ValueVisitor, a CRTP base class recursive
//  visitation of heavy:Value objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_VALUE_VISITOR_H
#define LLVM_HEAVY_VALUE_VISITOR_H

#include "heavy/Value.h"
#include <cassert>
#include <utility>

namespace heavy {
// ValueVisitor
// This will be the base class for evaluation and printing
template <typename Derived, typename RetTy = void>
class ValueVisitor {
#define DISPATCH(NAME) \
  return getDerived().Visit ## NAME(cast<NAME>(V), \
                                    std::forward<Args>(args)...)
#define VISIT_FN(NAME) \
  template <typename ...Args> \
  RetTy Visit ## NAME(cast_ty<NAME> V, Args&& ...args) { \
    return getDerived().VisitValue(V, std::forward<Args>(args)...); }

  Derived& getDerived() { return static_cast<Derived&>(*this); }
  Derived const& getDerived() const { return static_cast<Derived>(*this); }

protected:
  // Derived must implement VisitValue OR all of the
  // concrete visitors
  template <typename T>
  RetTy VisitValue(T* V) = delete;
  template <typename T>
  RetTy VisitValue(T V) = delete;

  // The default implementations for visiting
  // nodes is to call Derived::VisitValue

  VISIT_FN(Undefined)
  VISIT_FN(BigInt)
  VISIT_FN(Binding)
  VISIT_FN(Bool)
  VISIT_FN(Builtin)
  VISIT_FN(BuiltinSyntax)
  VISIT_FN(Char)
  VISIT_FN(Empty)
  VISIT_FN(EnvFrame)
  VISIT_FN(Environment)
  VISIT_FN(Error)
  VISIT_FN(Exception)
  VISIT_FN(Float)
  VISIT_FN(ForwardRef)
  VISIT_FN(Int)
  VISIT_FN(ImportSet)
  VISIT_FN(Lambda)
  VISIT_FN(Module)
  VISIT_FN(Operation)
  VISIT_FN(Pair)
  // VISIT_FN(PairWithSource) **PairWithSource Implemented below**
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
  RetTy Visit(Value V, Args&& ...args) {
    switch (V.getKind()) {
    case ValueKind::Undefined:      DISPATCH(Undefined);
    case ValueKind::BigInt:         DISPATCH(BigInt);
    case ValueKind::Binding:        DISPATCH(Binding);
    case ValueKind::Bool:        DISPATCH(Bool);
    case ValueKind::Builtin:        DISPATCH(Builtin);
    case ValueKind::BuiltinSyntax:  DISPATCH(BuiltinSyntax);
    case ValueKind::Char:           DISPATCH(Char);
    case ValueKind::Empty:          DISPATCH(Empty);
    case ValueKind::EnvFrame:       DISPATCH(EnvFrame);
    case ValueKind::Environment:    DISPATCH(Environment);
    case ValueKind::Error:          DISPATCH(Error);
    case ValueKind::Exception:      DISPATCH(Exception);
    case ValueKind::Float:          DISPATCH(Float);
    case ValueKind::ForwardRef:     DISPATCH(ForwardRef);
    case ValueKind::Int:            DISPATCH(Int);
    case ValueKind::ImportSet:      DISPATCH(ImportSet);
    case ValueKind::Lambda:         DISPATCH(Lambda);
    case ValueKind::Module:         DISPATCH(Module);
    case ValueKind::Operation:      DISPATCH(Operation);
    case ValueKind::Pair:           DISPATCH(Pair);
    case ValueKind::PairWithSource: DISPATCH(PairWithSource);
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
}

#endif
