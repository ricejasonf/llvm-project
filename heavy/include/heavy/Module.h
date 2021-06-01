//===----- Module.h - Classes for modules and import sets ---*- C++ -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines Module, ImportSet, and ImportSetIterator
//  Definitions reside in Context.cpp
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_MODULE_H
#define LLVM_HEAVY_MODULE_H

#include "heavy/Value.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"

namespace heavy {

class Module : public ValueBase {
  friend class Context;
  using MapTy = llvm::DenseMap<String*, Value>;
  using MapIteratorTy  = typename MapTy::iterator;
  MapTy Map;

public:
  Module()
    : ValueBase(ValueKind::Module)
    , Map()
  { }

  Binding* Insert(Binding* B) {
    Map.insert(std::make_pair(B->getName()->getString(), B));
    return B;
  }

  void Insert(ModuleInitListPairTy P) {
    Map.insert(P);
  }

  // Returns nullptr if not found
  Value Lookup(String* Str) {
    return Map.lookup(Str);
  }

  // Returns nullptr if not found
  Value Lookup(Symbol* Name) {
    return Map.lookup(Name->getString());
  }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::Module;
  }

  class Iterator : public llvm::iterator_facade_base<
                                              Iterator,
                                              std::forward_iterator_tag,
                                              Value>
  {
    friend class Module;
    using ItrTy = typename MapTy::iterator;
    ItrTy Itr;
    Iterator(ItrTy I) : Itr(I) { }

  public:
    Iterator& operator=(Iterator const& R) { Itr = R.Itr; return *this; }
    bool operator==(Iterator const& R) const { return Itr == R.Itr; }
    Value const& operator*() const { return (*Itr).getValue(); }
    Value& operator*() { return (*Itr).getValue(); }
    Iterator& operator++() { ++Itr; return *this; }
  };

  Iterator begin() {
    return Iterator(Map.begin());
  }

  Iterator end() {
    return Iterator(Map.end());
  }

  // Used by ImportSet::Iterator
  auto lookup_begin() { return Map.begin(); }
  auto lookup_end() { return Map.end(); }
};


class ImportSet : public ValueBase {
  friend class ImportSetIterator;

  enum class ImportKind {
    Library,
    Only,
    Except,
    Prefix,
    Rename,
  };

  // Parent - Parent may be nullptr in the case of Library
  ImportSet* Parent = nullptr;
  // Specifier - Refers directly to a subset of the AST for the
  //             import set syntax. Its representation is specific
  //             to the ImportKind and documented in Lookup.
  heavy::Value Specifier;
  ImportKind Kind;

public:
  ImportSet(ImportKind Kind, ImportSet* Parent, Value Specifier)
    : ValueBase(ValueKind::ImportSet),
      Parent(Parent),
      Specifier(Specifier),
      Kind(Kind)
  {
    assert((Parent || Kind == ImportKind::Library)
      && "parent cannot be null unless it is a library");
  }

  static bool classof(Value V) {
    return V.getKind() == ValueKind::ImportSet;
  }

  bool isInIdentiferList(Symbol* S) {
    Value Current = Specifier;
    while (Pair* P = dyn_cast<Pair>(Current)) {
      if (S->equals(cast<Symbol>(P->Car))) return true;
      Current = P->Cdr;
    }
    return false;
  }

  Value LookupFromPairs(heavy::Context& C, Symbol* S);
  Value Lookup(heavy::Context& C, Symbol* S);

  // FilterName - used for iteration of Module members
  //              filtered by import sets
  String* FilterFromPairs(heavy::Context& C, String* S);
  String* FilterName(heavy::Context&, String*);

  // IteratorValueTy - The String* member will be modified
  //                   to reflect the possibly renamed value
  //                   or nullptr if it was filtered out.
  //                   (this should eliminate redundant calls
  //                    to filter name and having to store
  //                    the end() of the map)
  using IteratorValueTy = std::pair<String*, Value>;
  class Iterator : public llvm::iterator_facade_base<
                                              Iterator,
                                              std::forward_iterator_tag,
                                              IteratorValueTy>
  {
    using ValueTy = IteratorValueTy;
    using ItrTy = typename MapTy::iterator;
    heavy::Context& Context; // for String lookup
    ImportSet Filter;
    ItrTy Itr;

    Iterator(heavy::Context& C, ItrTy I, ImportSet Filter)
      : Context(C),
        Filter(Filter),
        Itr(I)
    { }

    // returns the possibly renamed key
    // or nullptr if it is filtered
    String* getName() {
      String* Orig = (*Itr).getFirst();
      return Filter.FilterName(Context, Orig);
    }

    ValueTy getValue() {
      return ValueTy{getName(), (*Itr).getSecond()};
    }

  public:
    Iterator& operator=(Iterator const& R) {
      Context = R.Context
      Filter = R.Filter;
      Itr = R.Itr;
      return *this;
    }

    bool operator==(Iterator const& R) const { return Itr == R.Itr; }
    ValueTy operator*() const { return getValue(); }
    Iterator& operator++() { ++Itr; return *this; }
  };
};

}

#endif
