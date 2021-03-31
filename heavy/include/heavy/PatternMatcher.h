//===-- PatternMatcher.h - Class for Syntax Pattern Matching ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines heavy::PatternMatcher that maps Heavy Values to Values
//  for user defined syntax transformations via the `syntax-rules` syntax.
//
//  The result Value should be the matching template.
//
//===----------------------------------------------------------------------===//

#include "heavy/HeavyScheme.h"
#include "heavy/OpGen.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SmallVector.h"

namespace heavy {

#if 0

// stuff to handle

TransformerSpec
SyntaxRule
Pattern
Template
TemplateElement

#endif

class PatternMatcher : ValueVisitor<PatternMatcher, Value> {
  using KeywordRange = llvm::ArrayRef<Symbol*>;

  Symbol* Ellipsis;
  llvm::ArrayRef<Symbol*> Keywords;

  // P - the pattern node
  // E - the input to match

public:
  PatternMatcher(Symbol* Ellipsis,
                 llvm::ArrayRef<Symbol*> Ks)
    : Ellipsis(Ellipsis),
      Keywords(Ks)
  { }

  // returns true if insertion was successful
  bool bindPatternVar(Symbol* S, Value E) {
    if (PatternVars.lookup(S, E)) {
      // TODO set error cuz pattern variable was used twice
      return false;
    }
    PatternVars.insert(S, E);
    return true;
  }

  bool VisitValue(Value P, Value E) {
    // TODO set error invalid pattern node
    llvm_unreachable("TODO");
    return false;
  }

  bool VisitPair(Pair* P, Value E) {
    // (<pattern>*)
    // (<pattern>* <pattern> <ellipsis> <pattern>*)
    if (!isa<Pair>(E)) {
      // doesn't match
      return false;
    }
    unsigned PLen = P->getLength();
    unsigned ELen = P->getLength();
    bool Result = Visit(P->Car, cast<Pair>(E)->Car);

    // FIXME we need to allow improper lists
    while (Pair* P = dyn_cast<Pair>(P->Cdr)) {
      if (isSymbol(P->Car, Ellipsis)) {
        // TODO
        // not counting the ellipsis itself,
        // we need to walk the list to get where
        // we start matching the rest of the P
        // list against the corresponding nodes in E
      }
      Result = Visit(P->Car, EP->Car);
      if (!Result) return false;
    }
    return true;
  }

  bool VisitSymbol(Symbol* P, Value E) {
    // <pattern identifier> (literal identifier)
    // This is just a linear check against the string ref
    // as we don't have a fancy identifier table yet
    for (Symbol* S : Keywords) {
      if (P->equals(S)) return true;
    }
    // <underscore>
    if (P->equals("_")) {
      return true; 
    }
    // <ellipsis>
    if (P->equals(Ellipsis)) {
      llvm_unreachable("TODO");
      // this is an  error as <ellipsis> is not
      // a valid pattern by itself
    }
    // everything else is a pattern variable
    return bindPatternVar(P, E);
  }

  bool VisitString(String* P, Value E) {
    // <pattern datum> -> <string>
    return equal(P, E);
  }

  bool VisitBool(Bool P, Value E) {
    // <pattern datum> -> <boolean>
    return equal(P, E);
  }

  bool VisitInt(Int P, Value E) {
    // <pattern datum> -> <number>
    return equal(P, E);
  }

  bool VisitFloat(Float* P, Value E) {
    // <pattern datum> -> <number>
    return equal(P, E);
  }
};

}
