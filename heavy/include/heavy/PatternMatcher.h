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

#include "heavy/Context.h"
#include "heavy/OpGen.h"
#include "heavy/Value.h"
#include "heavy/ValueVisitor.h"
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

// PatternMatcher
//    - Detects if a pattern matches an input while generating a table
//      for pattern variables.
//    - Generates code to match pattern and bind pattern variables
//      See R7RS 4.3.2 Pattern Language.
class PatternMatcher : ValueVisitor<PatternMatcher, mlir::Value> {
public:
  using KeywordsTy = llvm::ArrayRef<std::pair<Symbol*, Binding*>>>;
private:

  Symbol* Ellipsis;
  Value Keywords; // literal identifiers (list)
  // PatternVars - maps symbols to input nodes
  llvm::StringMap<Value> PatternVars;

  // P - the pattern node
  // E - the input to match

public:
  PatternMatcher(Symbol* Ellipsis,
                 Value Ks)
    : Ellipsis(Ellipsis),
      Keywords(Ks)
  { }

  // returns true if insertion was successful
  mlir::Value bindPatternVar(Symbol* S, mlir::Value E) {
    bool Inserted;
    std::tie(std::ignore, Inserted) = PatternVars.insert({S->getView(), E})
    if (Inserted) return true;

    // TODO set error cuz pattern variable was used twice
    llvm_unreachable("TODO");
    return false;
  }

  mlir::Value VisitValue(Value P, mlir::Value E) {
    // TODO set error invalid pattern node
    llvm_unreachable("TODO");
    return false;
  }

  mlir::Value VisitPair(Pair* P, mlir::Value E) {
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
        // match the previous pattern until
        // we get a mismatch and then continue on to
        // the next pattern
      }
      Result = Visit(P->Car, EP->Car);
      if (!Result) return false;
    }
    return true;
  }

  mlir::Value VisitSymbol(Symbol* P, mlir::Value E) {
    // <pattern identifier> (literal identifier)
    // This is just a linear check against the string ref
    // as we don't have a fancy identifier table yet
    for (Symbol* S : Keywords) {
      // TODO we must perform a lookup to see that
      //      the lookup result is the same
      //      (the same lexical binding or not bound)
      //      Keywords must also store the Binding* result
      if (P->equals(S)) return true;
    }
    // <underscore>
    if (P->equals("_")) {
      return true; 
    }
    // <ellipsis>
    if (P->equals(Ellipsis)) {
      llvm_unreachable("TODO");
      // this is an error as <ellipsis> is not
      // a valid pattern by itself
    }
    // everything else is a pattern variable
    return bindPatternVar(P, E);
  }

  mlir::Value VisitString(String* P, mlir::Value E) {
    // <pattern datum> -> <string>
    return createEqual(createLiteral(P), E);
  }

  mlir::Value VisitBool(Bool P, mlir::Value E) {
    // <pattern datum> -> <boolean>
    return createEqual(createLiteral(P), E);
  }

  mlir::Value VisitInt(Int P, mlir::Value E) {
    // <pattern datum> -> <number>
    return createEqual(createLiteral(P), E);
  }

  mlir::Value VisitFloat(Float* P, mlir::Value E) {
    // <pattern datum> -> <number>
    return createEqual(createLiteral(P), E);
  }
};

}
