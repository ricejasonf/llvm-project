//===- Builtins.cpp - Builtin functions for HeavyScheme ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines builtins and builtin syntax for HeavyScheme
//
//===----------------------------------------------------------------------===//

#include "heavy/Builtins.h"
#include "heavy/Context.h"
#include "heavy/OpGen.h"
#include "llvm/Support/Casting.h"
#include "memory"

bool HEAVY_BASE_IS_LOADED = false;

heavy::ExternSyntax<>      HEAVY_BASE_VAR(define_library);
heavy::ExternSyntax<>      HEAVY_BASE_VAR(begin);
heavy::ExternSyntax<>      HEAVY_BASE_VAR(export);
heavy::ExternSyntax<>      HEAVY_BASE_VAR(include);
heavy::ExternSyntax<>      HEAVY_BASE_VAR(include_ci);
heavy::ExternSyntax<>      HEAVY_BASE_VAR(include_library_declarations);
heavy::ContextLocal        HEAVY_BASE_VAR(parse_source_file);

heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(cond_expand);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(define);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(define_syntax);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(syntax_rules);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(if);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(lambda);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(quasiquote);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(quote);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(set);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(source_loc);


heavy::ExternFunction HEAVY_BASE_VAR(add);
heavy::ExternFunction HEAVY_BASE_VAR(sub);
heavy::ExternFunction HEAVY_BASE_VAR(div);
heavy::ExternFunction HEAVY_BASE_VAR(mul);
heavy::ExternFunction HEAVY_BASE_VAR(gt);
heavy::ExternFunction HEAVY_BASE_VAR(lt);
heavy::ExternFunction HEAVY_BASE_VAR(list);
heavy::ExternFunction HEAVY_BASE_VAR(append);
heavy::ExternFunction HEAVY_BASE_VAR(dump);
heavy::ExternFunction HEAVY_BASE_VAR(eq);
heavy::ExternFunction HEAVY_BASE_VAR(equal);
heavy::ExternFunction HEAVY_BASE_VAR(eqv);
heavy::ExternFunction HEAVY_BASE_VAR(callcc);
heavy::ExternFunction HEAVY_BASE_VAR(with_exception_handler);
heavy::ExternFunction HEAVY_BASE_VAR(raise);
heavy::ExternFunction HEAVY_BASE_VAR(error);

heavy::ExternFunction HEAVY_BASE_VAR(eval);
heavy::ExternFunction HEAVY_BASE_VAR(op_eval);
heavy::ExternFunction HEAVY_BASE_VAR(compile);

namespace heavy { namespace base {

mlir::Value define(OpGen& OG, Pair* P) {
  Pair*   P2    = dyn_cast<Pair>(P->Cdr);
  Symbol* S     = nullptr;
  if (!P2) return OG.SetError("invalid define syntax", P);
  if (Pair* LambdaSpec = dyn_cast<Pair>(P2->Car)) {
    S = dyn_cast<Symbol>(LambdaSpec->Car);
  } else {
    S = dyn_cast<Symbol>(P2->Car);
  }
  if (!S) return OG.SetError("invalid define syntax", P);
  return OG.createDefine(S, P2, P);
}

mlir::Value define_syntax(OpGen& OG, Pair* P) {
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  if (!P2) return OG.SetError("invalid define-syntax syntax", P);
  Symbol* S = dyn_cast<Symbol>(P2->Car);
  if (!S) return OG.SetError("expecting name for define-syntax", P);

  return OG.createSyntaxSpec(P2, P);
}

mlir::Value syntax_rules(OpGen& OG, Pair* P) {
  // The input is the <Syntax Spec> (Keyword (syntax-rules ...))
  // <Syntax Spec> has its own checks in createSyntaxSpec
  Symbol* Keyword = dyn_cast<Symbol>(P->Car);
  Pair* SpecInput = dyn_cast<Pair>(P->Cdr.car().cdr());
  if (!SpecInput) return OG.SetError("invalid syntax-rules syntax", P);
  // Check for optional ellipsis identifier.
  Symbol* Ellipsis = dyn_cast<Symbol>(SpecInput->Car);
  if (Ellipsis) {
    Pair* Temp = dyn_cast<Pair>(SpecInput->Cdr);
    if (!Temp) return OG.SetError("invalid syntax-rules syntax.", SpecInput);
    SpecInput = Temp;
  } else {
    Ellipsis = OG.getContext().CreateSymbol("...");
  }
  return OG.createSyntaxRules(P->getSourceLocation(), Keyword, Ellipsis,
                              SpecInput->Car, SpecInput->Cdr);
}

mlir::Value lambda(OpGen& OG, Pair* P) {
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  Value Formals = P2->Car;
  Pair* Body = dyn_cast<Pair>(P2->Cdr);

  return OG.createLambda(Formals, Body, P->getSourceLocation());
}

mlir::Value if_(OpGen& OG, Pair* P) {
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  if (!P2) return OG.SetError("invalid if syntax", P);
  Value CondExpr = P2->Car;
  P2 = dyn_cast<Pair>(P2->Cdr);
  if (!P2) return OG.SetError("invalid if syntax", P);
  Value ThenExpr = P2->Car;
  P2 = dyn_cast<Pair>(P2->Cdr);
  if (!P2) return OG.SetError("invalid if syntax", P);
  Value ElseExpr = P2->Car;
  if (!isa<Empty>(P2->Cdr)) {
    return OG.SetError("invalid if syntax", P);
  }
  return OG.createIf(P->getSourceLocation(), CondExpr,
                     ThenExpr, ElseExpr);
}

mlir::Value set(OpGen& OG, Pair* P) {
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  if (!P2) return OG.SetError("invalid set syntax", P);
  Value S = P2->Car;
  if (!isa<Symbol, ExternName>(S))
    return OG.SetError("expecting symbol", P2);
  P2 = dyn_cast<Pair>(P2->Cdr);
  if (!P2) return OG.SetError("invalid set syntax", P2);
  heavy::Value Expr = P2->Car;
  if (!isa<Empty>(P2->Cdr)) return OG.SetError("invalid set syntax", P2);
  return OG.createSet(P->getSourceLocation(), S, Expr);
}

namespace {
  void import_helper(Context& C, ValueRefs Args) {
    if (Pair* P = dyn_cast<Pair>(Args[0])) {
      C.PushCont([](Context& C, ValueRefs) {
        Value Callee = C.getCapture(0);
        Value RecurseArgs[] = {C.getCapture(1)};
        C.Apply(Callee, RecurseArgs);
      }, CaptureList{C.getCallee(), P->Cdr});
      C.PushCont([](Context& C, ValueRefs Args) {
        auto* ImpSet = cast<ImportSet>(Args[0]);
        C.Import(ImpSet);
      }, CaptureList{});
      C.CreateImportSet(P->Car);
    } else if (isa<Empty>(Args[0])) {
      C.Cont();
    } else {
      C.SetError("expecting proper list for import syntax", Args[0]);
    }
  }
}
void import_(Context& C, ValueRefs Args) {
  Value ImportSpecs = cast<Pair>(Args[0])->Cdr;
  if (C.OpGen->isLibraryContext()) {
    C.OpGen->WithLibraryEnv(C.CreateLambda([](Context& C, ValueRefs) {
      Value ImportSpecs = C.getCapture(0);
      C.Apply(C.CreateLambda(import_helper, {}), ImportSpecs);
    }, CaptureList{ImportSpecs}));
  } else {
    C.Apply(C.CreateLambda(import_helper, {}), ImportSpecs);
  }
}

void export_(Context& C, ValueRefs Args) {
  if (!C.OpGen->isLibraryContext()) {
    return C.RaiseError("export must be in library context", Args[0]);
  }

  Value Input = cast<Pair>(Args[0])->Cdr;

  C.OpGen->WithLibraryEnv(C.CreateLambda([](Context& C, ValueRefs Args) {
    C.OpGen->Export(C.getCapture(0));
  }, CaptureList{Input}));
}

void define_library(Context& C, ValueRefs Args) {
  OpGen& OG = *C.OpGen;
  Pair* P = cast<Pair>(Args[0]);
  auto Loc = P->getSourceLocation();
  if (!OG.isTopLevel() || OG.isLibraryContext()) {
    C.SetError("unexpected library definition", P);
    return;
  }
  Value NameSpec;
  Pair* LibraryDecls;
  if (Pair* P2 = dyn_cast<Pair>(P->Cdr)) {
    NameSpec = P2->Car;
    LibraryDecls = dyn_cast<Pair>(P2->Cdr);
    if (!LibraryDecls) {
      C.SetError("expected library declarations", P2->Cdr);
      return;
    }
  }
  std::string MangledName = OG.mangleModule(NameSpec);
  if (MangledName.size() == 0) {
    C.SetError("library name is invalid");
    return;
  }
  OG.VisitLibrary(Loc, std::move(MangledName), LibraryDecls);
}

mlir::Value cond_expand(OpGen& OG, Pair* P) {
  return OG.SetError("TODO cond_expand", P);
}

namespace {
  void handleSequence(Context&C, heavy::SourceLocation Loc,
                                 heavy::Value Sequence) {
    OpGen& OG = *C.OpGen;
    if (OG.isTopLevel()) {
      OG.VisitTopLevelSequence(Sequence);
    } else {
      mlir::Value Result = OG.createSequence(Loc, Sequence);
      if (C.CheckError()) return;
      C.Cont(OpGen::fromValue(Result));
    }
  }
}

void begin(Context& C, ValueRefs Args) {
  Pair* P = cast<Pair>(Args[0]);
  handleSequence(C, P->getSourceLocation(), P->Cdr);
}

void include_(Context& C, ValueRefs Args) {
  heavy::SourceLocation Loc = Args[0].getSourceLocation();
  C.PushCont([Loc](Context& C, ValueRefs Args) {
    handleSequence(C, Loc, Args[0]);
  });

  auto* P = cast<Pair>(Args[0]);
  auto* P2 = cast<Pair>(P->Cdr);

  heavy::Value ParseSourceFile = HEAVY_BASE_VAR(parse_source_file).get(C);
  heavy::Value SourceVal = C.CreateSourceValue(Loc);
  heavy::Value Filename = C.RebuildLiteral(P2->Car);
  std::array<heavy::Value, 2> NewArgs = {SourceVal, Filename};
  C.Apply(ParseSourceFile, NewArgs);
}

void include_ci(Context& C, ValueRefs Args) {
  // TODO We need to dynamic-wind and set the
  //      context flag for case insensitive parsing
  //      and lookup. Then do what `include` does.
  C.RaiseError("TODO include-ci", Args[0]);
}

void include_library_declarations(Context& C, ValueRefs Args) {
  C.RaiseError("TODO include-library-declarations", Args[0]);
}

void parse_source_file(Context& C, ValueRefs Args) {
  // The user must use HeavyScheme::setParseSourceFileFn
  // to define how source files are loaded and stored.
  C.RaiseError("parse-source-file is undefined");
}

mlir::Value source_loc(OpGen& OG, Pair* P) {
  // Convert an unevaluated value into a SourceValue.
  // (or rather the operation to do this).
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  if (!P2 || !isa<Empty>(P2->Cdr)) {
    return OG.SetError("single argument required", P);
  }
  // Defer creating a SourceValue until evaluation.
  heavy::SourceLocation Loc = P2->Car.getSourceLocation();
  auto SourceLocOp = OG.create<heavy::SourceLocOp>(Loc);
  return SourceLocOp.getResult();
}

}} // end of namespace heavy::base

namespace heavy {
// TODO Replace NumberOp here with corresponding arithmetic ops in OpGen and OpEval
struct NumberOp {
  // These operations always mutate the first operand
  struct Add {
    static Int f(Context& C, Int X, Int Y) { return X + Y; }
    static void f(Context& C, Float* X, Value Y) {
      X->Val = llvm::APFloat(
          X->Val.convertToDouble() + Number::getAsDouble(Y));
    }
  };
  struct Sub {
    static Int f(Context& C, Int X, Int Y) { return X - Y; }
    static void f(Context& C, Float* X, Value Y) {
      X->Val = llvm::APFloat(
          X->Val.convertToDouble() - Number::getAsDouble(Y));
    }
  };
  struct Mul {
    static Int f(Context& C, Int X, Int Y) { return X * Y; }
    static void f(Context& C, Float* X, Value Y) {
      X->Val = llvm::APFloat(
          X->Val.convertToDouble() * Number::getAsDouble(Y));
    }
  };
  struct Div {
    static void f(Context& C, Float* X, Value Y) {
      X->Val = llvm::APFloat(
          X->Val.convertToDouble() / Number::getAsDouble(Y));
    }
    static Value f(Context& C, Int X, Int Y) {
      if (Number::isExactZero(Y)) {
        C.RaiseError("divide by exact zero", Value(Y));
        return nullptr;
      }
      // TODO Support exact rational numbers.
      // Treat all division as inexact.
      return C.CreateFloat(double{X + 0.0} / Y);
    }
  };
};

} // end namespace heavy

namespace heavy { namespace base {
void callcc(Context& C, ValueRefs Args) {
  unsigned Len = Args.size();
  assert(Len == 1 && "Invalid arity to builtin `callcc`");
  C.CallCC(Args[0]);
}

void dump(Context& C, ValueRefs Args) {
  if (Args.size() != 1) return C.RaiseError("invalid arity");
  Args[0].dump();
  C.Cont(heavy::Undefined());
}

template <typename Op>
heavy::Value arithmetic_reduce(Context& C, Value Accum, ValueRefs Args) {
  for (heavy::Value X : Args) {
    if (C.CheckNumber(X)) return nullptr;
    switch (Number::CommonKind(Accum, X)) {
      case ValueKind::Float: {
        // We can assume Float or Int for both values.
        // Make Accum a Float if it is not.
        if (!isa<Float>(Accum)) {
          Accum = C.CreateFloat(double{cast<Int>(Accum) + 0.0});
        }
        Op::f(C, cast<Float>(Accum), X);
        break;
      }
      case ValueKind::Int: {
        // We can assume they are both Int.
        Accum = Op::f(C, cast<Int>(Accum), cast<Int>(X));
        // The result might NOT be Int though.
        if (Accum == nullptr) return nullptr;
        break;
      }
      default:
        // CommonKind had an invalid number type
        // or an error was thrown.
        return nullptr;
    }
  }
  return Accum;
}

void add(Context& C, ValueRefs Args) {
  Value Result = arithmetic_reduce<NumberOp::Add>(C, Int{0}, Args);
  if (Result) {
    C.Cont(Result);
  }
}

void mul(Context&C, ValueRefs Args) {
  Value Result = arithmetic_reduce<NumberOp::Mul>(C, Int{1}, Args);
  if (Result) {
    C.Cont(Result);
  }
}

void sub(Context&C, ValueRefs Args) {
  Value Initial;

  if (Args.size() > 1) {
    Initial = Args[0];
    Args = Args.drop_front(); 
  } else {
    // Additive inverse
    Initial = Int{0};
  }

  Value Result = arithmetic_reduce<NumberOp::Sub>(C, Initial, Args);
  if (Result) {
    C.Cont(Result);
  }
}

void div(Context& C, ValueRefs Args) {
  Value Initial;

  if (Args.size() > 1) {
    Initial = Args[0];
    Args = Args.drop_front(); 
  } else {
    // Multiplicative inverse
    Initial = Int{1};
  }
  Value Result = arithmetic_reduce<NumberOp::Div>(C, Initial, Args);
  if (Result) {
    C.Cont(Result);
  }
}

void gt(Context& C, ValueRefs Args) {
  C.RaiseError("TODO gt");
}

void lt(Context& C, ValueRefs Args) {
  C.RaiseError("TODO lt");
}

void equal(Context& C, ValueRefs Args) {
  if (Args.size() != 2) return C.RaiseError("invalid arity");
  Value V1 = Args[0];
  Value V2 = Args[1];
  C.Cont(Bool(::heavy::equal(V1, V2)));
}

void eqv(Context& C, ValueRefs Args) {
  if (Args.size() != 2) return C.RaiseError("invalid arity");
  Value V1 = Args[0];
  Value V2 = Args[1];
  C.Cont(Bool(::heavy::eqv(V1, V2)));
}

void list(Context& C, ValueRefs Args) {
  // Returns a *newly allocated* list of its arguments.
  C.Cont(C.CreateList(Args));
}

void append(Context& C, ValueRefs Args) {
  C.RaiseError("TODO append");
}

void with_exception_handler(Context& C, ValueRefs Args) {
  if (Args.size() != 2) return C.RaiseError("invalid arity");
  C.WithExceptionHandler(Args[0], Args[1]);
}

void raise(Context& C, ValueRefs Args) {
  if (Args.size() != 1) return C.RaiseError("invalid arity");
  C.Raise(Args[0]);
}

void error(Context& C, ValueRefs Args) {
  if (Args.size() == 0) return C.RaiseError("invalid arity");
  if (C.CheckKind<String>(Args[0])) return;
  C.RaiseError(cast<String>(Args[0]), Args.drop_front());
}

void eval(Context& C, ValueRefs Args) {
  if (Args.size() != 2) {
    return C.RaiseError("invalid arity");
  }
  Value ExprOrDef       = Args[0];
  Value EnvSpec         = Args[1];
  Value Eval = Args.size() == 3 ? Args[2] : HEAVY_BASE_VAR(op_eval);

  std::array<Value, 3> NewArgs = {ExprOrDef, EnvSpec, Eval};
  C.Apply(HEAVY_BASE_VAR(compile), NewArgs);
}

void compile(Context& C, ValueRefs Args) {
  if (Args.size() != 3) {
    return C.RaiseError("invalid arity");
  }
  Value ExprOrDef       = Args[0];
  Value EnvSpec         = Args[1];
  Value TopLevelHandler = Args[2];

  // EnvPtr - Manage ownership of the Environment
  //          if we have to create it on the fly.
  std::unique_ptr<Environment> EnvPtr = nullptr;
  Environment* Env = nullptr;;

  if (auto* E = dyn_cast<Environment>(EnvSpec)) {
    Env = E;
  } else if (isa<ImportSet>(EnvSpec)) {
    auto EnvPtr = std::make_unique<Environment>(C);
    Env = EnvPtr.get();
  } else {
    return C.SetError("invalid environment specifier", EnvSpec);
  }

  Value Thunk = C.CreateLambda([](heavy::Context& C, ValueRefs) {
    Value EnvSpec = C.getCapture(0);
    C.PushCont([](heavy::Context& C, ValueRefs) {
      Value ExprOrDef = C.getCapture(0);
      Value TopLevelHandler = C.getCapture(1);
      C.OpGen->SetTopLevelHandler(TopLevelHandler);
      C.OpGen->VisitTopLevel(ExprOrDef); // calls Cont()
    }, C.getCaptures().drop_front());
    if (auto* ImpSet = dyn_cast<ImportSet>(EnvSpec)) {
      C.Import(ImpSet);
    } else {
      C.Cont();
    }
  }, CaptureList{EnvSpec, ExprOrDef, TopLevelHandler});

  C.WithEnv(std::move(EnvPtr), Env, Thunk);
}

}} // end of namespace heavy::base
