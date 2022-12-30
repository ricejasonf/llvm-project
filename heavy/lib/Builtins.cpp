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

// import must be pre-loaded
heavy::ExternSyntax<>      HEAVY_BASE_VAR(import);

heavy::ExternSyntax<>      HEAVY_BASE_VAR(define_library);
heavy::ExternSyntax<>      HEAVY_BASE_VAR(begin);
heavy::ExternSyntax<>      HEAVY_BASE_VAR(export);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(cond_expand);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(include);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(include_ci);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(include_library_declarations);
heavy::ExternSyntax<>      HEAVY_BASE_VAR(push_library_cleanup);

heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(define);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(define_syntax);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(syntax_rules);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(if);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(lambda);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(quasiquote);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(quote);
heavy::ExternBuiltinSyntax HEAVY_BASE_VAR(set);


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
  heavy::Symbol* S = dyn_cast<Symbol>(P2->Car);
  if (!S) return OG.SetError("expecting symbol", P2);
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

void begin(Context& C, ValueRefs Args) {
  OpGen& OG = *C.OpGen;
  Pair* P = cast<Pair>(Args[0]);
  auto Loc = P->getSourceLocation();
  if (OG.isTopLevel()) {
    OG.VisitTopLevelSequence(P->Cdr);
  } else {
    mlir::Value Result = OG.createSequence(Loc, P->Cdr);
    C.Cont(OpGen::fromValue(Result));
  }
}

mlir::Value cond_expand(OpGen& OG, Pair* P) {
  llvm_unreachable("TODO");
}

mlir::Value include_(OpGen& OG, Pair* P) {
  llvm_unreachable("TODO");
}

mlir::Value include_ci(OpGen& OG, Pair* P) {
  llvm_unreachable("TODO");
}

mlir::Value include_library_declarations(OpGen& OG, Pair* P) {
  llvm_unreachable("TODO");
}

// push_library_cleanup - Allow users to provide
//                        cleanups for unloading a module.
void push_library_cleanup(Context& C, ValueRefs Args) {
  if (!C.OpGen->isLibraryContext()) {
    return C.RaiseError("export must be in library context", Args[0]);
  }
  llvm_unreachable("TODO");
  // TODO Create OpGen ops to create the call to the push_module_cleanup
  //      run-time function.
}

}} // end of namespace heavy::base

namespace heavy {
// TODO Replace NumberOp here with corresponding arithmetic ops in OpGen and OpEval
struct NumberOp {
  // These operations always mutate the first operand
  struct Add {
    static Int f(Int X, Int Y) { return X + Y; }
    static void f(Float* X, Float *Y) { X->Val = X->Val + Y->Val; }
  };
  struct Sub {
    static Int f(Int X, Int Y) { return X - Y; }
    static void f(Float* X, Float *Y) { X->Val = X->Val - Y->Val; }
  };
  struct Mul {
    static Int f(Int X, Int Y) { return X * Y; }
    static void f(Float* X, Float *Y) { X->Val = X->Val * Y->Val; }
  };
  struct Div {
    static Int f(Int X, Int Y) { return X / Y; }
    static void f(Float* X, Float *Y) { X->Val = X->Val / Y->Val; }
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
heavy::Value operator_helper(Context& C, Value X, Value Y) {
  if (C.CheckNumber(X)) return nullptr;
  if (C.CheckNumber(X)) return nullptr;
  ValueKind CommonKind = Number::CommonKind(X, Y);
  switch (CommonKind) {
    case ValueKind::Float: {
      llvm_unreachable("TODO casting to float");
    }
    case ValueKind::Int: {
      // we can assume they are both Int
      return Op::f(cast<Int>(X), cast<Int>(Y));
    }
    default:
      llvm_unreachable("unsupported numeric type");
  }
}

void add(Context& C, ValueRefs Args) {
  Value Temp = Args[0];
  for (heavy::Value X : Args.drop_front()) {
    Temp = operator_helper<NumberOp::Add>(C, Temp, X);
    if (!Temp) return;
  }
  C.Cont(Temp);
}

void mul(Context&C, ValueRefs Args) {
  Value Temp = Args[0];
  for (heavy::Value X : Args.drop_front()) {
    Temp = operator_helper<NumberOp::Mul>(C, Temp, X);
    if (!Temp) return;
  }
  C.Cont(Temp);
}

void sub(Context&C, ValueRefs Args) {
  Value Temp = Args[0];
  for (heavy::Value X : Args.drop_front()) {
    Temp = operator_helper<NumberOp::Sub>(C, Temp, X);
    if (!Temp) return;
  }
  C.Cont(Temp);
}

void div(Context& C, ValueRefs Args) {
  Value Temp = Args[0];
  for (heavy::Value X : Args.drop_front()) {
    if (Number::isExactZero(X)) {
      C.RaiseError("divide by exact zero", X);
      return;
    }
    Temp = operator_helper<NumberOp::Div>(C, Temp, X);
    if (!Temp) return;
  }
  C.Cont(Temp);
}

void gt(Context& C, ValueRefs Args) {
  llvm_unreachable("TODO");
}

void lt(Context& C, ValueRefs Args) {
  llvm_unreachable("TODO");
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
  heavy::Value List = C.CreateEmpty();
  for (heavy::Value Arg : Args) {
    List = C.CreatePair(Arg, List);
  }
  C.Cont(List);
}

void append(Context& C, ValueRefs Args) {
  llvm_unreachable("TODO append");
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
