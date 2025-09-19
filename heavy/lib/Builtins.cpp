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
#include "heavy/Dialect.h"
#include "heavy/OpGen.h"
#include "heavy/Value.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "cassert"
#include "memory"

namespace mlir {
class Value;
}

namespace heavy::base_var {
heavy::ExternSyntax<>      define_library;
heavy::ExternSyntax<>      begin;
heavy::ExternSyntax<>      export_;
heavy::ExternSyntax<>      include;
heavy::ExternSyntax<>      include_ci;
heavy::ExternSyntax<>      include_library_declarations;
heavy::ContextLocal        parse_source_file;

heavy::ExternBuiltinSyntax cond_expand;
heavy::ExternBuiltinSyntax define;
heavy::ExternBuiltinSyntax define_syntax;
heavy::ExternBuiltinSyntax syntax_rules;
heavy::ExternBuiltinSyntax ir_macro_transformer;
heavy::ExternBuiltinSyntax if_;
heavy::ExternBuiltinSyntax lambda;
heavy::ExternBuiltinSyntax quasiquote;
heavy::ExternBuiltinSyntax quote;
heavy::ExternBuiltinSyntax set;
heavy::ExternBuiltinSyntax syntax_error;


heavy::ExternFunction add;
heavy::ExternFunction sub;
heavy::ExternFunction div;
heavy::ExternFunction mul;
heavy::ExternFunction gt;
heavy::ExternFunction lt;
heavy::ExternFunction list;
heavy::ExternFunction length;
heavy::ExternFunction cons;
heavy::ExternFunction car;
heavy::ExternFunction cdr;
heavy::ExternFunction append;
heavy::ExternFunction dump;
heavy::ExternFunction write;
heavy::ExternFunction newline;
heavy::ExternFunction string_append;
heavy::ExternFunction string_copy;
heavy::ExternFunction string_length;
heavy::ExternFunction number_to_string;
heavy::ExternFunction eq;
heavy::ExternFunction equal;
heavy::ExternFunction eqv;
heavy::ExternFunction call_cc;
heavy::ExternFunction values;
heavy::ExternFunction call_with_values;
heavy::ExternFunction with_exception_handler;
heavy::ExternFunction raise;
heavy::ExternFunction error;
heavy::ExternFunction dynamic_wind;
heavy::ExternFunction load_module;
heavy::ExternFunction source_loc;

heavy::ExternFunction eval;
heavy::ExternFunction op_eval;
heavy::ExternFunction compile;
heavy::ContextLocal   module_path;


// Type predicates
heavy::ExternFunction is_boolean;
heavy::ExternFunction is_bytevector;
heavy::ExternFunction is_char;
heavy::ExternFunction is_eof_object;
heavy::ExternFunction is_null;
heavy::ExternFunction is_number;
heavy::ExternFunction is_pair;
heavy::ExternFunction is_port;
heavy::ExternFunction is_procedure;
heavy::ExternFunction is_string;
heavy::ExternFunction is_symbol;
heavy::ExternFunction is_vector;
// Extended types.
heavy::ExternFunction is_mlir_operation;
heavy::ExternFunction is_source_value;

}

bool HEAVY_BASE_IS_LOADED = false;

namespace heavy::base {
// See Quasiquote.cpp
mlir::Value quote(OpGen& OG, Pair* P);
mlir::Value quasiquote(OpGen& OG, Pair* P);

// See OpEval.cpp
void op_eval(Context& C, ValueRefs Args);

mlir::Value define(OpGen& OG, Pair* P) {
  Pair*   P2    = dyn_cast<Pair>(P->Cdr);
  Value Id     = nullptr;

  if (!P2)
    return OG.SetError("invalid syntax for define", P);
  if (Pair* LambdaSpec = dyn_cast<Pair>(P2->Car))
    Id = LambdaSpec->Car;
  else
    Id = P2->Car;

  if (!isIdentifier(Id))
    return OG.SetError("invalid syntax for define", P);
  return OG.createDefine(Id, P2, P);
}

mlir::Value define_syntax(OpGen& OG, Pair* P) {
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  if (!P2) return OG.SetError("invalid define-syntax syntax", P);
  Value Id = P2->Car;
  if (!isIdentifier(Id))
    return OG.SetError("expecting name for define-syntax", P);

  return OG.createSyntaxSpec(P2, P);
}

mlir::Value syntax_rules(OpGen& OG, Pair* P) {
  // TODO Support SyntaxClosures.

  // The input is the <Syntax Spec> (Keyword (syntax-rules ...))
  // <Syntax Spec> has its own checks in createSyntaxSpec
  Symbol* Keyword = dyn_cast<Symbol>(P->Car);
  Pair* SpecInput = dyn_cast_or_null<Pair>(P->Cdr.car().cdr());
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

mlir::Value ir_macro_transformer(OpGen& OG, Pair* P) {
  llvm_unreachable("TODO");
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

  Value ElseExpr = nullptr;
  if (isa<Empty>(P2->Cdr))
    ElseExpr = Undefined();
  else if (Pair* P3 = dyn_cast<Pair>(P2->Cdr);
             P3 && isa<Empty>(P3->Cdr))
    ElseExpr = P3->Car;
  if (!ElseExpr)
    return OG.SetError("invalid if syntax", P);

  return OG.createIf(P->getSourceLocation(), CondExpr,
                     ThenExpr, ElseExpr);
}

mlir::Value set(OpGen& OG, Pair* P) {
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  if (!P2) return OG.SetError("invalid set syntax", P);

  Value S = P2->Car;

  if (!isa<Binding>(S) && !isIdentifier(S))
    return OG.SetError("expecting identifier", P2);

  P2 = dyn_cast<Pair>(P2->Cdr);
  if (!P2) return OG.SetError("invalid set syntax", P2);
  heavy::Value Expr = P2->Car;
  if (!isa<Empty>(P2->Cdr)) return OG.SetError("invalid set syntax", P2);
  return OG.createSet(P->getSourceLocation(), S, Expr);
}

mlir::Value syntax_error(OpGen& OG, Pair* P) {
  heavy::SourceLocation Loc = P->getSourceLocation();
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  if (!P2)
    return OG.SetError("invalid syntax-error... syntax", P);
  llvm::SmallVector<mlir::Value, 8> Args;
  for (auto [Loc, V] : WithSource(P2))
    Args.push_back(OG.createLiteral(Loc, V));
  return OG.createSyntaxError(Loc, Args);
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
      C.OpGen->SetError("expecting proper list for import syntax", Args[0]);
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
    C.OpGen->SetError("unexpected library definition", P);
    return;
  }
  Value NameSpec;
  Pair* LibraryDecls;
  if (Pair* P2 = dyn_cast<Pair>(P->Cdr)) {
    NameSpec = P2->Car;
    LibraryDecls = dyn_cast<Pair>(P2->Cdr);
    if (!LibraryDecls) {
      C.OpGen->SetError("expected library declarations", P2->Cdr);
      return;
    }
  }
  std::string MangledNameStr = OG.mangleModule(NameSpec);
  if (MangledNameStr.size() == 0) {
    C.OpGen->SetError("library name is invalid", NameSpec);
    return;
  }
  Symbol* MangledName = C.CreateSymbol(MangledNameStr);
  OG.VisitLibrary(Loc, MangledName, LibraryDecls);
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
      if (C.OpGen->CheckError()) return;
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

void source_loc(Context& C, ValueRefs Args) {
  heavy::SourceLocation Loc;
  // Take the first valid source location.
  for (heavy::Value Arg : Args) {
    Loc = Arg.getSourceLocation();
    if (Loc.isValid())
      break;
  }
  C.Cont(C.CreateSourceValue(Loc));
}

} // end of namespace heavy::base

namespace heavy {
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

namespace heavy::base {
void call_cc(Context& C, ValueRefs Args) {
  if (Args.size() != 1) return C.RaiseError("invalid arity");
  C.CallCC(Args[0]);
}

void values(Context& C, ValueRefs Args) {
  C.Cont(Args);
}

void call_with_values(Context& C, ValueRefs Args) {
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");
  heavy::Value Producer = Args[0];
  heavy::Value Consumer = Args[1];
  C.PushCont(Consumer);
  C.Apply(Producer, {});
}

void dump(Context& C, ValueRefs Args) {
  if (Args.size() != 1) return C.RaiseError("invalid arity");
  Args[0].dump();
  C.Cont(heavy::Undefined());
}

void write(Context& C, ValueRefs Args) {
  // TODO Write to specified output port.
  if (Args.size() == 2) return C.RaiseError("port argument unsupported");
  if (Args.size() != 1) return C.RaiseError("invalid arity");
  // TODO Write to specified output port.
  heavy::write(llvm::outs(), Args[0]);
  C.Cont(heavy::Undefined());
}

void newline(Context& C, ValueRefs Args) {
  // TODO Write to specified output port.
  if (Args.size() == 1) {
    return C.RaiseError("port argument unsupported");
  } else if (Args.size() != 0) {
    return C.RaiseError("invalid arity");
  }

  // heavy::write(llvm::outs(), heavy::Char('\n'));
  // If output port wraps an llvm::ostream then this would be fine.
  llvm::outs() << '\n';
  C.Cont(heavy::Undefined());
}

void string_append(Context& C, ValueRefs Args) {
  size_t TotalLength = 0;
  for (Value Arg : Args) {
    if (!isa<String, Symbol>(Arg))
      return C.RaiseError("expecting string-like object");
    TotalLength += Arg.getStringRef().size();
  }
  String* TargetStr = C.CreateString(TotalLength, '\0');
  llvm::MutableArrayRef<char> TargetRef = TargetStr->getMutableView();
  auto TargetItr = TargetRef.begin();
  auto TargetEndItr = TargetRef.end();
  for (Value Arg : Args) {
    llvm::StringRef CurRef = Arg.getStringRef();
    size_t Dist = std::distance(CurRef.begin(), CurRef.end());
    assert((TargetItr + Dist <= TargetEndItr) && "out of bounds copy");
    std::copy(CurRef.begin(), CurRef.end(), TargetItr);
    TargetItr += Dist;
  }
  C.Cont(TargetStr);
}

void string_length(Context& C, ValueRefs Args) {
  if (Args.size() != 1 || !isa<String, Symbol>(Args[0]))
    return C.RaiseError("expecting string-like object");
  int Size = static_cast<int>(Args[0].getStringRef().size());
  return C.Cont(Int(Size));
}

void string_copy(Context& C, ValueRefs Args) {
  if (Args.size() > 3)
    return C.RaiseError("invalid arity");
  if (Args.size() < 1 || !isa<String, Symbol>(Args[0]))
    return C.RaiseError("expecting string-like object");

  size_t StartPos = 0;
  size_t EndPos = ~size_t(0);  // npos

  if (Args.size() >= 2) {
    if (isa<Int>(Args[1]))
      StartPos = static_cast<size_t>(cast<Int>(Args[1]));
    else
      return C.RaiseError("expecting integer");
  }
  if (Args.size() == 3) {
    if (isa<Int>(Args[2]))
      EndPos = static_cast<size_t>(cast<Int>(Args[2]));
    else
      return C.RaiseError("expecting integer");
  }

  llvm::StringRef Substr = Args[0].getStringRef().slice(StartPos, EndPos);
  return C.Cont(C.CreateString(Substr));
}

void number_to_string(Context& C, ValueRefs Args) {
  if (Args.size() != 1 ||
      !isa<heavy::Int, heavy::Float>(Args[0]))
    return C.RaiseError("expecting number");
  std::string Str;
  llvm::raw_string_ostream Stream(Str);
  write(Stream, Args[0]);
  C.Cont(C.CreateString(llvm::StringRef(Str)));
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

void length(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");

  Value Cur = Args[0];
  Value CurFast = Cur;
  int32_t Count = 0;
  while (!isa<Empty>(Cur)) {
    Pair* P1 = dyn_cast<Pair>(Cur);
    Pair* P2 = dyn_cast_or_null<Pair>(CurFast);
    if (!P1)
      return C.RaiseError("expecting a list");

    Cur = P1->Cdr;
    CurFast = P2 ? P2->Cdr.cdr() : nullptr;

    if (Cur == CurFast)
      return C.RaiseError("cycle detected");

    ++Count;
  }

  return C.Cont(Int{Count});
}

void cons(Context& C, ValueRefs Args) {
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");
  C.Cont(C.CreatePair(Args[0], Args[1]));
}

void car(Context& C, ValueRefs Args) {
  if (Args.size() != 1 || !isa<heavy::Pair>(Args[0]))
    return C.RaiseError("expecting pair");
  C.Cont(cast<heavy::Pair>(Args[0])->Car);
}

void cdr(Context& C, ValueRefs Args) {
  if (Args.size() != 1 || !isa<heavy::Pair>(Args[0]))
    return C.RaiseError("expecting pair");
  C.Cont(cast<heavy::Pair>(Args[0])->Cdr);
}

namespace {
Value append_rec(Context& C, Value List, Value Cdr) {
  C.setLoc(List);
  if (isa<Empty>(List))
    return Cdr;
  if (Pair* P = dyn_cast<Pair>(List)) {
    if (Value V = append_rec(C, P->Cdr, Cdr))
      return C.CreatePair(P->Car, V);
  }
  return Value();
};
}
void append(Context& C, ValueRefs Args) {
  Value NewList = Args.empty() ? Empty() : Args.back();
  Args = Args.drop_back(1);
  for (auto Itr = Args.rbegin(); Itr != Args.rend(); ++Itr) {
    NewList = append_rec(C, *Itr, NewList);
    if (NewList == Value())
      return C.RaiseError("expecting proper list");
  }
  C.Cont(NewList);
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

void dynamic_wind(Context& C, ValueRefs Args) {
  if (Args.size() != 3) return C.RaiseError("invalid arity");
  C.DynamicWind(Args[0], Args[1], Args[2]);
}

// Dynamically and idempotently initialize a module by its mangled name.
void load_module(Context& C, ValueRefs Args) {
  if (Args.size() != 1) return C.RaiseError("invalid arity");
  heavy::Symbol* MangledName = dyn_cast<heavy::Symbol>(Args[0]);
  if (!MangledName)
    return C.RaiseError("module name should be a symbol");
  C.PushCont([](Context& C, ValueRefs) {
      heavy::Symbol* MangledName = cast<Symbol>(C.getCapture(0));
      C.InitModule(MangledName);
  }, CaptureList{MangledName});
  C.LoadModule(MangledName);
}

void eval(Context& C, ValueRefs Args) {
  if (Args.size() < 1)
    return C.RaiseError("invalid arity");

  Value ExprOrDef       = Args[0];
  Value EnvSpec         = Args.size() > 1 ? Args[1] : Undefined();
  Value EvalRaw = Args.size() == 3 ? Args[2] : HEAVY_BASE_VAR(op_eval);

  Value NewArgs[] = {ExprOrDef, EnvSpec, EvalRaw};
  C.Apply(HEAVY_BASE_VAR(compile), NewArgs);
}

void compile(Context& C, ValueRefs Args) {
  if (Args.size() < 1) {
    return C.RaiseError("invalid arity");
  }
  Value ExprOrDef       = Args[0];
  Value EnvSpec         = Args.size() > 1 ? Args[1] : Undefined();
  Value TopLevelHandler = Args.size() > 2 ? Args[2] : Undefined();

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
    C.OpGen->SetError("invalid environment specifier", EnvSpec);
    return;
  }

  // Clear any errors in the compiler too keep the dream alive.
  if (heavy::OpGen* OpGen = Env->getOpGen())
    OpGen->ClearError();

  // If there is a provided handler then we save the results in an
  // improper, reversed ordered list (stack) to pass to the handler.
  Value ResultHandler = Undefined();
  if (!isa<Undefined>(TopLevelHandler)) {
    Value Results = C.CreateBinding(Empty());
    ResultHandler = C.CreateLambda([](heavy::Context& C, ValueRefs Args) {
      // Push each result to a reverse ordered list (stack).
      assert(!Args.empty() && "compile should call handler with result");
      Value NewResult = Args[0];
      Binding* Results = cast<Binding>(C.getCapture(0));
      // Usually there is only a single result.
      if (isa<Empty>(Results->getValue()))
        Results->setValue(NewResult);
      else
        Results->setValue(C.CreatePair(NewResult, Results->getValue()));
      C.Cont();
    }, CaptureList{Results});
    C.PushCont([](Context& C, ValueRefs) {
      // The compiler should already be cleaned up at this point.
      Value TopLevelHandler = C.getCapture(0);
      Value Results = cast<Binding>(C.getCapture(1))->getValue();
      for (Value Result : Results) {
        if (isa<Empty>(Results))
          break;
        C.PushCont([](Context& C, ValueRefs) {
          Value TopLevelHandler = C.getCapture(0);
          Value Result = C.getCapture(1);
          C.Apply(TopLevelHandler, Result);
        }, CaptureList{TopLevelHandler, Result});
      }
      C.Cont();
    }, CaptureList{TopLevelHandler, Results});
  }

  Value Thunk = C.CreateLambda([](heavy::Context& C, ValueRefs) {
    Value EnvSpec = C.getCapture(0);
    C.PushCont([](heavy::Context& C, ValueRefs) {
      Value ExprOrDef = C.getCapture(0);
      Value ResultHandler = C.getCapture(1);
      if (!isa<Undefined>(ResultHandler))
        C.OpGen->SetTopLevelHandler(ResultHandler);
      C.OpGen->VisitTopLevel(ExprOrDef); // calls Cont()
    }, C.getCaptures().drop_front());
    if (auto* ImpSet = dyn_cast<ImportSet>(EnvSpec)) {
      C.Import(ImpSet);
    } else {
      C.Cont();
    }
  }, CaptureList{EnvSpec, ExprOrDef, ResultHandler});

  C.WithEnv(std::move(EnvPtr), Env, Thunk);
}

void is_boolean(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  C.Cont(Bool(isa<heavy::Bool>(Args[0])));
}
void is_bytevector(Context& C, ValueRefs Args) {
  return C.RaiseError("bytevector not supported");
}
void is_char(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  C.Cont(Bool(isa<heavy::Char>(Args[0])));
}
void is_eof_object(Context& C, ValueRefs Args) {
  return C.RaiseError("eof-object not supported");
}
void is_null(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  C.Cont(Bool(isa<heavy::Empty>(Args[0])));
}
void is_number(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  C.Cont(Bool(isa<heavy::Int, heavy::Float>(Args[0])));
}
void is_pair(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  C.Cont(Bool(isa<heavy::Pair, heavy::PairWithSource>(Args[0])));
}
void is_port(Context& C, ValueRefs Args) {
  return C.RaiseError("port not supported");
}
void is_procedure(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  C.Cont(Bool(isa<heavy::Lambda, heavy::Builtin>(Args[0])));
}
void is_string(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  C.Cont(Bool(isa<heavy::String>(Args[0])));
}
void is_symbol(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  C.Cont(Bool(isa<heavy::Symbol>(Args[0])));
}
void is_vector(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  C.Cont(Bool(isa<heavy::Vector>(Args[0])));
}
// Extended types.
void is_mlir_operation(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  C.Cont(Bool(isa<mlir::Operation>(Args[0])));
}

void is_source_value(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  C.Cont(Bool(isa<heavy::SourceValue>(Args[0])));
}
} // end of namespace heavy::base

// initialize the module for run-time independent of the compiler
void HEAVY_BASE_INIT(heavy::Context& Context) {
  Context.DialectRegistry->insert<heavy::Dialect>();
  // syntax
  HEAVY_BASE_VAR(define)          = heavy::base::define;
  HEAVY_BASE_VAR(define_syntax)   = heavy::base::define_syntax;
  HEAVY_BASE_VAR(syntax_rules)    = heavy::base::syntax_rules;
  HEAVY_BASE_VAR(ir_macro_transformer)
                                  = heavy::base::ir_macro_transformer;
  HEAVY_BASE_VAR(if_)             = heavy::base::if_;
  HEAVY_BASE_VAR(lambda)          = heavy::base::lambda;
  HEAVY_BASE_VAR(quasiquote)      = heavy::base::quasiquote;
  HEAVY_BASE_VAR(quote)           = heavy::base::quote;
  HEAVY_BASE_VAR(set)             = heavy::base::set;
  HEAVY_BASE_VAR(begin)           = heavy::base::begin;
  HEAVY_BASE_VAR(cond_expand)     = heavy::base::cond_expand;
  HEAVY_BASE_VAR(define_library)  = heavy::base::define_library;
  HEAVY_BASE_VAR(export_)         = heavy::base::export_;
  HEAVY_BASE_VAR(include)         = heavy::base::include_;
  HEAVY_BASE_VAR(include_ci)      = heavy::base::include_ci;
  HEAVY_BASE_VAR(include_library_declarations)
    = heavy::base::include_library_declarations;
  HEAVY_BASE_VAR(source_loc)      = heavy::base::source_loc;
  HEAVY_BASE_VAR(parse_source_file).init(Context);

  // functions
  HEAVY_BASE_VAR(add)     = heavy::base::add;
  HEAVY_BASE_VAR(sub)     = heavy::base::sub;
  HEAVY_BASE_VAR(div)     = heavy::base::div;
  HEAVY_BASE_VAR(mul)     = heavy::base::mul;
  HEAVY_BASE_VAR(gt)      = heavy::base::gt;
  HEAVY_BASE_VAR(lt)      = heavy::base::lt;
  HEAVY_BASE_VAR(list)    = heavy::base::list;
  HEAVY_BASE_VAR(length)  = heavy::base::length;
  HEAVY_BASE_VAR(cons)    = heavy::base::cons;
  HEAVY_BASE_VAR(car)     = heavy::base::car;
  HEAVY_BASE_VAR(cdr)     = heavy::base::cdr;
  HEAVY_BASE_VAR(append)  = heavy::base::append;
  HEAVY_BASE_VAR(dump)    = heavy::base::dump;
  HEAVY_BASE_VAR(write)   = heavy::base::write;
  HEAVY_BASE_VAR(newline) = heavy::base::newline;
  HEAVY_BASE_VAR(string_append) = heavy::base::string_append;
  HEAVY_BASE_VAR(string_copy) = heavy::base::string_copy;
  HEAVY_BASE_VAR(string_length) = heavy::base::string_length;
  HEAVY_BASE_VAR(number_to_string) = heavy::base::number_to_string;
  HEAVY_BASE_VAR(eq)      = heavy::base::eqv;
  HEAVY_BASE_VAR(equal)   = heavy::base::equal;
  HEAVY_BASE_VAR(eqv)     = heavy::base::eqv;
  HEAVY_BASE_VAR(call_cc) = heavy::base::call_cc;
  HEAVY_BASE_VAR(values)  = heavy::base::values;
  HEAVY_BASE_VAR(call_with_values) = heavy::base::call_with_values;
  HEAVY_BASE_VAR(with_exception_handler)
    = heavy::base::with_exception_handler;
  HEAVY_BASE_VAR(raise)   = heavy::base::raise;
  HEAVY_BASE_VAR(error)   = heavy::base::error;
  HEAVY_BASE_VAR(dynamic_wind) = heavy::base::dynamic_wind;

  HEAVY_BASE_VAR(eval)    = heavy::base::eval;
  HEAVY_BASE_VAR(op_eval) = heavy::base::op_eval;
  HEAVY_BASE_VAR(compile) = heavy::base::compile;
  HEAVY_BASE_VAR(module_path).init(Context);

  HEAVY_BASE_VAR(is_boolean) = heavy::base::is_boolean;
  HEAVY_BASE_VAR(is_bytevector) = heavy::base::is_bytevector;
  HEAVY_BASE_VAR(is_char) = heavy::base::is_char;
  HEAVY_BASE_VAR(is_eof_object) = heavy::base::is_eof_object;
  HEAVY_BASE_VAR(is_null) = heavy::base::is_null;
  HEAVY_BASE_VAR(is_number) = heavy::base::is_number;
  HEAVY_BASE_VAR(is_pair) = heavy::base::is_pair;
  HEAVY_BASE_VAR(is_port) = heavy::base::is_port;
  HEAVY_BASE_VAR(is_procedure) = heavy::base::is_procedure;
  HEAVY_BASE_VAR(is_string) = heavy::base::is_string;
  HEAVY_BASE_VAR(is_symbol) = heavy::base::is_symbol;
  HEAVY_BASE_VAR(is_vector) = heavy::base::is_vector;
  HEAVY_BASE_VAR(is_mlir_operation) = heavy::base::is_mlir_operation;
  HEAVY_BASE_VAR(is_source_value) = heavy::base::is_source_value;
}

// initializes the module and loads lookup information
// for the compiler
void HEAVY_BASE_LOAD_MODULE(heavy::Context& Context) {
  // Note: We call HEAVY_BASE_INIT in the constructor of Context.
  heavy::initModuleNames(Context, HEAVY_BASE_LIB_STR, {
    // syntax
    {"define",        HEAVY_BASE_VAR(define)},
    {"define-syntax", HEAVY_BASE_VAR(define_syntax)},
    {"if",            HEAVY_BASE_VAR(if_)},
    {"lambda",        HEAVY_BASE_VAR(lambda)},
    {"quasiquote",    HEAVY_BASE_VAR(quasiquote)},
    {"quote",         HEAVY_BASE_VAR(quote)},
    {"set!",          HEAVY_BASE_VAR(set)},
    {"syntax-rules",  HEAVY_BASE_VAR(syntax_rules)},
    {"ir-macro-transformer",
                      HEAVY_BASE_VAR(ir_macro_transformer)},
    {"begin",         HEAVY_BASE_VAR(begin)},
    {"cond-expand",   HEAVY_BASE_VAR(cond_expand)},
    {"define-library",HEAVY_BASE_VAR(define_library)},
    {"export",        HEAVY_BASE_VAR(export_)},
    {"include",       HEAVY_BASE_VAR(include)},
    {"include-ci",    HEAVY_BASE_VAR(include_ci)},
    {"include-library-declarations",
      HEAVY_BASE_VAR(include_library_declarations)},
    {"source-loc",    HEAVY_BASE_VAR(source_loc)},
    {"parse-source-file",
                      HEAVY_BASE_VAR(parse_source_file).get(Context)},

    // functions
    {"+",       HEAVY_BASE_VAR(add)},
    {"-",       HEAVY_BASE_VAR(sub)},
    {"/",       HEAVY_BASE_VAR(div)},
    {"*",       HEAVY_BASE_VAR(mul)},
    {">",       HEAVY_BASE_VAR(gt)},
    {"<",       HEAVY_BASE_VAR(lt)},
    {"list",    HEAVY_BASE_VAR(list)},
    {"length",  HEAVY_BASE_VAR(length)},
    {"cons",    HEAVY_BASE_VAR(cons)},
    {"car",     HEAVY_BASE_VAR(car)},
    {"cdr",     HEAVY_BASE_VAR(cdr)},
    {"append",  HEAVY_BASE_VAR(append)},
    {"dump",    HEAVY_BASE_VAR(dump)},
    {"write",   HEAVY_BASE_VAR(write)},
    {"newline", HEAVY_BASE_VAR(newline)},
    {"string-append", HEAVY_BASE_VAR(string_append)},
    {"string-copy", HEAVY_BASE_VAR(string_copy)},
    {"string-length", HEAVY_BASE_VAR(string_length)},
    {"number->string", HEAVY_BASE_VAR(number_to_string)},
    {"eq?",     HEAVY_BASE_VAR(eq)},
    {"equal?",  HEAVY_BASE_VAR(equal)},
    {"eqv?",    HEAVY_BASE_VAR(eqv)},
    {"call/cc", HEAVY_BASE_VAR(call_cc)},
    {"values", HEAVY_BASE_VAR(values)},
    {"call-with-values", HEAVY_BASE_VAR(call_with_values)},
    {"with-exception-handler", HEAVY_BASE_VAR(with_exception_handler)},
    {"raise", HEAVY_BASE_VAR(raise)},
    {"error", HEAVY_BASE_VAR(error)},
    {"dynamic-wind", HEAVY_BASE_VAR(dynamic_wind)},

    {"eval",    HEAVY_BASE_VAR(eval)},
    {"op-eval", HEAVY_BASE_VAR(op_eval)},
    {"compile", HEAVY_BASE_VAR(compile)},
    {"module-path", HEAVY_BASE_VAR(module_path).get(Context)},

    // Type predicates
    {"boolean?", HEAVY_BASE_VAR(is_boolean)},
    {"bytevector?", HEAVY_BASE_VAR(is_bytevector)},
    {"char?", HEAVY_BASE_VAR(is_char)},
    {"eof-object?", HEAVY_BASE_VAR(is_eof_object)},
    {"null?", HEAVY_BASE_VAR(is_null)},
    {"number?", HEAVY_BASE_VAR(is_number)},
    {"pair?", HEAVY_BASE_VAR(is_pair)},
    {"port?", HEAVY_BASE_VAR(is_port)},
    {"procedure?", HEAVY_BASE_VAR(is_procedure)},
    {"string?", HEAVY_BASE_VAR(is_string)},
    {"symbol?", HEAVY_BASE_VAR(is_symbol)},
    {"vector?", HEAVY_BASE_VAR(is_vector)},
    // Extended types.
    {"mlir-operation?", HEAVY_BASE_VAR(is_mlir_operation)},
    {"source-value?", HEAVY_BASE_VAR(is_source_value)},
  });
}

namespace heavy::detail {
// Borrowed from YAMLParser.
std::pair<uint32_t, unsigned> Utf8View::decode_front() const {
  StringRef::iterator Position= Range.begin();
  StringRef::iterator End = Range.end();
  // 1 byte: [0x00, 0x7f]
  // Bit pattern: 0xxxxxxx
  if (Position < End && (*Position & 0x80) == 0) {
    return std::make_pair(*Position, 1);
  }
  // 2 bytes: [0x80, 0x7ff]
  // Bit pattern: 110xxxxx 10xxxxxx
  if (Position + 1 < End && ((*Position & 0xE0) == 0xC0) &&
      ((*(Position + 1) & 0xC0) == 0x80)) {
    uint32_t codepoint = ((*Position & 0x1F) << 6) |
                          (*(Position + 1) & 0x3F);
    if (codepoint >= 0x80)
      return std::make_pair(codepoint, 2);
  }
  // 3 bytes: [0x8000, 0xffff]
  // Bit pattern: 1110xxxx 10xxxxxx 10xxxxxx
  if (Position + 2 < End && ((*Position & 0xF0) == 0xE0) &&
      ((*(Position + 1) & 0xC0) == 0x80) &&
      ((*(Position + 2) & 0xC0) == 0x80)) {
    uint32_t codepoint = ((*Position & 0x0F) << 12) |
                         ((*(Position + 1) & 0x3F) << 6) |
                          (*(Position + 2) & 0x3F);
    // Codepoints between 0xD800 and 0xDFFF are invalid, as
    // they are high / low surrogate halves used by UTF-16.
    if (codepoint >= 0x800 &&
        (codepoint < 0xD800 || codepoint > 0xDFFF))
      return std::make_pair(codepoint, 3);
  }
  // 4 bytes: [0x10000, 0x10FFFF]
  // Bit pattern: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
  if (Position + 3 < End && ((*Position & 0xF8) == 0xF0) &&
      ((*(Position + 1) & 0xC0) == 0x80) &&
      ((*(Position + 2) & 0xC0) == 0x80) &&
      ((*(Position + 3) & 0xC0) == 0x80)) {
    uint32_t codepoint = ((*Position & 0x07) << 18) |
                         ((*(Position + 1) & 0x3F) << 12) |
                         ((*(Position + 2) & 0x3F) << 6) |
                          (*(Position + 3) & 0x3F);
    if (codepoint >= 0x10000 && codepoint <= 0x10FFFF)
      return std::make_pair(codepoint, 4);
  }
  return std::make_pair(0, 0);
}

// Borrowed from YAMLParser.
// Encode \a UnicodeScalarValue in UTF-8 and append it to result.
void encode_utf8(uint32_t UnicodeScalarValue,
                 llvm::SmallVectorImpl<char> &Result) {
  if (UnicodeScalarValue <= 0x7F) {
    Result.push_back(UnicodeScalarValue & 0x7F);
  } else if (UnicodeScalarValue <= 0x7FF) {
    uint8_t FirstByte = 0xC0 | ((UnicodeScalarValue & 0x7C0) >> 6);
    uint8_t SecondByte = 0x80 | (UnicodeScalarValue & 0x3F);
    Result.push_back(FirstByte);
    Result.push_back(SecondByte);
  } else if (UnicodeScalarValue <= 0xFFFF) {
    uint8_t FirstByte = 0xE0 | ((UnicodeScalarValue & 0xF000) >> 12);
    uint8_t SecondByte = 0x80 | ((UnicodeScalarValue & 0xFC0) >> 6);
    uint8_t ThirdByte = 0x80 | (UnicodeScalarValue & 0x3F);
    Result.push_back(FirstByte);
    Result.push_back(SecondByte);
    Result.push_back(ThirdByte);
  } else if (UnicodeScalarValue <= 0x10FFFF) {
    uint8_t FirstByte = 0xF0 | ((UnicodeScalarValue & 0x1F0000) >> 18);
    uint8_t SecondByte = 0x80 | ((UnicodeScalarValue & 0x3F000) >> 12);
    uint8_t ThirdByte = 0x80 | ((UnicodeScalarValue & 0xFC0) >> 6);
    uint8_t FourthByte = 0x80 | (UnicodeScalarValue & 0x3F);
    Result.push_back(FirstByte);
    Result.push_back(SecondByte);
    Result.push_back(ThirdByte);
    Result.push_back(FourthByte);
  }
}

void encode_hex(uint32_t Code, llvm::SmallVectorImpl<char> &Result) {
  llvm::StringRef Digits = "0123456789ABCDEF";
  unsigned Digit = 0;

  // Interval [16^0, 16^8).
  for (unsigned i = 0; i < 8; ++i) {
    Digit = (Code >> (4 * (7 - i))) % 16;
    // Do not push leading zeros.
    if (Digit > 0 || !Result.empty() || i == 7)
      Result.push_back(Digits[Digit]);
  }
}

// Note that we do not validate UTF8 here. (Should we?)
std::pair</*CodePoint=*/uint32_t, /*IsError=*/bool>
from_hex(llvm::StringRef Chars) {
  llvm::StringRef Digits = "0123456789ABCDEF";
  uint32_t Output = 0;

  for (unsigned i = 0; i < Chars.size(); ++i) {
    size_t Digit = Digits.find(llvm::toUpper(Chars[Chars.size() - i - 1]));
    if (Digit == llvm::StringRef::npos)
      return {0, true};
    Output += Digit << (i * 4);
  }

  return {Output, false};
}

} // end of namespace heavy::detail
