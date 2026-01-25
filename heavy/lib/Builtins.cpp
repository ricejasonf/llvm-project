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

#include "TemplateGen.h"
#include "heavy/Builtins.h"
#include "heavy/Context.h"
#include "heavy/Dialect.h"
#include "heavy/OpGen.h"
#include "heavy/SourceManager.h"
#include "heavy/Value.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DynamicLibrary.h"
#include "cassert"
#include "memory"

namespace mlir {
class Value;
}

namespace heavy::builtins_var {
heavy::ExternSyntax<>      define_library;
heavy::ExternSyntax<>      begin;
heavy::ExternSyntax<>      export_;
heavy::ExternSyntax<>      include;
heavy::ExternSyntax<>      include_ci;
heavy::ExternSyntax<>      include_library_declarations;
heavy::ContextLocal        parse_source_file;

heavy::ExternBuiltinSyntax cond_expand;
heavy::ExternBuiltinSyntax define;
heavy::ExternBuiltinSyntax define_binding;
heavy::ExternBuiltinSyntax define_syntax;
heavy::ExternBuiltinSyntax syntax_rules;
heavy::ExternBuiltinSyntax syntax_fn;
heavy::ExternBuiltinSyntax if_;
heavy::ExternBuiltinSyntax lambda;
heavy::ExternBuiltinSyntax quasiquote;
heavy::ExternBuiltinSyntax quote;
heavy::ExternBuiltinSyntax set;
heavy::ExternBuiltinSyntax syntax_error;
heavy::ExternBuiltinSyntax case_lambda;

heavy::ExternFunction apply;
heavy::ExternFunction add;
heavy::ExternFunction sub;
heavy::ExternFunction div;
heavy::ExternFunction mul;
heavy::ExternFunction is_positive;
heavy::ExternFunction is_zero;
heavy::ExternFunction list;
heavy::ExternFunction length;
heavy::ExternFunction cons;
heavy::ExternFunction source_cons;
heavy::ExternFunction car;
heavy::ExternFunction cdr;
heavy::ExternFunction append;
heavy::ExternFunction vector;
heavy::ExternFunction make_vector;
heavy::ExternFunction vector_length;
heavy::ExternFunction vector_ref;
heavy::ExternFunction vector_set;
heavy::ExternFunction make_list;
heavy::ExternFunction list_set;
heavy::ExternFunction list_ref;
heavy::ExternFunction dump;
heavy::ExternFunction write;
heavy::ExternFunction newline;
heavy::ExternFunction string_append;
heavy::ExternFunction string_copy;
heavy::ExternFunction string_length;
heavy::ExternFunction string_ref;
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
heavy::ExternFunction source_loc_valid;
heavy::ExternFunction dump_source_loc;
heavy::ExternFunction make_syntactic_closure;
heavy::ExternFunction load_plugin;
heavy::ExternFunction load_builtin;

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

namespace heavy::builtins {
// See Quasiquote.cpp
mlir::Value quote(OpGen& OG, Pair* P);
mlir::Value quasiquote(OpGen& OG, Pair* P);

// See OpEval.cpp
void op_eval(Context& C, ValueRefs Args);

mlir::Value define(OpGen& OG, Pair* P) {
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  Value Id;

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

// Binds a dynamically loaded heavy::ContextLocal.
mlir::Value define_binding(OpGen& OG, Pair* P) {
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  Value Name;
  Value ExtName;

  if (P2) {
    Name = P2->Car;
    ExtName = P2->Cdr.car();
  }
  if (!P2 || !ExtName || !isa<Symbol>(Name) || !isa<Symbol, String>(ExtName))
    return OG.SetError("invalid syntax for define-binding", P);
  return OG.createExternalBinding(Name, ExtName);
}

mlir::Value define_syntax(OpGen& OG, Pair* P) {
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  if (!P2) return OG.SetError("invalid define-syntax syntax", P);
  Value Id = P2->Car;
  if (!isIdentifier(Id))
    return OG.SetError("expecting name for define-syntax", P);

  return OG.createSyntaxSpec(P2, P);
}

namespace {
std::pair<heavy::Value, heavy::Pair*> DestructureSyntaxSpec(OpGen& OG, Pair* P) {
  heavy::Value Keyword = P->Car;
  heavy::Pair* P2 = dyn_cast<Pair>(P->Cdr);
  if (P2) {
    heavy::Pair* Spec = dyn_cast<Pair>(P2->Car);
    if (Spec) {
      if (heavy::Pair* SpecInput = dyn_cast<Pair>(Spec->Cdr))
        return {Keyword, SpecInput};
    }
  }
  return {};
}
}  // end anon namespace

mlir::Value syntax_rules(OpGen& OG, Pair* P) {
  auto [Keyword, SpecInput] = DestructureSyntaxSpec(OG, P);

  if (!Keyword || !SpecInput)
    return OG.SetError("invalid syntax-rules syntax", P);

  // Check for optional ellipsis identifier.
  Value Ellipsis;
  if (isIdentifier(SpecInput->Car))
    Ellipsis = SpecInput->Car;
  if (Ellipsis) {
    Pair* Temp = dyn_cast<Pair>(SpecInput->Cdr);
    if (!Temp)
      return OG.SetError("invalid syntax-rules syntax.", SpecInput);
    SpecInput = Temp;
  } else {
    Ellipsis = OG.getContext().CreateSymbol("...");
  }
  return OG.createSyntaxRules(P->getSourceLocation(), Keyword, Ellipsis,
                              SpecInput->Car, SpecInput->Cdr);
}

// Convert lambda syntax into heavy::Syntax function
// (for use with define-syntax.)
mlir::Value syntax_fn(OpGen& OG, Pair* P) {
  auto [Keyword, SpecInput] = DestructureSyntaxSpec(OG, P);
  if (!Keyword || !SpecInput || !isa<Empty>(SpecInput->Cdr))
    return OG.SetError("invalid syntax for syntax-fn", P);
  heavy::SourceLocation Loc = P->getSourceLocation();
  heavy::Value ProcExpr = SpecInput->Car;
  heavy::FuncOp FuncOp = OG.createSyntaxFunction(Loc, ProcExpr);
  return OG.create<heavy::SyntaxOp>(Loc, FuncOp.getSymName());
}

mlir::Value lambda(OpGen& OG, Pair* P) {
  Pair* P2 = dyn_cast<Pair>(P->Cdr);
  if (!P2)
    return OG.SetError("invalid lambda syntax", P);
  Value Formals = P2->Car;
  Pair* Body = dyn_cast<Pair>(P2->Cdr);

  if (!Body)
    return OG.SetError("lambda syntax requires body");

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
  return OG.createError(Loc, Args);
}

mlir::Value case_lambda(OpGen& OG, Pair* P) {
  return OG.createCaseLambda(P);
}

namespace {
  void import_helper(Context& C, ValueRefs Args) {
    if (Pair* P = dyn_cast<Pair>(Args[0])) {
      C.PushCont([](Context& C, ValueRefs) {
        Value RecurseArgs = C.getCapture(0);
        import_helper(C, RecurseArgs);
      }, CaptureList{P->Cdr});
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
      import_helper(C, ImportSpecs);
    }, CaptureList{ImportSpecs}));
  } else {
    import_helper(C, ImportSpecs);
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

// Return the first valid source location or
// an invalid source location if none exists.
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

// Return the first valid source location or
// #f if none exists.
void source_loc_valid(Context& C, ValueRefs Args) {
  heavy::SourceLocation Loc;
  // Take the first valid source location.
  for (heavy::Value Arg : Args) {
    Loc = Arg.getSourceLocation();
    if (Loc.isValid())
      break;
  }
  if (!Loc.isValid())
    return C.Cont(Bool(false));
  C.Cont(C.CreateSourceValue(Loc));
}

void dump_source_loc(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  if (!C.SourceManager)
    return C.RaiseError("source manager not available");

  // TODO
  // This could be generalize to take an optional message to work with
  // errors, warnings, notes, etc.

  heavy::SourceLocation Loc = Args[0].getSourceLocation();
  heavy::FullSourceLocation SL = C.SourceManager->getFullSourceLocation(Loc);
  if (SL.isValid()) {
    heavy::SourceLineContext LineContext = SL.getLineContext();
    llvm::errs() << LineContext.FileName
                 << ':' << LineContext.LineNumber
                 << ':' << LineContext.Column
                 // TODO optional message could go here
                 << '\n'
                 << LineContext.LineRange << '\n';
    // Display the caret pointing to the point of interest.
    for (unsigned i = 1; i < LineContext.Column; i++) {
      llvm::errs() << ' ';
    }
    llvm::errs() << "^\n";
  } else {
    llvm::errs() << "<<INVALID SOURCE LOCATION>>\n";
  }
  C.Cont();
}

} // end of namespace heavy::builtins

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

namespace heavy::builtins {
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
  C.Cont(Args[0]);
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

  auto Utf8View = detail::Utf8View(Args[0].getStringRef());

  int Size = 0;
  while (Utf8View.drop_front())
    ++Size;

  return C.Cont(Int(Size));
}

void string_ref(Context& C, ValueRefs Args) {
  size_t K = 0;
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");
  if (!isa<String, Symbol>(Args[0]))
    return C.RaiseError("expecting string-like object");
  if (Args.size() == 2) {
    if (isa<Int>(Args[1]) && cast<Int>(Args[1]) >= 0)
      K = static_cast<size_t>(cast<Int>(Args[1]));
    else
      return C.RaiseError("expecting positive integer");
  }

  auto Utf8View = detail::Utf8View(Args[0].getStringRef());

  Value V;
  for (size_t I = 0; I <= K; ++I) {
    V = Utf8View.drop_front();
    if (!V) {
      return C.RaiseError("invalid index for string: {0} {1}",
                          {Args[1], Args[0]});
    }
  }

  return C.Cont(V);
}

void string_copy(Context& C, ValueRefs Args) {
  if (Args.size() > 3)
    return C.RaiseError("invalid arity");
  if (Args.size() < 1 || !isa<String, Symbol>(Args[0]))
    return C.RaiseError("expecting string-like object");

  size_t StartPos = 0;
  size_t EndPos = ~size_t(0);  // npos

  if (Args.size() >= 2) {
    if (isa<Int>(Args[1]) && cast<Int>(Args[1]) >= 0)
      StartPos = static_cast<size_t>(cast<Int>(Args[1]));
    else
      return C.RaiseError("expecting positive integer");
  }
  if (Args.size() == 3) {
    if (isa<Int>(Args[2]) && cast<Int>(Args[2]) >= 0)
      EndPos = static_cast<size_t>(cast<Int>(Args[2]));
    else
      return C.RaiseError("expecting positive integer");
  }

  auto View = detail::Utf8View(Args[0].getStringRef());
  size_t I = 0;
  for (; I < StartPos; ++I)
    View.drop_front();
  auto Back = View;
  llvm::StringRef ViewStr = View.Range;
  if (EndPos != ~size_t(0)) {
    for (; I < EndPos; ++I)
      Back.drop_front();
    ViewStr = ViewStr.drop_back(Back.Range.size());
  }

  return C.Cont(C.CreateString(ViewStr));
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

void is_positive(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");

  Value Input = Args[0];
  if (!isa<Int, Float>(Input))
    return C.RaiseError("expecting number");

  C.Cont(Bool(Number::getAsDouble(Input) > 0.0));
}

void is_zero(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");

  Value Input = Args[0];
  if (!isa<Int, Float>(Input))
    return C.RaiseError("expecting number");

  C.Cont(Bool(Number::getAsDouble(Input) == 0.0));
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

// Return length of proper list.
// Return negative value for circular lists.
// Return nullptr for improper list.
Value length_helper(Value Cur) {
  Value CurFast = Cur;
  int32_t Count = 0;
  while (!isa<Empty>(Cur)) {
    Pair* P1 = dyn_cast<Pair>(Cur);
    Pair* P2 = dyn_cast_or_null<Pair>(CurFast);
    if (!P1)
      return Value();

    Cur = P1->Cdr;
    CurFast = P2 ? P2->Cdr.cdr() : nullptr;

    // TODO Return length of cycle as negative integer.
    if (Cur == CurFast)
      return Value(Int(-1));

    ++Count;
  }

  return Int{Count};
}

// Return negative number if list is circular.
void length(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  Value Result = length_helper(Args[0]);
  if (!Result)
    return C.RaiseError("expecting a list: {}", Args[0]);
  C.Cont(Result);
}

void cons(Context& C, ValueRefs Args) {
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");
  C.Cont(C.CreatePair(Args[0], Args[1]));
}

// (source-cons car cdr src)
// Where `src` can be any value that might
// have a valid source location.
void source_cons(Context& C, ValueRefs Args) {
  if (Args.size() != 3)
    return C.RaiseError("invalid arity");
  C.Cont(C.CreatePair(Args[0], Args[1], Args[2]));
}

void car(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  if (auto* P = dyn_cast<heavy::Pair>(Args[0]))
    return C.Cont(P->Car);

  return C.RaiseError("expecting pair: {}", Args[0]);
}

void cdr(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");
  if (auto* P = dyn_cast<heavy::Pair>(Args[0]))
    return C.Cont(P->Cdr);

  return C.RaiseError("expecting pair: {}", Args[0]);
}

namespace {
Value append_rec(Context& C, Value List, Value Cdr) {
  C.setLoc(List);
  if (isa<Empty>(List))
    return Cdr;
  if (Pair* P = dyn_cast<Pair>(List)) {
    if (Value V = append_rec(C, P->Cdr, Cdr))
      return C.CreatePair(P->Car, V, P);
  }
  return Value();
}
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

void make_list(Context& C, ValueRefs Args) {
  if (Args.size() < 1 || Args.size() > 2)
    return C.RaiseError("invalid arity");

  Value Length = Args[0];
  if (!isa<Int>(Length) || cast<Int>(Length) < 0)
    return C.RaiseError("expecting positive integer");

  Value Default = Args.size() == 2 ? Args[1] : Value(Undefined());

  unsigned K = static_cast<unsigned>(cast<Int>(Length));
  Value Result = Empty();
  for (unsigned I = 0; I < K; I++)
    Result = C.CreatePair(Default, Result);

  C.Cont(Result);
}

void list_ref(Context& C, ValueRefs Args) {
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");

  Value List = Args[0];
  Value Index = Args[1];

  if (!isa<Pair>(List))
    return C.RaiseError("expecting non-empty list");

  int Len = 0;
  if (Value Length = length_helper(List))
    Len = cast<Int>(Length);
  else
    return;

  int K = 0;
  if (!isa<Int>(Index) || cast<Int>(Index) < 0)
    return C.RaiseError("expecting positive integer");
  K = cast<Int>(Index);

  if (Len >= 0 && K >= Len)
    return C.RaiseError("index is out of range");

  Pair* P = cast<Pair>(List);
  for (int I = 0; I < K; I++)
    P = cast<Pair>(P->Cdr);

  C.Cont(P->Car);
}

void list_set(Context& C, ValueRefs Args) {
  if (Args.size() != 3)
    return C.RaiseError("invalid arity");

  Value List = Args[0];
  Value Index = Args[1];
  Value Obj = Args[2];

  if (!isa<Pair>(List))
    return C.RaiseError("expecting non-empty list");

  int Len = 0;
  if (Value Length = length_helper(List))
    Len = cast<Int>(Length);
  else
    return;

  int K = 0;
  if (!isa<Int>(Index) || cast<Int>(Index) < 0)
    return C.RaiseError("expecting positive integer");
  K = cast<Int>(Index);

  // If Len is negative it has infinite length.
  if (Len >= 0 && K >= Len)
    return C.RaiseError("index is out of range");

  Pair* P = cast<Pair>(List);
  for (int I = 0; I < K; I++)
    P = cast<Pair>(P->Cdr);
  P->Car = Obj;
  C.Cont();
}

void vector(Context& C, ValueRefs Args) {
  C.Cont(C.CreateVector(Args));
}

void vector_length(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");

  Vector* Vec = dyn_cast<Vector>(Args[0]);
  if (!Vec)
    return C.RaiseError("expecting a vector");

  unsigned Len = Vec->getElements().size();
  Value Length = Int(static_cast<int32_t>(Len));

  C.Cont(Length);
}

void make_vector(Context& C, ValueRefs Args) {
  if (Args.size() < 1 || Args.size() > 2)
    return C.RaiseError("invalid arity");

  Value Length = Args[0];
  if (!isa<Int>(Length) || cast<Int>(Length) < 0)
    return C.RaiseError("expecting positive integer");

  Value Default = Args.size() == 2 ? Args[1] : Value(Undefined());

  unsigned K = static_cast<unsigned>(cast<Int>(Length));
  Vector* Vec = C.CreateVector(K, Default);

  C.Cont(Vec);
}

void vector_ref(Context& C, ValueRefs Args) {
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");

  Vector* Vec = dyn_cast<Vector>(Args[0]);
  Value Index = Args[1];

  if (!Vec)
    return C.RaiseError("expecting a vector");

  ValueRefs Xs = Vec->getElements();

  unsigned K = 0;
  if (!isa<Int>(Index) || cast<Int>(Index) < 0)
    return C.RaiseError("expecting positive integer");
  K = static_cast<unsigned>(cast<Int>(Index));

  if (K >= Xs.size())
    return C.RaiseError("index is out of range");

  Value Result = Xs[K];
  C.Cont(Result);
}

void vector_set(Context& C, ValueRefs Args) {
  if (Args.size() != 3)
    return C.RaiseError("invalid arity");

  Vector* Vec = dyn_cast<Vector>(Args[0]);
  if (!Vec)
    return C.RaiseError("expecting vector");
  ValueRefs Xs = Vec->getElements();

  Value Index = Args[1];
  if (!isa<Int>(Index) || cast<Int>(Index) < 0)
    return C.RaiseError("expecting positive integer");
  unsigned K = static_cast<unsigned>(cast<Int>(Index));

  if (K >= Xs.size())
    return C.RaiseError("index out of range");


  Xs[K] = Args[2];
  C.Cont();
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
  Value EnvSpec         = Args.size() > 1 ? Args[1] : Value(Undefined());
  Value EvalRaw = Args.size() == 3 ? Args[2] : HEAVY_BASE_VAR(op_eval);

  Value NewArgs[] = {ExprOrDef, EnvSpec, EvalRaw};
  C.Apply(HEAVY_BASE_VAR(compile), NewArgs);
}

void compile(Context& C, ValueRefs Args) {
  if (Args.size() < 1) {
    return C.RaiseError("invalid arity");
  }
  Value ExprOrDef       = Args[0];
  Value EnvSpec         = Args.size() > 1 ? Args[1] : Value(Undefined());
  Value TopLevelHandler = Args.size() > 2 ? Args[2] : Value(Undefined());

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

void apply(Context& C, ValueRefs Args) {
  llvm::SmallVector<heavy::Value, 8> NewArgs;

  if (Args.size() == 1)
    return C.Apply(Args[0], NewArgs);
  if (Args.size() < 2)
    return C.RaiseError("invalid arity");

  Value Fn = Args.front();
  Value LastArg = Args.back();
  Args = Args.drop_front().drop_back();

  for (heavy::Value Arg : Args)
    NewArgs.push_back(Arg);
  for (heavy::Value Arg : LastArg)
    NewArgs.push_back(Arg);

  C.Apply(Fn, NewArgs);
}

void make_syntactic_closure(Context& C, ValueRefs Args) {
  llvm_unreachable("make-syntactic-closure is not supported");
  if (Args.size() != 2)
    return C.RaiseError("invalid arity");
  Value Expr = Args[0];
  Value Env = Args[1];
  heavy::SourceLocation Loc = Expr.getSourceLocation();
  Value Result = C.CreateSyntaxClosure(Loc, Expr, Env);
  C.Cont(Result);
}

// Dynamically load a native shared library.
void load_plugin(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");

  llvm::StringRef Filename = Args.front().getStringRef();
  if (Filename.empty())
    return C.RaiseError("expecting nonempty string-like object: {}",
                        Args.front());

  std::string ErrMsg;
  if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(Filename.data(),
                                                        &ErrMsg)) {
    // TODO Make a "note" with Msg.
    String* Msg = C.CreateString(ErrMsg);
    return C.RaiseError("Failed to load plugin: {}\n{}", {Args.front(), Msg});
  }

  C.Cont();
}

// Dynamically load a Builtin from a heavy::ValueFn.
void load_builtin(Context& C, ValueRefs Args) {
  if (Args.size() != 1)
    return C.RaiseError("invalid arity");

  llvm::StringRef FuncName = Args.front().getStringRef();
  if (FuncName.empty())
    return C.RaiseError("expecting nonempty string-like object: {}",
                        Args.front());

  void* FuncVoidPtr
    = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(FuncName.data());
  if (FuncVoidPtr == nullptr)
    return C.RaiseError("unable to load builtin: {}", Args.front());
  heavy::ValueFn Fn = reinterpret_cast<heavy::ValueFn>(FuncVoidPtr);
  C.Cont(C.CreateBuiltin(Fn));
}
} // end of namespace heavy::builtins

// initialize the module for run-time independent of the compiler
void HEAVY_BASE_INIT(heavy::Context& Context) {
  Context.DialectRegistry->insert<heavy::HeavyDialect>();
  // syntax
  HEAVY_BASE_VAR(define)          = heavy::builtins::define;
  HEAVY_BASE_VAR(define_binding)  = heavy::builtins::define_binding;
  HEAVY_BASE_VAR(define_syntax)   = heavy::builtins::define_syntax;
  HEAVY_BASE_VAR(syntax_rules)    = heavy::builtins::syntax_rules;
  HEAVY_BASE_VAR(syntax_fn)       = heavy::builtins::syntax_fn;
  HEAVY_BASE_VAR(syntax_error)    = heavy::builtins::syntax_error;
  HEAVY_BASE_VAR(if_)             = heavy::builtins::if_;
  HEAVY_BASE_VAR(lambda)          = heavy::builtins::lambda;
  HEAVY_BASE_VAR(case_lambda)     = heavy::builtins::case_lambda;
  HEAVY_BASE_VAR(quasiquote)      = heavy::builtins::quasiquote;
  HEAVY_BASE_VAR(quote)           = heavy::builtins::quote;
  HEAVY_BASE_VAR(set)             = heavy::builtins::set;
  HEAVY_BASE_VAR(begin)           = heavy::builtins::begin;
  HEAVY_BASE_VAR(cond_expand)     = heavy::builtins::cond_expand;
  HEAVY_BASE_VAR(define_library)  = heavy::builtins::define_library;
  HEAVY_BASE_VAR(export_)         = heavy::builtins::export_;
  HEAVY_BASE_VAR(include)         = heavy::builtins::include_;
  HEAVY_BASE_VAR(include_ci)      = heavy::builtins::include_ci;
  HEAVY_BASE_VAR(include_library_declarations)
    = heavy::builtins::include_library_declarations;


  HEAVY_BASE_VAR(source_loc)      = heavy::builtins::source_loc;
  HEAVY_BASE_VAR(source_loc_valid) = heavy::builtins::source_loc_valid;
  HEAVY_BASE_VAR(dump_source_loc) = heavy::builtins::dump_source_loc;

  // functions
  HEAVY_BASE_VAR(add)     = heavy::builtins::add;
  HEAVY_BASE_VAR(sub)     = heavy::builtins::sub;
  HEAVY_BASE_VAR(div)     = heavy::builtins::div;
  HEAVY_BASE_VAR(mul)     = heavy::builtins::mul;
  HEAVY_BASE_VAR(is_positive) = heavy::builtins::is_positive;
  HEAVY_BASE_VAR(is_zero) = heavy::builtins::is_zero;
  HEAVY_BASE_VAR(list)    = heavy::builtins::list;
  HEAVY_BASE_VAR(length)  = heavy::builtins::length;
  HEAVY_BASE_VAR(cons)    = heavy::builtins::cons;
  HEAVY_BASE_VAR(source_cons) = heavy::builtins::source_cons;
  HEAVY_BASE_VAR(car)     = heavy::builtins::car;
  HEAVY_BASE_VAR(cdr)     = heavy::builtins::cdr;
  HEAVY_BASE_VAR(append)  = heavy::builtins::append;
  HEAVY_BASE_VAR(vector)  = heavy::builtins::vector;
  HEAVY_BASE_VAR(make_vector) = heavy::builtins::make_vector;
  HEAVY_BASE_VAR(vector_length) = heavy::builtins::vector_length;
  HEAVY_BASE_VAR(vector_ref) = heavy::builtins::vector_ref;
  HEAVY_BASE_VAR(vector_set) = heavy::builtins::vector_set;
  HEAVY_BASE_VAR(make_list) = heavy::builtins::make_list;
  HEAVY_BASE_VAR(list_ref) = heavy::builtins::list_ref;
  HEAVY_BASE_VAR(list_set) = heavy::builtins::list_set;
  HEAVY_BASE_VAR(dump)    = heavy::builtins::dump;
  HEAVY_BASE_VAR(write)   = heavy::builtins::write;
  HEAVY_BASE_VAR(newline) = heavy::builtins::newline;
  HEAVY_BASE_VAR(string_append) = heavy::builtins::string_append;
  HEAVY_BASE_VAR(string_copy) = heavy::builtins::string_copy;
  HEAVY_BASE_VAR(string_length) = heavy::builtins::string_length;
  HEAVY_BASE_VAR(string_ref) = heavy::builtins::string_ref;
  HEAVY_BASE_VAR(number_to_string) = heavy::builtins::number_to_string;
  HEAVY_BASE_VAR(eq)      = heavy::builtins::eqv;
  HEAVY_BASE_VAR(equal)   = heavy::builtins::equal;
  HEAVY_BASE_VAR(eqv)     = heavy::builtins::eqv;
  HEAVY_BASE_VAR(call_cc) = heavy::builtins::call_cc;
  HEAVY_BASE_VAR(values)  = heavy::builtins::values;
  HEAVY_BASE_VAR(call_with_values) = heavy::builtins::call_with_values;
  HEAVY_BASE_VAR(with_exception_handler)
    = heavy::builtins::with_exception_handler;
  HEAVY_BASE_VAR(raise)   = heavy::builtins::raise;
  HEAVY_BASE_VAR(error)   = heavy::builtins::error;
  HEAVY_BASE_VAR(dynamic_wind) = heavy::builtins::dynamic_wind;

  HEAVY_BASE_VAR(eval)    = heavy::builtins::eval;
  HEAVY_BASE_VAR(op_eval) = heavy::builtins::op_eval;
  HEAVY_BASE_VAR(compile) = heavy::builtins::compile;

  HEAVY_BASE_VAR(is_boolean) = heavy::builtins::is_boolean;
  HEAVY_BASE_VAR(is_bytevector) = heavy::builtins::is_bytevector;
  HEAVY_BASE_VAR(is_char) = heavy::builtins::is_char;
  HEAVY_BASE_VAR(is_eof_object) = heavy::builtins::is_eof_object;
  HEAVY_BASE_VAR(is_null) = heavy::builtins::is_null;
  HEAVY_BASE_VAR(is_number) = heavy::builtins::is_number;
  HEAVY_BASE_VAR(is_pair) = heavy::builtins::is_pair;
  HEAVY_BASE_VAR(is_port) = heavy::builtins::is_port;
  HEAVY_BASE_VAR(is_procedure) = heavy::builtins::is_procedure;
  HEAVY_BASE_VAR(is_string) = heavy::builtins::is_string;
  HEAVY_BASE_VAR(is_symbol) = heavy::builtins::is_symbol;
  HEAVY_BASE_VAR(is_vector) = heavy::builtins::is_vector;
  HEAVY_BASE_VAR(is_mlir_operation) = heavy::builtins::is_mlir_operation;
  HEAVY_BASE_VAR(is_source_value) = heavy::builtins::is_source_value;
  HEAVY_BASE_VAR(apply) = heavy::builtins::apply;
  HEAVY_BASE_VAR(make_syntactic_closure)
                        = heavy::builtins::make_syntactic_closure;
  HEAVY_BASE_VAR(load_plugin) = heavy::builtins::load_plugin;
  HEAVY_BASE_VAR(load_builtin) = heavy::builtins::load_builtin;
}

// initializes the module and loads lookup information
// for the compiler
void HEAVY_BASE_LOAD_MODULE(heavy::Context& Context) {
  // Note: We call HEAVY_BASE_INIT in the constructor of Context.
  heavy::initModuleNames(Context, HEAVY_BASE_LIB_STR, {
    // syntax
    {"define",        HEAVY_BASE_VAR(define)},
    {"define-binding",
                      HEAVY_BASE_VAR(define_binding)},
    {"define-syntax", HEAVY_BASE_VAR(define_syntax)},
    {"if",            HEAVY_BASE_VAR(if_)},
    {"lambda",        HEAVY_BASE_VAR(lambda)},
    {"quasiquote",    HEAVY_BASE_VAR(quasiquote)},
    {"quote",         HEAVY_BASE_VAR(quote)},
    {"set!",          HEAVY_BASE_VAR(set)},
    {"syntax-rules",  HEAVY_BASE_VAR(syntax_rules)},
    {"syntax-fn",     HEAVY_BASE_VAR(syntax_fn)},
    {"syntax-error",  HEAVY_BASE_VAR(syntax_error)},
    {"case-lambda",   HEAVY_BASE_VAR(case_lambda)},
    {"begin",         HEAVY_BASE_VAR(begin)},
    {"cond-expand",   HEAVY_BASE_VAR(cond_expand)},
    {"define-library",HEAVY_BASE_VAR(define_library)},
    {"export",        HEAVY_BASE_VAR(export_)},
    {"include",       HEAVY_BASE_VAR(include)},
    {"include-ci",    HEAVY_BASE_VAR(include_ci)},
    {"include-library-declarations",
      HEAVY_BASE_VAR(include_library_declarations)},

    {"source-loc",    HEAVY_BASE_VAR(source_loc)},
    {"source-loc-valid", HEAVY_BASE_VAR(source_loc_valid)},
    {"dump-source-loc", HEAVY_BASE_VAR(dump_source_loc)},
    {"parse-source-file",
                      HEAVY_BASE_VAR(parse_source_file).get(Context)},

    // functions
    {"+",       HEAVY_BASE_VAR(add)},
    {"-",       HEAVY_BASE_VAR(sub)},
    {"/",       HEAVY_BASE_VAR(div)},
    {"*",       HEAVY_BASE_VAR(mul)},
    {"positive?", HEAVY_BASE_VAR(is_positive)},
    {"zero?", HEAVY_BASE_VAR(is_zero)},
    {"list",    HEAVY_BASE_VAR(list)},
    {"length",  HEAVY_BASE_VAR(length)},
    {"cons",    HEAVY_BASE_VAR(cons)},
    {"source-cons", HEAVY_BASE_VAR(source_cons)},
    {"car",     HEAVY_BASE_VAR(car)},
    {"cdr",     HEAVY_BASE_VAR(cdr)},
    {"append",  HEAVY_BASE_VAR(append)},
    {"vector", HEAVY_BASE_VAR(vector)},
    {"make-vector", HEAVY_BASE_VAR(make_vector)},
    {"vector-length", HEAVY_BASE_VAR(vector_length)},
    {"vector-ref", HEAVY_BASE_VAR(vector_ref)},
    {"vector-set!", HEAVY_BASE_VAR(vector_set)},
    {"make-list", HEAVY_BASE_VAR(make_list)},
    {"list-ref", HEAVY_BASE_VAR(list_ref)},
    {"list-set!", HEAVY_BASE_VAR(list_set)},
    {"dump",    HEAVY_BASE_VAR(dump)},
    {"write",   HEAVY_BASE_VAR(write)},
    {"newline", HEAVY_BASE_VAR(newline)},
    {"string-append", HEAVY_BASE_VAR(string_append)},
    {"string-copy", HEAVY_BASE_VAR(string_copy)},
    {"string-length", HEAVY_BASE_VAR(string_length)},
    {"string-ref", HEAVY_BASE_VAR(string_ref)},
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
    {"apply", HEAVY_BASE_VAR(apply)},
    {"make-syntactic-closure", HEAVY_BASE_VAR(make_syntactic_closure)},
    {"load-plugin", HEAVY_BASE_VAR(load_plugin)},
    {"load-builtin", HEAVY_BASE_VAR(load_builtin)},
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
