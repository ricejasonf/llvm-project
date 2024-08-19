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
#include "llvm/ADT/StringExtras.h"
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
heavy::ExternFunction HEAVY_BASE_VAR(write);
heavy::ExternFunction HEAVY_BASE_VAR(newline);
heavy::ExternFunction HEAVY_BASE_VAR(eq);
heavy::ExternFunction HEAVY_BASE_VAR(equal);
heavy::ExternFunction HEAVY_BASE_VAR(eqv);
heavy::ExternFunction HEAVY_BASE_VAR(call_cc);
heavy::ExternFunction HEAVY_BASE_VAR(with_exception_handler);
heavy::ExternFunction HEAVY_BASE_VAR(raise);
heavy::ExternFunction HEAVY_BASE_VAR(error);
heavy::ExternFunction HEAVY_BASE_VAR(dynamic_wind);

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
  // Unwrap to check that we are looking at a symbol or binding.
  if (heavy::SyntaxClosure* SC = llvm::dyn_cast<SyntaxClosure>(S))
    S = SC->Node;

  if (!isa<Binding, Symbol, ExternName, SyntaxClosure>(S))
    return OG.SetError("expecting symbol", P2);

  // Go with the original unwrapped expression.
  S = P2->Car;

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
  std::string MangledName = OG.mangleModule(NameSpec);
  if (MangledName.size() == 0) {
    C.OpGen->SetError("library name is invalid", NameSpec);
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
void call_cc(Context& C, ValueRefs Args) {
  if (Args.size() != 1) return C.RaiseError("invalid arity");
  C.CallCC(Args[0]);
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

void dynamic_wind(Context& C, ValueRefs Args) {
  if (Args.size() != 3) return C.RaiseError("invalid arity");
  C.DynamicWind(Args[0], Args[1], Args[2]);
}

void eval(Context& C, ValueRefs Args) {
  if (Args.size() != 2) {
    return C.RaiseError("invalid arity");
  }
  Value ExprOrDef       = Args[0];
  Value EnvSpec         = Args[1];
  Value EvalRaw = Args.size() == 3 ? Args[2] : HEAVY_BASE_VAR(op_eval);
  // Wrap Eval as escape procedure to prevent the
  // user from capturing the environment with compiler objects
  // that will be deleted.
  Value Eval = C.CreateBinding(Undefined());

  // Push the compile step.
  C.PushCont([](heavy::Context& C, heavy::ValueRefs) {
    heavy::ValueRefs NewArgs = C.getCaptures();
    // Unwrap the escape proc.
    NewArgs[2] = cast<heavy::Binding>(NewArgs[2])->getValue();
    C.Apply(HEAVY_BASE_VAR(compile), NewArgs);
  }, CaptureList{ExprOrDef, EnvSpec, Eval});

  // Save the 
  C.SaveEscapeProc(Eval, [](heavy::Context& C, ValueRefs Args) {
    Value EvalRaw = C.getCapture(0);
    // Skip the compile step that we just pushed above.
    C.PopCont();
    C.Apply(EvalRaw, Args);
  }, CaptureList{EvalRaw});
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
    C.OpGen->SetError("invalid environment specifier", EnvSpec);
    return;
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
