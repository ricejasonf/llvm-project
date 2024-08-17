//===--- Context.cpp - HeavyScheme Context Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations for heavy scheme Context.
//
//===----------------------------------------------------------------------===//

#include "heavy/Builtins.h"
#include "heavy/Dialect.h"
#include "heavy/Context.h"
#include "heavy/Lexer.h"
#include "heavy/OpGen.h"
#include "heavy/Mangle.h"
#include "heavy/Source.h"
#include "heavy/Value.h"
#include "heavy/ValueVisitor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h" // TODO move to OpGen
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Unicode.h"
#include <algorithm>
#include <cstring>

using namespace heavy;

void ValueBase::dump() {
  write(llvm::errs(), this);
  llvm::errs() << '\n';
}

void Value::dump() {
  write(llvm::errs(), *this);
  llvm::errs() << '\n';
}

// NameForImportVar - Because we use String* for storing mangled
//                    names, we need to create one for "import"
//                    which is special. We also give it a relatively
//                    simple, readable symbol name.
heavy::ExternString<20> NameForImportVar;
heavy::ExternSyntax<> _HEAVY_import;

Context::Context()
  : ContinuationStack<Context>(),
    ContextLocalLookup(),
    Heap(MiB),
    EnvStack(Empty()),
    MLIRContext(std::make_unique<mlir::MLIRContext>()),
    OpGen(nullptr)
{
  NameForImportVar = "_HEAVY_import";
  _HEAVY_import = heavy::base::import_;
  RegisterModule(HEAVY_BASE_LIB_STR, HEAVY_BASE_LOAD_MODULE);
}

Context::~Context() = default;

Environment::Environment(Environment* Parent)
  : ValueBase(ValueKind::Environment),
    OpGen(nullptr),
    Parent(Parent),
    EnvMap(0)
{ }
Environment::Environment(Context& C, std::string ModulePrefix)
  : ValueBase(ValueKind::Environment),
    OpGen(std::make_unique<heavy::OpGen>(C, std::move(ModulePrefix))),
    Parent(nullptr),
    EnvMap(0)
{ }
Environment::~Environment() = default;

Environment* Context::getTopLevelEnvironment() {
  // EnvStack is always an Environment or an improper
  // list ending with an Environment.
  if (auto* E = dyn_cast<Environment>(EnvStack)) {
    return E;
  }
  while (Pair* P = dyn_cast<Pair>(EnvStack)) {
    if (auto* E = dyn_cast<Environment>(P->Cdr)) {
      return E;
    }
  }
  llvm_unreachable("EnvStack should have an Environment.");
}

Syntax* Context::CreateSyntaxWithOp(mlir::Operation* Op) {
  auto SyntaxOp = cast<heavy::SyntaxOp>(Op);
  auto Fn = [SyntaxOp](heavy::Context& C, ValueRefs Args) -> void {
    heavy::Value Input = Args[0];
    invokeSyntaxOp(C, SyntaxOp, Input);
    // The contained OpGenOp will call C.Apply(...).
  };
  return CreateSyntax(Fn);
}

Module* Context::RegisterModule(llvm::StringRef MangledName,
                                heavy::ModuleLoadNamesFn* LoadNames) {
  auto Result = Modules.try_emplace(MangledName,
        std::make_unique<Module>(*this, LoadNames));
  auto Itr = Result.first;
  bool DidInsert = Result.second;

  assert(DidInsert && "module should be created only once");
  return Itr->second.get();
}

#if 0 // TODO Make this create a module storing a function
              to initialize it from a ModuleOp
Module* Context::RegisterModule(mlir::Operation* ModuleOp) {
  // TODO Get the "load_module" FuncOp
  mlir::ModuleOp Op = cast<mlir::ModuleOp>(ModuleOp);
  llvm::StringRef MangledName = Op.getName().getValueOr("");
  return RegisterModule(MangledName,
      std::make_unique<Module>(*this, ModuleOp));
}
#endif

void Context::Import(heavy::ImportSet* ImportSet) {
  Environment* Env = dyn_cast<Environment>(EnvStack);
  if (!Env) {
    // Import doesn't work in local scope.
    SetError("unexpected import", ImportSet);
    return;
  }

  // This should probably be recursive scheme calls.
  for (EnvBucket ImportVal : *ImportSet) {
    String* Name = ImportVal.first;
    // If there is no name just skip it (it was filtered out).
    if (!Name) continue;
    if (!Env->ImportValue(ImportVal)) {
      String* ErrMsg = CreateString("imported name already exists: ",
                                    Name->getView());
      SetError(ErrMsg, ImportSet);
      return;
    }
  }

  Cont();
}

EnvFrame* Context::PushLambdaFormals(Value Formals,
                                     bool& HasRestParam) {
  llvm::SmallVector<Symbol*, 8> Names;
  HasRestParam = false;
  if (CheckLambdaFormals(Formals, Names,
                         HasRestParam)) return nullptr;

  llvm::SmallSet<llvm::StringRef, 8> NameSet;
  // ensure uniqueness of names
  for (Symbol* Name : Names) {
    auto Result = NameSet.insert(Name->getVal());
    if (!Result.second) {
      SetError("duplicate parameter name", Name);
    }
  }

  return PushEnvFrame(Names);
}

EnvFrame* Context::PushEnvFrame(llvm::ArrayRef<Symbol*> Names) {
  EnvFrame* E = CreateEnvFrame(Names);
  EnvStack = CreatePair(E, EnvStack);
  return E;
}

void Context::PopEnvFrame() {
  // We want to remove the current local scope
  // and assert that we aren't popping anything else
  // Walk through the local bindings
  Value Env = EnvStack;
  Pair* EnvPair;
  while ((EnvPair = dyn_cast<Pair>(Env))) {
    if (!isa<Binding>(EnvPair->Car)) break;
    Env = EnvPair->Cdr;
  }
  assert(isa<EnvFrame>(EnvPair->Car) &&
      "Scope must be in an EnvFrame");
  EnvStack = EnvPair->Cdr;
}

void Context::PushLocalBinding(Binding* B) {
  EnvStack = CreatePair(B, EnvStack);
}

// CheckLambdaFormals - Returns true on error
bool Context::CheckLambdaFormals(Value Formals,
                               llvm::SmallVectorImpl<Symbol*>& Names,
                               bool& HasRestParam) {
  Value V = Formals;
  if (isa<Empty>(V)) return false;
  if (isa<Symbol>(V)) {
    // If the formals are just a Symbol
    // or an improper list ending with a
    // Symbol, then that Symbol is a "rest"
    // parameter that binds remaining
    // arguments as a list.
    Names.push_back(cast<Symbol>(V));
    HasRestParam = true;
    return false;
  }

  Pair* P = dyn_cast<Pair>(V);
  if (!P || !isa<Symbol>(P->Car)) {
    SetError("invalid formals syntax", V);
    return true;
  }
  Names.push_back(cast<Symbol>(P->Car));
  return CheckLambdaFormals(P->Cdr, Names, HasRestParam);
}

// The Stack is an improper list ending with an Environment
EnvEntry Context::Lookup(Symbol* Name, Value Stack) {
  if (auto* E = dyn_cast<Environment>(Stack)) {
    EnvEntry Result = E->Lookup(Name);
    if (!Result && Name->equals("import")) {
      return EnvEntry{_HEAVY_import, NameForImportVar};
    }
    return Result;
  }
  EnvEntry Result = {};
  Value V    = cast<Pair>(Stack)->Car;
  Value Next = cast<Pair>(Stack)->Cdr;
  switch (V.getKind()) {
    case ValueKind::Binding:
      Result = cast<Binding>(V)->Lookup(Name);
      break;
    case ValueKind::EnvFrame:
      Result = cast<EnvFrame>(V)->Lookup(Name);
      break;
    case ValueKind::Module:
      Result = cast<Module>(V)->Lookup(Name);
      break;
    case ValueKind::ImportSet:
      Result = cast<ImportSet>(V)->Lookup(*this, Name);
      break;
    case ValueKind::Environment: {
      Result = cast<Environment>(V)->Lookup(Name);
      break;
    }
    default:
      llvm_unreachable("Invalid Lookup Type");
  }
  if (Result) return Result;
  return Lookup(Name, Next);
}

mlir::Operation* Context::getModuleOp() {
  return ModuleOp;
}
void Context::dumpModuleOp() {
  return cast<mlir::ModuleOp>(ModuleOp).dump();
}

void Context::verifyModule() {
  mlir::ModuleOp M = cast<mlir::ModuleOp>(ModuleOp);
  if (mlir::failed(mlir::verify(M))) {
    llvm::errs() << "error: verfication failed\n";
  }
}

void Context::PushTopLevel(heavy::Value V) {
  OpGen->VisitTopLevel(V);
}

namespace {
class Writer : public ValueVisitor<Writer> {
  friend class ValueVisitor<Writer>;
  unsigned IndentationLevel = 0;
  llvm::raw_ostream &OS;

  bool isIdentifier(heavy::String* Str) {
    // Use the lexer to identify an identifier.
    heavy::Lexer Lexer(Str->getView());
    heavy::Token Tok;
    Lexer.Lex(Tok);
    return (Tok.Kind == tok::identifier &&
            Tok.getLength() == Str->size());
  }

public:
  Writer(llvm::raw_ostream& OS)
    : OS(OS)
  { }

private:
  void PrintFormattedWhitespace() {
    // We could handle indentation and new lines
    // more dynamically here probably
    assert(IndentationLevel < 100 && "IndentationLevel overflow suspected");

    OS << '\n';
    for (unsigned i = 0; i < IndentationLevel; ++i) {
      OS << ' ';
    }
  }

  void VisitValue(Value V) {
    OS << "#<Value of Kind:"
       << getKindName(V.getKind())
       << ">";
  }

  void VisitBool(Bool V) {
    if (V)
      OS << "#t";
    else
      OS << "#f";
  }

  void VisitByteVector(ByteVector* BV) {
    OS << "#u8(";
    llvm::StringRef Bytes = BV->getView();
    llvm::SmallVector<char, 4> HexCode;
    auto encode_hex = [&HexCode](unsigned char c) {
      detail::encode_hex(c, HexCode);
      if (HexCode.size() == 1)
        HexCode.insert(HexCode.begin(), '0');
    };
    if (!Bytes.empty()) {
      unsigned char c = Bytes[0];
      encode_hex(c);
      OS << "#x" << HexCode;
      Bytes = Bytes.drop_front();
    }
    for (unsigned char c : Bytes) {
      HexCode.clear();
      encode_hex(c);
      OS << " #x" << HexCode;
    }
    OS << ")";
  }

  void VisitEmpty(Empty) {
    OS << "()";
  }

  void VisitError(Error* E) {
    OS << "(error ";
    Visit(E->getMessage());
    OS << " ";
    VisitCdr(E->getIrritants());
  }

  void VisitExternName(ExternName* E) {
    OS << "#_" << E->getView();
  }

  void VisitInt(Int V) { OS << int32_t{V}; }
  void VisitFloat(Float* V) {
    llvm::SmallVector<char, 16> Buffer;
    llvm::APFloat Val = V->getVal();
    if (!Val.isInfinity() && !Val.isNaN()) {
      Val.toString(Buffer);
      llvm::transform(Buffer, Buffer.begin(), llvm::toLower);
      OS << Buffer;
    } else if (Val.isInfinity() && !Val.isNegative()) {
      OS << "+inf.0";
    } else if (Val.isInfinity()) {
      OS << "-inf.0";
    } else if (Val.isNegative()) {
      OS << "-nan.0";
    } else {
      OS << "+nan.0";
    }
  }

  void VisitPair(Pair* P) {
    // Iterate the whole list to print
    // in standard list notation
    ++IndentationLevel;
    OS << '(';
    Visit(P->Car);
    VisitCdr(P->Cdr);
    --IndentationLevel;
  }

  // VisitCdr - Print the rest of a list
  void VisitCdr(Value Cdr) {
    while (Pair* P = dyn_cast<Pair>(Cdr)) {
      OS << ' ';
      //PrintFormattedWhitespace();
      Visit(P->Car);
      Cdr = P->Cdr;
    };

    if (!isa<Empty>(Cdr)) {
      OS << " . ";
      Visit(Cdr);
    }
    OS << ')';
  }

  void VisitQuote(Quote* Q) {
    OS << "(quote ";
    Visit(Q->Val);
    OS << ")";
  }

  void VisitVector(Vector* Vec) {
    OS << "#(";
    ArrayRef<Value> Xs = Vec->getElements();
    if (!Xs.empty()) {
      Visit(Xs[0]);
      Xs = Xs.drop_front(1);
      for (Value X : Xs) {
        OS << ' ';
        Visit(X);
      }
    }
    OS << ')';
  }

  void WriteLiteral(llvm::StringRef Str, char Delimiter) {
    OS << Delimiter;
    while (Str.size() > 0) {
      unsigned ConsumeLen = 1;
      // Escape all special characters.
      switch(Str[0]) {
        case '\a':
          OS << "\\a";
          break;
        case '\b':
          OS << "\\b";
          break;
        case '\t':
          OS << "\\t";
          break;
        case '\n':
          OS << "\\n";
          break;
        case '\r':
          OS << "\\r";
          break;
        case '"':
          if (Delimiter == '"')
            OS << "\\\"";
          else
            OS << '"';
          break;
        case '\\':
          // Output an escaped backslash.
          // ie Output 2 backslashes.
          OS << "\\\\";
          break;
        case '|':
          if (Delimiter == '|')
            OS << "\\|";
          else
            OS << '|';
          break;
        default: {
          auto [Decoded, Length] = detail::Utf8View(Str).decode_front();
          if (llvm::sys::unicode::isPrintable(Decoded)) {
            OS << Str.take_front(Length);
            ConsumeLen = Length;
          } else {
            // Handle non-printable characters as escaped
            // hexadecimal codes.
            llvm::SmallVector<char, 8> HexCode;
            detail::encode_hex(Decoded, HexCode);
            OS << "\\x" << HexCode << ';';
          }

        }
      }
      Str = Str.drop_front(ConsumeLen);
    }
    OS << Delimiter;
  }

  void VisitSymbol(Symbol* S) {
    // If the symbol is a valid identifer then print it raw.
    if (isIdentifier(S->getString())) {
      OS << S->getView();
    } else {
      // Otherwise it must be escaped via | |
      WriteLiteral(S->getView(), '|');
    }
  }

  void VisitString(String* S) {
    WriteLiteral(S->getView(), '"');
  }

  void VisitChar(heavy::Char Char) {
    uint32_t C = uint32_t(Char);
    OS << "#\\";
    switch (char(C)) {
      case '\0':
        OS << "null";
        break;
      case '\a':
        OS << "alarm";
        break;
      case '\b':
        OS << "backspace";
        break;
      case '\x7F':
        OS << "delete";
        break;
      case '\x1B':
        OS << "escape";
        break;
      case '\n':
        OS << "newline";
        break;
      case '\r':
        OS << "return";
        break;
      case ' ':
        OS << "space";
        break;
      case '\t':
        OS << "tab";
        break;
      default:
        if (llvm::sys::unicode::isPrintable(C) &&
            !llvm::sys::unicode::isFormatting(C)) {
          llvm::SmallVector<char, 4> ByteSequence;
          detail::encode_utf8(C, ByteSequence);
          OS << ByteSequence;
        } else {
          // Handle non-printable characters as escaped
          // hexadecimal codes.
          llvm::SmallVector<char, 8> HexCode;
          detail::encode_hex(C, HexCode);
          OS << "x" << HexCode;
        }
        break;
    }
  }
};

} // end anon namespace

namespace heavy {
void write(llvm::raw_ostream& OS, Value V) {
  Writer W(OS);
  return W.Visit(V);
}

void compile(Context& C, Value V, Value Env, Value Handler) {
  heavy::Value Args[3] = {V, Env, Handler};
  C.Apply(HEAVY_BASE_VAR(compile), ValueRefs(Args));
}

void eval(Context& C, Value V, Value Env) {
  heavy::Value Args[2] = {V, Env};
  C.Apply(HEAVY_BASE_VAR(eval), ValueRefs(Args));
}

// this handles non-immediate values
// and assumes the values have the same kind
bool equal_slow(Value V1, Value V2) {
  assert(V1.getKind() == V2.getKind() &&
      "inputs are expected to have same kind");

  switch (V1.getKind()) {
  case ValueKind::String:
    return cast<String>(V1)->equals(cast<String>(V2));
  case ValueKind::Pair:
  case ValueKind::PairWithSource: {
    Pair* P1 = cast<Pair>(V1);
    Pair* P2 = cast<Pair>(V2);
    // FIXME this does not handle cyclic refs
    return equal(P1->Car, P2->Car) &&
           equal(P1->Cdr, P2->Cdr);
  }
  case ValueKind::Vector:
    llvm_unreachable("TODO");
    return false;
  default:
      return eqv(V1, V2);
  }
}

// this handles non-immediate values
// and assumes the values have the same kind
bool eqv_slow(Value V1, Value V2) {
  assert(V1.getKind() == V2.getKind() &&
      "inputs are expected to have same kind");
  switch (V1.getKind()) {
  case ValueKind::Symbol:
    return cast<Symbol>(V1)->equals(
              cast<Symbol>(V2));
  case ValueKind::Float:
    return cast<Float>(V1)->getVal() ==
              cast<Float>(V2)->getVal();
  default:
      return false;
  }
}

} // end namespace heavy

void Context::EmitStackSpaceError() {
  // TODO We should probably clear the stack
  //      and run this as a continuation
  SetError("insufficient stack space");
}

EnvEntry ImportSet::Lookup(heavy::Context& C, Symbol* S) {
  switch(Kind) {
  case ImportKind::Library:
    return cast<Module>(Specifier)->Lookup(S);
  case ImportKind::Only:
    // Specifier is a list of Symbols
    return isInIdentiferList(S) ? Parent->Lookup(C, S) : EnvEntry{};
  case ImportKind::Except:
    // Specifier is a list of Symbols
    return !isInIdentiferList(S) ? Parent->Lookup(C, S) : EnvEntry{};
  case ImportKind::Prefix: {
    llvm::StringRef Str = S->getVal();
    // Specifier is just a Symbol
    llvm::StringRef Prefix = cast<Symbol>(Specifier)->getVal();
    if (!Str.consume_front(Prefix)) return {};
    S = C.CreateSymbol(Str, S->getSourceLocation());
    return Parent->Lookup(C, S);
  }
  case ImportKind::Rename:
    // Specifier is a list of pairs of symbols
    return LookupFromPairs(C, S);
  }
  llvm_unreachable("invalid import kind");
}

EnvEntry ImportSet::LookupFromPairs(heavy::Context& C, Symbol* S) {
  // The syntax of Specifier should be checked already
  assert(Kind == ImportKind::Rename && "expecting import rename");
  Value CurrentRow = Specifier;
  while (Pair* P = dyn_cast<Pair>(CurrentRow)) {
    Pair* Row = cast<Pair>(P->Car);
    Symbol* Key   = cast<Symbol>(Row->Car);
    Symbol* Value = cast<Symbol>(cast<Pair>(Row->Cdr)->Car);
    if (S->equals(Value)) return Parent->Lookup(C, Key);
    CurrentRow = P->Cdr;
  }
  return {};
}

String* ImportSet::FilterName(heavy::Context& C, String* S) {
  // recurse all the way to the module and
  // traverse back down removing or replacing the String
  if (Kind == ImportKind::Library) {
    Module* M = cast<Module>(Specifier);
    assert(M->Lookup(S) && "filtered name not in library");
    return S;
  }
  S = Parent->FilterName(C, S);
  switch(Kind) {
  case ImportKind::Only:
    // filter if not in the list
    return isInIdentiferList(S) ? S : nullptr;
  case ImportKind::Except:
    // filter if its in the list
    return !isInIdentiferList(S) ? S : nullptr;
  case ImportKind::Prefix: {
    // add the prefix
    llvm::StringRef Str = S->getView();
    llvm::StringRef Prefix = cast<Symbol>(Specifier)->getVal();
    return C.CreateIdTableEntry(Prefix, Str);
  }
  case ImportKind::Rename:
    return FilterFromPairs(C, S);
  default:
    llvm_unreachable("invalid import kind");
  }
}

String* ImportSet::FilterFromPairs(heavy::Context& C, String* S) {
  assert(Kind == ImportKind::Rename && "expecting import rename");
  Value CurrentRow = Specifier;
  while (Pair* P = dyn_cast<Pair>(CurrentRow)) {
    Pair* Row = cast<Pair>(P->Car);
    String* Key   = cast<Symbol>(Row->Car)->getString();
    String* Value = cast<Symbol>(cast<Pair>(Row->Cdr)->Car)->getString();
    if (S->equals(Key)) return Value;
    CurrentRow = P->Cdr;
  }
  return S;
}

// CreateImportSet - Return a created import set from an import-spec,
//                   or nullptr if a continuation was called
//                   (ie for an error or LoadModule).
//
void Context::CreateImportSet(Value Spec) {
  ImportSet::ImportKind Kind;
  Symbol* Keyword = dyn_cast_or_null<Symbol>(Spec.car());
  if (!Keyword) {
    SetError("expecting import set", Spec);
    return;
  }
  // TODO perhaps we could intern these keywords in the IdTable
  if (Keyword->equals("only")) {
    Kind = ImportSet::ImportKind::Only;
  } else if (Keyword->equals("except")) {
    Kind = ImportSet::ImportKind::Except;
  } else if (Keyword->equals("rename")) {
    Kind = ImportSet::ImportKind::Rename;
  } else if (Keyword->equals("prefix")) {
    Kind = ImportSet::ImportKind::Prefix;
  } else {
    // ImportSet::ImportKind::Library;
    PushCont([this](Context& C, ValueRefs Args) {
      Module* M = cast<heavy::Module>(Args[0]);
      C.Cont(new (*this) ImportSet(M));
    });
    LoadModule(Spec);
    return;
  }

  Value ParentSpec = cadr(Spec);
  Spec = Spec.cdr().cdr();

  PushCont([this, Kind](Context& C, ValueRefs Args) {
    ImportSet* Parent = cast<ImportSet>(Args[0]);
    Value Spec = C.getCapture(0);

    if (Kind == ImportSet::ImportKind::Prefix) {
      Symbol* Prefix = dyn_cast_or_null<Symbol>(C.car(Spec));
      if (!Prefix) {
        C.SetError("expected identifier for prefix", Spec);
        return;
      }
      if (!isa_and_nonnull<Empty>(C.cdr(Spec))) {
        C.SetError("expected end of list", Spec);
        return;
      }
      C.Cont(new (*this) ImportSet(Kind, Parent, Prefix));
      return;
    }

    // Identifier (pairs) list
    // Check the syntax and that each provided name
    // exists in the parent import set
    if (!isa_and_nonnull<Pair>(Spec)) {
      C.SetError("expecting list");
      return;
    }
    // Spec is now the list of ids to be used by ImportSet

    Value Current = Spec;
    Symbol* Name = nullptr;
    while (Pair* P = dyn_cast<Pair>(Current)) {
      switch (Kind) {
      case ImportSet::ImportKind::Only:
      case ImportSet::ImportKind::Except:
        Name = dyn_cast<Symbol>(P->Car);
        break;
      case ImportSet::ImportKind::Rename: {
        Pair* P2 = dyn_cast<Pair>(P->Car);
        if (!P2) {
         C.SetError("expected pair", P);
         return;
        }
        Name = dyn_cast<Symbol>(P2->Car);
        Symbol* Rename = dyn_cast_or_null<Symbol>(Value(P2).cadr());
        if (Name || Rename) {
          C.SetError("expected pair of identifiers", P2);
          return;
        }
        break;
      }
      default:
        llvm_unreachable("invalid import set kind");
      }

      EnvEntry LookupResult = Parent->Lookup(C, Name);
      if (!LookupResult) {
        C.SetError("name does not exist in import set");
        return;
      }
      Current = P->Cdr;
    }
    C.Cont(new (*this) ImportSet(Kind, Parent, Spec));
  }, CaptureList{Spec});
  CreateImportSet(ParentSpec);
}

void Context::LoadModule(Value Spec, bool IsFileLoaded) {
  heavy::SourceLocation Loc = Spec.getSourceLocation();
  setLoc(Loc);
  heavy::Mangler Mangler(*this);
  // Name - The mangled module name
  std::string Name = Mangler.mangleModule(Spec);
  if (CheckError())
    return;

  std::unique_ptr<Module>& M = Modules[Name];
  if (M) {
    // Idempotently register the names of the globals.
    // (For external modules)
    M->LoadNames();
    Cont(M.get());
    return;
  }

  // Lookup the Module in ModuleOp
  mlir::ModuleOp TopOp = cast<mlir::ModuleOp>(ModuleOp);
  if (auto Op = dyn_cast_or_null<mlir::ModuleOp>(TopOp.lookupSymbol(Name))) {
    auto M = std::make_unique<Module>(*this);
    Module* MPtr = M.get();
    Modules[Name] = std::move(M);
    Value Args[] = {Op.getOperation()};
    PushCont([](Context& C, ValueRefs) {
      C.Cont(C.getCapture(0));
    }, CaptureList{MPtr});
    Apply(HEAVY_BASE_VAR(op_eval), Args);
    return;
  }

  // Load the module from an sld file.
  // Spec is validated by the above call to mangleModule.
  // Expect valid data, but use dyn_cast just in case.
  heavy::String* Filename = nullptr;
  {
    llvm::SmallString<128> FilenameBuffer;
    heavy::Value Current = Spec;
    auto* P = dyn_cast<heavy::Pair>(Current);
    auto* Symbol = dyn_cast<heavy::Symbol>(P->Car);
    FilenameBuffer += Symbol->getView();
    Current = P->Cdr;
    while ((P = dyn_cast<heavy::Pair>(Current))) {
      FilenameBuffer += '/';
      auto* Symbol = dyn_cast<heavy::Symbol>(P->Car);
      FilenameBuffer += Symbol->getView();
      Current = P->Cdr;
    }
    FilenameBuffer += ".sld";
    Filename = CreateString(FilenameBuffer);
  }

  if (IsFileLoaded) {
    // The file was already loaded, but we still do not have a module.
    heavy::String* ErrMsg = CreateString(Filename->getView(),
                              ": library was not defined");
    return RaiseError(ErrMsg, Value(Filename));
  }

  PushCont(
    [Loc](heavy::Context& C, heavy::ValueRefs Args) {
      Value Spec = C.getCapture(0);
      // Require the module this time.
      C.LoadModule(Spec, /*IsFileLoaded=*/true);
    }, CaptureList{Spec});
  IncludeModuleFile(Loc, Filename, std::move(Name));
}

// Allow the user to add cleanup routines to unload a module/library.
// It is currently used for destroying the instance of OpEval.
// TODO expose this via run-time function accessible in scheme land.
void Context::PushModuleCleanup(llvm::StringRef MangledName, Value Fn) {
  std::unique_ptr<Module>& M = Modules[MangledName];
  if (!M) {
    return RaiseError("module not loaded",
        Value(CreateString(MangledName)));
  }
  if (heavy::Lambda* Lambda = dyn_cast<heavy::Lambda>(Fn)) {
    M->PushCleanup(Lambda);
  } else {
    return RaiseError("expecting function", Fn);
  }
}

// IncludeModuleFile - Fulfill loading a module for `import`.
//                   - Create a temporary environment with
//                     the define-library syntax and library
//                     declarations already defined.
void Context::IncludeModuleFile(heavy::SourceLocation Loc,
                                heavy::String* Filename,
                                std::string ModuleMangledName) {
  heavy::Value Thunk = CreateLambda(
    [Loc](heavy::Context& C, heavy::ValueRefs Args) {
      C.PushCont(
        [](heavy::Context& C, heavy::ValueRefs Args) {
          C.OpGen->VisitTopLevelSequence(Args[0]);
        });
      heavy::Value Parse = HEAVY_BASE_VAR(parse_source_file).get(C);
      heavy::Value SourceVal = C.CreateSourceValue(Loc);
      heavy::Value Filename = C.getCapture(0);
      std::array<heavy::Value, 2> NewArgs = {SourceVal, Filename};
      C.Apply(Parse, NewArgs);
    }, CaptureList{Filename});

  // Make a new environment with `define-library`
  // and related syntax preloaded.
  heavy::Module* Base = Modules[HEAVY_BASE_LIB_STR].get();
  // The ModuleMangledName becomes the name of the module that
  // is generated from the top level of the file.
  ModuleMangledName += "__module_file";
  auto Env = std::make_unique<heavy::Environment>(*this, ModuleMangledName);
  for (llvm::StringRef Name : {"define-library",
                               "export",
                               "begin",
                               "include",
                               "include-ci",
                               "include-library-declarations",
                               "cond-expand"}) {
    String* Id = CreateIdTableEntry(Name);
    EnvEntry Entry = Base->Lookup(Id);
    if (Entry && Entry.MangledName != nullptr)
      Env->ImportValue(EnvBucket{Id, Entry});
  }

  heavy::String* MetaModuleName = CreateString(ModuleMangledName);
  PushCont([](heavy::Context& C, ValueRefs) {
    // Delete the meta module file module.
    heavy::String* ModuleName = cast<heavy::String>(C.getCapture(0));
    mlir::ModuleOp TopOp = cast<mlir::ModuleOp>(C.ModuleOp);
    if (mlir::Operation* Op = TopOp.lookupSymbol(ModuleName->getView())) {
      Op->erase();
    }

    C.Cont();
  }, CaptureList{MetaModuleName});
  WithEnv(std::move(Env), Thunk);
}

void Context::AddKnownAddress(llvm::StringRef MangledName, heavy::Value Value) {
  String* Name = CreateIdTableEntry(MangledName);
  KnownAddresses[Name] = Value;
}

Value Context::GetKnownValue(llvm::StringRef MangledName) {
  String* Name = CreateIdTableEntry(MangledName);
  // Could we possibly use dlsym or something here
  // for actual external values?
  heavy::Value Result = KnownAddresses.lookup(Name);
  return Result;
}

void heavy::initModule(heavy::Context& C, llvm::StringRef ModuleMangledName,
                  ModuleInitListTy InitList) {
  Module* M = C.Modules[ModuleMangledName].get();
  assert(M && "module must be registered");
  heavy::Mangler Mangler(C);
  for (ModuleInitListPairTy const& X : InitList) {
    Value Val = X.second;
    llvm::StringRef Id = X.first;
    std::string MangledName = Mangler.mangleVariable(ModuleMangledName, Id);
    registerModuleVar(C, M, MangledName, Id, Val);
    // Track valid values by their mangled names
    if (Val) {
      C.AddKnownAddress(MangledName, Val);
    }
  }
}

void heavy::registerModuleVar(heavy::Context& C,
                              heavy::Module* M,
                              llvm::StringRef VarSymbol,
                              llvm::StringRef VarId,
                              Value Val) {
  String* Id = C.CreateIdTableEntry(VarId);
  String* MangledName = C.CreateIdTableEntry(VarSymbol);
  M->Insert(EnvBucket{Id, EnvEntry{Val, MangledName}});
}

bool Context::CheckKind(ValueKind VK, Value V) {
  if (V.getKind() == VK) return false;
  String* S = CreateString(llvm::StringRef("invalid type "),
                           getKindName(V.getKind()),
                           llvm::StringRef(", expecting "),
                           getKindName(VK));
  RaiseError(S, V);
  return true;
}
bool Context::CheckNumber(Value V) {
  if (V.isNumber()) return false;
  String* S = CreateString(llvm::StringRef("invalid type "),
                           getKindName(V.getKind()),
                           llvm::StringRef(", expecting number"));
  RaiseError(S, V);
  return true;
}

void Context::SetError(Value E) {
  assert(isa<Error>(E) || isa<Exception>(E));
  Err = E;
  Raise(E);
}

void Context::SetErrorHandler(Value Handler) {
  assert((isa<Empty>(ExceptionHandlers) ||
          isa<Empty>(ExceptionHandlers.cdr())) &&
    "error handler may only be set at the bottom of the handler stack");

  ExceptionHandlers = Handler;
}

void Context::WithExceptionHandlers(Value NewHandlers, Value Thunk) {
  Value PrevHandlers = ExceptionHandlers;
  Value Before = CreateLambda([this](Context& C, ValueRefs) {
    Value NewHandlers = C.getCapture(0);
    this->ExceptionHandlers = NewHandlers;
    C.Cont(Undefined());
  }, {NewHandlers});
  Value After = CreateLambda([this](Context& C, ValueRefs) {
    Value PrevHandlers = C.getCapture(0);
    this->ExceptionHandlers = PrevHandlers;
    C.Cont(Undefined());
  }, {PrevHandlers});
  DynamicWind(Before, Thunk, After);
}

void Context::WithExceptionHandler(Value Handler, Value Thunk) {
  Value NewHandlers = CreatePair(Handler, ExceptionHandlers);
  WithExceptionHandlers(NewHandlers, Thunk);
}

void Context::Raise(Value Obj) {
  if (isa<Empty>(ExceptionHandlers)) {
    ClearStack();
    return Cont();
  }
  if (!isa<Pair>(ExceptionHandlers)) {
    if (isa<heavy::Error>(Obj)) {
      Value BottomHandler = ExceptionHandlers;
      return Apply(BottomHandler, Obj);
    } else {
      std::string Msg;
      llvm::raw_string_ostream Stream(Msg);
      write(Stream << "uncaught exception: ", Obj);
      return SetError(Msg, Obj);
    }
  }
  Pair* P = cast<Pair>(ExceptionHandlers);
  Value Handler = P->Car;
  Value PrevHandlers = P->Cdr;
  WithExceptionHandlers(PrevHandlers,
                        CreateLambda([](Context& C, ValueRefs) {
    Value Handler = C.getCapture(0);
    Value Obj = C.getCapture(1);
    C.PushCont([](Context& C, ValueRefs Args) {
      if (Args.size() != 1)
        C.RaiseError("exception handler expecting single argument");
      else
        C.Raise(Args[0]);
    });
    C.Apply(Handler, Obj);
  }, {Handler, Obj}));
}

void Context::RaiseError(String* Msg, llvm::ArrayRef<Value> IrrArgs) {
  heavy::SourceLocation Loc = this->Loc;
  Value IrrList = Empty();
  while (!IrrArgs.empty()) {
    Value Irr = IrrArgs.front();
    if (!Loc.isValid()) {
      Loc = Irr.getSourceLocation();
    }
    IrrList = CreatePair(Irr, IrrList);
    IrrArgs = IrrArgs.drop_front();
  }
  Value Error = CreateError(Loc, Msg, IrrList);
  Raise(Error);
}

namespace {
class LibraryEnv {
  // EnvPtr is meant to just keep the
  // object alive for the user when needed.
  std::unique_ptr<heavy::Environment> EnvPtr;
  heavy::Environment* Env;

public:
  LibraryEnv(std::unique_ptr<Environment> EP,
             heavy::Environment* E)
    : EnvPtr(std::move(EP)),
      Env(E)
  { }

  static void Wind(heavy::Context& Context,
                   std::unique_ptr<heavy::Environment> EnvPtr,
                   heavy::Environment* Env,
                   Value Thunk) {
    auto Ptr = std::make_unique<LibraryEnv>(std::move(EnvPtr), Env);
    heavy::OpGen* OpGen = Ptr->Env->GetOpGen();

    Value PrevEnv = Context.getEnvironment();
    heavy::OpGen* PrevOpGen = Context.OpGen;

    // Capture Env via scheme lambda so its references are checked
    // during garbage collection.
    Value Before = Context.CreateLambda([OpGen](heavy::Context& C,
                                                 ValueRefs) {
      Value Env = C.getCapture(0);
      C.setEnvironment(Env);
      C.OpGen = OpGen;
      C.Cont();
    }, CaptureList{Value((Ptr->Env))});

    // Since we don't own PrevEnv do not capture it in scheme land.
    Value After = Context.CreateLambda([PrevOpGen, PrevEnv, OpGen, Env](heavy::Context& C,
                                                            ValueRefs) {
      C.setEnvironment(PrevEnv);
      C.OpGen = PrevOpGen;
      C.Cont();
    }, {});

    Context.DynamicWind(std::move(Ptr), Before, Thunk, After);
  }
};
} // namespace

void Context::WithEnv(std::unique_ptr<heavy::Environment> EnvPtr,
                      heavy::Environment* Env, Value Thunk) {
  LibraryEnv::Wind(*this, std::move(EnvPtr), Env, Thunk);
}

// RunSync - Call a function and run the continuation loop
//           synchronously breaking once the function or any
//           called escape procedure is complete.
//           This is used for *nested* calls in C++ when finishing
//           the operation on the current C++ call stack is needed.
Value Context::RunSync(Value Callee, Value SingleArg) {
  PushBreak();
  heavy::Value Thunk = CreateLambda([](heavy::Context& C,
                                       heavy::ValueRefs) {
    heavy::Value Callee = C.getCapture(0);
    heavy::Value SingleArg = C.getCapture(1);
    C.Apply(Callee, ValueRefs{SingleArg});
  }, CaptureList{Callee, SingleArg});
  heavy::Value HandleError = CreateLambda([](heavy::Context& C,
                                            heavy::ValueRefs Args) {
    // Defer error propagation to the parent loop.
    C.PushCont([](heavy::Context& C, ValueRefs) {
      heavy::ValueRefs ErrorArgs = C.getCaptures();
      C.Cont(ErrorArgs);
    }, Args);
    C.Yield(Args);
  });
  WithExceptionHandler(HandleError, Thunk);
  Resume();
  return getCurrentResult();
}

namespace {
class LiteralRebuilder : public ValueVisitor<LiteralRebuilder, Value> {
  friend class ValueVisitor<LiteralRebuilder, Value>;
  heavy::Context& Context;

public:
  LiteralRebuilder(heavy::Context& C)
    : Context(C)
  { }

private:
  Value VisitValue(Value V) {
    return V;
  }

  Value VisitSyntaxClosure(SyntaxClosure* SC) {
    // Unwrap the SyntaxClosure.
    return Visit(SC->Node);
  }

  Value VisitPair(Pair* P) {
    Value Car = Visit(P->Car);
    Value Cdr = Visit(P->Cdr);
    if (Car == P->Car && Cdr == P->Cdr)
      return P;
    return Context.CreatePair(Car, Cdr);
  }

  Value VisitVector(Vector* V) {
    bool NeedsRebuild = false;
    llvm::SmallVector<Value, 16> TempVec{};
    for (Value X : V->getElements()) {
      Value X2 = Visit(X);
      if (X2 != X) NeedsRebuild = true;
      TempVec.push_back(Visit(X));
    }
    if (!NeedsRebuild) return V;

    return Context.CreateVector(TempVec);
  }
};
} // namespace

// Rebuild the parts of a literal if it contains
// an unexpanded SyntaxClosure.
// Otherwise, it is idempotent.
Value Context::RebuildLiteral(Value V) {
  LiteralRebuilder Rebuilder(*this);
  return Rebuilder.Visit(V);
}

// ContextLocal

heavy::Value ContextLocal::init(heavy::Context& C, heavy::Value Value) {
  heavy::ContextLocalLookup& CL = C;
  heavy::Value& Result = CL.LookupTable[key()];
  // Allow init to be called after the user already called init.
  if (Result != nullptr && Value == nullptr)
    return Result;
  assert(Result == nullptr &&
      "context local should be initialized only once");
  Result = Value ? Value : C.CreateBinding(heavy::Undefined());
  return Result;
}

void ContextLocal::set(heavy::ContextLocalLookup& C,
                       heavy::Value Value) {
  heavy::Value& Result = C.LookupTable[key()];
  auto* Binding = dyn_cast<heavy::Binding>(Result);
  assert(Binding && "context local must be initialized as binding");
  Binding->setValue(Value);
}

// Return the value pointed to by Binding or nullptr.
heavy::Value ContextLocal::get(heavy::ContextLocalLookup const& C) const {
  heavy::Value Value = C.LookupTable.lookup(key());
  if (auto* Binding = dyn_cast<heavy::Binding>(Value))
    return Binding->getValue();
  return Value;
}


// Module

Module::~Module() {
  // Call cleanups pushed within the module
  // when the module is unloading.
  if (Cleanup) {
    Context.ApplyThunk(Cleanup);
  }
}

// Push a cleanup function to be called when the
// module is unloaded/destroyed.
void Module::PushCleanup(heavy::Lambda* Fn) {
  if (!Cleanup) {
    Cleanup = Fn;
    return;
  }
  Cleanup = Context.CreateLambda([](heavy::Context& C, ValueRefs) {
    heavy::Lambda* Next = cast<Lambda>(C.getCapture(0));
    heavy::Lambda* Fn   = cast<Lambda>(C.getCapture(1));
    C.PushCont(Value(Next));
    C.ApplyThunk(Fn);
  }, CaptureList{Cleanup, Fn});
}

// Create Functions

template <typename Allocator, typename ...StringRefs>
static String* CreateStringHelper(Allocator& Alloc, StringRefs ...S) {
  std::array<unsigned, sizeof...(S)> Sizes{static_cast<unsigned>(S.size())...};
  unsigned TotalLen = 0;
  for (unsigned Size : Sizes) {
    TotalLen += Size;
  }

  unsigned MemSize = String::sizeToAlloc(TotalLen);
  void* Mem = Alloc.BigAllocate(MemSize, alignof(String));

  return new (Mem) String(TotalLen, S...);
}

String* Context::CreateString(unsigned Length, char InitChar) {
  unsigned MemSize = String::sizeToAlloc(Length);
  void* Mem = BigAllocate(MemSize, alignof(String));
  return new (Mem) String(Length, InitChar);
}

String* Context::CreateString(llvm::StringRef S) {
  return CreateStringHelper(getAllocator(), S);
}

String* Context::CreateString(llvm::StringRef S1, StringRef S2) {
  return CreateStringHelper(getAllocator(), S1, S2);
}

String* Context::CreateString(llvm::StringRef S1,
                           llvm::StringRef S2,
                           llvm::StringRef S3) {
  return CreateStringHelper(getAllocator(), S1, S2, S3);
}

String* Context::CreateString(llvm::StringRef S1,
                           llvm::StringRef S2,
                           llvm::StringRef S3,
                           llvm::StringRef S4) {
  return CreateStringHelper(getAllocator(), S1, S2, S3, S4);
}

Value Context::CreateList(llvm::ArrayRef<Value> Vs) {
  // Returns a *newly allocated* list of its arguments.
  heavy::Value List = CreateEmpty();
  for (auto Itr = Vs.rbegin(); Itr != Vs.rend(); ++Itr) {
    List = CreatePair(*Itr, List);
  }
  return List;
}

Float* Context::CreateFloat(llvm::APFloat Val) {
  return new (getAllocator()) Float(Val);
}

Vector* Context::CreateVector(unsigned N) {
  return new (getAllocator(), N) Vector(Undefined(), N);
}

Vector* Context::CreateVector(ArrayRef<Value> Xs) {
  return new (getAllocator(), Xs) Vector(Xs);
}

ByteVector* Context::CreateByteVector(ArrayRef<Value> Xs) {
  // Convert the Values to bytes.
  // Invalid inputs will be set to zero.
  heavy::String* String = CreateString(Xs.size(), '\0');
  ByteVector* BV =  new (*this) ByteVector(String);
  llvm::MutableArrayRef<char> Data = String->getMutableView();
  for (unsigned I = 0; I < Xs.size(); ++I) {
    if (isa<heavy::Int>(Xs[I])) {
      auto Byte = cast<heavy::Int>(Xs[I]);
      Data[I] = int32_t(Byte);
    }
  }
  return BV;
}

EnvFrame* Context::CreateEnvFrame(llvm::ArrayRef<Symbol*> Names) {
  unsigned MemSize = EnvFrame::sizeToAlloc(Names.size());

  void* Mem = Allocate(MemSize, alignof(EnvFrame));

  EnvFrame* E = new (Mem) EnvFrame(Names.size());
  auto Bindings = E->getBindings();
  for (unsigned i = 0; i < Bindings.size(); i++) {
    Bindings[i] = CreateBinding(Names[i], CreateUndefined());
  }
  return E;
}
