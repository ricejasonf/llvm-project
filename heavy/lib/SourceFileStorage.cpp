//===--- SourceFileStorage.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// System touching implementations for the ParseSourceFile hook in HeavyScheme
// and others.
//
//===----------------------------------------------------------------------===//

#include "heavy/Builtins.h"
#include "heavy/HeavyScheme.h"
#include "heavy/SourceFileStorage.h"
#include "heavy/SourceManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace heavy {

// HeavyScheme

void HeavyScheme::InitSourceFileStorage() {
  using PtrTy = decltype(SourceFileStoragePtr);
  SourceFileStoragePtr = PtrTy(new SourceFileStorage(),
                    [](SourceFileStorage* S) { delete S; });
  heavy::builtins::InitParseSourceFile(getContext(), 
                                   [this](heavy::Context& C,
                                          heavy::SourceLocation Loc,
                                          heavy::String* Filename) {
    heavy::Value Result = this->ParseSourceFile(Loc, Filename->getView());
    // The parser already raised an error.
    if (llvm::isa<heavy::Undefined>(Result))
      return;
    C.Cont(Result);
  });
}

void HeavyScheme::ProcessTopLevelCommands(llvm::StringRef Filename,
                          llvm::function_ref<ValueFnTy> ExprHandler,
                          llvm::function_ref<ErrorHandlerFn> ErrorHandler) {
  heavy::SourceManager& SM = getSourceManager();
  if (!SourceFileStoragePtr) {
    ErrorHandler("source file storage not initialized",
                 SM.getFullSourceLocation({}));
    return;
  }
  std::string ErrorMessage;
  llvm::ErrorOr<heavy::SourceFile>
    FileResult = SourceFileStoragePtr->Open(SM, heavy::SourceLocation(),
                                            Filename, ErrorMessage);
  if (!FileResult) {
    ErrorHandler(ErrorMessage, SM.getFullSourceLocation({}));
    return;
  }
  heavy::Lexer Lexer(FileResult.get());
  ProcessTopLevelCommands(Lexer, ExprHandler, ErrorHandler, tok::eof);
}

// Indicate an error was raised by returning an empty string.
static std::string SearchIncludePath(heavy::Context& C,
                                     llvm::StringRef Filename) {
  heavy::Value IncludePaths = HEAVY_BASE_VAR(include_paths).get(C);
  for (heavy::Value Path : IncludePaths) {
    llvm::StringRef PathStr = Path.getStringRef();
    if (PathStr.empty()) {
      C.RaiseError(
          "expecting nonempty string-like object in include-paths: {}", Path);
      return std::string();
    }
    llvm::Twine T2(Filename);
    llvm::Twine T1(PathStr, "/");
    if (llvm::sys::fs::exists(T1.concat(T2)))
      return T1.concat(T2).str();
  }

  C.RaiseError("file not found using include-paths: {}",
      Value(C.CreateString(Filename)));
  return std::string();
}

// This overload provides the default file system
// access for opening source files.
// The user must call InitSourceFileStorage.
heavy::Value HeavyScheme::ParseSourceFile(heavy::SourceLocation Loc,
                                          llvm::StringRef Filename) {
  assert(SourceFileStoragePtr &&
      "source file storage not initialized");

  std::string FullPath = SearchIncludePath(getContext(), Filename);
  if (FullPath.empty())
    return Undefined();

  std::string ErrorMessage;
  llvm::ErrorOr<heavy::SourceFile>
    FileResult = SourceFileStoragePtr->Open(getSourceManager(), Loc,
                                            FullPath, ErrorMessage);
  if (!FileResult) {
    heavy::Context& C = getContext();
    C.RaiseError(ErrorMessage);
    return Undefined();
  }
  heavy::Lexer Lexer(FileResult.get());
  return ParseSourceFile(Lexer);
}

llvm::ErrorOr<heavy::SourceFile>
SourceFileStorage::Open(heavy::SourceManager& SM,
                        heavy::SourceLocation IncludeLoc,
                        llvm::StringRef Filename,
                        std::string& ErrorMessage) {
  llvm::SmallString<128> FilePath;
  // Module files already have an absolute path.
  if (!Filename.ends_with(".sld")) {
    // Include Loc may be invalid.
    heavy::SourceFile IncludeSF = SM.getFile(IncludeLoc);
    if (IncludeSF.isValid() && IncludeSF.Name != "-") {
      // Get the file path.
      FilePath = IncludeSF.Name;
      llvm::sys::path::remove_filename(FilePath);
    }
  }

  FilePath += Filename;

  // Check LookupCache
  heavy::SourceLocation CachedLoc = LookupCache.lookup(FilePath); 
  if (CachedLoc.isValid()) {
    heavy::SourceFile CachedSF = SM.getFile(CachedLoc);
    assert(CachedSF.isValid() && "cached source file should be valid");
    return CachedSF;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> File =
    llvm::MemoryBuffer::getFileOrSTDIN(FilePath);
  if (std::error_code ec = File.getError()) {
    ErrorMessage = (llvm::Twine("opening ", FilePath) + 
                             llvm::Twine(": ", ec.message())).str();
    return File.getError();
  }

  // Ensure the name is stable in memory.
  NodeTy& Node = Storage.emplace_back(std::move(*File));
  return SM.createEntry(Node->getBuffer(), FilePath);
}

}
