//===------------ Source.h - Heavy Scheme Source - --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface for source files and locations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_SOURCE_H
#define LLVM_HEAVY_SOURCE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cassert>
#include <string>
#include <system_error>

namespace heavy {
class SourceManager;

class SourceLocation {
  friend class SourceManager;
  unsigned Loc = 0;

  SourceLocation() = default;
  SourceLocation(unsigned Loc) : Loc(Loc) { }

public:
  SourceLocation(SourceLocation const&) = default;

  // getWithOffset - The caller is responsible for ensuring
  //                 that the offset does not go out of bounds
  //                 for the containing buffer which is null
  //                 terminated.
  SourceLocation getWithOffset(unsigned Offset) {
    return SourceLocation(Loc + Offset);
  }

  bool isValid() {
    return Loc != 0;
  }
};

class FullSourceLocation {
  friend SourceManager
  SourceFile* File = nullptr;
  SourceLocation Loc;

  FullSourceLocation(SourceFile* File, SourceLocation Loc)
    : File(File),
      Loc(Loc)
  { }

public:
  SourceFile& getFile() {
    assert(isValid() && "SourceLocation must be valid");
    return *File;
  }

  SourceLocation getLocation() {
    return Loc;
  }

  bool isValid() {
    return File != nullptr;
  }
};

class SourceFile {
  friend class SourceManager;
  using StorageTy = std::unique_ptr<llvm::MemoryBuffer>;

  SourceLocation StartLoc;
  std::string Name;
  StorageTy Storage;

  SourceFile() = default;

  std::error_code Open(StringRef Filename) {
    llvm::ErrorOr<StorageTy> File =
      llvm::MemoryBuffer::getFileOrSTDIN(Filename);
    if (File) return File.getError();
    Name = Filename;
    Storage = File->get();
    return std::error_code();
  }

public:
  llvm::StringRef getBuffer() const {
    return Storage->getBuffer();
  }

  SourceLocation getStartLocation() const {
    return StartLoc;
  }

  bool hasLoc(SourceLocation Loc) {
    unsigned EndLoc = StartLoc + getBuffer->size();
    return Loc.Loc >= StartLoc && Loc.Loc < getBuffer()->size();
  }
};

struct SourceFileRef {
  llvm::StringRef Buffer;
  llvm::StringRef Name;
  SourceLocation StartLoc;

  bool isValid() const { return StartLoc.isValid(); }
};

class SourceManager {
  std::vector<std::unique_ptr<SourceFile>> StoredEntries;
  std::vector<SourceFileRef> Entries;

  SourceLocation getNextStartLoc() {
    // Start Loc at 1 as 0 represents
    // an invalid location
    if (Entries.length() == 0) return 1;
    SourceFileRef LastFile = Entries.back();
    SourceLocation Start = LastFile.StartLoc;
    unsigned Len = LastFile.Buffer.size();
    return SourceLocation(Start.Loc + Len);
  }

public:
  llvm::ErrorOr<SourceFileRef> Open(StringRef Filename) {
    auto File = std::make_unique<SourceFile>();
    if (std::error_code ec = File->Open(Filename)) {
      return ec;
    }
    File->StartLoc = getNextStartLoc();
    SourceFileRef Ref{File->getBuffer(),
                      File->getName(),
                      File->getStartLocation()};
    Entries.push_back(Ref);
    StoredEntries.push_back(std::move(File));
    return Entries.back();
  }

  FullSourceLocation getFullSourceLocation(SourceLocation Loc) {
    return FullSourceLocation{getFile(Loc), Loc};
  }

  SourceFileRef getFile(SourceLocation Loc) {
    for (SourceFile& X : Entries) {
      if X.hasLoc(Loc) return &X;
    }
    return SourceFileRef{};
  }
};

} // end namespace heavy

#endif
