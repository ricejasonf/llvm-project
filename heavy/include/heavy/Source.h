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
#include <unordered_map>

namespace heavy {
class SourceManager;

// This exists for the reinterpret_cast<uintptr_t>
// stuff that mlir::OpaqueLoc does.
// (because they expect a pointer)
struct SourceLocationEncoding;

class SourceLocation {
  friend class SourceManager;
  unsigned Loc = 0;

  SourceLocation(unsigned Loc) : Loc(Loc) { }

public:
  SourceLocation() = default;
  SourceLocation(SourceLocation const&) = default;

  SourceLocation(SourceLocationEncoding* E)
    : Loc(reinterpret_cast<uintptr_t>(E))
  { }

  // getLocWithOffset - The caller is responsible for ensuring
  //                    that the offset does not go out of bounds
  //                    for the containing buffer which is null
  //                    terminated.
  SourceLocation getLocWithOffset(unsigned Offset) const {
    return SourceLocation(Loc + Offset);
  }

  bool isValid() const {
    return Loc != 0;
  }

  unsigned getEncoding() const { return Loc; }
  SourceLocationEncoding* getOpaqueEncoding() const {
    uintptr_t E = Loc;
    return reinterpret_cast<SourceLocationEncoding*>(E);
  };
};

class SourceFileStorage {
  friend class SourceManager;
  using StorageTy = std::unique_ptr<llvm::MemoryBuffer>;

  StorageTy Storage = {};
  std::string Name = {};

  llvm::StringRef getBuffer() const {
    return Storage->getBuffer();
  }

  std::error_code Open(llvm::StringRef Filename) {
    llvm::ErrorOr<StorageTy> File =
      llvm::MemoryBuffer::getFileOrSTDIN(Filename);
    if (!File) return File.getError();
    Name = Filename.str();
    Storage = std::move(File.get());
    return std::error_code();
  }

public:
  SourceFileStorage() = default;
};

struct SourceFile {
  llvm::StringRef Buffer;
  llvm::StringRef Name;
  SourceLocation StartLoc;
  // ExternalLocRawEncoding - Used to store clang::SourceLocation
  uintptr_t ExternalRawEncoding = 0;

  bool isValid() const { return StartLoc.isValid(); }

  bool hasLoc(SourceLocation Loc) {
    unsigned L = Loc.getEncoding();
    unsigned Start = StartLoc.getEncoding();
    unsigned End = Start + Buffer.size();
    return L >= Start && L < End;
  }

  SourceLocation getStartLocation() const {
    return StartLoc;
  }
};

class FullSourceLocation {
  friend SourceManager;
  SourceFile File;
  SourceLocation Loc;

  FullSourceLocation(SourceFile File, SourceLocation Loc)
    : File(File),
      Loc(Loc)
  { }

public:
  SourceFile getFile() const {
    assert(isValid() && "SourceLocation must be valid");
    return File;
  }

  SourceLocation getLocation() const {
    return Loc;
  }

  bool isValid() const {
    return Loc.isValid();
  }
};

class SourceManager {
  std::vector<std::unique_ptr<SourceFileStorage>> StoredEntries;
  std::vector<SourceFile> Entries;
  std::unordered_map<uintptr_t, SourceFile> ExternalLookup;

  SourceLocation getNextStartLoc() {
    // Start Loc at 1 as 0 represents
    // an invalid location
    if (Entries.size() == 0) return 1;
    SourceFile LastFile = Entries.back();
    SourceLocation Start = LastFile.StartLoc;
    unsigned Len = LastFile.Buffer.size();
    return SourceLocation(Start.Loc + Len);
  }

  SourceFile createEntry(llvm::StringRef Buffer,
                         llvm::StringRef Name,
                         uintptr_t ExternalRawEncoding = 0) {
    SourceLocation StartLoc = getNextStartLoc();
    Entries.push_back(SourceFile{Buffer, Name, StartLoc,
                                 ExternalRawEncoding});
    return Entries.back();
  }


public:

  SourceManager() = default;
  SourceManager(SourceManager const&) = delete;

  llvm::ErrorOr<SourceFile> Open(llvm::StringRef Filename) {
    auto File = std::make_unique<SourceFileStorage>();
    if (std::error_code ec = File->Open(Filename)) {
      return ec;
    }
    SourceFile Entry = createEntry(File->getBuffer(), File->Name);
    StoredEntries.push_back(std::move(File));
    return Entry;
  }

  FullSourceLocation getFullSourceLocation(SourceLocation Loc) {
    return FullSourceLocation{getFile(Loc), Loc};
  }

  SourceFile getFile(SourceLocation Loc) {
    for (SourceFile X : Entries) {
      if (X.hasLoc(Loc)) return X;
    }
    return SourceFile{};
  }

  SourceFile getOrCreateExternal(uintptr_t RawEncoding,
                                 llvm::StringRef Buffer,
                                 llvm::StringRef Name) {
    auto itr = ExternalLookup.find(RawEncoding);
    if (itr == ExternalLookup.end()) {
      SourceFile File = createEntry(Buffer, Name, RawEncoding);
      ExternalLookup.insert({RawEncoding, File});
      return File;
    }
    return (*itr).second;
  }
};

} // end namespace heavy

#endif
