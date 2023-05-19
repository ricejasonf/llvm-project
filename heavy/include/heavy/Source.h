//===------------ Source.h - Heavy Scheme Source - --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface for source locations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_HEAVY_SOURCE_H
#define LLVM_HEAVY_SOURCE_H

#include "llvm/ADT/StringRef.h"
#include <utility>
#include <cassert>

namespace heavy {
class SourceManager;

// SourceLineContext - Represent a position in source with everything
//                     needed to render a location in source.
struct SourceLineContext {
  llvm::StringRef FileName = {};
  llvm::StringRef LineRange = {};
  // Column - The offset of the location of interest in LineBuffer.
  unsigned Column = 0;
  unsigned LineNumber = 0;
};

// This exists for the reinterpret_cast<uintptr_t>
// stuff that mlir::OpaqueLoc does.
// (because they expect a pointer)
struct SourceLocationEncoding;

class SourceLocation {
  friend class SourceManager;
  friend class SourceFile;
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

  unsigned getOffsetFrom(SourceLocation Start) const {
    assert(Start.Loc < Loc && "location must exist after start location");
    return Loc - Start.Loc;
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

struct SourceFile {
  llvm::StringRef Buffer;
  llvm::StringRef Name;
  SourceLocation StartLoc;
  // ExternalLocRawEncoding - Used to store clang::SourceLocation
  uintptr_t ExternalRawEncoding = 0;

  bool isValid() const { return StartLoc.isValid() || isExternal(); }

  bool hasLoc(SourceLocation Loc) const {
    unsigned L = Loc.getEncoding();
    unsigned Start = StartLoc.getEncoding();
    unsigned End = Start + Buffer.size();
    return L >= Start && L < End;
  }

  SourceLocation getStartLocation() const {
    return StartLoc;
  }

  bool isExternal() const { return ExternalRawEncoding > 0; }
};

class FullSourceLocation {
  friend SourceManager;
  SourceManager& Manager;
  SourceFile File;
  SourceLocation Loc;

  FullSourceLocation(SourceManager& SM, SourceFile File, SourceLocation Loc)
    : Manager(SM),
      File(File),
      Loc(Loc)
  { }

public:
  SourceLocation getLocation() const {
    return Loc;
  }

  // Define the functions in SourceManager.cpp.

  uintptr_t getExternalRawEncoding() const {
    return File.ExternalRawEncoding;
  }

  // getOffset - returns offset of location from
  //             the start of the file
  unsigned getOffset() const {
    return Loc.getOffsetFrom(File.StartLoc);
  }

  bool isValid() const {
    return Loc.isValid() && File.isValid() && !File.isExternal();
  }

  // This is a potentially expensive operation
  // so only call it if we intend to display
  // an error to the user.
  SourceLineContext getLineContext() const;
};

} // end namespace heavy

#endif
