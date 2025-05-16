//=== - MangleTest.cpp - Tests for heavy::Mangler --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------===//

#include "heavy/Context.h"
#include "heavy/Mangle.h"

#include <gtest/gtest.h>
#include <string>

namespace {

TEST(MangleTest, mangleLibraryName) {
  auto Context = heavy::Context();
  auto Mangler = heavy::Mangler(Context);
  auto mangle = [&](llvm::StringRef Name) {
    heavy::Value Spec = Context.ParseLiteral(Name);
    return Mangler.mangleModule(Spec);
  };

  EXPECT_EQ(mangle("(heavy base)"), "_HEAVYL5SheavyL4Sbase");
  EXPECT_EQ(mangle("(heavy clang)"), "_HEAVYL5SheavyL5Sclang");
  EXPECT_EQ(mangle("(heavy mlir)"), "_HEAVYL5SheavyL4Smlir");
  EXPECT_EQ(mangle("(heavy foo-bar)"), "_HEAVYL5SheavyL3Sfoomi3Sbar");
  EXPECT_EQ(mangle("(heavy foo.bar 42)"), "_HEAVYL5SheavyL3Sfoodt3SbarL2S42");
  EXPECT_EQ(mangle("(foo_bar3)"), "_HEAVYL8Sfoo_bar3");
  EXPECT_EQ(mangle("(foo_bar3 (woof))"), "");
  EXPECT_EQ(mangle("(rmrf *)"), "_HEAVYL4SrmrfLml");
}

TEST(MangleTest, parseLibraryName) {
  auto Context = heavy::Context();
  auto Mangler = heavy::Mangler(Context);

  std::string MangledNameStr;
  auto mangle = [&](llvm::StringRef Name) {
    heavy::Value Spec = Context.ParseLiteral(Name);
    MangledNameStr = Mangler.mangleModule(Spec);
    auto MangledName = llvm::StringRef(MangledNameStr);
    // Strip the _HEAVY prefix.
    MangledName.consume_front(Mangler.getManglePrefix());
    return MangledName;
  };
  llvm::SmallString<32> Output;
  llvm::StringRef MangledName;

  MangledName = mangle("(heavy base)");
  EXPECT_TRUE(Mangler.parseLibraryName(MangledName, Output));
  EXPECT_EQ(std::string(Output), "heavy");
  Output.clear();
  EXPECT_TRUE(Mangler.parseLibraryName(MangledName, Output));
  EXPECT_EQ(std::string(Output), "base");
  Output.clear();

  MangledName = mangle("(heavy clang)");
  EXPECT_TRUE(Mangler.parseLibraryName(MangledName, Output));
  EXPECT_EQ(std::string(Output), "heavy");
  Output.clear();
  EXPECT_TRUE(Mangler.parseLibraryName(MangledName, Output));
  EXPECT_EQ(std::string(Output), "clang");
  Output.clear();

  MangledName = mangle("(heavy foo-bar)");
  EXPECT_TRUE(Mangler.parseLibraryName(MangledName, Output));
  EXPECT_EQ(std::string(Output), "heavy");
  Output.clear();
  EXPECT_TRUE(Mangler.parseLibraryName(MangledName, Output));
  EXPECT_EQ(std::string(Output), "foo-bar");
  Output.clear();

  MangledName = mangle("(heavy foo.bar 42)");
  EXPECT_TRUE(Mangler.parseLibraryName(MangledName, Output));
  EXPECT_EQ(std::string(Output), "heavy");
  Output.clear();
  EXPECT_TRUE(Mangler.parseLibraryName(MangledName, Output));
  EXPECT_EQ(std::string(Output), "foo.bar");
  Output.clear();
  EXPECT_TRUE(Mangler.parseLibraryName(MangledName, Output));
  EXPECT_EQ(std::string(Output), "42");
  Output.clear();

  MangledName = mangle("(foo_bar3)");
  EXPECT_TRUE(Mangler.parseLibraryName(MangledName, Output));
  EXPECT_EQ(std::string(Output), "foo_bar3");
  Output.clear();

  MangledName = mangle("(rmrf *)");
  EXPECT_TRUE(Mangler.parseLibraryName(MangledName, Output));
  EXPECT_EQ(std::string(Output), "rmrf");
  Output.clear();
  EXPECT_TRUE(Mangler.parseLibraryName(MangledName, Output));
  EXPECT_EQ(std::string(Output), "*");
  Output.clear();
}

}
