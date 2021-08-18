// RUN: %clang_cc1 -std=c++17 -emit-codegen-only -verify %s
// Don't crash (Bugzilla - Bug 45964).

// non-dependent ArrayInitLoop should not, upon instantiation,
// contain an OpaqueValueExpr with a nested OpaqueValueExpr or an
// uninstantiated source expr. (triggers asserts in CodeGen)

// expected-no-diagnostics
int a[1];
template <int> void b() {
  auto [c] = a;
}
void (*d)(){b<0>};
