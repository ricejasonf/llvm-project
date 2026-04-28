// RUN: clang++ -I %S/Inputs -fsyntax-only -fplugin=HeavyClang.so -Xclang -verify %s

#include <type_traits>

// Note that a note requires an error or a warning in order to be displayed.
// expected-warning@13{{this is a foo}}
// expected-error@13{{this is still a foo}}
// expected-note@15{{this is a bar in a list}}
namespace foo {
#pragma heavy_scheme
{
(import (heavy builtins) (heavy clang))
(define foo 'foo)
(define bar "bar")
(define baz '(foo bar))

(diag-warning "this is a foo" (source-loc foo))
(diag-error "this is still a foo" (source-loc bar foo))
(diag-note "this is a bar in a list" (source-loc (cdr baz) bar foo))

(if (source-loc-valid "strings don't have source locations")
  (error "expecting an invalid source location"))
(if (eq? #f (source-loc-valid 'SymbolsDo))
  (error "expecting a valid source location"))
}
}
