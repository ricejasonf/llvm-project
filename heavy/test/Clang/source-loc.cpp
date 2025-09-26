// RUN: clang++ -I %S/Inputs -fsyntax-only -Xclang -fheavy -Xclang -verify %s

#include <type_traits>

// Note that a note requires an error or a warning in order to be displayed.
// expected-warning@12{{this is a foo}}
// expected-error@12{{this is still a foo}}
// expected-note@14{{this is a bar in a list}}
namespace foo {
heavy_scheme {
(import (heavy builtins) (heavy clang))
(define foo 'foo)
(define bar "bar")
(define baz '(foo bar))

(diag-warning "this is a foo" (source-loc foo))
(diag-error "this is still a foo" (source-loc bar foo))
(diag-note "this is a bar in a list" (source-loc (cdr baz) bar foo))
}
}
