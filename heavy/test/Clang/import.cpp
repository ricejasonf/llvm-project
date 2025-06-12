// RUN: clang++ -I %S/Inputs -fsyntax-only -Xclang -fheavy -Xclang -verify %s

#include <type_traits>

namespace foo {
// expected-error@+6{{expected unqualified-id}}
heavy_scheme {
(import (my lib))
(import (heavy base))

(lol-trait 'woof)
(lol-trait 'my-trait)
(define hello-foo "hello foo!")
(write hello-foo)
}
}

static_assert(std::is_empty_v<foo::woof>);

namespace bar {
heavy_scheme {
; // FIXME Should we have separate scopes in different namespaces?
(import (heavy base))
(write hello-foo)
(import (my lib)) ; FIXME
(if (is-empty 'foo::woof)
  (lol-trait 'bark)
  (raise-error "expecting empty woof"))
}
}

static_assert(std::is_empty_v<bar::bark>);
static_assert(bar::bark::is_bark);
