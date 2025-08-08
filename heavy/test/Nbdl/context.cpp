// RUN: clang++ --std=c++23 -I %S/Inputs -fsyntax-only -Xclang -fheavy -Xclang -verify %s
// expected-no-diagnostics

namespace moo {
  struct move_only {
    int value;

    move_only(int v) : value(v) { };
    move_only(move_only const&) = delete;
    move_only(move_only&&) = default;
  };

  struct foo_t {
    int value;
    foo_t(int v) : value(v) { }
  };

  struct bar_t { };

  struct baz_t {
    int value;
    template <typename V>
    baz_t(V&& v)
      : value(v.value) { }
  };
}

heavy_scheme {
(import (heavy base)
        (heavy clang)
        (heavy mlir)
        (nbdl comp))

(load-dialect "func")
(load-dialect "heavy")
(load-dialect "nbdl")

(define !nbdl.opaque (type "!nbdl.opaque"))
(define !nbdl.store (type "!nbdl.store"))
(define !nbdl.tag (type "!nbdl.tag"))
(define !nbdl.symbol (type "!nbdl.symbol"))
(define !nbdl.empty (type "!nbdl.empty"))
(define i32 (type "i32"))

(define (build-member-name name)
  (result
    (create-op "nbdl.member_name"
               (attributes
                 `("name", (string-attr name)))
               (result-types !nbdl.symbol))))

(%build-context
  'my_context
  1 ; num_params
  (lambda (BazArg)
    (define parent
      (result
        (create-op "nbdl.empty"
                   (result-types !nbdl.empty))))
    (define (build-member key typename . init-args)
      (define store
        (result
          (create-op "nbdl.store"
                    (attributes `("name", (flat-symbolref-attr typename)))
                    (operands init-args)
                    (result-types !nbdl.store)
                    )))
      (set! parent
        (result
          (create-op "nbdl.store_compose"
                     (operands key store parent)
                     (result-types !nbdl.store)))))
    (define foo-input
      (result
        (create-op "nbdl.literal"
                   (attributes
                     `("value", (attr "42" i32)))
                   (result-types !nbdl.opaque))))

    (build-member (build-member-name 'foo)
                  '::moo::foo_t
                  foo-input)
    (build-member (build-member-name 'bar) '::moo::bar_t)
    (build-member (build-member-name 'baz)
                  '::moo::baz_t
                  BazArg)
    (create-op "nbdl.cont"
               (operands parent))
  ))
  (define my_context
    (module-lookup current-nbdl-module "my_context"))
  (translate-cpp my_context lexer-writer)
}

int main() {
  auto mc = my_context(moo::move_only(5));
}
