// RUN: clang++ --std=c++23 -I %heavy_module_path -fsyntax-only -Xclang -fheavy -Xclang -verify %s

heavy_scheme {
(import (heavy builtins)
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
(define !nbdl.unit (type "!nbdl.unit"))
(define i32 (type "i32"))

(define (build-member-name name)
  (result
    (old-create-op "nbdl.member_name"
               (attributes
                 `("name", (string-attr name)))
               (result-types !nbdl.symbol))))

(%build-context
  'my_context
  0 ; num_params
  (lambda ()
    (define parent
      (result
        (old-create-op "nbdl.unit"
                   (result-types !nbdl.unit))))
    (define (build-member key typename . init-args)
      (define store
        (result
          (old-create-op "nbdl.store"
                    (loc typename)
                    (attributes `("name", (flat-symbolref-attr typename)))
                    (operands init-args)
                    (result-types !nbdl.store)
                    )))
      (set! parent
        (result
          (old-create-op "nbdl.store_compose"
                     (loc typename)
                     (operands key store parent)
                     (result-types !nbdl.store)))))
    (define foo-input
      (result
        (old-create-op "nbdl.literal"
                   (attributes
                     `("value", (attr "42" i32)))
                   (result-types !nbdl.opaque))))

    (build-member (build-member-name 'foo)
                  '::moo::foo_t)
    (old-create-op "nbdl.cont"
               (operands parent))
  ))
  (define my_context
    (module-lookup current-nbdl-module "my_context"))
  (translate-cpp my_context lexer-writer)
}
// expected-error@-8 {{no member named 'moo'}}

int main() { }
