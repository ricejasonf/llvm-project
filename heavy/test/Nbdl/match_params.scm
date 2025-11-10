; RUN: heavy-scheme --module-path=%heavy_module_path %s 2>&1 | FileCheck %s
(import (heavy builtins)
        (heavy mlir)
        (nbdl comp))

(load-dialect "func")
(load-dialect "heavy")
(load-dialect "nbdl")

(define !nbdl.store (type "!nbdl.store"))
(define !nbdl.tag (type "!nbdl.tag"))
(define !nbdl.symbol (type "!nbdl.symbol"))
(define !nbdl.unit (type "!nbdl.unit"))
(define i32 (type "i32"))

(define (resolve . args)
  (define Result
    (result
      (create-op "nbdl.visit"
                 (loc: 0)
                 (operands: args)
                 (attributes:)
                 (result-types: !nbdl.unit)
                 )))
  (create-op "nbdl.discard"
             (loc: 0)
             (operands: Result)
             (attributes:)
             (result-types:)))

(%build-match-params
  'my_match_params
  1 ; num-params
  (lambda (store fn)
    (define key1
      (result
        (old-create-op "nbdl.constexpr"
                   (attributes
                     `("expr", (string-attr "::my::tag{}")))
                   (result-types !nbdl.tag))))
    (define key2
      (result
        (old-create-op "nbdl.member_name"
                   (attributes
                     `("name", (string-attr "bar")))
                   (result-types !nbdl.symbol))))
    (define key3
      (result
        (old-create-op "nbdl.literal"
                   (attributes
                     `("value", (attr "42" i32)))
                   (result-types !nbdl.store))))
    (define foo
      (result
        (old-create-op "nbdl.get"
                   (operands store key1)
                   (result-types !nbdl.store))))
    (define value
      (result
        (old-create-op "nbdl.get"
                   (operands store key2)
                   (result-types !nbdl.store))))
    (define the-match-op
      (old-create-op "nbdl.match"
                 (regions 1)
                 (operands value key3)))
    (if (value? "!nbdl.store" key3)
      'ok
      (error "key should be a an !nbdl.store type"))
    (if (value? !nbdl.store foo)
      'ok
      (error "foo should be a nbdl store"))
    (if (value? value)
      'ok
      (error "value should be a value"))
    (with-builder
      (lambda ()
        (at-block-begin (entry-block the-match-op))
        (%build-overload 'loc "::my::first_t const&"
          (lambda (my-first)
            (resolve fn my-first)))
        (%build-overload 'loc "uint32_t"
          (lambda (input)
            (define pred1 (result
                (old-create-op "nbdl.constexpr"
                         (attributes
                           `("expr", (string-attr "::my::greater_than{42}")))
                         (result-types !nbdl.store))))
            (define pred2 (result
              (old-create-op "nbdl.constexpr"
                         (attributes
                           `("expr", (string-attr "::my::greater_than{5}")))
                         (result-types !nbdl.store))))
            (%build-match-if
              'loc input pred1
              (lambda()
                (resolve fn input))
              (lambda()
                (%build-match-if
                  'loc input pred2
                  (lambda()
                    (resolve fn input))
                  (lambda ()
                    (define some-fn (result
                        (old-create-op
                          "nbdl.constexpr"
                          (attributes
                            `("expr", (string-attr "::my::some_fn")))
                          (result-types !nbdl.store))))
                    (define some-fn-result
                      ; Intermediate visit call with result as operand
                      ; to "resolving" visit/discard.
                      (result
                        (old-create-op "nbdl.visit"
                                       (operands some-fn input)
                                       (result-types !nbdl.store))))
                    (resolve fn some-fn-result)
                    ))))))
        (%build-overload 'loc "")
        ))))

; CHECK: #op{module @nbdl_gen_module {
; CHECK: func.func @my_match_params(
(verify current-nbdl-module)
(write current-nbdl-module)

(define my_match_params
  (module-lookup current-nbdl-module "my_match_params"))


(translate-cpp my_match_params)
(newline)
