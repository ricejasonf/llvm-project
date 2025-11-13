; RUN: heavy-scheme --module-path=%heavy_module_path %s 2>&1 | FileCheck %s
(import (heavy base)
        (heavy mlir)
        (nbdl comp))

(load-dialect "func")
(load-dialect "heavy")
(load-dialect "nbdl")

(define !nbdl.store (type "!nbdl.store"))
(define !nbdl.unit (type "!nbdl.unit"))
(define !nbdl.member_name (type "!nbdl.member_name"))
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

(define my_match_params
  (%build-match-params
    'my_match_params
    1 ; num-params
    (lambda (Store Fn)
      (define UnitKey
        (result
          (create-op "nbdl.unit"
            (loc: 0)
            (operands:)
            (attributes:)
            (result-types: !nbdl.unit))))
      (define Num42
        (result
          (create-op "nbdl.literal"
            (loc: 0)
            (operands:)
            (attributes:
              ("value" (attr "42" i32)))
            (result-types: !nbdl.store))))
      (define Foo
        (result
          (create-op "nbdl.get"
            (loc: 0)
            (operands: Store Num42)
            (attributes:)
            (result-types: !nbdl.store))))
      (close-previous-scope) ; Should do nothing.
      (create-op "nbdl.match"
        (loc: 0)
        (operands: Store UnitKey)
        (attributes:)
        (result-types:)
        (region: "overloads" ()
          (create-op "nbdl.overload"
            (loc: 0)
            (operands:)
            (attributes:
              ("type" (string-attr "std::string")))
            (result-types:)
            (region: "body" ((MatchedStore : !nbdl.store))
              (resolve MatchedStore Foo)
                ))))
      ; Foo is not allowed after this (because it could be invalidated.)
      (close-previous-scope)
      (close-previous-scope) ; Should do nothing.
      (let ((SomeTag (result
                        (create-op "nbdl.constexpr"
                          (loc: 0)
                          (operands:)
                          (attributes:
                            ("expr" (string-attr "::some_tag{}")))
                          (result-types: !nbdl.store)))))
        (resolve
          Store SomeTag)))))

(unless (verify current-nbdl-module)
  (error "verification failed {}" my_match_params))

; TODO Handle perfect forwarding.
; CHECK: arg_3(get_2);
; CHECK: arg_0(::some_tag{});
(translate-cpp my_match_params)
(newline)
