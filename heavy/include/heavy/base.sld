(import (heavy builtins))

(define-library (heavy base)
  (import (heavy builtins))
  #;(begin
    ) ; end of begin
  (export
    ; syntax
    define
    define-syntax
    if
    lambda
    quasiquote
    quote
    set!
    syntax-rules
    ;ir-macro-transformer
    begin
    cond-expand
    define-library
    export
    include
    include-ci
    include-library-declarations
    source-loc
    parse-source-file
    +
    -
    /
    *
    >
    <
    apply
    append
    call-with-values
    call/cc
    car
    cdr
    cons
    dump
    dynamic-wind
    eq?
    equal?
    eqv?
    error
    length
    list
    newline
    number->string
    raise
    string-append
    string-copy
    string-length
    values
    with-exception-handler
    write

    compile
    eval
    module-path
    op-eval

    ; Type predicates
    boolean?
    bytevector?
    char?
    eof-object?
    null?
    number?
    pair?
    port?
    procedure?
    string?
    symbol?
    vector?

    ; Extended types.
    mlir-operation?
    source-value?
    make-syntactic-closure
    )
)
