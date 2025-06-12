(import (heavy base))

(define-library (my lib)
  (import (heavy base))
  (import (heavy clang))
  (begin
    (define (lol-trait name)
      (define loc (source-loc name))
      (write-lexer loc 'struct)
      (write-lexer loc
        (string-append
          name
          " { static constexpr bool is_"
          name
          " = true; };")))
    (define (is-empty cpp-typename)
      (expr-eval
        (source-loc cpp-typename)
        (string-append "std::is_empty_v<"
                       cpp-typename
                       ">")))
    (write "end of my lib init")
    (newline))
  (export is-empty lol-trait)
)
(write "end of my lib module")
(newline)
