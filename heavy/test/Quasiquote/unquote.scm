; RUN: heavy-scheme %s 2>&1 | FileCheck  %s --strict-whitespace --match-full-lines
(import (heavy base))

(define foo 5)
(define bar "six")
(define baz 'seven)

(dump `23)
; CHECK:23
(dump `moo)
; CHECK:moo

(dump `(This is just a literal))
; CHECK:(This is just a literal)
(dump `#(This is just a literal))
; CHECK:#(This is just a literal)

(dump `,foo)
; CHECK:5
(dump `(,foo))
; CHECK:(5)

(dump `(,foo This is just a ,bar literal ,baz))
; CHECK:(5 This is just a "six" literal seven)

(dump `#(,foo))
; CHECK:#(5)

(dump `#(,foo ,bar))
; CHECK:#(5 "six")

(dump `#(,foo This is just a ,bar literal ,baz))
; CHECK:#(5 This is just a "six" literal seven)
