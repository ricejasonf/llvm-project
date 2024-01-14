; RUN: heavy-scheme %s 2>&1 | FileCheck  %s --strict-whitespace --match-full-lines
(import (heavy base))

(define empty-list ())
(define foo-list '(foo bar baz 5))
(define moo-list '(moo boo 7 noo))

(dump `())
(dump `(,@'()))
(dump `(,@'() ,@'()))
(dump `(,@'() ,@'() ,@'()))
(dump `(,@empty-list))
(dump `(,@empty-list ,@empty-list))
(dump `(,@empty-list ,@empty-list ,@empty-list))
(dump `(,@empty-list ,@empty-list ,@empty-list ,@'()))
; CHECK-COUNT-8:()

(dump `(,@foo-list))
; CHECK:(foo bar baz 5)

(dump `(This is just ,@foo-list))
; CHECK:(This is just foo bar baz 5)

(dump `(,@moo-list This is just ,@foo-list))
; CHECK:(moo boo 7 noo This is just foo bar baz 5)

(dump `((,@moo-list (This is just) ,@foo-list)))
; CHECK:((moo boo 7 noo (This is just) foo bar baz 5))

; Splice directly quoted list with vars.
(dump `(,@moo-list ,@'(This is just) ,@foo-list))
; CHECK:(moo boo 7 noo This is just foo bar baz 5)

(dump `(,@moo-list ,@'() ,@foo-list))
; CHECK:(moo boo 7 noo foo bar baz 5)
