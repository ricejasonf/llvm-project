; RUN: heavy-scheme -fread-only %s
; this is a comment
5 ; this is also a comment
#|this is a block comment |#
#| #|
  this is a nested block comment
|# |#

("hello" #| world |#)

; commented datums are actually handled in the parser
(+ 5 6 x #;("this string in a list is a comment"))
#(+ 5 6 x #;9 #;("this string in a list is a comment"))
