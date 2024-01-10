; RUN: heavy-scheme --mode=read %s 2>&1 | FileCheck  %s --strict-whitespace --match-full-lines

""
; CHECK:""

"foo"
; CHECK:"foo"
"bar"
; CHECK:"bar"
"foo\\nbar"
; CHECK:"foo\\nbar"
"escape\nnewline"
; CHECK:"escape\nnewline"

; Support mnemonic escapes
"\a\b\t\n\r\"\\\|"
; CHECK:"\a\b\t\n\r\"\\|"

; All relevant hex codes should
; normalize to relevant escape codes
; or a hex code if there is no escape code.
; Normalize to uppercase escape code.
"\x007;\x8;\x9;\xA;\xd;\x5C;"
; CHECK:"\a\b\t\n\r\\"

; Normalize to uppercase escape code
; with no leading zeros.
"\x00c;"
; CHECK:"\xC;"

; Normalize printable UTF8 codepoints.
"\x2615;"
; CHECK:"☕"

; Omit escaped whitespace
"Hello, \  
  world!"
; CHECK:"Hello, world!"

; Just render characters that do no represent
; an escape character;
"\m\l\n"
; CHECK:"ml\n"

; Support actual newlines
"Hello,

    World!"
; CHECK:"Hello,\n\n    World!"
