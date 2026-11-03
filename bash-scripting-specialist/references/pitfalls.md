# Pitfalls: wrong form, right form, why

Read this when writing or reviewing any script. Most rows come from Greg's Wiki
BashPitfalls (BP #N = entry number, mywiki.wooledge.org/BashPitfalls) and the BashFAQ
pages named; the ShellCheck code that catches each one is given where one exists, so
you can tell which mistakes the linter will find for you and which it won't.

## Quoting and word splitting

An unquoted `$var` or `$(cmd)` is split on `IFS` (space, tab, newline by default) and each
piece is then glob-expanded. Quote every expansion unless you want both of those.

| Wrong | Right | Why | Catches |
|---|---|---|---|
| `cp $file $dest` | `cp -- "$file" "$dest"` | splitting and globbing of the values | SC2086 |
| `for a in $*` / `for a in $@` | `for a in "$@"` (or `for a; do`) | only `"$@"` keeps each argument whole | SC2048 / SC2068 |
| `cmd $(other)` | `out=$(other); cmd "$out"`, or an array | splitting, and the exit status is lost | SC2046, SC2312 |
| `arr=($(cmd))` | `mapfile -t arr < <(cmd)` (4.0+) or a read loop | splits on spaces, not lines, and globs | SC2207 |
| `echo $var` | `printf '%s\n' "$var"` | echo eats `-n`/`-e` values and mangles whitespace | BP #14 |
| `printf "$var"` | `printf '%s' "$var"` | `%` and `\` in data become format directives | SC2059 |
| `rm $f` where f is `-rf` | `rm -- "$f"`, globs as `./*` | a leading dash becomes an option | BP #3 |

Places where quotes are optional: the right side of a plain assignment (`a=$b`), the
word after `case`, and inside `[[ ]]` except the right side of `==`/`=~`. Quoting there
anyway is harmless.

## Tests and conditionals

| Wrong | Right | Why | Catches |
|---|---|---|---|
| `[ $foo = bar ]` | `[[ $foo == bar ]]` | empty or multi-word `$foo` breaks `[` | SC2086 |
| `[ -n $foo ]` | `[[ -n $foo ]]` | unquoted empty value makes it `[ -n ]`, which is true | BP #36 |
| `[[ $x == "$pat" ]]` meaning a pattern | `[[ $x == $pat ]]` | quoted right side is literal, unquoted is a glob | BP #34 |
| `[[ $x =~ "^a.*$" ]]` | `re='^a.*$'; [[ $x =~ $re ]]` | quoted regex is matched literally | SC2076 |
| `[[ $a > 7 ]]` | `((a > 7))` | `>` in `[[` compares strings | BP #7 |
| `a && b \|\| c` as if/else | `if a; then b; else c; fi` | `c` also runs when `b` fails | SC2015 |
| `cmd; if [ $? -ne 0 ]` | `if ! cmd; then` | `[` overwrites `$?`; checks drift away from commands | SC2181 |
| `case` with no `*)` | add `*) die "unexpected: $x" ;;` | silent fallthrough on bad input | optional SC2249 |

## Reading input

| Wrong | Right | Why | Catches |
|---|---|---|---|
| `for line in $(cat f)` | `while IFS= read -r line; do ...; done < f` | splits words, not lines; globs | BP #1, SC2013 |
| `read line` | `IFS= read -r line` | without `-r` backslashes vanish; without `IFS=` edge whitespace is stripped | SC2162 |
| last line has no trailing newline and is dropped | `while IFS= read -r line \|\| [[ -n $line ]]` | `read` returns 1 on an unterminated final line (*tested*) | BashFAQ/001 |
| `cmd \| while read ...; do n=$((n+1)); done; echo "$n"` | `while ...; done < <(cmd)` | each pipeline stage is a subshell, so `n` is lost (*tested*: prints 0) | BP #8 |
| a loop body that runs `ssh`/`ffmpeg` eats the loop's input | `while IFS= read -r -u 9 l; do ...; done 9< f` | those commands read stdin too | BashFAQ/001 |
| `IFS=, read -ra f <<<"a,b,"` loses the empty last field | append a delimiter: `<<<"$csv,"` | trailing empty field is dropped | BP #47 |
| `done <<<"$(cmd)"` | `done < <(cmd)` | here-string drops NULs and trailing newlines, buffers everything | BP #63 |

`shopt -s lastpipe` (4.2+, non-interactive scripts only) also keeps the last pipeline
stage in the current shell; *tested* to work on 4.3+, absent on 3.2.

## Files, globs and find

| Wrong | Right | Why | Catches |
|---|---|---|---|
| `for f in $(ls *.txt)` | `for f in ./*.txt; do [[ -e $f ]] \|\| continue; ...` | `ls` output has no safe delimiter | SC2045 |
| glob with no matches iterates once over the literal `*.txt` | `shopt -s nullglob` (no match → nothing) or the `-e` guard above | default glob behavior | Greg's Wiki glob |
| `find . -name x \| xargs rm` | `find . -name x -exec rm -- {} +` or `-print0 \| xargs -0 rm --` | xargs splits on whitespace and quotes | BP #56 |
| `for f in $(find ...)` | `while IFS= read -r -d '' f; do ...; done < <(find ... -print0)` | names can contain spaces and newlines | SC2044 |
| `find . -exec sh -c 'echo {}' \;` | `find . -exec sh -c 'for f; do echo "$f"; done' _ {} +` | the filename becomes shell code | BP #52 |
| `rm -rf "$dir/"*` | `rm -rf -- "${dir:?}/"*` | empty `dir` turns it into `/*` | SC2115 |
| `cd "$d"; rm *` | `cd -- "$d" \|\| exit` | a failed cd runs the rest in the wrong place | SC2164 |
| `sed 's/a/b/' f > f` | `tmp=$(mktemp "$f.XXXXXX") && sed 's/a/b/' "$f" > "$tmp" && mv -- "$tmp" "$f"` | `> f` truncates before sed reads; a temp file in the same directory makes the `mv` an atomic rename (one in `/tmp` may be on another filesystem) | BP #13 |
| `cmd 2>&1 >>log` | `cmd >>log 2>&1` | redirections apply left to right | BP #43 |
| `sudo cmd > /root/f` | `cmd \| sudo tee /root/f >/dev/null` | the redirect runs as you, not root | SC2024 |

With `nullglob` on, a command given only an unmatched glob runs with no arguments:
`ls *.nope` becomes `ls` and lists the current directory (*tested*). Check the count before using
the expansion as arguments: `files=(./*.txt); ((${#files[@]})) || die "no input"`.

## Command substitution and status

| Wrong | Right | Why | Catches |
|---|---|---|---|
| `local x=$(cmd)` | `local x; x=$(cmd)` | `local`'s status hides `cmd`'s | SC2155 |
| `x=$(cmd)` keeps trailing newlines | they are stripped; if they matter: `x=$(cmd; printf .); x=${x%.}` | `$(...)` deletes all trailing newlines (*tested*) | BP #41 |
| `[[ $(cmd) ]]` to test success | `if cmd; then` | tests output, not exit status | |
| process substitution failure ignored | check with `wait $!` (4.4+) or use a temp file | `<(cmd)` status is never collected | SC2312 |

## Arithmetic

| Wrong | Right | Why | Catches |
|---|---|---|---|
| `$((m + 1))` with `m=08` | `$((10#$m + 1))` | leading 0 means octal; `08` is an error (*tested*) | BP #59 |
| `((i++))` under `set -e` | `i=$((i + 1))` | exits when `i` was 0 (bash 4.1+) | audit B103 |
| `$(( $n + 1 ))` with untrusted `n` | validate first, see security.md | arithmetic evaluates `a[$(cmd)]` | audit B307 |
| `let`, `expr`, `$[ ]` | `(( ))`, `$(( ))` | slower, deprecated, or both | Google style |
| `[[ 010 -eq 8 ]]` vs `[ 010 -eq 8 ]` | strip zeros with `10#` | `[[`/`((` read octal, `[` reads decimal | BP #7 |
| floating point in `$(( ))` | `awk`, `bc`, or Python | bash arithmetic is integer only | |

## Commands in variables, arrays, eval

Variables hold data; functions and arrays hold commands (BashFAQ/050).

| Wrong | Right | Why |
|---|---|---|
| `cmd="ls -l '$dir'"; $cmd` | `args=(-l -- "$dir"); ls "${args[@]}"` | quotes inside a variable are not syntax |
| building options by string concatenation | `opts=(); if ((verbose)); then opts+=(-v); fi; tool "${opts[@]}"` | each element stays one argument (on 3.2 with `set -u`, `${opts[@]+"${opts[@]}"}`) |
| `eval "$var=\$value"` | `printf -v "$var" '%s' "$value"` after validating the name | `eval` runs data as code |
| `for i in {1..$n}` | `for ((i = 1; i <= n; i++))` | brace expansion happens before `$n` expands (BP #33) |
| `unset a[0]` | `unset -v 'a[0]'` | `a[0]` is a glob that can match a file named `a0` (BP #57) |
| `OIFS=$IFS; IFS=,; ...; IFS=$OIFS` | `IFS=, read -r ...` or `local IFS=,` | restoring an originally unset IFS leaves it set and empty (BP #49) |

## Locale

Character ranges, `sort` order, and `tr`/`grep` classes depend on the locale. When a
script parses command output or needs reproducible ordering, set the locale for those
commands: `LC_ALL=C sort`, `LC_ALL=C grep '[a-z]'`, or `export LC_ALL=C` near the top of
a script that never handles non-ASCII text. Error messages from tools also change with
the locale, so never match on their wording; check exit codes.

## Environment and misc

| Wrong | Right | Why |
|---|---|---|
| `cd somedir` in a script when the user exported `CDPATH` | `CDPATH='' cd -P -- somedir` | CDPATH can send `cd` elsewhere and makes it print (BP #48) |
| `ps ax \| grep foo` | `pgrep foo` | matches the grep itself (BP #31) |
| two `date` calls around midnight | one call, store the result | values from different days (BP #58) |
| `which cmd` | `command -v cmd` | `which` is external and inconsistent (SC2230, optional) |
| `function f()` | `f() { ...; }` | mixes two syntaxes (BP #25) |
| alias in a script | a function | aliases are off in non-interactive shells (Google style bans them) |
| `#!/bin/sh` with bash features | `#!/usr/bin/env bash` | `/bin/sh` is dash on Debian/Ubuntu |
| a comment that starts `# shellcheck ...` | write `# ShellCheck ...` or rephrase | ShellCheck parses it as a directive and errors (SC1073; *tested* on 0.11.0) |
