# Portability: bash versions, macOS, GNU vs BSD, POSIX sh

Read this before writing a script that will run anywhere other than the machine it was
written on, and whenever the audit reports B2xx.

## Pick the target first

The oldest bash the script must run on decides which features are allowed. Decide it
before writing, and state it in the script header and in `# bash-target: X.Y` so the
audit checks against it.

| Where it runs | Bash there (Oct 2026) | Target |
|---|---|---|
| Stock macOS, `#!/bin/bash`, cron/launchd, other people's Macs | 3.2.57 | 3.2 |
| Your Mac with Homebrew bash first on PATH | 5.3.x | still 3.2 if anyone else runs it |
| Ubuntu 22.04 / RHEL 9 family | 5.1 | 5.1 |
| Ubuntu 24.04, Debian 12 and 13 | 5.2 | 5.2 |
| Ubuntu 26.04, Fedora 43+, Alpine 3.23+ (after `apk add bash`) | 5.3 | 5.3 |
| Mixed Linux fleet with older long-term systems | varies | 4.4 is a reasonable floor |

Sources: distro versions from the Repology API (repology.org/api/v1/project/bash, read
2026-10-22); macOS from Apple's bash source (github.com/apple-oss-distributions/bash,
`version.h` says 3.2, patch level 57). Latest upstream release: bash 5.3, announced
2025-07-05, with official patches through bash53-020 (ftp.gnu.org/gnu/bash). No 5.4 or
6.0 pre-release existed as of 2026-10-22.

If the script needs newer features, fail early with a clear message instead of failing
halfway with a syntax error. The guard itself must parse on 3.2:

```bash
if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4))); then
  printf '%s: needs bash 4.4+, this is %s\n' "${0##*/}" "$BASH_VERSION" >&2
  printf 'On macOS: brew install bash, and run it with that bash.\n' >&2
  exit 1
fi
```

Place it right after the shebang and before any newer syntax. Bash parses a script one
command at a time as it runs, so code later in the file is not parsed before the guard
runs, but a newer construct *on the same line or in the same compound command* as the
guard would be.

## macOS specifics

- `/bin/bash` is 3.2.57. bash 3.2 is GPLv2 and bash 4.0 is GPLv3 (both COPYING files);
  that this is *why* Apple stopped updating is widely reported but not stated by Apple.
- zsh has been the default interactive shell since macOS 10.15 (Apple support article
  102360). That changes nothing for scripts with a bash shebang, but it means commands
  typed into Terminal run in zsh, not bash. *Tested in zsh 5.9:* unquoted `$var` is not
  word-split, `$arr[1]` is the first element, and a pasted line starting with `#` fails
  with "command not found: #" unless `setopt interactivecomments` is on. So shell
  snippets meant for pasting into a terminal contain no comment lines, and a command
  that works in one shell can fail in the other.
- Homebrew bash: `/opt/homebrew/bin/bash` on Apple Silicon, `/usr/local/bin/bash` on Intel
  (docs.brew.sh/Installation).
- `#!/bin/bash` always runs 3.2. `#!/usr/bin/env bash` runs the first bash on `PATH`, which
  is Homebrew's only when its directory comes before `/bin`. Launchers with a minimal PATH
  (cron, launchd, some IDE task runners, `ssh host cmd`) may still find 3.2. Use
  `#!/usr/bin/env bash` and the version guard above; the official `bash` Docker images
  also install bash at `/usr/local/bin/bash`, so `#!/bin/bash` fails there.
- `/bin/sh` on macOS is bash 3.2 in POSIX mode, so bashisms in `#!/bin/sh` scripts work on
  a Mac and break on Debian/Ubuntu (dash) and Alpine (BusyBox ash).

### Behavior that differs on 3.2 (tested on 3.2.57 vs 4.4+, October 2026, machine not recorded)

| Construct | bash 3.2 | bash 4.4+ |
|---|---|---|
| `set -u; a=(); "${a[@]}"` | "unbound variable", script dies | expands to nothing |
| `set -e; i=0; ((i++))` | continues | exits (bash 4.1 made `((` and `[[` subject to `set -e`) |
| `set -e; [[ -f missing ]]` as a statement | continues | exits |
| `local -` | "not a valid identifier" | scopes `set` options to the function |
| `${a[-1]}` | "bad array subscript" | last element (4.2+) |
| `x=$(false; echo hi)` with `inherit_errexit` | option does not exist | `-e` applies inside |

## Feature → minimum bash version

From the bash NEWS file and Greg's Wiki BashFAQ/061, cross-checked by running each on
bash 3.2.57, 4.3.30, 4.4.18, 5.2.21 and 5.3.20. Everything at 4.0 or later fails on stock
macOS.

| Feature | Version | 3.2-compatible alternative |
|---|---|---|
| `[[ ]]`, `$(< file)`, extglob | 2.02 | |
| `<<<` here-string | 2.05b | |
| `[[ =~ ]]`, `BASH_REMATCH`, `pipefail`, `failglob`, `{1..9}` | 3.0 | |
| `printf -v var`, `+=` | 3.1 | |
| `declare -A` associative arrays | 4.0 | parallel indexed arrays, or a `case` lookup |
| `mapfile` / `readarray` | 4.0 | `while IFS= read -r l \|\| [[ -n $l ]]; do a+=("$l"); done < file` |
| `${v,,}` `${v^^}`, `declare -l/-u` | 4.0 | `tr '[:upper:]' '[:lower:]'` |
| `;&` and `;;&` in case | 4.0 | repeat the commands |
| `\|&`, `&>>` | 4.0 | `2>&1 \|`, `>>f 2>&1` |
| `globstar` (`**`) | 4.0 | `find` |
| `coproc`, `$BASHPID`, `read -t 0.5`, `{01..10}` | 4.0 | |
| `{fd}>file`, `BASH_XTRACEFD`, `read -N` | 4.1 | fixed fd numbers (`exec 3>file`) |
| `[[ -v var ]]`, `declare -g`, lastpipe | 4.2 | `[[ -n ${var+x} ]]`; `< <(cmd)` instead of lastpipe |
| `printf '%(%F %T)T'` | 4.2 | `date '+%F %T'` |
| `${a[-1]}`, `${v:0:-1}` | 4.2 | `${a[${#a[@]}-1]}`, `${v:0:${#v}-1}` |
| `a[-1]=x`, `unset 'a[-1]'` | 4.3 | |
| `declare -n` namerefs, `wait -n` | 4.3 | `printf -v "$name"` after validating the name |
| `${v@Q}` `@E @P @A @a`, `local -`, `inherit_errexit`, `mapfile -d` | 4.4 | `printf %q` |
| empty `"${a[@]}"` allowed under `set -u` | 4.4 | `${a[@]+"${a[@]}"}` |
| `EPOCHSECONDS`, `EPOCHREALTIME`, `BASH_ARGV0`, `wait -f` | 5.0 | `date +%s` |
| `SRANDOM`, `wait -p`, `${v@U}` `@u @L @K` | 5.1 | |
| `${v@k}`, `patsub_replacement` (on by default) | 5.2 | |
| `${ cmd; }`, `${\| cmd; }`, `GLOBSORT`, `source -p`, `array_expand_once` | 5.3 | `$(cmd)` |

`scripts/audit_bash.py --target X.Y` flags these (rule B201).

### bash 5.2 changed `${var/pat/rep}`

`patsub_replacement` is on by default from 5.2: an unquoted `&` in the replacement
means "the matched text". *Tested:*

```bash
x='a=b' r='X&Y'
echo "${x/=/ & }"    # 3.2-5.1: "a & b"     5.2+: "a = b"
echo "${x/=/$r}"     # 3.2-5.1: "aX&Yb"     5.2+: "aX=Yb"
echo "${x/=/"$r"}"   # 3.2: 'a"X&Y"b'       4.3+: "aX&Yb"
echo "${x/=/\&}"     # 3.2: 'a\&b'          4.3+: "a&b"
```

No single spelling gives a literal `&` on every version. For scripts that must also run
on 3.2, put `shopt -u patsub_replacement 2>/dev/null || true` at the top and write the
replacement unquoted; for 4.3+ only, quote the replacement (`"$r"`); bash 4.3 began
running the replacement through quote removal (COMPAT file, item 52). Audit rule B205.

## GNU vs BSD (macOS) commands

macOS behavior read from Apple's source (text_cmds, shell_cmds, file_cmds, tagged
2026-04-17); GNU from the man7.org/gnu.org pages; BusyBox tested on Alpine 3.24.3.

| Need | GNU (Linux) | macOS | Portable choice |
|---|---|---|---|
| edit in place | `sed -i 's/a/b/' f` | `sed -i '' 's/a/b/' f` (suffix is a separate, required argument) | `sed -i.bak 's/a/b/' f && rm f.bak`, or write a temp file and `mv` |
| ERE in sed | `-E` (or `-r`) | `-E` | `sed -E` |
| parse a date | `date -d '2026-01-02'` | `date -j -f '%Y-%m-%d' 2026-01-02` | avoid; do date math in Python, or branch on `date --version` |
| now as epoch / formatted | `date +%s` | `date +%s` | `date +%s`, `date '+%F %T'` |
| file size, mtime | `stat -c %s f` | `stat -f %z f` (`-f` means *filesystem* on GNU) | `wc -c < f` for size; branch on `stat --version` otherwise |
| PCRE grep | `grep -P` | absent | `grep -E`, or `[[ $x =~ re ]]` |
| find output format | `find -printf` | absent | `find ... -exec stat ... {} +` or `-print0 \| xargs -0` |
| canonical path | `readlink -f`, `realpath` (many options) | `readlink -f` since macOS 12.3; `realpath` (only `-q`) on recent macOS | for directories: `CDPATH='' cd -P -- "$d" && pwd -P` |
| run with time limit | `timeout 5 cmd` | not installed (`gtimeout` from Homebrew coreutils) | `command -v timeout \|\| command -v gtimeout`, or a background job plus `kill` |
| lock a file | `flock` | not installed | `mkdir "$lockdir"` (atomic) plus an EXIT trap |
| long-option parsing | util-linux `getopt -o ... --long ...` | old BSD `getopt`, no long options, breaks on spaces | a `while`/`case` loop (see `assets/template.sh`) |
| temp file | `mktemp`, `mktemp -d` | same; `-t` means something else | `mktemp`, `mktemp -d`, or `mktemp -d "${TMPDIR:-/tmp}/name.XXXXXX"` |
| base64 decode | `base64 -d` | `-d` on current macOS (`-D` on old) | `base64 -d` |
| skip empty xargs input | `xargs -r` | `-r` accepted; empty input already skipped | `xargs -r` |
| version sort | `sort -V` | `sort -V` | `sort -V` |
| print text | `echo -e` works in bash | under macOS `/bin/sh`, echo ignores options and always expands escapes | `printf` |

Audit rule B204 flags the GNU-only forms; `--platform linux` turns it off for scripts that
never run on macOS.

## When to write POSIX sh instead

Write `#!/bin/sh` (POSIX, no bash features) when bash may not exist: Alpine-based Docker
images (no bash by default; `/bin/sh` is BusyBox), initramfs, Debian maintainer scripts,
and installers meant to run anywhere. Everywhere else, write bash with a bash shebang and
never invoke it as `sh script.sh`, which ignores the shebang.

- `/bin/sh` is dash on Debian and Ubuntu, BusyBox ash on Alpine, bash 3.2 in POSIX mode
  on macOS. BusyBox ash accepts some bash syntax (`[[ ]]`, `${v//a/b}`, `$'..'`) and
  rejects other parts (arrays, `<<<`, `${v,,}`, `{1..3}`), so "works in ash" proves
  neither POSIX nor bash compatibility. Test sh scripts under dash.
- Check with `shellcheck -s sh` and `checkbashisms` (Debian devscripts).
- POSIX.1-2024 added `set -o pipefail` and `$'...'`, but dash 0.5.12 (Ubuntu 24.04) still
  rejects both; don't rely on them in sh scripts yet.
- Common replacements: `[ ]` with quoted operands instead of `[[ ]]`; `=` not `==`;
  `f() { ...; }` not `function f`; `.` not `source`; `printf` not `echo -e`; positional
  parameters (`set -- a b c`) instead of arrays; `>f 2>&1` not `&>f`.

## Testing on the versions you target

- Run the script and its tests with an old bash first on PATH. On a Mac, `/bin/bash`
  *is* the 3.2 test:
  `/bin/bash script.sh args`, or for bats, `PATH=/bin:$PATH bats test/`.
- `scripts/test_matrix.sh` runs one command under several bash versions, using the
  official Docker images (`bash:3.2`, `bash:4.4`, `bash:5.3`; also 3.0 through 5.2 and
  `devel`) or local binaries via `--bash PATH`. The images are Alpine with BusyBox tools,
  so they also expose GNU-only flags.
