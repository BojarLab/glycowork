#!/usr/bin/env python
"""glycowork coding-standard checker.

Rules enforced (beyond the pycodestyle subset below):
  GW1  no blank line inside a function body (a blank line bracketing a nested def/class is fine)
  GW2  keyword arguments are written with spaces around '=' (f(x = 1), not f(x=1))
  GW3  parameters that have a default must be passed by name, never positionally

Usage:
  python check_style.py --diff origin/dev...HEAD   # only lines touched by the diff
  python check_style.py --all glycowork tests      # whole files
  python check_style.py --fix --all glycowork      # fix GW1, GW2 and the pycodestyle subset in place first
"""
import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from collections import defaultdict
from pathlib import Path

PYCODESTYLE_SELECT = 'E225,E231,E301,E302,E303,E305,E306,W291,W293'
SKIP_DIRS = {'.git', 'build', 'dist', '_proc', '_docs', '.ipynb_checkpoints', 'glycowork.egg-info'}


def py_files(paths):
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out += [f for f in p.rglob('*.py') if not SKIP_DIRS & set(f.parts)]
        elif p.suffix == '.py' and p.exists():
            out.append(p)
    return sorted(set(out))


def changed_lines(diff_range):
    """{path: {line numbers added/modified}} for the given git diff range."""
    out = defaultdict(set)
    raw = subprocess.run(['git', 'diff', '--unified=0', '--diff-filter=d', diff_range],
                         capture_output = True, encoding = 'utf-8', errors = 'replace', check = True).stdout
    path = None
    for line in raw.splitlines():
        if line.startswith('+++ b/'):
            path = line[6:]
        elif line.startswith('@@') and path:
            m = re.search(r'\+(\d+)(?:,(\d+))?', line)
            start, count = int(m.group(1)), int(m.group(2) or 1)
            out[path].update(range(start, start + count))
    return {k: v for k, v in out.items() if k.endswith('.py')}


def check_blank_lines(tree, lines, tokens, report):
    protected, depth, opened, starts, nested_bounds = set(), {}, 0, [], []
    for tok in tokens:
        depth.setdefault(tok.start[0], opened)
        kind = tokenize.tok_name[tok.type]
        if kind == 'STRING':
            protected.update(range(tok.start[0], tok.end[0] + 1))
        elif kind in ('FSTRING_START', 'TSTRING_START'):  # Python 3.12+ splits f-strings into tokens, so a blank line inside one starts no token
            starts.append(tok.start[0])
        elif kind in ('FSTRING_END', 'TSTRING_END'):
            protected.update(range(starts.pop(), tok.end[0] + 1))
        elif kind == 'OP' and tok.string in '([{':
            opened += 1
        elif kind == 'OP' and tok.string in ')]}':
            opened -= 1
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for fn in funcs:
        for sub in fn.body:
            for node in ast.walk(sub):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not fn:
                    start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                    while start > 1 and lines[start - 2].strip().startswith('#'):  # comments belong to the nested def
                        start -= 1
                    nested_bounds.append((start, node.end_lineno))
    for fn in funcs:
        for i in range(fn.body[0].lineno, fn.end_lineno):
            if lines[i - 1].strip() or i in protected or depth.get(i, 0) > 0:
                continue
            nxt = next((j for j in range(i + 1, fn.end_lineno + 1) if lines[j - 1].strip()), None)
            brackets = any(nxt == s or i == e + 1 for s, e in nested_bounds)
            if not brackets:
                report(i, 'GW1', 'blank line inside function body')


def kwarg_equals(tokens):
    """The '=' tokens that bind a keyword argument or a parameter default."""
    prev, stack = None, []
    for tok in tokens:
        if tok.type == tokenize.OP:
            if tok.string in '([{':
                stack.append(tok.string == '(' and prev is not None and (prev.type == tokenize.NAME or prev.string in ')]'))
            elif tok.string in ')]}':
                if stack:
                    stack.pop()
            elif tok.string == '=' and stack and stack[-1]:
                yield tok
        if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            prev = tok


def check_kwarg_spacing(tokens, report):
    for tok in kwarg_equals(tokens):
        before = tok.line[:tok.start[1]]
        after = tok.line[tok.end[1]:]
        if not before.endswith(' ') or not after.startswith(' '):
            report(tok.start[0], 'GW2', "keyword argument '=' needs spaces around it")


def build_signatures(paths):
    """{function name: number of parameters that have no default}, ambiguous names resolved conservatively."""
    sigs = defaultdict(set)
    for f in py_files(paths):
        try:
            tree = ast.parse(f.read_text(encoding = 'utf-8'))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            if args.vararg:
                sigs[node.name].add(None)
                continue
            pos = args.posonlyargs + args.args
            required = len(pos) - len(args.defaults)
            if pos and pos[0].arg in ('self', 'cls'):
                continue  # a method is only ever called as an attribute, which GW3 skips, and one named like a builtin (GlycoDataFrame.filter) must not claim bare filter() calls
            sigs[node.name].add(max(required, 0))
    return {k: (None if None in v else max(v)) for k, v in sigs.items()}


def check_positional_kwargs(tree, sigs, report):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or any(isinstance(a, ast.Starred) for a in node.args):
            continue  # attribute calls are skipped: obj.step() cannot be resolved to a package function by name alone
        if not isinstance(node.func, ast.Name) or node.func.id.startswith('__'):
            continue
        name = node.func.id
        required = sigs.get(name)
        if required is not None and len(node.args) > required:
            report(node.lineno, 'GW3', f"'{name}' takes {required} positional argument(s); pass the rest by name")


def run_pycodestyle(files):
    if not files:
        return []
    cmd = [sys.executable, '-m', 'pycodestyle', f'--select={PYCODESTYLE_SELECT}'] + [str(f) for f in files]
    proc = subprocess.run(cmd, capture_output = True, text = True)
    if proc.returncode and not proc.stdout:  # pycodestyle missing or crashed: failing loudly beats a silent 'Style check passed' without the E/W rules
        sys.exit(proc.stderr.strip())
    out = []
    for line in proc.stdout.splitlines():
        m = re.match(r'^(.*?):(\d+):\d+: (\S+) (.*)$', line)
        if m:
            out.append((m.group(1), int(m.group(2)), m.group(3), m.group(4)))
    return out


def pycodestyle_findings(source):
    """[(line, column, code, message)] for the pycodestyle subset, checked in-process on a string (for editors and the fixer)."""
    import pycodestyle
    found = []

    class Collect(pycodestyle.BaseReport):
        def error(self, line_number, offset, text, check):
            if super().error(line_number, offset, text, check):
                found.append((line_number, offset, text[:4], text[5:]))
    style = pycodestyle.StyleGuide(select = PYCODESTYLE_SELECT.split(','), reporter = Collect)
    pycodestyle.Checker(lines = io.StringIO(source).readlines(), options = style.options).check_all()
    return found


def fix_pycodestyle(source):
    """One pass of fixes at the positions pycodestyle reports (fix_source repeats it until nothing changes)."""
    lines = io.StringIO(source).readlines()  # split on '\n' only, so the numbering matches tokenize and '\r\n' survives
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    in_string, starts = set(), []
    for tok in tokens:
        kind = tokenize.tok_name[tok.type]
        if kind == 'STRING':
            in_string.update(range(tok.start[0], tok.end[0]))
        elif kind in ('FSTRING_START', 'TSTRING_START'):
            starts.append(tok.start[0])
        elif kind in ('FSTRING_END', 'TSTRING_END'):
            in_string.update(range(starts.pop(), tok.end[0]))
    ops = {}
    for tok in tokens:
        if tok.type == tokenize.OP and tok.start[0] == tok.end[0]:
            ops.setdefault(tok.start, tok)
            ops.setdefault(tok.end, tok)
    done = set()
    for row, col, code, msg in sorted(pycodestyle_findings(source), reverse = True):  # bottom-right first, so earlier positions stay valid
        line = lines[row - 1]
        eol = line[len(line.rstrip('\r\n')):] or '\n'
        if code in ('W291', 'W293'):
            if row not in in_string:  # trailing whitespace inside a multi-line string is part of its value
                lines[row - 1] = line.rstrip('\r\n').rstrip(' \t\x0c') + line[len(line.rstrip('\r\n')):]
        elif code == 'E231':
            lines[row - 1] = line[:col + 1] + ' ' + line[col + 1:]
        elif code == 'E225':
            tok = ops.get((row, col))
            if tok is None or tok.start in done:
                continue
            done.add(tok.start)
            start, end = tok.start[1], tok.end[1]
            if line[end:end + 1] not in (' ', '\r', '\n', ''):
                line = line[:end] + ' ' + line[end:]
            if start and line[start - 1] != ' ':
                line = line[:start] + ' ' + line[start:]
            lines[row - 1] = line
        elif code == 'E303':
            allowed = 1 if line[:1].isspace() else 2
            above = row - 1
            while above > 0 and not lines[above - 1].strip():
                above -= 1
            del lines[above:above + (row - 1 - above) - allowed]
        else:  # E301, E302, E305, E306: 'expected N blank lines, found K'
            expected, found = map(int, re.findall(r'\d+', msg)[:2])
            at = row - 1
            while at > 0 and lines[at - 1].lstrip().startswith('#'):  # a comment directly above belongs to the line it precedes
                at -= 1
            lines[at:at] = [eol] * (expected - found)
    return ''.join(lines)


def fix_blank_lines(source):
    lines = io.StringIO(source).readlines()  # split on '\n' only, so the numbering matches tokenize and '\r\n' survives
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    blank = set()
    check_blank_lines(ast.parse(source), [line.rstrip('\r\n') for line in lines], tokens, lambda ln, code, msg: blank.add(ln))
    return ''.join(line for i, line in enumerate(lines, 1) if i not in blank)


def fix_kwarg_spacing(source):
    lines = io.StringIO(source).readlines()
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    for tok in sorted(kwarg_equals(tokens), key = lambda t: t.start, reverse = True):
        row, col = tok.start
        line = lines[row - 1]
        if line[col + 1:col + 2] not in (' ', '\r', '\n', '\\', ''):  # a line break after '=' cannot take a space without adding W291
            line = line[:col + 1] + ' ' + line[col + 1:]
        if col and line[col - 1] != ' ':
            line = line[:col] + ' ' + line[col:]
        lines[row - 1] = line
    return ''.join(lines)


def fix_source(source):
    """source with GW1, GW2 and the pycodestyle subset fixed; GW3 is left alone, since only the author knows which name was meant."""
    for _ in range(5):  # one fix can enable another, so repeat until nothing changes
        fixed = fix_kwarg_spacing(fix_blank_lines(fix_pycodestyle(source)))
        if fixed == source:
            break
        source = fixed
    return source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs = '*', default = ['glycowork'])
    ap.add_argument('--diff', help = 'git diff range; only lines touched there are reported')
    ap.add_argument('--all', action = 'store_true', help = 'report on whole files instead of the diff')
    ap.add_argument('--fix', action = 'store_true', help = 'fix GW1, GW2 and the pycodestyle subset in place before checking; with --diff, only the touched lines (including uncommitted edits)')
    args = ap.parse_args()
    diff = args.diff
    if args.fix and diff:  # fixes edit the working tree, so touched lines are read off it, from the merge base for an A...B range
        left, dots, right = diff.partition('...')
        diff = subprocess.run(['git', 'merge-base', left, right or 'HEAD'], capture_output = True, encoding = 'utf-8', check = True).stdout.strip() if dots else left.split('..')[0]
    scope = None if args.all or not diff else changed_lines(diff)
    files = py_files(args.paths) if scope is None else py_files(scope)
    if args.fix:
        for f in files:
            source = f.read_bytes().decode('utf-8')  # bytes keep the file's line endings, which read_text/write_text would translate
            try:
                fixed = fix_source(source)
            except (SyntaxError, tokenize.TokenError):
                continue  # the check below reports it as GW0
            if scope is not None:  # with --diff, fix only touched lines: fixes change whitespace alone, so each non-blank line and the blank lines before it pair up in order
                allowed, runs = scope.get(str(f).replace('\\', '/'), set()), []
                for text in (source, fixed):
                    runs.append([[]])
                    for i, line in enumerate(text.splitlines(keepends = True), 1):
                        runs[-1][-1].append((i, line))
                        if line.strip():
                            runs[-1].append([])
                fixed = ''.join(line for old, new in zip(*runs, strict = True) for _, line in (new if allowed & {i for i, _ in old} else old))
            if fixed != source:
                f.write_bytes(fixed.encode('utf-8'))
                print(f'fixed {f}')
        if scope is not None:
            scope = changed_lines(diff)  # GW1 fixes shift line numbers
    sigs = build_signatures(args.paths)
    findings = []
    for f in files:
        allowed = None if scope is None else scope.get(str(f).replace('\\', '/'), set())
        if allowed is not None and not allowed:
            continue
        source = f.read_text(encoding = 'utf-8')
        lines = source.splitlines()
        try:
            tree = ast.parse(source)
            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        except (SyntaxError, tokenize.TokenError) as e:
            findings.append((str(f), 0, 'GW0', f'cannot parse file: {e}'))
            continue
        report = lambda ln, code, msg: findings.append((str(f), ln, code, msg))
        check_blank_lines(tree, lines, tokens, report)
        check_kwarg_spacing(tokens, report)
        check_positional_kwargs(tree, sigs, report)
    findings += run_pycodestyle(files)
    if scope is not None:
        findings = [x for x in findings if x[2] == 'GW0' or x[1] in scope.get(x[0].replace('\\', '/'), set())]
    for path, ln, code, msg in sorted(set(findings)):
        print(f'{path}:{ln}: {code} {msg}')
    if findings:
        print(f'\n{len(set(findings))} style violation(s). See CONTRIBUTING.md for the coding standard.')
        sys.exit(1)
    print('Style check passed.')


if __name__ == '__main__':
    main()
