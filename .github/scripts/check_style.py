#!/usr/bin/env python
"""glycowork coding-standard checker.

Rules enforced (beyond the pycodestyle subset below):
  GW1  no blank line inside a function body (a blank line bracketing a nested def/class is fine)
  GW2  keyword arguments are written with spaces around '=' (f(x = 1), not f(x=1))
  GW3  parameters that have a default must be passed by name, never positionally

Usage:
  python check_style.py --diff origin/dev...HEAD   # only lines touched by the diff
  python check_style.py --all glycowork tests      # whole files
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
                         capture_output = True, text = True, check = True).stdout
    path = None
    for line in raw.splitlines():
        if line.startswith('+++ b/'):
            path = line[6:]
        elif line.startswith('@@') and path:
            m = re.search(r'\+(\d+)(?:,(\d+))?', line)
            start, count = int(m.group(1)), int(m.group(2) or 1)
            out[path].update(range(start, start + count))
    return {k: v for k, v in out.items() if k.endswith('.py')}


def string_lines(tokens):
    """Line numbers that live inside a (multi-line) string literal."""
    out = set()
    for tok in tokens:
        if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
            out.update(range(tok.start[0], tok.end[0] + 1))
    return out


def bracket_depth_by_line(tokens):
    """Depth of open brackets at the start of each line."""
    depth, out = 0, {}
    for tok in tokens:
        out.setdefault(tok.start[0], depth)
        if tok.type == tokenize.OP:
            if tok.string in '([{':
                depth += 1
            elif tok.string in ')]}':
                depth -= 1
    return out


def check_blank_lines(tree, lines, tokens, report):
    protected, nested_bounds = string_lines(tokens), []
    depth = bracket_depth_by_line(tokens)
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


def check_kwarg_spacing(tokens, report):
    prev, depth, stack = None, 0, []
    for tok in tokens:
        if tok.type == tokenize.OP:
            if tok.string in '([{':
                depth += 1
                stack.append(tok.string == '(' and prev is not None and (prev.type == tokenize.NAME or prev.string in ')]'))
            elif tok.string in ')]}':
                depth -= 1
                if stack:
                    stack.pop()
            elif tok.string == '=' and depth and stack and stack[-1]:
                before = tok.line[:tok.start[1]]
                after = tok.line[tok.end[1]:]
                if not before.endswith(' ') or not after.startswith(' '):
                    report(tok.start[0], 'GW2', "keyword argument '=' needs spaces around it")
        if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            prev = tok
    return


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
                required -= 1
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
    raw = subprocess.run(cmd, capture_output = True, text = True).stdout
    out = []
    for line in raw.splitlines():
        m = re.match(r'^(.*?):(\d+):\d+: (\S+) (.*)$', line)
        if m:
            out.append((m.group(1), int(m.group(2)), m.group(3), m.group(4)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs = '*', default = ['glycowork'])
    ap.add_argument('--diff', help = 'git diff range; only lines touched there are reported')
    ap.add_argument('--all', action = 'store_true', help = 'report on whole files instead of the diff')
    args = ap.parse_args()
    scope = None if args.all or not args.diff else changed_lines(args.diff)
    files = py_files(args.paths) if scope is None else py_files(scope)
    sigs = build_signatures(args.paths if args.paths else ['glycowork'])
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
        findings = [x for x in findings if x[1] in scope.get(x[0].replace('\\', '/'), set())]
    for path, ln, code, msg in sorted(set(findings)):
        print(f'{path}:{ln}: {code} {msg}')
    if findings:
        print(f'\n{len(set(findings))} style violation(s). See CONTRIBUTING.md for the coding standard.')
        sys.exit(1)
    print('Style check passed.')


if __name__ == '__main__':
    main()
