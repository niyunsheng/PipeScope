/**
 * Tiny arithmetic expression language for user-written schedule formulas
 * (e.g. the number of warmup micro-batches per rank). Deliberately not
 * `eval`: a hand-written recursive-descent parser gives precise error
 * positions and admits nothing but arithmetic.
 *
 * Grammar (lowest to highest precedence):
 *   expr   := cmp
 *   cmp    := sum (('==' | '!=' | '<' | '<=' | '>' | '>=') sum)?    -> 1 or 0
 *   sum    := prod (('+' | '-') prod)*
 *   prod   := unary (('*' | '/' | '%') unary)*
 *   unary  := '-' unary | atom
 *   atom   := number | name | name '(' expr (',' expr)* ')' | '(' expr ')'
 * Functions: min, max, floor, ceil, abs.
 */

export type Vars = Record<string, number>;

type Node =
  | { t: 'num'; v: number }
  | { t: 'var'; name: string; pos: number }
  | { t: 'neg'; a: Node }
  | { t: 'bin'; op: string; a: Node; b: Node }
  | { t: 'call'; name: string; args: Node[]; pos: number };

const FUNCS: Record<string, (...xs: number[]) => number> = {
  min: Math.min,
  max: Math.max,
  floor: Math.floor,
  ceil: Math.ceil,
  abs: Math.abs,
};

export class FormulaError extends Error {
  /** 0-based offset into the source where the problem was found. */
  pos: number;
  constructor(message: string, pos: number) {
    super(`${message} (at position ${pos + 1})`);
    this.pos = pos;
  }
}

export interface Formula {
  source: string;
  /** Variable names referenced, for validation against the available set. */
  vars: string[];
  eval(vars: Vars): number;
}

export function parseFormula(source: string): Formula {
  let i = 0;
  const s = source;
  const skip = () => {
    while (i < s.length && /\s/.test(s[i])) i++;
  };
  const peek = (str: string) => s.startsWith(str, i);
  const eat = (str: string) => {
    skip();
    if (!peek(str)) return false;
    i += str.length;
    return true;
  };
  const used = new Set<string>();

  const atom = (): Node => {
    skip();
    if (i >= s.length) throw new FormulaError('unexpected end of formula', i);
    const c = s[i];
    if (/[0-9.]/.test(c)) {
      const m = /^[0-9]*\.?[0-9]+|^[0-9]+\.?/.exec(s.slice(i))!;
      i += m[0].length;
      return { t: 'num', v: Number(m[0]) };
    }
    if (/[A-Za-z_]/.test(c)) {
      const start = i;
      while (i < s.length && /[A-Za-z0-9_]/.test(s[i])) i++;
      const name = s.slice(start, i);
      if (eat('(')) {
        const args: Node[] = [expr()];
        while (eat(',')) args.push(expr());
        if (!eat(')')) throw new FormulaError(`expected ')' after arguments of ${name}`, i);
        if (!(name in FUNCS)) throw new FormulaError(`unknown function ${name}`, start);
        return { t: 'call', name, args, pos: start };
      }
      used.add(name);
      return { t: 'var', name, pos: start };
    }
    if (eat('(')) {
      const e = expr();
      if (!eat(')')) throw new FormulaError("expected ')'", i);
      return e;
    }
    throw new FormulaError(`unexpected character '${c}'`, i);
  };
  const unary = (): Node => (eat('-') ? { t: 'neg', a: unary() } : atom());
  const prod = (): Node => {
    let a = unary();
    for (;;) {
      if (eat('*')) a = { t: 'bin', op: '*', a, b: unary() };
      else if (eat('/')) a = { t: 'bin', op: '/', a, b: unary() };
      else if (eat('%')) a = { t: 'bin', op: '%', a, b: unary() };
      else return a;
    }
  };
  const sum = (): Node => {
    let a = prod();
    for (;;) {
      if (eat('+')) a = { t: 'bin', op: '+', a, b: prod() };
      else if (eat('-')) a = { t: 'bin', op: '-', a, b: prod() };
      else return a;
    }
  };
  const cmp = (): Node => {
    const a = sum();
    for (const op of ['==', '!=', '<=', '>=', '<', '>']) {
      if (eat(op)) return { t: 'bin', op, a, b: sum() };
    }
    return a;
  };
  const expr = cmp;

  const root = expr();
  skip();
  if (i < s.length) throw new FormulaError(`unexpected '${s[i]}'`, i);

  const ev = (n: Node, vars: Vars): number => {
    switch (n.t) {
      case 'num':
        return n.v;
      case 'var':
        if (!(n.name in vars)) throw new FormulaError(`unknown variable ${n.name}`, n.pos);
        return vars[n.name];
      case 'neg':
        return -ev(n.a, vars);
      case 'call':
        return FUNCS[n.name](...n.args.map((a) => ev(a, vars)));
      case 'bin': {
        const a = ev(n.a, vars);
        const b = ev(n.b, vars);
        switch (n.op) {
          case '+': return a + b;
          case '-': return a - b;
          case '*': return a * b;
          case '/': return a / b;
          case '%': return a % b;
          case '==': return a === b ? 1 : 0;
          case '!=': return a !== b ? 1 : 0;
          case '<': return a < b ? 1 : 0;
          case '<=': return a <= b ? 1 : 0;
          case '>': return a > b ? 1 : 0;
          default: return a >= b ? 1 : 0;
        }
      }
    }
  };
  return { source, vars: [...used], eval: (vars) => ev(root, vars) };
}
