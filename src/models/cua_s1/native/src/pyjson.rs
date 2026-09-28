//! JSON the way the Python worker sees it.
//!
//! - [`parse`] follows CPython 3.12's `json.loads` together with the checks that
//!   `contract.parse_body` adds: duplicate keys, `NaN`/`Infinity`, numbers that are
//!   out of range for a float, lone surrogates and very deep nesting are all rejected,
//!   and errors surface in the same order as in Python.
//! - [`dumps`] follows `json.dumps(value, ensure_ascii=False)` (and the compact form
//!   Starlette uses for responses).
//! - [`repr`] follows Python's `repr`, which the error messages quote.

use std::collections::HashSet;
use std::fmt::Write as _;

use crate::printable::NON_PRINTABLE;

/// Nesting limits of the Python worker, measured on CPython 3.12.13 under uvicorn.
/// Both come from interpreter recursion limits, so they depend on the call stack and
/// are not documented constants.
///
/// Parsing fails when a container would open more than this many levels deep (the
/// `json` C scanner's recursion check).
pub const MAX_PARSE_DEPTH: usize = 9990;

/// After parsing, `parse_body` walks the value with a recursive Python function
/// (`_check_unicode`), one call per value, containers and scalars alike. A walk that
/// needs more nested calls than this fails.
pub const MAX_CHECK_CALLS: usize = 969;

/// `sys.int_info.default_max_str_digits`: longer integers fail to convert in Python.
const MAX_INT_DIGITS: usize = 4300;

/// A string as Python holds it: a sequence of code points that may include lone
/// surrogates, stored as generalized UTF-8.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PyStr(Vec<u8>);

impl PyStr {
    pub fn new(s: &str) -> Self {
        PyStr(s.as_bytes().to_vec())
    }

    fn push(&mut self, cp: u32) {
        let b = &mut self.0;
        match cp {
            0..=0x7f => b.push(cp as u8),
            0x80..=0x7ff => {
                b.push(0xc0 | (cp >> 6) as u8);
                b.push(0x80 | (cp & 0x3f) as u8);
            }
            0x800..=0xffff => {
                b.push(0xe0 | (cp >> 12) as u8);
                b.push(0x80 | ((cp >> 6) & 0x3f) as u8);
                b.push(0x80 | (cp & 0x3f) as u8);
            }
            _ => {
                b.push(0xf0 | (cp >> 18) as u8);
                b.push(0x80 | ((cp >> 12) & 0x3f) as u8);
                b.push(0x80 | ((cp >> 6) & 0x3f) as u8);
                b.push(0x80 | (cp & 0x3f) as u8);
            }
        }
    }

    /// The text, or None if it holds a lone surrogate.
    pub fn as_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.0).ok()
    }

    pub fn code_points(&self) -> impl Iterator<Item = u32> + '_ {
        let b = &self.0;
        let mut i = 0;
        std::iter::from_fn(move || {
            let lead = *b.get(i)? as u32;
            let (len, init) = match lead {
                0..=0x7f => (1, lead),
                0xc0..=0xdf => (2, lead & 0x1f),
                0xe0..=0xef => (3, lead & 0x0f),
                _ => (4, lead & 0x07),
            };
            let cp = b[i + 1..i + len]
                .iter()
                .fold(init, |acc, &c| (acc << 6) | (c as u32 & 0x3f));
            i += len;
            Some(cp)
        })
    }
}

impl PartialEq<str> for PyStr {
    fn eq(&self, other: &str) -> bool {
        self.0 == other.as_bytes()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// An integer as Python would print it (JSON integers never have leading zeros,
    /// so this is the source text, with `-0` read as `0`).
    Int(String),
    Float(f64),
    Str(PyStr),
    Array(Vec<Value>),
    /// Key order is kept; keys are unique once parsing succeeds.
    Object(Vec<(PyStr, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

/// Values can nest thousands of levels deep before the depth check rejects them, so
/// they are dropped without recursion.
impl Drop for Value {
    fn drop(&mut self) {
        let mut stack: Vec<Value> = Vec::new();
        take_children(self, &mut stack);
        while let Some(mut v) = stack.pop() {
            take_children(&mut v, &mut stack);
        }
    }
}

fn take_children(value: &mut Value, stack: &mut Vec<Value>) {
    match value {
        Value::Array(items) => stack.append(items),
        Value::Object(pairs) => stack.extend(pairs.drain(..).map(|(_, v)| v)),
        _ => {}
    }
}

/// Why a body was rejected as JSON; every case is a 400.
#[derive(Debug, PartialEq)]
pub enum JsonError {
    Utf8,
    Syntax,
    Depth,
    Duplicate(PyStr),
    Constant(&'static str),
    Range(String),
    NotObject,
}

impl JsonError {
    /// The `detail` the Python worker returns for this error.
    pub fn message(&self) -> String {
        match self {
            JsonError::Utf8 => "request body must be valid UTF-8 text".into(),
            JsonError::Syntax => "request body must be valid JSON".into(),
            JsonError::Depth => "request body is nested too deeply".into(),
            JsonError::Duplicate(key) => {
                format!("duplicate key {} in a JSON object", repr_str(key))
            }
            JsonError::Constant(name) => format!("{name} is not valid JSON"),
            JsonError::Range(text) => format!("number {text} is out of range"),
            JsonError::NotObject => "request body must be a JSON object".into(),
        }
    }
}

/// Decode a request body into its top-level object.
pub fn parse(raw: &[u8]) -> Result<Vec<(PyStr, Value)>, JsonError> {
    let text = std::str::from_utf8(raw).map_err(|_| JsonError::Utf8)?;
    let mut p = Parser {
        s: text.as_bytes(),
        i: 0,
    };
    p.ws();
    let mut value = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(JsonError::Syntax);
    }
    check(&value, 1)?;
    match &mut value {
        Value::Object(pairs) => Ok(std::mem::take(pairs)),
        _ => Err(JsonError::NotObject),
    }
}

/// `_check_unicode` from `contract.py`: visit values in order (an object's key before
/// its value) and fail at the first lone surrogate or the first call past
/// [`MAX_CHECK_CALLS`], whichever comes first.
fn check(value: &Value, calls: usize) -> Result<(), JsonError> {
    if calls > MAX_CHECK_CALLS {
        return Err(JsonError::Depth);
    }
    match value {
        Value::Str(s) if s.as_str().is_none() => Err(JsonError::Utf8),
        Value::Array(items) => items.iter().try_for_each(|v| check(v, calls + 1)),
        Value::Object(pairs) => pairs.iter().try_for_each(|(k, v)| {
            if k.as_str().is_none() {
                return Err(JsonError::Utf8);
            }
            check(v, calls + 1)
        }),
        _ => Ok(()),
    }
}

/// An array or object that has been opened but not closed yet.
enum Open {
    Array(Vec<Value>),
    /// the members so far, and the key whose value is being parsed
    Object(Vec<(PyStr, Value)>, PyStr),
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn rest_starts_with(&self, lit: &[u8]) -> bool {
        self.s[self.i..].starts_with(lit)
    }

    fn ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.i += 1;
        }
    }

    /// One JSON value. Containers are parsed with an explicit stack, so deep nesting
    /// cannot overflow the thread's stack before [`MAX_PARSE_DEPTH`] stops it.
    fn value(&mut self) -> Result<Value, JsonError> {
        let mut open: Vec<Open> = Vec::new();
        loop {
            // the start of a value: open a container or read a scalar
            let mut done = match self.peek() {
                Some(b'{') => {
                    if open.len() + 1 > MAX_PARSE_DEPTH {
                        return Err(JsonError::Depth);
                    }
                    self.i += 1;
                    self.ws();
                    if self.peek() == Some(b'}') {
                        self.i += 1;
                        Value::Object(Vec::new())
                    } else {
                        let key = self.key()?;
                        open.push(Open::Object(Vec::new(), key));
                        continue;
                    }
                }
                Some(b'[') => {
                    if open.len() + 1 > MAX_PARSE_DEPTH {
                        return Err(JsonError::Depth);
                    }
                    self.i += 1;
                    self.ws();
                    if self.peek() == Some(b']') {
                        self.i += 1;
                        Value::Array(Vec::new())
                    } else {
                        open.push(Open::Array(Vec::new()));
                        continue;
                    }
                }
                _ => self.scalar()?,
            };
            // hand the finished value to its container, closing containers that end
            loop {
                match open.last_mut() {
                    None => return Ok(done),
                    Some(Open::Array(items)) => {
                        items.push(done);
                        self.ws();
                        match self.peek() {
                            Some(b',') => {
                                self.i += 1;
                                self.ws();
                                break;
                            }
                            Some(b']') => {
                                self.i += 1;
                                let Some(Open::Array(items)) = open.pop() else {
                                    unreachable!()
                                };
                                done = Value::Array(items);
                            }
                            _ => return Err(JsonError::Syntax),
                        }
                    }
                    Some(Open::Object(pairs, key)) => {
                        pairs.push((std::mem::take(key), done));
                        self.ws();
                        match self.peek() {
                            Some(b',') => {
                                self.i += 1;
                                self.ws();
                                *key = self.key()?;
                                break;
                            }
                            Some(b'}') => {
                                self.i += 1;
                                let Some(Open::Object(pairs, _)) = open.pop() else {
                                    unreachable!()
                                };
                                check_duplicates(&pairs)?;
                                done = Value::Object(pairs);
                            }
                            _ => return Err(JsonError::Syntax),
                        }
                    }
                }
            }
        }
    }

    /// A member name and its colon.
    fn key(&mut self) -> Result<PyStr, JsonError> {
        if self.peek() != Some(b'"') {
            return Err(JsonError::Syntax);
        }
        self.i += 1;
        let key = self.string()?;
        self.ws();
        if self.peek() != Some(b':') {
            return Err(JsonError::Syntax);
        }
        self.i += 1;
        self.ws();
        Ok(key)
    }

    fn scalar(&mut self) -> Result<Value, JsonError> {
        match self.peek() {
            Some(b'"') => {
                self.i += 1;
                Ok(Value::Str(self.string()?))
            }
            Some(b'n') if self.rest_starts_with(b"null") => {
                self.i += 4;
                Ok(Value::Null)
            }
            Some(b't') if self.rest_starts_with(b"true") => {
                self.i += 4;
                Ok(Value::Bool(true))
            }
            Some(b'f') if self.rest_starts_with(b"false") => {
                self.i += 5;
                Ok(Value::Bool(false))
            }
            Some(b'N') if self.rest_starts_with(b"NaN") => Err(JsonError::Constant("NaN")),
            Some(b'I') if self.rest_starts_with(b"Infinity") => {
                Err(JsonError::Constant("Infinity"))
            }
            Some(b'-') if self.rest_starts_with(b"-Infinity") => {
                Err(JsonError::Constant("-Infinity"))
            }
            _ => self.number(),
        }
    }

    fn digits(&mut self) {
        while let Some(b'0'..=b'9') = self.peek() {
            self.i += 1;
        }
    }

    fn number(&mut self) -> Result<Value, JsonError> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        match self.peek() {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => self.digits(),
            _ => return Err(JsonError::Syntax),
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') && matches!(self.s.get(self.i + 1), Some(b'0'..=b'9')) {
            self.i += 1;
            self.digits();
            is_float = true;
        }
        if let Some(b'e' | b'E') = self.peek() {
            let e_start = self.i;
            self.i += 1;
            if let Some(b'+' | b'-') = self.peek() {
                self.i += 1;
            }
            let digits_start = self.i;
            self.digits();
            if self.i > digits_start {
                is_float = true;
            } else {
                // not an exponent after all; what follows is left for the caller
                self.i = e_start;
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).expect("ASCII");
        if is_float {
            let v: f64 = text.parse().map_err(|_| JsonError::Syntax)?;
            if !v.is_finite() {
                return Err(JsonError::Range(text.to_string()));
            }
            Ok(Value::Float(v))
        } else {
            let digits = text.strip_prefix('-').unwrap_or(text);
            if digits.len() > MAX_INT_DIGITS {
                return Err(JsonError::Syntax);
            }
            Ok(Value::Int(if digits == "0" {
                "0".into()
            } else {
                text.into()
            }))
        }
    }

    fn hex4(&self, at: usize) -> Option<u32> {
        let h = self.s.get(at..at + 4)?;
        let mut v = 0;
        for &c in h {
            v = v * 16 + (c as char).to_digit(16)?;
        }
        Some(v)
    }

    /// The rest of a string whose opening quote has been read.
    fn string(&mut self) -> Result<PyStr, JsonError> {
        let mut out = PyStr::default();
        loop {
            match self.peek().ok_or(JsonError::Syntax)? {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    let esc = *self.s.get(self.i + 1).ok_or(JsonError::Syntax)?;
                    self.i += 2;
                    let cp = match esc {
                        b'"' => '"' as u32,
                        b'\\' => '\\' as u32,
                        b'/' => '/' as u32,
                        b'b' => 0x08,
                        b'f' => 0x0c,
                        b'n' => '\n' as u32,
                        b'r' => '\r' as u32,
                        b't' => '\t' as u32,
                        b'u' => {
                            let mut c = self.hex4(self.i).ok_or(JsonError::Syntax)?;
                            self.i += 4;
                            // a high surrogate joins a directly following low one
                            if (0xd800..=0xdbff).contains(&c)
                                && self.rest_starts_with(b"\\u")
                                && let Some(c2) = self.hex4(self.i + 2)
                                && (0xdc00..=0xdfff).contains(&c2)
                            {
                                c = 0x10000 + ((c - 0xd800) << 10) + (c2 - 0xdc00);
                                self.i += 6;
                            }
                            c
                        }
                        _ => return Err(JsonError::Syntax),
                    };
                    out.push(cp);
                }
                0x00..=0x1f => return Err(JsonError::Syntax),
                _ => {
                    let start = self.i;
                    while let Some(c) = self.peek() {
                        if c == b'"' || c == b'\\' || c < 0x20 {
                            break;
                        }
                        self.i += 1;
                    }
                    out.0.extend_from_slice(&self.s[start..self.i]);
                }
            }
        }
    }
}

/// Python's object_pairs_hook runs when an object closes and reports the first key
/// that repeats an earlier one.
fn check_duplicates(pairs: &[(PyStr, Value)]) -> Result<(), JsonError> {
    let mut seen = HashSet::with_capacity(pairs.len());
    for (key, _) in pairs {
        if !seen.insert(key) {
            return Err(JsonError::Duplicate(key.clone()));
        }
    }
    Ok(())
}

/// Python's `repr(float)`: the shortest digits that round-trip, in fixed notation
/// for exponents from -5 to 15 and scientific notation otherwise.
pub fn float_repr(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let (neg, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let (digits, exp) = break_tie_to_even(x, digits, exp);
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    let decpt = exp + 1;
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let sign = if exp < 0 { '-' } else { '+' };
        write!(out, "e{sign}{:02}", exp.unsigned_abs()).unwrap();
    } else if decpt <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-decpt) as usize));
        out.push_str(&digits);
    } else if decpt as usize >= digits.len() {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', decpt as usize - digits.len()));
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    out
}

/// When `x` lies exactly halfway between two shortest round-tripping decimals, Python
/// (David Gay's dtoa) takes the one with an even last digit, while Rust's shortest
/// formatting may take the other. Ties need an exact decimal expansion only one
/// digit longer than the shortest, which takes 16 or more significant digits.
fn break_tie_to_even(x: f64, digits: String, exp: i32) -> (String, i32) {
    let n = digits.len();
    if n < 15 {
        return (digits, exp);
    }
    // every finite double has at most 767 significant decimal digits
    let exact = format!("{:.800e}", x.abs());
    let (mantissa, e) = exact.split_once('e').expect("{:e} has an exponent");
    let e: i32 = e.parse().expect("integer exponent");
    let full: String = mantissa.chars().filter(|c| *c != '.').collect();
    let full = full.trim_end_matches('0');
    if full.len() != n + 1 || !full.ends_with('5') {
        return (digits, exp);
    }
    let floor = &full[..n];
    let last = floor.as_bytes()[n - 1] - b'0';
    let even = if last.is_multiple_of(2) {
        floor.to_string()
    } else if last == 9 {
        // the even choice would carry, which exact ties (values in [2^50, 2^51)
        // ending in .25 or .75) never need
        return (digits, exp);
    } else {
        // the even choice is floor + 1 in the last digit
        let mut up = floor.as_bytes().to_vec();
        up[n - 1] += 1;
        String::from_utf8(up).expect("ASCII digits")
    };
    // At a power of two the double below is half as far away, so one of the two
    // candidates may read back as a different double; then it is not a tie.
    let back: f64 = format!("{}.{}e{}", &even[..1], &even[1..], e)
        .parse()
        .expect("decimal digits");
    if back.to_bits() != x.abs().to_bits() {
        return (digits, exp);
    }
    (even, e)
}

/// A JSON string literal as `json.dumps(s, ensure_ascii=False)` writes it.
pub fn write_json_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `json.dumps(value, ensure_ascii=False)`, with the default separators when
/// `compact` is false and `(",", ":")` when it is true. The value must not hold
/// lone surrogates (`parse` rejects them).
pub fn dumps(value: &Value, compact: bool) -> String {
    let mut out = String::new();
    write_value(value, compact, &mut out);
    out
}

fn write_value(value: &Value, compact: bool, out: &mut String) {
    let (item_sep, key_sep) = if compact { (",", ":") } else { (", ", ": ") };
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Int(text) => out.push_str(text),
        Value::Float(x) => out.push_str(&float_repr(*x)),
        Value::Str(s) => write_json_str(s.as_str().expect("checked UTF-8"), out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                write_value(item, compact, out);
            }
            out.push(']');
        }
        Value::Object(pairs) => {
            out.push('{');
            for (i, (key, item)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                write_json_str(key.as_str().expect("checked UTF-8"), out);
                out.push_str(key_sep);
                write_value(item, compact, out);
            }
            out.push('}');
        }
    }
}

fn is_printable(cp: u32) -> bool {
    NON_PRINTABLE
        .binary_search_by(|&(lo, hi)| {
            if hi < cp {
                std::cmp::Ordering::Less
            } else if lo > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_err()
}

/// Python's `repr(str)`.
pub fn repr_str(s: &PyStr) -> String {
    let cps: Vec<u32> = s.code_points().collect();
    let squote = cps.contains(&('\'' as u32));
    let dquote = cps.contains(&('"' as u32));
    let quote = if squote && !dquote { '"' } else { '\'' };
    let mut out = String::with_capacity(cps.len() + 2);
    out.push(quote);
    for cp in cps {
        match cp {
            _ if cp == quote as u32 || cp == '\\' as u32 => {
                out.push('\\');
                out.push(char::from_u32(cp).unwrap());
            }
            0x09 => out.push_str("\\t"),
            0x0a => out.push_str("\\n"),
            0x0d => out.push_str("\\r"),
            0..=0x1f | 0x7f => write!(out, "\\x{cp:02x}").unwrap(),
            0x20..=0x7e => out.push(char::from_u32(cp).unwrap()),
            _ if is_printable(cp) => out.push(char::from_u32(cp).unwrap()),
            0x80..=0xff => write!(out, "\\x{cp:02x}").unwrap(),
            0x100..=0xffff => write!(out, "\\u{cp:04x}").unwrap(),
            _ => write!(out, "\\U{cp:08x}").unwrap(),
        }
    }
    out.push(quote);
    out
}

/// Python's `repr` of a decoded JSON value.
pub fn repr(value: &Value) -> String {
    let mut out = String::new();
    write_repr(value, &mut out);
    out
}

fn write_repr(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
        Value::Int(text) => out.push_str(text),
        Value::Float(x) => out.push_str(&float_repr(*x)),
        Value::Str(s) => out.push_str(&repr_str(s)),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_repr(item, out);
            }
            out.push(']');
        }
        Value::Object(pairs) => {
            out.push('{');
            for (i, (key, item)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&repr_str(key));
                out.push_str(": ");
                write_repr(item, out);
            }
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(body: &str) -> String {
        parse(body.as_bytes()).unwrap_err().message()
    }

    #[test]
    fn float_repr_matches_python_examples() {
        let cases = [
            (1.0, "1.0"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1e-5, "1e-05"),
            (1e-4, "0.0001"),
            (-0.0, "-0.0"),
            (1e22, "1e+22"),
            (3.14e-07, "3.14e-07"),
            (0.1, "0.1"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            // 2^-24: a digit string halfway to the next decimal, but the lower one
            // reads back as the double below
            (5.960464477539063e-08, "5.960464477539063e-08"),
            (123456789.0, "123456789.0"),
            (0.0024726232513785362, "0.0024726232513785362"),
            // exactly halfway between two 17-digit candidates: Python takes the even one
            (f64::from_bits(0xc31d5a973d792fa1), "-2065594985630696.2"),
        ];
        for (x, want) in cases {
            assert_eq!(float_repr(x), want, "{x:e}");
        }
    }

    #[test]
    #[ignore = "needs CUA_S1_FLOAT_VECTORS"]
    fn float_repr_matches_python_vectors() {
        // tests/make_float_vectors.py writes "<bits as hex> <repr(float)>" lines
        let path = std::env::var("CUA_S1_FLOAT_VECTORS")
            .expect("CUA_S1_FLOAT_VECTORS must name a file from tests/make_float_vectors.py");
        let text = std::fs::read_to_string(path).unwrap();
        let mut n = 0;
        for line in text.lines() {
            let (bits, want) = line.split_once(' ').unwrap();
            let x = f64::from_bits(u64::from_str_radix(bits, 16).unwrap());
            assert_eq!(float_repr(x), want, "bits {bits}");
            n += 1;
        }
        assert!(n > 1000);
    }

    #[test]
    fn dumps_matches_python() {
        let v = parse(br#"{"a": [1.0, 1e16, 1e-5, 0.0001, -0.0, 1e22, 123456789012345678, 3.14e-07, -0, true, null]}"#).unwrap();
        let v = Value::Object(v);
        assert_eq!(
            dumps(&v, false),
            r#"{"a": [1.0, 1e+16, 1e-05, 0.0001, -0.0, 1e+22, 123456789012345678, 3.14e-07, 0, true, null]}"#
        );
        let s = Value::Str(PyStr::new("\u{0}\u{1f}\u{7f}\u{2028}\"\\/\t\u{8}\u{c}é😀"));
        assert_eq!(
            dumps(&s, false),
            "\"\\u0000\\u001f\u{7f}\u{2028}\\\"\\\\/\\t\\b\\fé😀\""
        );
    }

    #[test]
    fn repr_matches_python() {
        let s = PyStr::new("a\u{7f}\u{a0}\u{2028}😀é");
        assert_eq!(repr_str(&s), "'a\\x7f\\xa0\\u2028😀é'");
        assert_eq!(repr_str(&PyStr::new("it's")), "\"it's\"");
        assert_eq!(repr_str(&PyStr::new("both'\"")), "'both\\'\"'");
        let v = Value::Object(parse(br#"{"k": [1, 2.5, null, true, "x"], "e": {}}"#).unwrap());
        assert_eq!(repr(&v), "{'k': [1, 2.5, None, True, 'x'], 'e': {}}");
    }

    #[test]
    fn surrogates() {
        let v = parse(b"{\"a\": \"\\ud83d\\ude00\"}").unwrap();
        assert_eq!(v[0].1, Value::Str(PyStr::new("😀")));
        assert_eq!(
            err(r#"{"a": "\ud800x"}"#),
            "request body must be valid UTF-8 text"
        );
        assert_eq!(
            err(r#"{"a": "\ude00\ud83d"}"#),
            "request body must be valid UTF-8 text"
        );
        // duplicate-key errors come first, and quote the surrogate
        assert_eq!(
            err(r#"{"\ud800": 1, "\ud800": 2}"#),
            "duplicate key '\\ud800' in a JSON object"
        );
    }

    #[test]
    fn errors() {
        assert_eq!(err("[]"), "request body must be a JSON object");
        assert_eq!(err("\u{feff}{}"), "request body must be valid JSON");
        assert_eq!(err(r#"{"a": NaN}"#), "NaN is not valid JSON");
        assert_eq!(err(r#"{"a": -Infinity}"#), "-Infinity is not valid JSON");
        assert_eq!(err(r#"{"a": 1e400}"#), "number 1e400 is out of range");
        assert_eq!(err(r#"{"a": 1e400x"#), "number 1e400 is out of range");
        assert_eq!(
            err(r#"{"a": 1, "b": 2, "a": 3}"#),
            "duplicate key 'a' in a JSON object"
        );
        assert_eq!(err(r#"{"a": [1,]}"#), "request body must be valid JSON");
        assert_eq!(err("{\"a\": \"\u{1}\"}"), "request body must be valid JSON");
        assert_eq!(err(r#"{"a": 01}"#), "request body must be valid JSON");
        assert_eq!(err(r#"{"a": 1.}"#), "request body must be valid JSON");
        assert_eq!(err(r#"{"a": "\x"}"#), "request body must be valid JSON");
        assert_eq!(err(r#"{} x"#), "request body must be valid JSON");
        assert!(parse(br#"{"a": 1e-400}"#).is_ok());
        let long = format!("{{\"a\": {}}}", "1".repeat(4300));
        assert!(parse(long.as_bytes()).is_ok());
        let long = format!("{{\"a\": -{}}}", "1".repeat(4301));
        assert_eq!(err(&long), "request body must be valid JSON");
        assert_eq!(parse(b"{\"a\": \"\xff\"}").unwrap_err(), JsonError::Utf8);
    }

    /// The Python worker's behaviour, measured over HTTP: `state` nested `d` lists deep
    /// inside the top-level object.
    #[test]
    fn depth() {
        let body = |d: usize, leaf: &str, tail: &str| {
            format!(
                "{{\"model\": \"m\", \"state\": {}{leaf}{}{tail}}}",
                "[".repeat(d),
                "]".repeat(d)
            )
        };
        let deep = "request body is nested too deeply";
        // a scalar leaf takes one more call of the check than an empty container
        assert!(parse(body(967, "\"x\"", "").as_bytes()).is_ok());
        assert_eq!(err(&body(968, "\"x\"", "")), deep);
        assert!(parse(body(968, "", "").as_bytes()).is_ok());
        assert_eq!(err(&body(969, "", "")), deep);
        assert!(parse(body(967, "{}", "").as_bytes()).is_ok());
        assert_eq!(err(&body(968, "{}", "")), deep);
        // up to the parse limit, later parse errors win over depth
        assert_eq!(
            err(&(body(9989, "1", "") + "x")),
            "request body must be valid JSON"
        );
        assert_eq!(err(&(body(9990, "1", "") + "x")), deep);
        assert_eq!(
            err(&body(5000, "1", ", \"state\": 1")),
            "duplicate key 'state' in a JSON object"
        );
        // after parsing, the first problem in walk order wins
        let deep_state = format!("{}1{}", "[".repeat(1500), "]".repeat(1500));
        assert_eq!(
            err(&format!(
                "{{\"model\": \"\\ud800\", \"state\": {deep_state}}}"
            )),
            "request body must be valid UTF-8 text"
        );
        assert_eq!(
            err(&format!("{{\"state\": {deep_state}, \"z\": \"\\ud800\"}}")),
            deep
        );
        assert_eq!(
            err(&format!("{{\"a\": {deep_state}, \"\\ud800\": 1}}")),
            deep
        );
        // far past the limit: fails cleanly, without overflowing the stack
        assert_eq!(err(&"[".repeat(1_000_000)), deep);
        let wide = format!("{{\"a\": {}1{}}}", "[".repeat(9989), "]".repeat(9989));
        assert_eq!(err(&wide), deep);
    }
}
