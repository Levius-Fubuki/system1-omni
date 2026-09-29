"""Request bodies for comparing the Python and native Cua-S1 text workers.

    python recipe/cua_s1/diff_corpus.py tests/cua_s1/data/text_inputs.json corpus.jsonl [n_fuzz]

Each line is {"name": ..., "body": <base64 of the raw request body>}. The set has
the fixed input set, hand-written edge cases for every error path of
`contract.parse_body` / `map_request` / the server, and seeded random bodies.
"""

from __future__ import annotations

import base64
import json
import random
import sys

M = "cua-s1-4b-0.2"


def req(state="Button: OK", questions=None, model=M, **extra):
    body = {"model": model, "state": state}
    body["questions"] = questions if questions is not None else {"q": choice()}
    body.update(extra)
    return body


def choice(instructions="Press OK.", criteria=None, type_="choice"):
    q = {"type": type_, "instructions": instructions}
    q["criteria"] = (
        criteria if criteria is not None else {"ok": "OK", "cancel": "Cancel"}
    )
    return q


def build(inputs_path: str, n_fuzz: int) -> list[tuple[str, bytes]]:
    cases: list[tuple[str, bytes]] = []

    def add(name, body):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, str):
            body = body.encode("utf-8", "surrogatepass")
        cases.append((name, body))

    fixtures = json.load(open(inputs_path, encoding="utf-8"))
    for name, body in fixtures.items():
        add(f"fixture/{name}", body)
        add(
            f"fixture_ascii_compact/{name}",
            json.dumps(body, ensure_ascii=True, separators=(",", ":")),
        )

    # --- JSON syntax and decoding ---
    ok = json.dumps(req())
    raw = {
        "empty": "",
        "space": " ",
        "open": "{",
        "close": "}",
        "array": "[]",
        "null": "null",
        "number": "1",
        "string": '"x"',
        "true": "true",
        "extra": ok + " x",
        "bom": "﻿" + ok,
        "ws": " \t\n" + ok + "\r\n",
        "trailing_comma_obj": ok[:-1] + ",}",
        "single_quotes": ok.replace('"', "'"),
        "comment": "// c\n" + ok,
        "nan": ok.replace('"Button: OK"', "NaN"),
        "inf": ok.replace('"Button: OK"', "Infinity"),
        "neg_inf": ok.replace('"Button: OK"', "-Infinity"),
        "nan_nested": ok.replace('"OK"', "[1, NaN]"),
        "float_range": ok.replace('"OK"', "[1e400]"),
        "float_range_neg": ok.replace('"OK"', "[-1E309]"),
        "float_range_then_garbage": ok.replace('"OK"', "[1e400x]"),
        "float_underflow": ok.replace('"OK"', "[1e-400, 1.0e+308, -0.0, 5e-324]"),
        "int_4300": ok.replace('"OK"', "[" + "9" * 4300 + "]"),
        "int_4301": ok.replace('"OK"', "[" + "9" * 4301 + "]"),
        "int_neg_4301": ok.replace('"OK"', "[-" + "9" * 4301 + "]"),
        "leading_zero": ok.replace('"OK"', "[01]"),
        "dot_no_digit": ok.replace('"OK"', "[1.]"),
        "exp_no_digit": ok.replace('"OK"', "[1e]"),
        "exp_sign_no_digit": ok.replace('"OK"', "[1e+]"),
        "minus_alone": ok.replace('"OK"', "[-]"),
        "plus": ok.replace('"OK"', "[+1]"),
        "hex": ok.replace('"OK"', "[0x10]"),
        "raw_control": ok.replace("Press OK.", "Press\u0001OK."),
        "raw_tab": ok.replace("Press OK.", "Press\tOK."),
        "raw_del": ok.replace("Press OK.", "Press\u007fOK."),
        "bad_escape": ok.replace("Press OK.", "Press \\x OK."),
        "short_u": ok.replace("Press OK.", "\\u12"),
        "bad_u": ok.replace("Press OK.", "\\uZZZZ"),
        "escapes": ok.replace("Press OK.", '\\/\\b\\f\\n\\r\\t\\"\\\\ \\u00e9\\u0000'),
        "missing_colon": ok.replace('"model":', '"model"'),
        "unterminated": ok[:-3],
        "dup_top": '{"model": "'
        + M
        + '", "model": "'
        + M
        + '", "state": "s", "questions": {}}',
        "dup_criteria": ok.replace('"cancel": "Cancel"', '"ok": "Again"'),
        "dup_then_syntax": ok.replace('"cancel": "Cancel"', '"ok": "Again"') + "x",
        "syntax_inside_dup_object": ok.replace(
            '"cancel": "Cancel"', '"ok": "Again", x'
        ),
        "dup_state_nested": ok.replace('"Button: OK"', '{"a": {"b": 1, "b": 2}}'),
        "dup_after_nan": ok.replace('"cancel": "Cancel"', '"ok": NaN'),
        "lone_high": ok.replace("Press OK.", "\\ud800"),
        "lone_low": ok.replace("Press OK.", "\\udc00x"),
        "pair": ok.replace("Press OK.", "\\ud83d\\ude00"),
        "reversed_pair": ok.replace("Press OK.", "\\ude00\\ud83d"),
        "high_then_bmp": ok.replace("Press OK.", "\\ud800\\u0041"),
        "lone_in_key": ok.replace('"ok": "OK"', '"\\udfff": "OK"'),
        "lone_dup_key": ok.replace(
            '"ok": "OK", "cancel": "Cancel"', '"\\ud800": 1, "\\ud800": 2'
        ),
        "lone_then_syntax": ok.replace("Press OK.", "\\ud800") + ",",
    }
    for name, text in raw.items():
        add(f"json/{name}", text)
    add("json/invalid_utf8", ok.encode().replace(b"Press", b"Pr\xffss"))
    add("json/overlong_utf8", ok.encode().replace(b"Press", b"Pr\xc0\xafss"))
    add("json/utf8_surrogate", ok.encode().replace(b"Press", b"Pr\xed\xa0\x80ss"))
    add("json/utf8_bom_bytes", b"\xef\xbb\xbf" + ok.encode())
    add("json/latin1", ok.encode().replace(b"Press", b"Pr\xe9ss"))

    def nested(d, leaf='"x"', open_="[", close="]"):
        return open_ * d + leaf + close * d

    for d in (1, 50, 966, 967, 968, 969, 1500):
        add(f"depth/state_list_{d}", ok.replace('"Button: OK"', nested(d)))
    for d in (964, 965, 966):
        add(f"depth/criteria_{d}", ok.replace('"OK"', nested(d)))
        add(f"depth/type_{d}", ok.replace('"choice"', nested(d)))
        add(
            f"depth/instructions_obj_{d}",
            ok.replace('"Press OK."', nested(d, '"x"', '{"a":', "}")),
        )

    head = '{"model": "cua-s1-4b-0.2", "state": '
    tail = ', "questions": {"q": {"type": "choice", "instructions": "go", "criteria": {"a": "A"}}}}'
    for d in (967, 968, 969, 970):
        add(f"depth2/empty_list_leaf_{d}", head + nested(d, "") + tail)
        add(f"depth2/empty_obj_leaf_{d}", head + nested(d, "{}") + tail)
        add(f"depth2/number_leaf_{d}", head + nested(d, "1") + tail)
    for d in (9988, 9989, 9990, 9991):
        add(f"depth2/parse_limit_garbage_{d}", head + nested(d, "1") + tail + "x")
    add("depth2/deep_then_dup", head + nested(5000, "1") + ', "state": 1' + tail)
    add("depth2/deep_then_nan", head + nested(3000, "1") + ', "z": NaN' + tail)
    add(
        "depth2/surrogate_before_deep",
        '{"model": "\\ud800", "state": ' + nested(1500, "1") + tail,
    )
    add(
        "depth2/deep_before_surrogate",
        head + nested(1500, "1") + ', "z": "\\ud800"' + tail,
    )
    add(
        "depth2/deep_before_surrogate_key",
        '{"a": ' + nested(1500, "1") + ', "\\ud800": 1}',
    )
    add("depth2/hundred_thousand", "[" * 100000)
    add("depth2/deep_top_level_array", nested(5000, "1"))
    sa = {
        "_sample": choice(criteria={"_save": "Save", "_sa": None, "b": "B"}),
        "q": choice(),
    }
    add("keys/_sa_names_and_options", req(questions=sa))
    add("keys/_sa_state_keys", req(state={"_sa": 1, "_sample": [2]}))

    # --- mapping errors, in Python's check order ---
    sem = {
        "no_model": {"state": "s", "questions": {"q": choice()}},
        "model_null": req(model=None),
        "model_case": req(model="Cua-S1-4B-0.2"),
        "model_space": req(model=M + " "),
        "model_number": req(model=1),
        "model_wrong_and_no_state": {"model": "x"},
        "no_state": {"model": M, "questions": {"q": choice()}},
        "state_null": req(state=None),
        "state_true": req(state=True),
        "state_zero": req(state=0),
        "state_float": req(state=1.5),
        "state_empty": req(state=""),
        "state_empty_obj": req(state={}),
        "state_empty_list": req(state=[]),
        "state_spaces": req(state="   "),
        "state_obj": req(
            state={
                "a": None,
                "b": [1, 2.5, -0.0, 1e22, 12345678901234567890, True],
                "é": " ",
            }
        ),
        "state_list": req(state=["Button: OK", {"k": "v"}, 3.14e-07]),
        "no_questions": {"model": M, "state": "s"},
        "questions_null": req(questions=None) | {"questions": None},
        "questions_list": req(questions=[]),
        "questions_empty": req(questions={}),
        "questions_str": req(questions="q"),
        "questions_65": req(questions={f"q{i}": choice() for i in range(65)}),
        "questions_65_bad_types": req(
            questions={f"q{i}": {"type": "score"} for i in range(65)}
        ),
        "questions_12": req(
            questions={f"q{i}": choice(f"Pick {i}.") for i in range(12)}
        ),
        "question_list": req(questions={"q": []}),
        "question_null": req(questions={"q": None}),
        "question_str": req(questions={"q": "choice"}),
        "score_second": req(
            questions={"a": {"type": "choice"}, "b": {"type": "score"}}
        ),
        "noul": req(questions={"a": {"type": "noul"}}),
        "type_missing": req(
            questions={"a": {"instructions": "x", "criteria": {"a": "A"}}}
        ),
        "missing_instructions_then_score": req(
            questions={"a": {"type": "choice"}, "b": {"type": "noul"}}
        ),
        "no_instructions": req(
            questions={"a": {"type": "choice", "criteria": {"x": "X"}}}
        ),
        "instructions_null": req(questions={"a": choice(None)}),
        "instructions_empty": req(questions={"a": choice("")}),
        "instructions_zero": req(questions={"a": choice(0)}),
        "instructions_false": req(questions={"a": choice(False)}),
        "instructions_obj": req(questions={"a": choice({"x": 1.5e-7, "y": [None]})}),
        "instructions_empty_obj": req(questions={"a": choice({})}),
        "instructions_list": req(questions={"a": choice([])}),
        "no_criteria": req(questions={"a": {"type": "choice", "instructions": "x"}}),
        "criteria_null": req(
            questions={"a": choice(criteria=None) | {"criteria": None}}
        ),
        "criteria_list": req(questions={"a": choice(criteria=[])}),
        "criteria_empty": req(questions={"a": choice(criteria={})}),
        "criteria_26": req(
            questions={
                "a": choice(criteria={f"o{i}": f"Option {i}" for i in range(26)})
            }
        ),
        "criteria_27": req(
            questions={
                "a": choice(criteria={f"o{i}": f"Option {i}" for i in range(27)})
            }
        ),
        "criteria_values": req(
            questions={
                "a": choice(
                    criteria={
                        "n": None,
                        "o": {"k": [1, 2]},
                        "l": [],
                        "e": {},
                        "q": 'quote"s',
                        "b": "back\\slash",
                        "nl": "new\nline",
                        "t": "tab\t",
                        "z": "\u0000",
                        "ls": " ",
                        "é": "ünï 中文 😀",
                    }
                )
            }
        ),
        "criteria_bool": req(questions={"a": choice(criteria={"x": "X", "y": True})}),
        "criteria_number": req(questions={"a": choice(criteria={"x": 1})}),
        "second_question_bad": req(questions={"a": choice(), "b": choice(criteria={})}),
    }
    for name, body in sem.items():
        add(f"map/{name}", body)

    weird_names = [
        "it's",
        'say "hi"',
        "both'\"",
        "back\\slash",
        "new\nline",
        "nul\u0000",
        "del\u007f",
        "nbsp ",
        "ls ",
        "zw​",
        "tag\U000e0001",
        "pua",
        "unassigned͸",
        "emoji😀",
        "é",
        "combining é",
        "rtl א",
        "soft­",
        "space ",
        "",
        "\t",
    ]
    for i, name in enumerate(weird_names):
        add(f"names/type_{i}", req(questions={name: {"type": name}}))
        add(f"names/option_{i}", req(questions={name: choice(criteria={name: 1})}))
        add(
            f"names/ok_{i}",
            req(questions={name: choice(criteria={name: None, "other": "Other"})}),
        )
    for i, kind in enumerate(
        [
            "",
            "Choice",
            1,
            1.5,
            1e16,
            -0.0,
            1e-5,
            True,
            None,
            [],
            {},
            {"a": [1, {"b": None}]},
            12345678901234567890,
            2.5e-310,
        ]
    ):
        add(f"types/{i}", req(questions={"q": {"type": kind}}))

    # --- prompt length ---
    long_state = "Row: item " * 3000  # a little over 16384 tokens
    add("limit/prompt_too_long", req(state=long_state))
    add(
        "limit/second_prompt_too_long",
        req(questions={"a": choice(), "b": choice("x " * 17000)}),
    )
    add("limit/prompt_just_under", req(state="Row: item " * 2600))

    # --- seeded random bodies ---
    rnd = random.Random(0)
    pool = (
        "abcXYZ019 _-.:/\\\"'\n\t\r{}[]<>|é中文😀 ​ ́א\u0000\u001f\u007f"
        "<|im_start|><|im_end|><think>"
    )

    def rstr(n=12):
        return "".join(rnd.choice(pool) for _ in range(rnd.randint(0, n)))

    def rnum():
        k = rnd.random()
        if k < 0.3:
            return rnd.randint(-(10 ** rnd.randint(1, 30)), 10 ** rnd.randint(1, 30))
        if k < 0.9:
            return rnd.uniform(-1, 1) * 10 ** rnd.randint(-320, 300)
        return rnd.choice([0.0, -0.0, 5e-324, 1e16, 1e-5, 0.1, 1 / 3])

    def rval(depth=0):
        k = rnd.random()
        if depth > 3 or k < 0.35:
            return rstr()
        if k < 0.5:
            return rnum()
        if k < 0.55:
            return rnd.choice([None, True, False])
        if k < 0.75:
            return [rval(depth + 1) for _ in range(rnd.randint(0, 4))]
        return {rstr(6): rval(depth + 1) for _ in range(rnd.randint(0, 4))}

    for i in range(n_fuzz):
        qs = {}
        for _ in range(rnd.randint(1, 3)):
            crit = {
                rstr(8) or "k": rnd.choice([rstr(), None, rval(), rval()])
                for _ in range(rnd.randint(1, 6))
            }
            qs[rstr(8)] = {
                "type": "choice" if rnd.random() < 0.9 else rval(),
                "instructions": rnd.choice([rstr(40), None, rval(), ""]),
                "criteria": crit,
            }
        body = req(state=rnd.choice([rstr(200), rval(), rval()]), questions=qs)
        text = json.dumps(body, ensure_ascii=rnd.random() < 0.3)
        add(f"fuzz/{i}", text)
        # byte-level damage to a copy
        b = bytearray(text.encode("utf-8", "surrogatepass"))
        for _ in range(rnd.randint(1, 3)):
            op, pos = rnd.random(), rnd.randrange(len(b))
            if op < 0.4:
                del b[pos]
            elif op < 0.8:
                b.insert(pos, rnd.choice(b'{}[],:"\\ 0e-.\x00\xff'))
            else:
                b[pos] = rnd.randrange(256)
        add(f"fuzz_damaged/{i}", bytes(b))
    return cases


def main() -> None:
    n_fuzz = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    cases = build(sys.argv[1], n_fuzz)
    with open(sys.argv[2], "w") as f:
        for name, body in cases:
            f.write(
                json.dumps({"name": name, "body": base64.b64encode(body).decode()})
                + "\n"
            )
    print(f"{len(cases)} bodies")


if __name__ == "__main__":
    main()
