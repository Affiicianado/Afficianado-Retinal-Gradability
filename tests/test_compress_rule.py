"""The deployment rule, fed the numbers retina-08-compression actually measured."""
from src.deploy.compress import choose_variant

MEASURED = {
    "float32": dict(size_mb=17.36, ms=17.8, confident_agreement=1.000,
                    worst_reject_as_good=0.0855, grade_vs_reject=0.481,
                    grade_vs_defocus=0.516),
    "float16": dict(size_mb=8.74, ms=18.1, confident_agreement=1.000,
                    worst_reject_as_good=0.0843, grade_vs_reject=0.482,
                    grade_vs_defocus=0.515),
    "dynamic": dict(size_mb=4.88, ms=32.6, confident_agreement=0.949,
                    worst_reject_as_good=0.0914, grade_vs_reject=0.507,
                    grade_vs_defocus=0.550),
    "int8": dict(size_mb=5.25, ms=18.7, confident_agreement=0.602,
                 worst_reject_as_good=0.0333, grade_vs_reject=0.338,
                 grade_vs_defocus=0.167),
}
UNCOMPRESSED = dict(worst_reject_as_good=0.0855, grade_vs_reject=0.481,
                    grade_vs_defocus=0.516)


def test_rule_ships_float16():
    chosen, eligible, why = choose_variant(MEASURED, UNCOMPRESSED)
    assert chosen == "float16"
    assert sorted(eligible) == ["float16", "float32"]


def test_int8_is_rejected_despite_the_better_safety_number():
    _, _, why = choose_variant(MEASURED, UNCOMPRESSED)
    assert any("agreement" in f for f in why["int8"])
    assert any("grade-vs-defocus" in f for f in why["int8"])


def test_the_old_tie_break_would_have_shipped_float32():
    """Defect 17: sorting eligible variants on (latency, size) picked a model that
    was barely compressed. Kept as a reminder of why the rule is size-first."""
    _, eligible, _ = choose_variant(MEASURED, UNCOMPRESSED)
    old = min(eligible, key=lambda n: (MEASURED[n]["ms"], MEASURED[n]["size_mb"]))
    assert old == "float32"
