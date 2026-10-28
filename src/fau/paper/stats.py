"""The statistics behind every reported number: paired class-bootstrap gaps, recovery shares, and number formatting.

A gap A - B averages each class's top-1 over the classifier runs of A and of B, takes the per-class differences, and
reports their mean with a 95% percentile interval from 10,000 bootstrap resamples of the classes (Python's
`random.Random(0)`, reset for every comparison). The interval conditions on the generated datasets and fitted adapters.
"""
import random
import statistics

RESAMPLES = 10000


def class_means(per_class):
    """{run: [per-class top-1]} -> per-class top-1 averaged over the runs."""
    return [statistics.mean(v) for v in zip(*per_class.values())]


def bootstrap(d, resamples=RESAMPLES, seed=0, exact_mean=False):
    """Mean of the per-class differences `d` (points) and its 95% percentile interval over classes."""
    rng, k = random.Random(seed), len(d)
    boot = sorted(sum(d[rng.randrange(k)] for _ in range(k)) / k for _ in range(resamples))
    mean = statistics.mean(d) if exact_mean else sum(d) / k
    return mean, boot[int(0.025 * resamples)], boot[int(0.975 * resamples) - 1]


def gap(a, b, exact_mean=False):
    """Paired gap of two {run: [per-class top-1 in 0..1]} records, in points: (mean, low, high)."""
    ma, mb = class_means(a), class_means(b)
    assert len(ma) == len(mb), "the two groups are evaluated on different classes"
    return bootstrap([100 * (x - y) for x, y in zip(ma, mb)], exact_mean=exact_mean)


def gap_by_class(a, b):
    """Paired gap of two {class: top-1} records over the classes both cover (OOD sets and transfer targets)."""
    cs = sorted(set(a) & set(b), key=int)
    return bootstrap([100 * (a[c] - b[c]) for c in cs])


def share(arm, reference, released):
    """The share (%) of the reference's gain over the released model that the arm recovers."""
    return 100 * (arm - released) / (reference - released)


# Formatting ----------------------------------------------------------------------------------------------------------
def signed(x, digits=1):
    """+1.2 / $-$1.2, as the paper prints gaps."""
    return f"{x:+.{digits}f}".replace("-", "$-$")


def interval(lo, hi, digits=1):
    return f"[{signed(lo, digits)}, {signed(hi, digits)}]"


def pm(mean, sd, bold=False):
    """A table cell: mean (bold if asked) with its standard deviation in the uncertainty font."""
    m = f"{mean:.1f}"
    return (rf"\textbf{{{m}}}" if bold else m) + rf"\,{{\tableuncert$\pm${sd:.1f}}}"


def gap_cell(mean, lo, hi, bold_if_positive=True):
    """A gap with its interval in the uncertainty font, bold when the interval lies above zero."""
    m = signed(mean)
    if bold_if_positive and lo > 0:
        m = rf"\textbf{{{m}}}"
    return m + rf"\,{{\tableuncert{interval(lo, hi)}}}"
