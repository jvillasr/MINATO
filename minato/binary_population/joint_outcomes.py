"""Experimental joint RV/period/mass-ratio response-bank inference.

A row is one complete survey outcome, not an independent period likelihood.
The caller must run the same blind measurement/selection pipeline on banks and
observations. Missing periods and mass ratios are explicit outcomes. Intrinsic
coordinates used for importance weights must precede any Roche-period mapping.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp, ndtr


def powerlaw_logpdf(values, exponent, lower, upper):
    """Normalised density proportional to x**exponent on positive bounds."""
    x = np.asarray(values, dtype=float)
    if not 0 < lower < upper or not np.isfinite(exponent):
        raise ValueError('Positive ordered bounds and a finite exponent are required')
    a = exponent + 1.0
    log_range = np.log(upper / lower)
    if abs(a) < 1e-10:
        log_norm = np.log(log_range)
    elif a > 0:
        log_norm = a*np.log(upper) + np.log(-np.expm1(-a*log_range)) - np.log(a)
    else:
        log_norm = a*np.log(lower) + np.log(-np.expm1(a*log_range)) - np.log(-a)
    safe = np.where(x > 0, x, 1.0)
    return np.where((x >= lower) & (x <= upper) & np.isfinite(x),
                    exponent*np.log(safe)-log_norm, -np.inf)


def mass_ratio_efficiency(q, maximum=1.0, minimum_q=0.3):
    """Assumed linear q-measurement efficiency; not an instrument calibration."""
    if not 0 <= maximum <= 1 or not 0 <= minimum_q < 1:
        raise ValueError('Invalid efficiency bounds')
    q = np.asarray(q, dtype=float)
    if np.any(~np.isfinite(q)) or np.any((q < 0) | (q > 1)):
        raise ValueError('Intrinsic mass ratios must be finite and in [0, 1]')
    return maximum*np.clip((q-minimum_q)/(1-minimum_q), 0, 1)


@dataclass(frozen=True)
class OutcomeAxis:
    """Bin a measured quantity, reserving code zero for NaN when missing=True."""
    name: str
    edges: tuple
    missing: bool = False

    def __post_init__(self):
        edges = np.asarray(self.edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or np.isnan(edges).any() or not np.all(np.diff(edges) > 0):
            raise ValueError('Edges must be ordered without NaNs')

    @property
    def size(self):
        return len(self.edges)-1+int(self.missing)

    def encode(self, values):
        x = np.asarray(values, dtype=float)
        absent = np.isnan(x)
        if absent.any() and not self.missing:
            raise ValueError(f'{self.name} does not allow missing values')
        valid = ~absent
        if np.any(~np.isfinite(x[valid])) or np.any(x[valid] < self.edges[0]) or np.any(x[valid] > self.edges[-1]):
            raise ValueError(f'{self.name} contains values outside its bins')
        code = np.searchsorted(self.edges, x, side='right')-1
        code[x == self.edges[-1]] = len(self.edges)-2
        return np.where(absent, 0, code+int(self.missing)).astype(int)


@dataclass
class OutcomeBank:
    """Forward-measured rows; binaries also carry pre-guard logP, q and proposal log density.

    proposal_log_density is the normalised *joint* sampling density in raw
    log10(P/day), q, including all proposal-mixture components. Other latent
    distributions and observing/selection models must stay fixed when reweighting.
    """
    values: dict
    condition: np.ndarray
    raw_log_period: np.ndarray | None = None
    mass_ratio: np.ndarray | None = None
    proposal_log_density: np.ndarray | None = None

    response_weight: np.ndarray | None = None
    source_index: np.ndarray | None = None

    def __post_init__(self):
        self.condition = np.asarray(self.condition)
        if self.condition.ndim != 1 or not len(self.condition) or not np.issubdtype(self.condition.dtype, np.integer) or (self.condition < 0).any():
            raise ValueError('Non-negative integer condition labels are required')
        self.values = {k: np.asarray(v, dtype=float) for k, v in self.values.items()}
        if any(v.shape != self.condition.shape for v in self.values.values()):
            raise ValueError('Every outcome column must match the bank length')
        self.response_weight = (np.ones(len(self.condition)) if self.response_weight is None
                                else np.asarray(self.response_weight, dtype=float))
        self.source_index = (np.arange(len(self.condition)) if self.source_index is None
                             else np.asarray(self.source_index))
        if (self.response_weight.shape != self.condition.shape or
                not np.isfinite(self.response_weight).all() or (self.response_weight <= 0).any()):
            raise ValueError('Response weights must be finite and strictly positive')
        if (self.source_index.shape != self.condition.shape or
                not np.issubdtype(self.source_index.dtype, np.integer)):
            raise ValueError('Integer source indices must match bank rows')
        latent = [self.raw_log_period, self.mass_ratio, self.proposal_log_density]
        if any(x is not None for x in latent):
            if any(x is None for x in latent):
                raise ValueError('Binary banks require all three intrinsic/proposal arrays')
            for name in ['raw_log_period', 'mass_ratio', 'proposal_log_density']:
                value = np.asarray(getattr(self, name), dtype=float)
                if value.shape != self.condition.shape or not np.isfinite(value).all():
                    raise ValueError('Invalid intrinsic/proposal coordinates')
                setattr(self, name, value)


class JointSurveyOutcomeLikelihood:
    """One joint categorical outcome per star, with self-normalised bank weights.

    No probability floors or implicit pseudocounts: unsupported observed cells
    return -inf. Check bank convergence and support before interpreting a fit.
    Conditions must be selected independently of the RV/period/q outcomes.
    Dropping axes marginalises the same banks exactly, enabling fair comparisons.
    This class returns a likelihood only; parameter priors belong to the caller.
    Currently supports direct p(logP) proportional to logP**pi and p(q) to q**kappa.
    """
    def __init__(self, observed, condition, single_bank, binary_bank, axes,
                 log_period_bounds, mass_ratio_bounds):
        self.axes = tuple(axes)
        if not self.axes or len({a.name for a in self.axes}) != len(self.axes):
            raise ValueError('Distinct outcome axes are required')
        self.log_period_bounds = tuple(log_period_bounds)
        self.mass_ratio_bounds = tuple(mass_ratio_bounds)
        for bounds in [self.log_period_bounds, self.mass_ratio_bounds]:
            if len(bounds) != 2 or not 0 < bounds[0] < bounds[1]:
                raise ValueError('Direct-power-law bounds must be positive and ordered')
        self.observed = OutcomeBank(observed, np.asarray(condition))
        self.single_bank, self.binary_bank = single_bank, binary_bank
        if binary_bank.raw_log_period is None or single_bank.raw_log_period is not None:
            raise ValueError('Provide a single bank and a binary importance bank')
        for x,bounds in [(binary_bank.raw_log_period,self.log_period_bounds),
                         (binary_bank.mass_ratio,self.mass_ratio_bounds)]:
            if ((x < bounds[0]) | (x > bounds[1])).any():
                raise ValueError('Intrinsic bank values exceed the target support')
        self.conditions = np.unique(self.observed.condition)
        self.n_cells = int(np.prod([a.size for a in self.axes]))
        self.codes = [self._encode(b.values) for b in [self.observed,single_bank,binary_bank]]
        self.observed_counts = np.array([np.bincount(self.codes[0][self.observed.condition == c],minlength=self.n_cells) for c in self.conditions])
        self.single_probability = self._probabilities(single_bank,self.codes[1],np.zeros(len(single_bank.condition)))[0]

    def _encode(self, values):
        return np.ravel_multi_index(tuple(a.encode(values[a.name]) for a in self.axes),
                                    tuple(a.size for a in self.axes))

    def _probabilities(self, bank, code, log_weight):
        probabilities=[];ess=[]
        for c in self.conditions:
            mask = bank.condition == c
            if not mask.any():
                raise ValueError(f'No simulated support for condition {c}')
            lw = log_weight[mask]+np.log(bank.response_weight[mask]);norm = logsumexp(lw)
            if not np.isfinite(norm):
                raise ValueError('No finite importance weights')
            w = np.exp(lw-norm)
            probabilities.append(np.bincount(code[mask],weights=w,minlength=self.n_cells))
            # Analytically split outcomes do not create independent simulated stars.
            _, inverse = np.unique(bank.source_index[mask], return_inverse=True)
            source_weights = np.bincount(inverse, weights=w)
            ess.append(1/np.sum(source_weights*source_weights))
        return np.asarray(probabilities), np.asarray(ess)

    def probabilities(self, f_bin, pi, kappa=0.0):
        if not np.isfinite(f_bin) or not 0 <= f_bin <= 1:
            raise ValueError('f_bin must be in [0, 1]')
        b=self.binary_bank
        lw=(powerlaw_logpdf(b.raw_log_period,pi,*self.log_period_bounds)
            +powerlaw_logpdf(b.mass_ratio,kappa,*self.mass_ratio_bounds)-b.proposal_log_density)
        binary,ess=self._probabilities(b,self.codes[2],lw)
        return (1-f_bin)*self.single_probability+f_bin*binary,ess

    def __call__(self, f_bin, pi, kappa=0.0):
        probability,_=self.probabilities(f_bin,pi,kappa)
        used=self.observed_counts>0
        if np.any(probability[used] <= 0):
            return -np.inf
        return float(np.sum(self.observed_counts[used]*np.log(probability[used])))

    def diagnostics(self, f_bin, pi, kappa=0.0):
        probability,ess=self.probabilities(f_bin,pi,kappa)
        return dict(binary_effective_sample_size=ess.tolist(),
                    observed_unsupported_cells=int(np.sum((self.observed_counts>0)&(probability==0))),
                    probability_sums=probability.sum(axis=1).tolist(),
                    observed_systems=int(self.observed_counts.sum()))


def integrate_mass_ratio_response(bank, axis, maximum, sigma=0.05, eligible=None, minimum_q=0.3):
    """Split each binary into missing/q-bin outcomes with exact Gaussian weights.

    This marginalises the assumed selection and measurement noise, rather than
    drawing another Monte Carlo mark. Rows retain their intrinsic importance
    coordinates and source identity. q values represent bins, not precise data.
    minimum_q sets the zero-efficiency endpoint; the default preserves 0.3.
    """
    if bank.mass_ratio is None or not axis.missing or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('A binary bank, missing-aware axis and positive sigma are required')
    edges=np.asarray(axis.edges)
    if edges[0] != -np.inf or edges[-1] != np.inf:
        raise ValueError('Gaussian q-measurement bins must include both infinite tails')
    eligible=np.ones(len(bank.condition),bool) if eligible is None else np.asarray(eligible,bool)
    if eligible.shape != bank.condition.shape:
        raise ValueError('Eligibility must match bank rows')
    efficiency=mass_ratio_efficiency(bank.mass_ratio,maximum,minimum_q)*eligible
    cdf=ndtr((edges[None,:]-bank.mass_ratio[:,None])/sigma)
    probability=np.column_stack([1-efficiency,efficiency[:,None]*np.diff(cdf,axis=1)])
    if not np.allclose(probability.sum(axis=1),1):
        raise RuntimeError('Mass-ratio response is not normalised')
    source,code=np.nonzero(probability>0)
    representatives=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        representatives.append(0. if np.isneginf(lo) and np.isposinf(hi) else (lo+hi)/2 if np.isfinite(lo+hi) else
                               hi-1 if np.isneginf(lo) else lo+1)
    values={name:value[source] for name,value in bank.values.items()}
    values[axis.name]=np.asarray([np.nan]+representatives)[code]
    return OutcomeBank(values,bank.condition[source],bank.raw_log_period[source],
        bank.mass_ratio[source],bank.proposal_log_density[source],
        bank.response_weight[source]*probability[source,code],bank.source_index[source])
