# Development-only shift-and-add adaptation

This directory contains local adaptations that wrap spectral-disentangling
code as a Python class for use during MINATO development. It is not a MINATO
module or product, and it is excluded from MINATO packages and releases.

The original implementation is the
[Disentangling_Shift_And_Add repository](https://github.com/TomerShenar/Disentangling_Shift_And_Add),
written by Tomer Shenar with contributions from Matthias Fabry and Julia
Bodensteiner. Credit for the method and code belongs to that project.

Follow the upstream repository for current usage instructions, permissions,
and citation guidance. Its README asks users to cite:

- Gonzalez & Levato (2006), A&A, 448, 283, for the shift-and-add algorithm;
- Shenar et al. (2020), A&A, 639, A6;
- Shenar et al. (2022), A&A, 665, A148.

The upstream repository does not currently publish a software licence file.
Keeping a copy on a public development branch may itself require permission
from the upstream owner. Record that permission or a suitable upstream licence
before treating this copy as redistributable. It must not be distributed under
MINATO's MIT licence or included in a MINATO source archive, wheel, or release
branch without a separately reviewed licence decision.

Within a development checkout, the adapted class is available as:

```python
from minato.contrib.spdis import SpecDisent
```

Do not document or import it as `minato.spdis`; the `minato.contrib` namespace
marks it as external, development-only code.
