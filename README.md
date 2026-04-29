# matilija

This repository uses DHSVM version 3.2 from the PNNL DHSVM upstream release.

## Building the Docker image

The Dockerfile automatically downloads and compiles DHSVM v3.2 as part of the build process:

```bash
docker build -t dhsvm-matilija ./src
```

This produces a multi-stage image optimized for size and runtime efficiency. The builder stage compiles DHSVM with all required dependencies; the final image includes only runtime libraries.

## Citation

Use the following citation when referring to DHSVM in publications:

Wigmosta, M. S., Vail, L. W., & Lettenmaier, D. P. (1994). A distributed hydrology–vegetation model for complex terrain. *Water Resources Research, 30*(6), 1665–1679.
