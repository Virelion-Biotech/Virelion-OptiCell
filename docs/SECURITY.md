# Security Policy

## Scope

OptiCell is research software for microscopy quality control and quantitative image analysis. It is not a clinical device, diagnostic system, or security-sensitive execution environment.

## Supported versions

Security and dependency issues are addressed on the current `main` release line. Older releases may not receive fixes.

## Reporting a vulnerability

Please do not disclose an undisclosed security vulnerability in a public issue. Contact the repository maintainers privately through the security contact mechanisms provided by GitHub for this repository.

When reporting, include:

- affected OptiCell version or commit;
- affected module or workflow;
- minimal reproducible example, when safe to provide;
- expected and observed behavior;
- dependency and operating-system information.

Do not include patient data, proprietary microscopy data, credentials, access tokens, or other sensitive information in reports.

## Research-data safety

OptiCell can process user-supplied image files and metadata. Users are responsible for validating provenance, access controls, and de-identification before processing sensitive datasets. Never commit confidential experimental data, credentials, or generated datasets containing sensitive information to the repository.
