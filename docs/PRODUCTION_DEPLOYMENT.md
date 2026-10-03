# Production deployment

OptiCell 2.19 adds a deployable single-node research workbench. The supported
reference deployment is the supplied Docker image with a persistent workspace
volume.

## Start

```bash
docker compose up --build -d
```

Open port 8501. The container health check uses Streamlit's health endpoint.

## Authentication

Authentication is optional by default for local research use. To require local
accounts, set:

```bash
export OPTICELL_REQUIRE_AUTH=1
export OPTICELL_ADMIN_USER=admin
export OPTICELL_ADMIN_PASSWORD='use-a-long-random-password'
docker compose up --build -d
```

The bootstrap password is used only to create the account if it does not
already exist. Passwords are stored as PBKDF2-HMAC-SHA256 hashes with unique
salts. This is a local-account implementation, not an enterprise SSO/OIDC
provider.

## Persistence and backups

The Docker volume `opticell_workspace` holds the SQLite database, projects,
model checkpoints, corrected masks and run artifacts. Create a consistent DB
backup with:

```bash
python scripts/backup_workbench.py
```

Back up the entire workspace volume as well if project input/artifact retention
matters to you.

## Quotas and audit

Users can have per-account run quotas. Project creation, run creation, logins
and human review decisions are appended to the audit log. Quotas are enforced
when a run is persisted.

## Monitoring

The Operations page reports runtime/GPU/Cellpose capability, workspace
writability and disk space. Docker additionally exposes a health check. For a
real multi-user deployment, connect container logs and health status to your
normal monitoring stack.

## GPU / Cellpose image

Build with Cellpose dependencies:

```bash
docker build --build-arg OPTICELL_EXTRAS=ui,cellpose -t opticell:cellpose .
```

CUDA-enabled deployment still requires a compatible host NVIDIA stack,
container runtime and PyTorch build. The default image deliberately does not
pretend GPU support exists when the host does not provide it.

## Scaling boundary

SQLite is deliberately a single-node store. Do not mount one SQLite database
read/write across multiple application replicas. A multi-node SaaS deployment
would need an external transactional database, object storage, centralized
identity, secrets management and external observability.
