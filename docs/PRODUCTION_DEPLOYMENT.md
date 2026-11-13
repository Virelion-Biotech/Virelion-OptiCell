# Production deployment

OptiCell 2.19 provides a deployable single-node research workbench. The
reference Docker deployment is **authenticated by default** and binds to
localhost unless an operator explicitly changes the bind address.

## Configure the initial account

Before starting the container, set a username and a long, random bootstrap
password. The password is used only to create the initial account if that
account does not already exist.

```bash
export OPTICELL_ADMIN_USER=admin
export OPTICELL_ADMIN_PASSWORD="$(openssl rand -hex 32)"
docker compose up --build -d
```

Save the generated password in an appropriate secrets manager. Passwords are
stored as PBKDF2-HMAC-SHA256 hashes with unique salts. The local-account
implementation is not an enterprise SSO/OIDC provider.

## Start and access

The default Compose bind address is `127.0.0.1`, so the service is only
reachable from the host. Open `http://127.0.0.1:8501`.

For access from another machine, keep `OPTICELL_REQUIRE_AUTH=1`, set
`OPTICELL_BIND_ADDRESS=0.0.0.0`, and place the service behind a properly
configured TLS reverse proxy or VPN. Do not expose the raw Streamlit port to
the public internet.

Authentication can be disabled only by an explicit override:

```bash
OPTICELL_REQUIRE_AUTH=0 docker compose up --build -d
```

Use that mode only for a trusted, single-user local test. Do not combine it
with a network-reachable bind address.

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
