# Deploying to Azure (AKS)

These manifests pass strict schema validation (`kubeconform`, Kubernetes 1.30 + KEDA CRDs) but have **not been applied to a real cluster yet**. Expect to adjust names, sizes and ingress details on the first deploy.

## Managed services to create first

| Component | Azure service | Settings that matter |
|---|---|---|
| Kubernetes | **AKS**, 3 zones; system pool + user pool (D4s/D8s); optional GPU pool for reranking | Install **KEDA** (AKS add-on) and **ingress-nginx** + **cert-manager** |
| Images | **Azure Container Registry** | Attach to AKS (`az aks update --attach-acr`) |
| Postgres | **Azure Database for PostgreSQL Flexible Server**, zone-redundant HA | Point-in-time restore on; PgBouncer on; `max_connections` ≥ 500 |
| Redis | **Azure Cache for Redis** (Standard/Premium) | `maxmemory-policy noeviction` (queued jobs must never be evicted); TLS port 6380 |
| Vectors | **Qdrant Cloud** on Azure (or the Qdrant Helm chart), 3 nodes | `QDRANT_SHARDS=6`, `QDRANT_REPLICAS=2`, set **before** the first bootstrap |
| Raw files | **Azure Blob Storage** (ZRS), container `rag-raw` | `BLOB_BACKEND=azure` |
| LLMs | **Azure OpenAI**: embedding, chat, fast (mini) and vision deployments | TPM quota or PTU sized for your peak (see the main README) |
| Secrets | **Key Vault** + Secrets Store CSI driver | Or `kubectl create secret` for a first test |
| Monitoring | **Azure Monitor managed Prometheus + Grafana** | Pods are annotated for scraping `/metrics` |

## Deploy

```bash
# 1. Image
az acr build -r <registry> -t node-rag:1.0.0 .
# set the registry and tag in deploy/k8s/kustomization.yaml (images:)

# 2. Secret (values: see secret.example.yaml)
kubectl create namespace node-rag
kubectl -n node-rag create secret generic rag-secrets --from-env-file=prod.env

# 3. Everything else (the migrate Job runs bootstrap: migrations + collection)
kubectl apply -k deploy/k8s
kubectl -n node-rag wait --for=condition=complete job/migrate --timeout=300s

# 4. First admin key
kubectl -n node-rag exec deploy/api -- python scripts/api_keys.py create --name ops --tenant default --role admin
```

Release: build a new tag, update `kustomization.yaml`, delete the old `migrate` Job, `kubectl apply -k deploy/k8s`.

## Scaling behaviour

| Workload | Scales on | Range | Why |
|---|---|---|---|
| `api` | CPU 60% (HPA) | 3–40 pods | about 40 new questions/s per pod (measured, fake models) |
| `worker-parse` | Redis `parse_cpu` length (KEDA), +1 per 4 waiting docs | 2–30 | Docling about 2.5 s/page per core |
| `worker-ocr` | Redis `ocr_vision` length (KEDA) | 1–8 | Limited by `VISION_RPM_LIMIT`, not CPU |
| `rerank` | CPU 70% (HPA) | 2–20 | Moves reranking CPU off the API pods |

The real ceiling at high chat load is **Azure OpenAI throughput**, not pods. `CHAT_RPM_LIMIT` and `CHAT_TPM_LIMIT` make pods wait for a slot instead of hammering the API; they answer 503 with `Retry-After` when no slot frees up within `CHAT_RATE_WAIT_S`.
