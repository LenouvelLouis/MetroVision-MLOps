# Roadmap sécurité — MetroVision-MLOps

Feuille de route en petites itérations indépendantes pour amener le projet d'un état "dev local" à une mise en production durcie.

Chaque étape :
- Se fait en moins de 2 h
- Est testable isolément
- Peut être livrée dans un commit / PR distinct
- Ne casse pas les étapes précédentes

L'ordre est pensé pour que chaque étape débloque la suivante.

---

## Phase 0 — Hygiène immédiate (30 min chacune)

### 0.1 — Bind localhost uniquement
- Modifier `docker-compose.yml` : `"127.0.0.1:PORT:PORT"` sur les 4 services
- **Test** : `nmap localhost` depuis une autre machine du LAN → rien
- **Valeur** : ferme l'exposition LAN accidentelle

### 0.2 — Sortir le mot de passe Grafana du repo
- Créer `.env.example` (committé) + `.env` (gitignored)
- Référencer `${GF_ADMIN_PASSWORD}` dans `docker-compose.yml:54`
- Ajouter `.env` au `.gitignore`
- **Test** : `git ls-files | grep -v example | xargs grep -l metrovision` → vide

### 0.3 — Ajouter gitleaks en pre-commit
- Étendre `.pre-commit-config.yaml` avec le hook gitleaks
- **Test** : commit d'un faux secret → bloqué

### 0.4 — Pin des images Docker par digest
- Remplacer `prom/prometheus:v3.4.1` par `prom/prometheus@sha256:...` (idem MLflow, Grafana)
- **Test** : `docker compose pull` reproductible

---

## Phase 1 — Durcir l'API (1-2 h chacune)

### 1.1 — Limiter la taille des uploads
- Middleware FastAPI qui rejette si `Content-Length > 10 MB`
- Ajouter test pytest dans `tests/`
- **Valeur** : tue le DoS trivial

### 1.2 — Valider le type de fichier réellement
- Vérifier les magic bytes (lib `python-magic` ou `imghdr`) dans `predict.py`
- Allowlist : `image/jpeg`, `image/png`
- **Test** : POST d'un `.txt` renommé `.jpg` → 415

### 1.3 — Gérer les exceptions proprement
- Catch `PIL.UnidentifiedImageError`, `cv2.error` → renvoyer 400 propre
- Incrémenter le compteur `status="error"` dans tous les chemins d'erreur (`api/routes/predict.py:39`)
- **Test** : les erreurs apparaissent dans `/metrics`

### 1.4 — Timeout sur le traitement
- Wrapper `asyncio.wait_for(processOneMetroImage, timeout=30)`
- **Test** : image piégée → 504 propre, pas de worker bloqué

### 1.5 — Rate limiting
- Ajouter `slowapi`, limite 10 req/min par IP sur `/predict`
- **Test** : 11ᵉ requête → 429

### 1.6 — Security headers
- Middleware qui ajoute `X-Content-Type-Options`, `X-Frame-Options`, `Strict-Transport-Security`
- **Test** : `curl -I` montre les headers

### 1.7 — Logging structuré sans PII
- Passer en JSON logs (`structlog` ou `python-json-logger`)
- Ne plus logger le nom des fichiers utilisateur, uniquement un hash
- **Valeur** : logs prêts pour Loki / ELK

---

## Phase 2 — Auth & accès (2 h chacune)

### 2.1 — API keys sur l'API
- Header `X-API-Key`, vérifié via dépendance FastAPI
- Stocker les hashes dans un fichier monté (pas de DB encore)
- **Test** : sans clé → 401 ; avec clé valide → 200

### 2.2 — Auth basic sur MLflow
- Activer `mlflow server --app-name basic-auth`
- Adapter `register_baseline.py` pour lire `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`
- **Test** : appel sans creds → 401

### 2.3 — Reverse proxy Caddy ou Traefik
- Ajouter un service `proxy` qui termine TLS + route vers api / grafana / mlflow
- Cert auto-signé pour le dev, Let's Encrypt prévu pour la prod
- Retirer les `ports:` des services internes (sauf le proxy)
- **Test** : `curl -k https://localhost/api/health` OK, `localhost:5000` ne répond plus

### 2.4 — OIDC sur Grafana
- Configurer Grafana avec un IdP de test (Keycloak en local ou compte dev d'Auth0)
- Désactiver l'utilisateur `admin` après bootstrap
- **Test** : login redirige vers IdP

---

## Phase 3 — Conteneurs sécurisés (1 h chacune)

### 3.1 — Resource limits sur tous les services
- Ajouter `deploy.resources.limits` (CPU / mem) dans `docker-compose.yml`
- **Valeur** : un service ne peut plus tuer l'hôte

### 3.2 — Read-only root filesystem pour l'API
- `read_only: true` + `tmpfs:` pour `/tmp` si nécessaire
- **Test** : la stack tourne toujours

### 3.3 — Drop capabilities
- `cap_drop: [ALL]`, `security_opt: [no-new-privileges:true]`
- **Test** : stack OK, conteneur ne peut plus `mount`, `ptrace`, etc.

### 3.4 — Image distroless pour l'API
- Stage runtime basé sur `gcr.io/distroless/python3` au lieu de `python:3.11-slim`
- **Test** : image plus petite, scan trivy avec moins de CVEs

### 3.5 — MLflow en non-root
- Build d'une image custom MLflow qui tourne en USER 10001
- Ou utiliser une image communautaire rootless si dispo

---

## Phase 4 — Supply chain & CI (1-2 h chacune)

### 4.1 — Trivy sur le repo en CI
- Ajouter job GitHub Actions : `trivy fs --severity CRITICAL,HIGH`
- Fail si CVE critique sur les deps

### 4.2 — Trivy sur l'image construite
- Job qui build l'image puis scan
- **Valeur** : repère les CVEs OS + Python

### 4.3 — pip-audit dans le venv
- Step dédié dans la CI
- Snapshot des CVEs connues acceptées dans `.pip-audit-ignore`

### 4.4 — Lock file avec hashes
- Migrer vers `uv` + `uv.lock`
- `pip install --require-hashes`
- **Test** : build reproductible bit-à-bit

### 4.5 — Pin des GitHub Actions par SHA
- Remplacer `@v4` par `@<sha>` dans tous les workflows
- Renovate / Dependabot pour les mises à jour

### 4.6 — SBOM à chaque build
- `docker buildx build --sbom=true --provenance=true`
- Upload SBOM comme artefact

### 4.7 — Cosign : signer les images
- Sign en CI avec keyless OIDC (GitHub)
- Vérification au déploiement

---

## Phase 5 — ML security (1-2 h chacune)

### 5.1 — Checksum des modèles au chargement
- Stocker les SHA256 attendus dans un manifest committé
- Vérifier dans `api/model_manager.py:33` avant `load_models()`
- **Test** : remplacer un `.joblib` → l'API refuse de démarrer

### 5.2 — Documenter le risque joblib.load
- ADR dans `docs/` expliquant pourquoi `joblib.load` est OK ici (modèles signés / checksummés)
- Refuser de loader un modèle dont l'origine n'est pas vérifiée

### 5.3 — Tests adversariaux basiques
- Ajouter `tests/test_adversarial.py` : images corrompues, géantes, formats exotiques
- **Valeur** : non-régression sécurité

### 5.4 — Evidently drift comme signal sécurité
- Configurer une alerte si drift > seuil → peut indiquer poisoning
- Reporter dans Prometheus → Alertmanager

### 5.5 — Model registry workflow
- Documenter le passage `None → Staging → Production` dans MLflow
- Aucun modèle en `Production` sans review humaine

---

## Phase 6 — Infra production (2-4 h chacune)

### 6.1 — Postgres pour MLflow
- Remplacer SQLite par Postgres dans `docker-compose.yml:24`
- Volume persistant + backup
- **Test** : restart MLflow → données toujours là

### 6.2 — S3 / MinIO pour les artefacts MLflow
- Ajouter MinIO local, configurer MLflow avec `--serve-artifacts` + `--artifacts-destination s3://...`
- Fixe le bug d'artefacts inaccessibles depuis le client

### 6.3 — Manifests K8s durcis
- Reprendre `k8s/` existant, ajouter `securityContext` partout
- NetworkPolicies entre services
- PodDisruptionBudgets

### 6.4 — Secrets via External Secrets Operator
- Vault local (dev mode) → ESO → K8s Secrets
- Plus aucun secret dans les manifests

### 6.5 — Ingress avec cert-manager
- TLS automatique via Let's Encrypt
- Force HTTPS (redirect 301)

### 6.6 — Backups automatisés
- CronJob K8s qui dump Postgres MLflow + snapshot du PVC Grafana
- Restore testé et documenté

---

## Phase 7 — Observabilité sécurité (1-2 h chacune)

### 7.1 — Logs centralisés
- Loki + Promtail (s'intègre à Grafana déjà présent)
- Toute la stack envoie ses logs

### 7.2 — Alertes sécurité dans Prometheus
- Compléter `monitoring/alerts.yml` : pic de 4xx, latence anormale, taux d'erreur
- Routes Alertmanager → Slack / email

### 7.3 — Audit log applicatif
- Middleware FastAPI qui logge : qui (API key hash), quoi (endpoint), quand, résultat
- Loggé en JSON structuré, ingéré par Loki

### 7.4 — Falco sur les nœuds
- DaemonSet K8s qui surveille les syscalls suspects
- Alertes sur shell dans conteneur, écriture dans `/etc`, etc.

### 7.5 — Tracing OpenTelemetry
- Instrumenter l'API avec `opentelemetry-instrumentation-fastapi`
- Export vers Tempo ou Jaeger
- Trace une requête de bout en bout

---

## Phase 8 — Gouvernance (1 h chacune)

### 8.1 — `SECURITY.md`
- Politique de divulgation, contact, SLA de réponse

### 8.2 — Threat model documenté
- STRIDE sur les 4 composants principaux dans `docs/threat-model.md`

### 8.3 — ADRs sécurité
- Une ADR par décision structurante (choix de l'IdP, du WAF, etc.)

### 8.4 — Pen test interne
- ZAP en CI sur l'API (scan passif au minimum)
- Rapport archivé

### 8.5 — Runbook incident
- Que faire si un secret leak, si un modèle est compromis, si l'API est DDoS

---

## Suggestion d'ordre d'exécution

| Sprint | Durée | Contenu | Résultat |
|--------|-------|---------|----------|
| 1 | 1 semaine | 0.1 → 0.4 + 1.1 → 1.3 | Quick wins, gain immédiat |
| 2 | 1 semaine | 1.4 → 1.7 | API solide |
| 3 | 1 semaine | 2.1 → 2.3 | Plus rien d'anonyme |
| 4 | 1 semaine | 3.x + 4.1 → 4.3 | Conteneurs + CI sécurisés |
| 5 | 1 semaine | 5.x | Spécificités ML |
| 6+ | À la demande | 6.x, 7.x, 8.x | Selon priorités prod |

À la fin de chaque sprint, le projet est dans un état meilleur qu'avant, déployable, et la décision de continuer ou de s'arrêter peut être prise librement.
