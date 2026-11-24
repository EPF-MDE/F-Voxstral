# Voxtral Live : voix → texte traduit en temps réel

Déploiement de **Voxtral Mini 3B** (`mistralai/Voxtral-Mini-3B-2507`) servi par **vLLM** sur une VM GPU partagée, avec une application web qui transcrit et traduit le micro en direct.

## 1. Architecture

```
Navigateur (micro)                         VM GPU partagée (Docker Compose)
┌──────────────────┐  WebSocket   ┌───────────────────────────┐   HTTP (API OpenAI)   ┌────────────────────────┐
│ AudioWorklet     │  PCM 16 kHz  │ app (FastAPI, :8080)       │ ───────────────────► │ backend (vLLM, :8000)  │
│ → Int16, 100 ms  │ ───────────► │ 1. VAD (énergie) → segments│  /v1/audio/          │ Voxtral-Mini-3B-2507   │
│                  │ ◄─────────── │ 2. transcription           │   transcriptions     │ GPU (GPU_ID,           │
│ affiche source + │   JSON       │ 3. traduction (même modèle)│  /v1/chat/completions│  GPU_MEM_UTIL)         │
│ traduction       │              └───────────────────────────┘                       └────────────────────────┘
└──────────────────┘          ▲ exposé uniquement sur 127.0.0.1 → accès par tunnel SSH
```

Le pipeline traite chaque segment de parole ainsi :
1. Le navigateur envoie l'audio du micro par blocs de 100 ms (PCM 16 bits, 16 kHz, mono).
2. Le serveur découpe le flux en énoncés : il attend 700 ms de silence, ou coupe à 12 s maximum.
3. Il appelle `POST /v1/audio/transcriptions` (Voxtral, mode transcription) et renvoie aussitôt le texte source.
4. Il appelle `POST /v1/chat/completions` avec une consigne de traduction (Voxtral, mode texte). Les 2 phrases précédentes sont passées en contexte.
5. Le navigateur affiche la traduction et la latence mesurée.

**Pourquoi « transcription puis traduction » plutôt qu'un seul appel audio→traduction ?** Le texte source s'affiche immédiatement, ce qui donne une meilleure impression de temps réel. La transcription elle-même est plus fiable. Et l'appel de traduction est purement textuel, donc peu coûteux.

### Contraintes « GPU partagé » prises en compte

| Problème | Solution |
|---|---|
| Un seul GPU pour plusieurs utilisateurs/services | `GPU_MEM_UTIL` (par défaut 0,45) limite la VRAM réservée par vLLM. `GPU_ID` choisit la carte. |
| Vérifier qu'il reste de la place avant de lancer | `deploy.sh gpu` lit `nvidia-smi` et refuse de démarrer si la VRAM libre ne suffit pas. |
| Conflits de ports avec les autres utilisateurs | `APP_PORT` et `VLLM_PORT` se règlent dans `.env`. |
| Sécurité sur une machine partagée | Ports liés à `127.0.0.1`. vLLM est protégé par `--api-key` (clé générée automatiquement). L'accès passe par un tunnel SSH. |
| Saturation du GPU | `MAX_CONCURRENT_REQUESTS` (sémaphore global côté app) et `MAX_NUM_SEQS` (côté vLLM). Si la file d'une session déborde, le segment le plus ancien est abandonné pour rester en temps réel. |
| Téléchargement répété des poids (~9 Go) | Volume Docker `hf-cache`. |

## 2. Arborescence

```
.
├── docker-compose.yml     # service app (commun)
├── compose.gpu.yml        # backend = vLLM + Voxtral (VM GPU)
├── compose.mock.yml       # backend = mock API OpenAI (test local sans GPU)
├── .env.example           # toute la configuration
├── app/                   # FastAPI + front (HTML/JS) + tools/smoke_test.py
├── vllm/Dockerfile        # vllm/vllm-openai + extras audio (mistral_common[audio], librosa)
├── mock/                  # faux vLLM : mêmes endpoints, réponses factices
└── scripts/
    ├── setup_vm.sh        # installe Docker et NVIDIA Container Toolkit (une seule fois)
    └── deploy.sh          # build, démarrage, attente du healthcheck, test de bout en bout
```

## 3. Test local avec Docker (sans GPU)

Prérequis : Docker Desktop (Windows/macOS) ou Docker Engine (Linux) avec Compose v2, et un shell bash (Git Bash sous Windows).

```bash
./scripts/deploy.sh mock
```

Le script effectue les étapes suivantes :
1. Il crée `.env` à partir de `.env.example` s'il n'existe pas.
2. Il construit et démarre `app` et `backend` (le mock).
3. Il attend que `GET /healthz` réponde OK (l'app interroge `/v1/models` sur le backend).
4. Il lance `tools/smoke_test.py` dans le conteneur. Ce script envoie un signal audio sur le WebSocket en temps réel et vérifie qu'on reçoit des traductions (code retour ≠ 0 en cas d'échec).

Ouvrir ensuite `http://localhost:8080` (ou le port défini par `APP_PORT`), cliquer sur **Démarrer** et parler. Avec le mock, les textes sont factices, mais tout le reste est réel : capture micro, streaming, découpage, appels API, affichage.

Arrêt : `./scripts/deploy.sh down mock`

**Résultat du test local** (Windows 11, Docker 29.8, Compose v5.5) :
```
OK: 3 segment(s) translated, latency avg 617 ms, max 621 ms
```

> Le mock permet de valider l'image, le réseau Compose, le healthcheck, le protocole WebSocket et le découpage. Il ne valide pas le modèle : celui-ci est testé sur la VM GPU (section 4).

## 4. Déploiement sur la VM GPU partagée

Prérequis : Ubuntu 22.04/24.04, pilote NVIDIA installé (`nvidia-smi` fonctionne), droits sudo pour l'installation initiale, environ 20 Go de disque libre. VRAM : environ 9,5 Go pour les poids en bf16, plus le cache KV.

```bash
# depuis ton PC
scp -r Cloud_Project user@VM:~/voxtral-live      # ou git clone
ssh user@VM
cd ~/voxtral-live

# une seule fois par VM (si Docker / nvidia-container-toolkit ne sont pas déjà installés)
./scripts/setup_vm.sh && newgrp docker

# vérifier l'occupation du GPU et ajuster .env (GPU_ID, GPU_MEM_UTIL, ports)
nvidia-smi
cp .env.example .env && nano .env   # facultatif : deploy.sh crée .env et génère une clé API

# déploiement et test automatique
./scripts/deploy.sh gpu sample.wav  # sample.wav : un fichier de parole en WAV 16 bits (facultatif)
```

Le premier démarrage télécharge le modèle (10 minutes ou plus). Le healthcheck de vLLM accorde 15 minutes (`start_period: 900s`). Pour suivre la progression : `docker compose -f docker-compose.yml -f compose.gpu.yml logs -f backend`.

Pour préparer un fichier de test : `ffmpeg -i parole.mp3 -ac 1 -ar 16000 sample.wav`.

### Accès depuis ton navigateur (tunnel SSH)

Le micro du navigateur n'est autorisé que sur `https://` ou `localhost`. Le tunnel SSH répond à cette contrainte tout en gardant l'app privée :

```bash
ssh -N -L 8080:localhost:8080 user@VM
# puis ouvrir http://localhost:8080
```

### Réglage de `GPU_MEM_UTIL` selon la carte

| GPU | VRAM | `GPU_MEM_UTIL` conseillé | Remarque |
|---|---|---|---|
| A100 / H100 80 Go | 80 Go | 0,15 à 0,2 | |
| A100 40 Go / A6000 48 Go | 40 à 48 Go | 0,3 | |
| L4 / A10 / RTX 4090 24 Go | 24 Go | 0,45 à 0,6 | |
| T4 16 Go | 16 Go | 0,85 | Pas de bf16 : `VLLM_EXTRA_ARGS=--dtype half`. À peine assez de place pour partager. |

Si vLLM échoue avec « No available memory for the cache blocks », augmenter `GPU_MEM_UTIL` ou réduire `MAX_MODEL_LEN` (4096 suffit pour des segments de 12 s).

## 5. Configuration (`.env`)

| Variable | Défaut | Rôle |
|---|---|---|
| `MODEL_NAME` | `mistralai/Voxtral-Mini-3B-2507` | Modèle servi |
| `VLLM_VERSION` | `v0.10.1.1` | Tag de l'image `vllm/vllm-openai` (support de Voxtral depuis la 0.10) |
| `VLLM_API_KEY` | générée | Clé partagée entre l'app et vLLM |
| `GPU_ID` | `0` | Index du GPU (`nvidia-smi`) |
| `GPU_MEM_UTIL` | `0.45` | Fraction de VRAM réservée par vLLM |
| `MAX_MODEL_LEN` / `MAX_NUM_SEQS` | `8192` / `8` | Contexte maximum et nombre de séquences simultanées |
| `APP_PORT` / `APP_BIND` / `VLLM_PORT` | `8080` / `127.0.0.1` / `8000` | Ports exposés sur l'hôte |
| `MAX_CONCURRENT_REQUESTS` | `4` | Nombre maximum de requêtes simultanées vers le GPU (toutes sessions confondues) |
| `VAD_RMS_THRESHOLD` | `400` | Seuil d'énergie de parole. L'augmenter si le bruit déclenche des segments. |
| `VAD_SILENCE_MS` | `700` | Durée de silence qui termine un segment. Plus bas = plus réactif, mais phrases coupées. |
| `VAD_MAX_SEGMENT_MS` | `12000` | Coupe forcée des longs monologues |

## 6. Vérifications et dépannage

| Symptôme | Cause / solution |
|---|---|
| `port is already allocated` | Port déjà pris (cas fréquent sur une VM partagée) : changer `APP_PORT` dans `.env`. |
| `/healthz` répond `backend_unavailable` | vLLM charge encore le modèle : consulter `logs backend`. |
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit absent : lancer `scripts/setup_vm.sh`. |
| CUDA out of memory au démarrage | Le GPU est occupé par d'autres utilisateurs : baisser `GPU_MEM_UTIL`, changer `GPU_ID`, ou attendre. |
| Aucun segment ne s'affiche | Le micro est trop faible : regarder la jauge « Niveau micro » et baisser `VAD_RMS_THRESHOLD`. |
| Le bouton micro ne fait rien | Page ouverte via `http://IP_VM` au lieu de `localhost` : passer par le tunnel SSH. |
| Avertissement « GPU busy, segment dropped » | GPU saturé : réduire le nombre d'utilisateurs ou augmenter `MAX_NUM_SEQS`/`GPU_MEM_UTIL`. |

Commandes utiles :
```bash
docker compose -f docker-compose.yml -f compose.gpu.yml ps
docker compose -f docker-compose.yml -f compose.gpu.yml logs -f
curl -s localhost:8080/healthz
curl -s localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 2
```

## 7. Références
- Tutoriel DataCamp « Voxtral de Mistral » : https://www.datacamp.com/fr/tutorial/voxtral-mistral (commande `vllm serve` avec `--tokenizer_mode/--config_format/--load_format mistral`, client OpenAI)
- Fiche modèle : https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
- Documentation vLLM (API OpenAI, transcription) : https://docs.vllm.ai
