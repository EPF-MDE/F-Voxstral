# Voxtral Live : voix → texte traduit en temps réel

Voxtral Mini 3B servi par vLLM sur une VM GPU partagée, avec une app web (FastAPI + WebSocket) qui transcrit et traduit le micro en direct.

```bash
./scripts/deploy.sh mock              # test local sans GPU (backend simulé)
./scripts/setup_vm.sh                 # VM : installation de Docker et du NVIDIA toolkit (une seule fois)
./scripts/deploy.sh gpu sample.wav    # VM : déploiement réel et test de bout en bout
```

Documentation complète : [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
