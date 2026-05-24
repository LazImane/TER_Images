# TER Master 1 IA Application de Traitement d'Images

Cette application Flask regroupe trois fonctionnalités majeures d'Intelligence Artificielle développées dans le cadre du TER (Travail d'Étude et de Recherche) en Master 1 IA.

## Fonctionnalités

### 1. Classification d'Images
- **Description** : Identification automatique d'animaux parmi 3 classes.
- **Classes supportées** : Chat, Mouton, Éléphant.
- **Modèle** : Basé sur MobileNetV2 fine-tuné (`modele_finetuning.keras`).
- **Entrée** : Image de chat, mouton, éléphant ou autre.

### 2. Colorisation (Auto-encodeur)
- **Description** : Transformation d'images en niveaux de gris en images colorisées.
- **Modèle** : Architecture U-Net (`unet_best_blackwhite.keras`).
- **Processus** : L'application convertit l'image en niveaux de gris, puis le modèle prédit les couleurs manquantes.
- **Entrée** : Image en noir et blanc.

### 3. Génération d'Images (VAE)
- **Description** : Génération de nouvelles images de chats à partir d'un espace latent.
- **Modèle** : Variational AutoEncoder (`catGeneratingVAE2.keras`).
- **Processus** : Échantillonnage d'un vecteur latent aléatoire (taille 16) passé au décodeur du VAE.

## Installation

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/LazImane/TER_Images.git
   cd TER_Images
   git checkout application
   ```

2. **Créer un environnement virtuel** :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

## Lancement

Pour démarrer l'application :
```bash
python app.py
```
L'interface sera accessible à l'adresse : `http://127.0.0.1:5000`

## Structure du Projet

- `app.py` : Serveur Flask et logique d'inférence des modèles.
- `models/` : Contient les fichiers `.keras` des modèles entraînés.
- `templates/` : Interface utilisateur (HTML/JS/Bootstrap).
- `static/` : Fichiers statiques et dossier d'uploads.
- `requirements.txt` : Liste des dépendances Python.
