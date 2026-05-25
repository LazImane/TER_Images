# TER Images - Explorez l'IA créative : classez, colorisez et inventez des images uniques

**Université de Montpellier - Travail Encadré de Recherche 2025-2026**  
Massila LAKAF, Imane LAZIZI, Kanzy YOUSSEF  
Encadrant : M. Pascal Poncelet

---

## Rapport & Application

- Rapport final (PDF) : disponible dans le dossier [`rapports/`](./rapports/)
- Rapport rédigé sur : [PLMLaTeX](https://plmlatex.math.cnrs.fr/project/699741cebef35e808a2ad6e9)
- Application web : voir la branche [`application`](https://github.com/LazImane/TER_Images/tree/application) - elle contient son propre README pour l'installation

---

## Organisation du dépôt

```
TER_Images/
├── 1.Classification avec CNN/
├── 2. Colorisation avec les Autoencodeurs/
├── 3. Generation avec les VAE/
├── Decouverte_TP1/          # Premieres experimentations (hors TER)
└── rapports/                # Rapport final
```

---

## Guide de lecture des notebooks

Les notebooks sont disponibles en deux formats : `.ipynb` (exécutable) et `.pdf` (lecture seule).  
Chaque fichier PDF correspond à une étape de notre progression et est directement lié aux résultats 
présentés dans le rapport final. Voici l'ordre de lecture recommandé.

---

### 1. Classification d'images par CNN
> *Correspond à la Section 3 du rapport*

#### Classification binaire (2 classes)

| Fichier | Description |
|--------|-------------|
| [cnn_etape2.pdf](https://github.com/LazImane/TER_Images/blob/main/1.Classification%20avec%20CNN/cnn_etape2.pdf) | **Notebook principal** - Premier modèle CNN baseline sur le dataset éléphant/non-éléphant. Contient toutes les variations d'hyperparamètres (epochs, learning rate, batch size, architecture complexe) rapportées dans le rapport. |
| [HumanCatClassification.pdf](https://github.com/LazImane/TER_Images/blob/main/1.Classification%20avec%20CNN/HumanCatClassification.pdf) | Classification binaire chat/humain - exploration complémentaire avec évaluation croisée (StratifiedKFold). |

#### Classification multi-classes (3 classes) & Augmentation de données

| Fichier | Description |
|--------|-------------|
| [Sheep_Cat_Elephant.pdf](https://github.com/LazImane/TER_Images/blob/main/1.Classification%20avec%20CNN/Sheep_Cat_Elephant.pdf) | **Notebook principal** - Classification mouton/chat/éléphant. Contient le modèle baseline 3 classes, les améliorations architecturales, l'augmentation de données et une première exploration du transfer learning. C'est ce notebook qui a motivé tous les résultats des Sections 3.3 et 3.4 du rapport. |
| [TER_TransferLearning.pdf](https://github.com/LazImane/TER_Images/blob/main/1.Classification%20avec%20CNN/TER_TransferLearning.pdf) | **Notebook principal Transfer Learning** - Feature extraction et fine-tuning avec MobileNetV2. Contient tous les résultats rapportés dans la Section 3.5 du rapport. |

---

### 2. Restauration et colorisation d'images
> *Correspond à la Section 4 du rapport*

A lire dans cet ordre :

| Ordre | Fichier | Description |
|-------|--------|-------------|
| 1 | [correction_Colorisation_decouverte.pdf](https://github.com/LazImane/TER_Images/blob/main/2.%20Colorisation%20avec%20les%20Autoencodeurs/correction_Colorisation_decouverte.pdf) | Découverte des autoencodeurs - implémentation de l'architecture simple (encodeur/décodeur classique). Correspond à la Section 4.5 du rapport. |
| 2 | [Test_collorisation_correction.pdf](https://github.com/LazImane/TER_Images/blob/main/2.%20Colorisation%20avec%20les%20Autoencodeurs/Test_collorisation_correction.pdf) | Introduction du U-Net et premiers essais avec les skip connections. Correspond aux Sections 4.8 et 4.9.1 du rapport. |
| 3 | [Colorisation_final_model.pdf](https://github.com/LazImane/TER_Images/blob/main/2.%20Colorisation%20avec%20les%20Autoencodeurs/Colorisation_final_model.pdf) | **Modèle final** - U-Net avec warmup progressif de la fonction de perte (LAB + SSIM). Contient les résultats finaux (PSNR ≈ 27 dB, SSIM ≈ 0.95) rapportés dans les Sections 4.9.2, 4.9.3 et 4.10 du rapport. |

Ces trois notebooks, lus dans l'ordre, reflètent exactement la progression décrite dans la Section 4 du rapport.

---

### 3. Génération d'images par VAE
> *Correspond à la Section 5 du rapport*

| Fichier | Description |
|--------|-------------|
| [Generation images.pdf](https://github.com/LazImane/TER_Images/blob/main/3.Generation%20avec%20les%20VAE/Generation%20images.pdf) | **Notebook principal VAE** - Implémentation complète du VAE, visualisation de l'espace latent (t-SNE), et génération d'images de chats. Contient tous les résultats rapportés dans la Sections 5 du rapport. |

---

### Dossier `Decouverte_TP1`

Ce dossier contient nos toutes premières expérimentations sur les réseaux de neurones, 
réalisées en début de TER pour prendre en main les outils. Ces fichiers ne sont pas 
directement liés aux résultats du rapport final, ils témoignent simplement de notre 
progression dans l'apprentissage.

---

## Liens utiles

| Ressource | Lien |
|-----------|------|
| Rapport final | [`rapports/`](./rapports/) |
| Rapport PLMLaTeX | [Accéder](https://plmlatex.math.cnrs.fr/project/699741cebef35e808a2ad6e9) |
| Application web | [Branche `application`](https://github.com/LazImane/TER_Images/tree/application) |
