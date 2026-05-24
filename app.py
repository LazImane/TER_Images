import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

class TERClassifier:
    def __init__(self):
        # Chargement de votre modèle fine-tuné
        model_path = os.path.join('models', 'modele_finetuning.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le modèle {model_path} est introuvable.")
        
        self.model = load_model(model_path)
        # Mapping des classes selon vos spécifications
        # 0: Mouton, 1: Chat, 2: Éléphant
        self.class_names = ['mouton', 'chat', 'éléphant']

    def predict(self, img_path_or_stream):
        # Taille d'entrée spécifiée : 128x128
        img = Image.open(img_path_or_stream).convert('RGB')
        img = img.resize((128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        # Prétraitement MobileNetV2 (important)
        x = preprocess_input(x)
        
        preds = self.model.predict(x)
        
        # Extraction des scores (vecteur de 3 probabilités)
        results = {}
        for i, label in enumerate(self.class_names):
            results[label] = float(preds[0][i])
        
        best_index = np.argmax(preds[0])
        best_class = self.class_names[best_index]
        best_score = float(preds[0][best_index])
        
        return best_class, best_score, results

class ImageColorizer:
    def __init__(self):
        model_path = os.path.join('models', 'unet_best_blackwhite.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le modèle {model_path} est introuvable.")
        # On charge avec compile=False car la fonction de perte custom 'warmup_mix_loss' 
        # n'est pas nécessaire pour l'inférence.
        self.model = load_model(model_path, compile=False)
        self.input_size = (128, 128)

    def process(self, img_stream):
        # 1. Charger et préparer l'image
        # Le modèle attend 3 canaux (RGB) même pour du noir et blanc
        # On convertit d'abord en 'L' (gris) pour garantir que l'entrée est sans couleur,
        # puis en 'RGB' pour avoir les 3 canaux attendus par le modèle (R=G=B).
        img_orig = Image.open(img_stream)
        img_gray = img_orig.convert('L')
        img_rgb = img_gray.convert('RGB')
        
        original_size = img_orig.size
        img_resized = img_rgb.resize(self.input_size)
        
        # 2. Prétraitement (Normalisation 0-1)
        x = image.img_to_array(img_resized)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        
        # 3. Prédiction
        pred = self.model.predict(x)[0]
        
        # 4. Post-traitement
        # Si le modèle sort des valeurs entre 0-1, on remet en 0-255
        pred = (pred * 255).astype(np.uint8)
        
        # Créer l'image colorisée
        # On redimensionne à la taille d'origine pour l'affichage
        res_img = Image.fromarray(pred)
        res_img = res_img.resize(original_size)
        
        # 5. Conversion en base64 pour le frontend
        buffered = io.BytesIO()
        res_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # On renvoie aussi l'image originale en NB pour comparaison
        orig_buffered = io.BytesIO()
        img_gray.save(orig_buffered, format="PNG")
        orig_str = base64.b64encode(orig_buffered.getvalue()).decode()
        
        return img_str, orig_str

class ImageGenerator:
    def __init__(self):
        model_path = os.path.join('models', 'catGeneratingVAE.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le modèle {model_path} est introuvable.")
        
        # Définition de la fonction de sampling pour le VAE
        @tf.keras.utils.register_keras_serializable(package="Custom")
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # Chargement du modèle complet
        self.model = load_model(model_path, custom_objects={'sampling': sampling}, compile=False)
        
        # On cherche la partie décodeur à l'intérieur du modèle
        self.decoder = None
        
        # 1. Chercher une couche qui s'appelle 'decoder'
        for layer in self.model.layers:
            if 'decoder' in layer.name.lower():
                self.decoder = layer
                break
        
        # 2. Si non trouvé, on utilise le modèle complet (cas où le fichier est déjà le décodeur)
        if self.decoder is None:
            self.decoder = self.model

        # 3. Dimension latente (Forcée à 16 car vue dans vos captures d'échantillonnage)
        # Si votre modèle utilise une autre taille, il faudra l'ajuster ici
        self.latent_dim = 16

    def generate(self):
        # 1. Générer un vecteur latent aléatoire
        random_latent_vector = np.random.normal(size=(1, self.latent_dim))
        
        # 2. Prédire l'image (décodage)
        prediction = self.decoder.predict(random_latent_vector)
        
        # 3. Post-traitement
        # On enlève la dimension de batch et on remet en 0-255
        img_array = prediction[0]
        img_array = (img_array * 255).astype(np.uint8)
        
        # Gérer les cas N&B (1 canal) ou Couleur (3 canaux)
        if img_array.shape[-1] == 1:
            res_img = Image.fromarray(img_array.squeeze(), mode='L')
        else:
            res_img = Image.fromarray(img_array, mode='RGB')
            
        # Redimensionner pour un meilleur affichage si nécessaire (ex: 256x256)
        res_img = res_img.resize((256, 256), Image.NEAREST)
        
        # 4. Conversion en base64
        buffered = io.BytesIO()
        res_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str

# Instances globales
classifier = None
colorizer = None
generator = None

def get_classifier():
    global classifier
    if classifier is None:
        classifier = TERClassifier()
    return classifier

def get_colorizer():
    global colorizer
    if colorizer is None:
        colorizer = ImageColorizer()
    return colorizer

def get_generator():
    global generator
    if generator is None:
        generator = ImageGenerator()
    return generator

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image fournie'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    try:
        model = get_classifier()
        best_class, best_score, all_scores = model.predict(file.stream)
        
        warning = None
        if best_score < 0.5:
            warning = "Attention : le score de confiance est inférieur à 50%. Le modèle ne reconnaît que les chats, moutons et éléphants."

        return jsonify({
            'label': best_class,
            'scores': all_scores,
            'warning': warning
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/colorize', methods=['POST'])
def colorize():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image fournie'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    try:
        model = get_colorizer()
        color_b64, gray_b64 = model.process(file.stream)
        
        return jsonify({
            'original': gray_b64,
            'colorized': color_b64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    try:
        model = get_generator()
        img_b64 = model.generate()
        
        return jsonify({
            'image': img_b64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)
