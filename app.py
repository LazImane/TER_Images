import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import io

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

# Instance globale du classifieur
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        classifier = TERClassifier()
    return classifier

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
    return jsonify({'error': 'Modèle non disponible'}), 503

@app.route('/generate', methods=['POST'])
def generate():
    return jsonify({'error': 'Modèle non disponible'}), 503

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)
