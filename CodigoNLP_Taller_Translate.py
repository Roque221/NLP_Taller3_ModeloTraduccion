from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from threading import Thread

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Cargar los modelos de traducción
models = {
    "en-es": {
        "model": AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es"),
        "tokenizer": AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    },
    "es-en": {
        "model": AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en"),
        "tokenizer": AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    }
}


html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Servicio de Traducción</title>
    <script>
        async function sendTranslation() {
            const text = document.getElementById("inputText").value;
            const lang = document.getElementById("languageSelect").value;
            
            const response = await fetch("http://localhost:5555/translate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text, lang: lang })
            });

            const data = await response.json();
            document.getElementById("translationResult").innerText = `Traducción: ${data.translation}`;
        }
    </script>
</head>
<body>
    <div style="max-width: 600px; margin: 0 auto; text-align: center;">
        <h1>Servicio de Traducción</h1>
        <label for="languageSelect">Selecciona el idioma de entrada:</label>
        <select id="languageSelect">
            <option value="en-es">Inglés → Español</option>
            <option value="es-en">Español → Inglés</option>
        </select>
        <br><br>
        <textarea id="inputText" rows="4" cols="50" placeholder="Escribe el texto aquí..."></textarea>
        <br><br>
        <button onclick="sendTranslation()">Traducir</button>
        <p id="translationResult" style="margin-top: 20px; font-size: 18px; font-weight: bold;"></p>
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    return html_content

# Ruta para la traducción
@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data.get("text", "")
    lang = data.get("lang", "en-es")  # Idioma por defecto: inglés → español

    if not text:
        return jsonify({"error": "No se proporcionó texto para traducir"}), 400

    # Seleccionar el modelo correcto según el idioma
    if lang not in models:
        return jsonify({"error": "Idioma no soportado"}), 400

    tokenizer = models[lang]["tokenizer"]
    model = models[lang]["model"]

    # Tokenización y generación de traducción
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        translated_tokens = model.generate(**inputs)

    # Decodificación del resultado
    translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return jsonify({"translation": translation})

# Ejecutar Flask
def run_flask():
    app.run(host="0.0.0.0", port=5555)


thread = Thread(target=run_flask)
thread.start()

