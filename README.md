# 🧠 Emotion Detector – Detecção de Emoções Faciais com Deep Learning

Este projeto consiste em um sistema de **visão computacional** com **deep learning**, capaz de detectar emoções humanas (como **feliz**, **triste**, **raiva**, entre outras) a partir de imagens faciais. O modelo é treinado com um conjunto de dados rotulado com expressões faciais humanas, usando **TensorFlow/Keras**.

---

## 🚀 Tecnologias Utilizadas

- Python 3.10
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- PIL (Pillow)
- Dataset de emoções faciais (baixado manualmente)

## 📥 Como Instalar e Rodar Localmente

> Pré-requisito: Python 3.10 instalado.

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/Emotion-Detector.git
cd Emotion-Detector
``` 

### 2. Crie um ambiente virtual com Python 3.10
```bash
py -3.10 -m venv venv
```

### 3.  Ative o ambiente virtual (Windows)
```bash
.\venv\Scripts\Activate.ps1
```

### 4. Instale as dependências
```bash
pip install -r requirements.txt
```

### 5. Treine o modelo
```bash
python train.py
```

### 6. Detecção em tempo real
```bash
python main.py
```

