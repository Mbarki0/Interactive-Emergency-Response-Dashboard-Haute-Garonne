#!/bin/bash

# Script d'installation simple pour le projet pompiers
set -e

echo "🚒 Installation du projet..."

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 requis"
    exit 1
fi

# Créer environnement virtuel
if [ ! -d "venv" ]; then
    echo "Création environnement virtuel..."
    python3 -m venv venv
fi

source venv/bin/activate

# Installer dépendances
echo "Installation dépendances..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Créer dossiers
echo "Création dossiers..."
mkdir -p Data Data_preprocessed Data_trained Data_sampled Data_environment SVG_model Plots Reward_weights

# Test rapide
echo "Test installation..."
python3 -c "
import torch, pandas, numpy
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print('Installation réussie!')
"

# Créer script d'activation
cat > activate.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
echo "Environnement activé - Prêt à travailler!"
EOF
chmod +x activate.sh

echo ""
echo "Installation terminée!"
echo "Pour commencer: ./activate.sh"