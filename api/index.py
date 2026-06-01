"""
Ponto de entrada serverless para Vercel.

Este arquivo segue o padrão de rotas serverless do Vercel para Python:
- O diretório api/ contém as funções serverless.
- Cada arquivo .py exporta um objeto ASGI (no caso, a aplicação FastAPI).
- O Vercel detecta automaticamente o framework e configura as rotas.

Para deploy no Vercel:
    1. Configure o build command: pip install -r requirements.txt
    2. O Vercel usará este arquivo como entrypoint para /api/*
    3. As rotas definidas em app.py ficam acessíveis como:
       /api/health, /api/run-attack, /api/run-defense, /api/federated

Configuração recomendada do vercel.json:
{
    "builds": [
        {
            "src": "api/index.py",
            "use": "@vercel/python"
        }
    ],
    "routes": [
        {
            "src": "/api/(.*)",
            "dest": "api/index.py"
        }
    ]
}
"""

import sys
import os

# Garantir que o diretório raiz está no path para importações
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar a aplicação FastAPI do app principal
from app import app

# O Vercel espera um objeto ASGI chamado 'app' ou 'handler'
# FastAPI já é ASGI-compatível
handler = app
