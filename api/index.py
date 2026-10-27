"""Legacy ASGI alias for tools that still import api.index.

Vercel Services uses app:app from the repository root. Public requests enter
through the gateway service and its runtime binding; see vercel.json and
DEPLOYMENT.md. Do not restore the former top-level builds/routes config.
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
