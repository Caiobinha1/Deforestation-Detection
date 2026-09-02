"""
Wrapper para manter compatibilidade com chamadas ao antigo Inference.py.
Redireciona para a implementação otimizada BatchInference.py.
"""
from BatchInference import main

if __name__ == "__main__":
    main()