"""
Proje yolu ve veri seti konfigürasyonu.

Tüm script ve notebook'lar bu modülden path'leri alabilir.
Böylece farklı makinelerde veya dizin yapısında tek yerden güncelleme yeterli.
"""

import os

# Proje kök dizini (config.py'nin bulunduğu klasör)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Veri dizinleri
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# RadioML 2016.10a - .dat veya .pkl uzantılı olabilir
RML2016_FILENAME = "RML2016.10a_dict.pkl"  # veya .dat
RML2016_PATH = os.path.join(RAW_DIR, RML2016_FILENAME)

# Tekrarlanabilirlik için sabit seed
RANDOM_SEED = 42
