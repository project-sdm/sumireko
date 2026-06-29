import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer

TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9]+")
