from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def ozet_yap(metin, cumle_sayisi=2):
    cumleler = metin.split('.')
    cumleler = [c.strip() for c in cumleler if len(c) > 1]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrisi = vectorizer.fit_transform(cumleler)
    
    puanlar = np.array(tfidf_matrisi.sum(axis=1)).flatten()
    
    en_iyi_cumle_indeksleri = puanlar.argsort()[-cumle_sayisi:]
    en_iyi_cumle_indeksleri.sort()
    
    ozet = [cumleler[i] for i in en_iyi_cumle_indeksleri]
    return ". ".join(ozet) + "."

uzun_metin = """

"""

print("ÖZET:")
print(ozet_yap(uzun_metin, 2))
