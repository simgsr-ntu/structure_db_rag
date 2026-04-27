from src.ingestion.metadata_extractor import MetadataExtractor

text = """Member’s Gui Theme Topic Speaker Date The spea physical function, other gro A similar together, through t the colou In this me important 4 Messa 1) W W ou 2) Fo CELL ide Effectiv We are Elder G 07th & 0 Spiri ker begins body in 1 the comm w in knowle analogy mi interlockin the hands o rful tapestr essage, the t it is that w ages on th We are the B We are joine urselves up our Characte a. Unity is wit the Sp b. Divers wante there c. Harm equal part is d. Matu L GU ve Priests an One Body oh Hock"""

extractor = MetadataExtractor()
result = extractor.extract(text)
print(f"RESULT: {result}")
