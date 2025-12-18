from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "/home/kun/models/nllb-200"

tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("NLLB load OK")
