import json
import random
from faker import Faker
from tqdm import tqdm

fake = Faker()

def get_noisy_text(text):
    # Simulate simple STT errors or filler words
    if random.random() < 0.2:
        text = text.replace(" is ", " is uh ")
    if random.random() < 0.2:
        text = text.replace(" and ", " n ")
    if random.random() < 0.1:
        text = text.replace(" ", "  ") # Double spaces
    return text.lower()

def create_example(id_num):
    # 20% chance of a sentence with NO entities (Background noise)
    if random.random() < 0.2:
        text = fake.sentence(nb_words=10)
        return {"id": f"utt_{id_num:04d}", "text": text.lower(), "entities": []}

    entities = []
    template_type = random.choice(["intro", "payment", "contact", "appointment"])
    
    if template_type == "intro":
        name = fake.name()
        loc = fake.city()
        # Varied templates
        phrases = [
            f"hi this is {name} calling from {loc}",
            f"my name is {name} and i live in {loc}",
            f"speaking with {name} from {loc} today"
        ]
        text = random.choice(phrases)
        
        # Recalculate offsets because text is random
        if name in text:
            start = text.find(name)
            entities.append({"start": start, "end": start + len(name), "label": "PERSON_NAME"})
        if loc in text:
            start = text.find(loc)
            entities.append({"start": start, "end": start + len(loc), "label": "CITY"})

    elif template_type == "payment":
        cc = fake.credit_card_number()
        # Noisy CC (STT often spaces out digits)
        cc_spaced = cc.replace("-", " ")
        text = f"please charge my card {cc_spaced} thanks"
        
        start = text.find(cc_spaced)
        entities.append({"start": start, "end": start + len(cc_spaced), "label": "CREDIT_CARD"})

    elif template_type == "contact":
        phone = fake.phone_number().replace("x", "").split(" ")[0] # Simplify
        email = fake.email().replace("@", " at ").replace(".", " dot ")
        text = f"reach me at {phone} or {email}"
        
        if phone in text:
            start = text.find(phone)
            entities.append({"start": start, "end": start + len(phone), "label": "PHONE"})
        if email in text:
            start = text.find(email)
            entities.append({"start": start, "end": start + len(email), "label": "EMAIL"})
            
    elif template_type == "appointment":
        date = fake.date()
        text = f"schedule it for {date}"
        start = text.find(date)
        entities.append({"start": start, "end": start + len(date), "label": "DATE"})

    return {
        "id": f"utt_{id_num:04d}",
        "text": get_noisy_text(text),
        "entities": entities
    }

def generate_file(filename, count):
    with open(filename, 'w') as f:
        for i in tqdm(range(count)):
            data = create_example(i)
            f.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    print("Generating harder train.jsonl...")
    generate_file("data/train.jsonl", 1000)
    print("Generating harder dev.jsonl...")
    generate_file("data/dev.jsonl", 200)
    print("Generating stress.jsonl...")
    generate_file("data/stress.jsonl", 100)