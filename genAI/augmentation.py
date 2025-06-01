from faker import Faker
import random
import pandas as pd

fake = Faker()

def generate_fake_passenger():
    return {
        'Age': round(random.uniform(1, 80), 1),
        'Fare': round(random.uniform(10, 100), 2),
        'Sex': random.choice([0, 1]),          # 0 = femme, 1 = homme ou selon ton encodage
        'sibsp': random.randint(0, 3),
        'Parch': random.randint(0, 2),
        'Pclass': random.choice([1, 2, 3]),
        'Embarked': random.choice([0, 1, 2]),  # 0 = S, 1 = C, 2 = Q (adapte si différent)
        'Survived': random.choice([0, 1])
    }

def create_synthetic_dataset(n=10):
    data = [generate_fake_passenger() for _ in range(n)]
    return pd.DataFrame(data)
