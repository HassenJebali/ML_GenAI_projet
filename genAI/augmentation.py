from faker import Faker
import random
import pandas as pd

fake = Faker()

def generate_fake_passenger():
    return {
        'Pclass': random.choice([1, 2, 3]),
        'Sex': random.choice(['male', 'female']),
        'Age': round(random.uniform(1, 80), 1),
        'SibSp': random.randint(0, 3),
        'Parch': random.randint(0, 2),
        'Fare': round(random.uniform(10, 100), 2),
        'Embarked': random.choice(['S', 'C', 'Q']),
        'Survived': random.choice([0, 1])
    }

def create_synthetic_dataset(n=10):
    data = [generate_fake_passenger() for _ in range(n)]
    return pd.DataFrame(data)
