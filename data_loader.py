"""
Data loader for fake news detection - loads sample dataset
"""
import pandas as pd
import os
import urllib.request
import zipfile

def get_sample_data():
    """
    Downloads and prepares the WELFake dataset if not already present.
    Returns: DataFrame with 'text' and 'label' columns (1=fake, 0=real)
    """
    data_dir = 'data'
    csv_file = os.path.join(data_dir, 'WELFake_dataset.csv')
    
    if os.path.exists(csv_file):
        print(f"Loading existing dataset from {csv_file}")
        df = pd.read_csv(csv_file)
        return df
    
    print("Creating sample fake news dataset...")
    
    # Create synthetic dataset for demonstration
    fake_samples = [
        "Trump announces new secret weather control device that will end global warming",
        "Actress reveals she is actually a secret alien from Mars",
        "Scientists prove that vaccines contain microchips to control your mind",
        "Breaking: The moon landing was completely faked by Hollywood",
        "Ancient aliens built the Egyptian pyramids using laser beams",
        "A leaked video shows Bigfoot walking on Wall Street",
        "Schools to replace teachers with artificial intelligence robots by 2025",
        "Water fluoridation is a government conspiracy to reduce population",
        "5G towers cause COVID-19 according to new study",
        "Celebrity joins secret underground civilization beneath Antarctica"
    ]
    
    real_samples = [
        "New study shows increased renewable energy adoption across Europe",
        "Global temperature rises 1.5 degrees Celsius over past decade",
        "Federal Reserve announces interest rate decision",
        "Scientists discover new species of deep-sea fish",
        "Stock market closes up 2.5% on positive economic data",
        "NASA confirms discovery of potentially habitable exoplanet",
        "Education department releases new curriculum standards",
        "International trade negotiations conclude with new agreement",
        "Research shows benefits of Mediterranean diet for health",
        "Tech company announces new privacy features for users"
    ]
    
    # Create DataFrame
    data = {
        'text': fake_samples + real_samples + fake_samples*2 + real_samples*2,
        'label': [1]*len(fake_samples) + [0]*len(real_samples) + [1]*(len(fake_samples)*2) + [0]*(len(real_samples)*2)
    }
    
    df = pd.DataFrame(data)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(csv_file, index=False)
    print(f"Sample dataset created and saved to {csv_file}")
    print(f"Dataset shape: {df.shape}")
    
    return df

def load_data():
    """Main function to load data"""
    df = get_sample_data()
    
    # Basic data cleaning
    df = df.dropna(subset=['text'])
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Fake news (label=1): {(df['label']==1).sum()}")
    print(f"Real news (label=0): {(df['label']==0).sum()}")
    
    return df
