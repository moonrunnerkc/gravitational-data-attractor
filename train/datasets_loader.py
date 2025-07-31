"""
Dataset loaders for multi-modal GDA training.
Supports text, image, and audio data with various augmentations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import random
from PIL import Image
import torchaudio
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Union


class TextAugmentation:
    """Text augmentation techniques."""
    
    def __init__(self, config):
        self.random_deletion_prob = config.get('random_deletion', 0.1)
        self.random_swap_prob = config.get('random_swap', 0.1)
    
    def __call__(self, text):
        """Apply text augmentations."""
        words = text.split()
        
        # Random deletion
        if random.random() < self.random_deletion_prob:
            words = [w for w in words if random.random() > 0.1]
        
        # Random swap
        if random.random() < self.random_swap_prob and len(words) > 1:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)


class ImageAugmentation:
    """Image augmentation pipeline."""
    
    def __init__(self, config):
        transforms_list = []
        
        if config.get('random_crop', True):
            transforms_list.append(transforms.RandomResizedCrop(224))
        else:
            transforms_list.append(transforms.Resize(256))
            transforms_list.append(transforms.CenterCrop(224))
        
        if config.get('horizontal_flip', 0.5) > 0:
            transforms_list.append(
                transforms.RandomHorizontalFlip(p=config['horizontal_flip'])
            )
        
        if config.get('color_jitter', 0) > 0:
            transforms_list.append(
                transforms.ColorJitter(
                    brightness=config['color_jitter'],
                    contrast=config['color_jitter'],
                    saturation=config['color_jitter'],
                    hue=config['color_jitter'] / 2
                )
            )
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.transform = transforms.Compose(transforms_list)
    
    def __call__(self, image):
        """Apply image augmentations."""
        return self.transform(image)


class AudioAugmentation:
    """Audio augmentation techniques."""
    
    def __init__(self, config):
        self.time_stretch_factor = config.get('time_stretch', 0.2)
        self.pitch_shift_steps = config.get('pitch_shift', 2)
        self.noise_level = config.get('add_noise', 0.1)
    
    def __call__(self, waveform, sample_rate):
        """Apply audio augmentations."""
        # Time stretching
        if self.time_stretch_factor > 0 and random.random() < 0.5:
            stretch_factor = 1 + random.uniform(-self.time_stretch_factor, 
                                              self.time_stretch_factor)
            waveform = torchaudio.functional.phase_vocoder(
                torch.stft(waveform, n_fft=2048, return_complex=True),
                stretch_factor
            ).abs()
        
        # Pitch shifting
        if self.pitch_shift_steps > 0 and random.random() < 0.5:
            n_steps = random.randint(-self.pitch_shift_steps, self.pitch_shift_steps)
            waveform = torchaudio.functional.pitch_shift(waveform, sample_rate, n_steps)
        
        # Add noise
        if self.noise_level > 0 and random.random() < 0.5:
            noise = torch.randn_like(waveform) * self.noise_level
            waveform = waveform + noise
        
        return waveform


class MultiModalDataset(Dataset):
    """Multi-modal dataset for GDA training."""
    
    def __init__(self, data_path, split='train', config=None, transform=None):
        self.data_path = data_path
        self.split = split
        self.config = config or {}
        
        # Load metadata
        metadata_path = os.path.join(data_path, f'{split}_metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Setup augmentations
        if transform:
            self.transform = transform
        else:
            self.setup_default_transforms()
        
        # Modality flags
        self.modalities = config.get('modalities', ['text', 'image', 'audio'])
        
    def setup_default_transforms(self):
        """Setup default transformations for each modality."""
        aug_config = self.config.get('augmentation', {})
        
        self.text_transform = TextAugmentation(aug_config.get('text', {}))
        self.image_transform = ImageAugmentation(aug_config.get('image', {}))
        self.audio_transform = AudioAugmentation(aug_config.get('audio', {}))
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Get multi-modal sample."""
        sample_info = self.metadata[idx]
        sample = {'idx': idx}
        
        # Load available modalities
        inputs = {}
        
        # Text
        if 'text' in self.modalities and 'text_path' in sample_info:
            text = self.load_text(sample_info['text_path'])
            if self.split == 'train':
                text = self.text_transform(text)
            inputs['text'] = text
        
        # Image
        if 'image' in self.modalities and 'image_path' in sample_info:
            image = self.load_image(sample_info['image_path'])
            image = self.image_transform(image)
            inputs['image'] = image
        
        # Audio
        if 'audio' in self.modalities and 'audio_path' in sample_info:
            audio, sr = self.load_audio(sample_info['audio_path'])
            if self.split == 'train':
                audio = self.audio_transform(audio, sr)
            inputs['audio'] = audio
        
        sample['inputs'] = inputs
        
        # Labels/targets if available
        if 'label' in sample_info:
            sample['label'] = sample_info['label']
        
        # For reconstruction tasks, targets are same as inputs
        sample['targets'] = inputs
        
        # Modality information
        sample['modality'] = list(inputs.keys())[0] if len(inputs) == 1 else 'multi'
        
        return sample
    
    def load_text(self, path):
        """Load text from file."""
        full_path = os.path.join(self.data_path, path)
        with open(full_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return text
    
    def load_image(self, path):
        """Load image from file."""
        full_path = os.path.join(self.data_path, path)
        image = Image.open(full_path).convert('RGB')
        return image
    
    def load_audio(self, path):
        """Load audio from file."""
        full_path = os.path.join(self.data_path, path)
        waveform, sample_rate = torchaudio.load(full_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        return waveform, sample_rate


class CollateFunction:
    """Custom collate function for multi-modal batches."""
    
    def __init__(self, pad_text=True, stack_images=True, pad_audio=True):
        self.pad_text = pad_text
        self.stack_images = stack_images
        self.pad_audio = pad_audio
    
    def __call__(self, batch):
        """Collate batch of multi-modal samples."""
        collated = {
            'idx': torch.tensor([s['idx'] for s in batch]),
            'modality': [s['modality'] for s in batch]
        }
        
        # Collate inputs by modality
        inputs = {}
        
        # Text
        text_samples = [s['inputs'].get('text') for s in batch if 'text' in s['inputs']]
        if text_samples:
            inputs['text'] = text_samples  # Keep as list for encoder
        
        # Images
        image_samples = [s['inputs'].get('image') for s in batch if 'image' in s['inputs']]
        if image_samples and self.stack_images:
            inputs['image'] = torch.stack(image_samples)
        
        # Audio
        audio_samples = [s['inputs'].get('audio') for s in batch if 'audio' in s['inputs']]
        if audio_samples:
            if self.pad_audio:
                # Pad to same length
                max_len = max(a.shape[-1] for a in audio_samples)
                padded = []
                for audio in audio_samples:
                    if audio.shape[-1] < max_len:
                        padding = max_len - audio.shape[-1]
                        audio = torch.nn.functional.pad(audio, (0, padding))
                    padded.append(audio)
                inputs['audio'] = torch.stack(padded)
            else:
                inputs['audio'] = audio_samples
        
        collated['inputs'] = inputs
        
        # Collate targets (same structure as inputs for reconstruction)
        collated['targets'] = inputs
        
        # Labels if available
        if 'label' in batch[0]:
            collated['labels'] = torch.tensor([s['label'] for s in batch])
        
        return collated


def create_dataloaders(config):
    """Create train and validation dataloaders."""
    data_config = config.get('data', {})
    train_config = config.get('training', {})
    
    # Create datasets
    train_dataset = MultiModalDataset(
        data_path=data_config['dataset_path'],
        split='train',
        config=data_config
    )
    
    val_dataset = MultiModalDataset(
        data_path=data_config['dataset_path'],
        split='val',
        config=data_config
    )
    
    # Create collate function
    collate_fn = CollateFunction()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


# Synthetic data generation for testing
def generate_synthetic_dataset(output_path, num_samples=1000):
    """Generate synthetic multi-modal dataset for testing."""
    os.makedirs(output_path, exist_ok=True)
    
    # Create directories
    for modality in ['text', 'image', 'audio']:
        os.makedirs(os.path.join(output_path, modality), exist_ok=True)
    
    # Generate samples
    metadata = {'train': [], 'val': [], 'test': []}
    
    for i in range(num_samples):
        sample = {'id': i}
        
        # Determine split
        if i < int(0.8 * num_samples):
            split = 'train'
        elif i < int(0.9 * num_samples):
            split = 'val'
        else:
            split = 'test'
        
        # Generate text
        text_path = f"text/sample_{i}.txt"
        with open(os.path.join(output_path, text_path), 'w') as f:
            f.write(f"This is synthetic text sample number {i}. " * random.randint(5, 20))
        sample['text_path'] = text_path
        
        # Generate image (random noise)
        image_path = f"image/sample_{i}.png"
        image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        image.save(os.path.join(output_path, image_path))
        sample['image_path'] = image_path
        
        # Generate audio (sine wave)
        audio_path = f"audio/sample_{i}.wav"
        duration = random.uniform(1, 3)
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = random.uniform(200, 1000)
        waveform = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        torchaudio.save(
            os.path.join(output_path, audio_path),
            torch.from_numpy(waveform).unsqueeze(0),
            sample_rate
        )
        sample['audio_path'] = audio_path
        
        # Add label
        sample['label'] = random.randint(0, 9)
        
        metadata[split].append(sample)
    
    # Save metadata
    for split, data in metadata.items():
        with open(os.path.join(output_path, f'{split}_metadata.json'), 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"Synthetic dataset created at {output_path}")


if __name__ == "__main__":
    # Generate test dataset
    generate_synthetic_dataset("data/multimodal_dataset", num_samples=100)