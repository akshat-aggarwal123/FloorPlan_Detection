import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.multi_task_net import DeepFloorplanNet
from data.datasets import FloorplanDataset
from config import Config
import os
from tqdm import tqdm

def weighted_cross_entropy_loss(predictions, targets, weights):
    """Compute weighted cross entropy loss"""
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    losses = loss_fn(predictions, targets)
    
    # Apply weights based on class frequency
    weighted_losses = losses * weights[targets]
    return weighted_losses.mean()

def train_model():
    config = Config()
    
    # Enhanced CUDA setup
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        torch.backends.cudnn.benchmark = True  
        torch.backends.cudnn.deterministic = False  
        torch.cuda.empty_cache()
    else:
        print("CUDA is not available. Using CPU.")
    
    # ✅ Correct dataset paths
    print("Loading datasets...")
    train_dataset = FloorplanDataset('data/data/processed/train', config)
    val_dataset   = FloorplanDataset('data/data/processed/val', config)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
        prefetch_factor=2 if config.NUM_WORKERS > 0 else 2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
        prefetch_factor=2 if config.NUM_WORKERS > 0 else False
    )
    
    # Model
    print("Initializing model...")
    model = DeepFloorplanNet(config.NUM_BOUNDARY_CLASSES, config.NUM_ROOM_CLASSES)
    model.to(config.DEVICE)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Updated AMP API
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    use_amp = torch.cuda.is_available() and hasattr(torch.amp, 'autocast')
    
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) for faster training")
    
    best_val_loss = float('inf')
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]', leave=False, ncols=50)
        
        for batch_idx, (images, boundary_labels, room_labels) in enumerate(train_pbar):
            try:
                images = images.to(config.DEVICE, non_blocking=True)
                boundary_labels = boundary_labels.to(config.DEVICE, non_blocking=True)
                room_labels = room_labels.to(config.DEVICE, non_blocking=True)
                
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        boundary_loss = nn.CrossEntropyLoss()(outputs['boundary'], boundary_labels)
                        room_loss = nn.CrossEntropyLoss()(outputs['room'], room_labels)
                        total_loss = (config.BOUNDARY_WEIGHT * boundary_loss + 
                                     config.ROOM_WEIGHT * room_loss)
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    boundary_loss = nn.CrossEntropyLoss()(outputs['boundary'], boundary_labels)
                    room_loss = nn.CrossEntropyLoss()(outputs['room'], room_labels)
                    total_loss = (config.BOUNDARY_WEIGHT * boundary_loss + 
                                 config.ROOM_WEIGHT * room_loss)
                    total_loss.backward()
                    optimizer.step()
                
                train_loss += total_loss.item()
                train_pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Avg Loss': f'{train_loss/(batch_idx+1):.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            except RuntimeError as e:
                print(f"\nError in training batch {batch_idx}: {e}")
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("GPU out of memory. Try reducing batch size or image size.")
                raise e
        
        train_pbar.close()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]', leave=False, ncols=50)
        
        with torch.no_grad():
            for batch_idx, (images, boundary_labels, room_labels) in enumerate(val_pbar):
                try:
                    images = images.to(config.DEVICE, non_blocking=True)
                    boundary_labels = boundary_labels.to(config.DEVICE, non_blocking=True)
                    room_labels = room_labels.to(config.DEVICE, non_blocking=True)
                    
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            outputs = model(images)
                            boundary_loss = nn.CrossEntropyLoss()(outputs['boundary'], boundary_labels)
                            room_loss = nn.CrossEntropyLoss()(outputs['room'], room_labels)
                            total_loss = (config.BOUNDARY_WEIGHT * boundary_loss + 
                                         config.ROOM_WEIGHT * room_loss)
                    else:
                        outputs = model(images)
                        boundary_loss = nn.CrossEntropyLoss()(outputs['boundary'], boundary_labels)
                        room_loss = nn.CrossEntropyLoss()(outputs['room'], room_labels)
                        total_loss = (config.BOUNDARY_WEIGHT * boundary_loss + 
                                     config.ROOM_WEIGHT * room_loss)
                    
                    val_loss += total_loss.item()
                    val_pbar.set_postfix({
                        'Loss': f'{total_loss.item():.4f}',
                        'Avg Loss': f'{val_loss/(batch_idx+1):.4f}'
                    })
                
                except RuntimeError as e:
                    print(f"\nError in validation batch {batch_idx}: {e}")
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print("GPU out of memory during validation.")
                    raise e
        
        val_pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f'  GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config.__dict__,
            }, 'best_model.pth')
            print(f'  ★ New best model saved! Val Loss: {best_val_loss:.4f}')
        
        scheduler.step()
        
        if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
        
        print('-' * 60)
    
    print('\n🎉 Training completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train_model()