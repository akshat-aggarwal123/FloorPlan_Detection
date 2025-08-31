import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from models.multi_task_net import DeepFloorplanNet
from data.datasets import FloorplanDataset
from config import Config
import os
from tqdm import tqdm
import time
import gc

# ---------------- Optimized Metrics ---------------- #
def compute_metrics_fast(preds, labels, num_classes):
    """
    Faster metrics computation using torch operations
    """
    with torch.no_grad():
        preds = torch.argmax(preds, dim=1)
        
        # Compute accuracy on GPU
        correct = (preds == labels).float()
        acc = correct.mean().item()
        
        # Only compute detailed metrics every few batches to save time
        return acc, 0, 0, 0  # Return simplified metrics during training


def compute_detailed_metrics(preds, labels, num_classes):
    """
    Detailed metrics for validation only
    """
    with torch.no_grad():
        preds = torch.argmax(preds, dim=1).view(-1).cpu().numpy()
        labels = labels.view(-1).cpu().numpy()

        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="macro", zero_division=0)
        recall = recall_score(labels, preds, average="macro", zero_division=0)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return acc, precision, recall, f1


def train_model():
    config = Config()
    
    # Enhanced CUDA setup
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True  
        torch.backends.cudnn.deterministic = False
        # Enable TensorFloat-32 for faster training on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
    else:
        print("CUDA is not available. Using CPU.")
    
    # ✅ Optimized Datasets
    print("Loading datasets...")
    train_dataset = FloorplanDataset('data/data/processed/train', config, is_train=True)
    val_dataset   = FloorplanDataset('data/data/processed/val', config, is_train=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Optimized DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE * 2,  # Increase batch size if memory allows
        shuffle=True, 
        num_workers=min(8, os.cpu_count()),  # Optimize workers
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,
        prefetch_factor=4,  # Increased prefetch
        drop_last=True  # Avoid variable batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE * 4,  # Larger batch for validation
        shuffle=False, 
        num_workers=min(4, os.cpu_count()),
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False
    )
    
    # Model with compilation (PyTorch 2.0+)
    print("Initializing model...")
    model = DeepFloorplanNet(config.NUM_BOUNDARY_CLASSES, config.NUM_ROOM_CLASSES)
    model.to(config.DEVICE)
    
    # Try to compile model for faster inference (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print("Model compiled with torch.compile for faster training")
    except:
        print("torch.compile not available, using standard model")
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    # Optimized optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE * 2,  # Increase learning rate with larger batch
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # More aggressive scheduler for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config.LEARNING_RATE * 2,
        epochs=config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    # Enhanced AMP setup
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    use_amp = torch.cuda.is_available()
    
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) for faster training")
    
    # Pre-compiled loss functions
    boundary_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    room_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_loss = float('inf')
    print(f"\nStarting optimized training for {config.NUM_EPOCHS} epochs...")
    
    # Training metrics tracking
    metrics_update_freq = max(1, len(train_loader) // 10)  # Update metrics less frequently
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        
        train_loss = 0.0
        train_acc_sum = 0.0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]', leave=False)
        
        for batch_idx, (images, boundary_labels, room_labels) in enumerate(train_pbar):
            batch_start = time.time()
            
            images = images.to(config.DEVICE, non_blocking=True)
            boundary_labels = boundary_labels.to(config.DEVICE, non_blocking=True)
            room_labels = room_labels.to(config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(images)
                    boundary_loss = boundary_criterion(outputs['boundary'], boundary_labels)
                    room_loss = room_criterion(outputs['room'], room_labels)
                    total_loss = (config.BOUNDARY_WEIGHT * boundary_loss + 
                                 config.ROOM_WEIGHT * room_loss)
                
                scaler.scale(total_loss).backward()
                
                # Gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                boundary_loss = boundary_criterion(outputs['boundary'], boundary_labels)
                room_loss = room_criterion(outputs['room'], room_labels)
                total_loss = (config.BOUNDARY_WEIGHT * boundary_loss + 
                             config.ROOM_WEIGHT * room_loss)
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()  # Step after each batch for OneCycleLR
            
            train_loss += total_loss.item()
            
            # Compute simplified metrics less frequently
            if batch_idx % metrics_update_freq == 0:
                acc, _, _, _ = compute_metrics_fast(outputs['room'], room_labels, config.NUM_ROOM_CLASSES)
                train_acc_sum += acc
            
            num_batches += 1
            
            # Update progress bar less frequently
            if batch_idx % 5 == 0:
                train_pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'Time/batch': f'{time.time() - batch_start:.2f}s'
                })
        
        train_pbar.close()
        
        # Validation (less frequent for speed)
        if epoch % 2 == 0 or epoch == config.NUM_EPOCHS - 1:  # Validate every 2 epochs
            model.eval()
            val_loss = 0.0
            val_metrics = []
            
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]', leave=False)
            
            with torch.no_grad():
                for batch_idx, (images, boundary_labels, room_labels) in enumerate(val_pbar):
                    images = images.to(config.DEVICE, non_blocking=True)
                    boundary_labels = boundary_labels.to(config.DEVICE, non_blocking=True)
                    room_labels = room_labels.to(config.DEVICE, non_blocking=True)
                    
                    if use_amp:
                        with torch.amp.autocast('cuda', dtype=torch.float16):
                            outputs = model(images)
                            boundary_loss = boundary_criterion(outputs['boundary'], boundary_labels)
                            room_loss = room_criterion(outputs['room'], room_labels)
                            total_loss = (config.BOUNDARY_WEIGHT * boundary_loss + 
                                         config.ROOM_WEIGHT * room_loss)
                    else:
                        outputs = model(images)
                        boundary_loss = boundary_criterion(outputs['boundary'], boundary_labels)
                        room_loss = room_criterion(outputs['room'], room_labels)
                        total_loss = (config.BOUNDARY_WEIGHT * boundary_loss + 
                                     config.ROOM_WEIGHT * room_loss)
                    
                    val_loss += total_loss.item()
                    
                    # Compute detailed metrics only for a subset of batches
                    if batch_idx % 3 == 0:
                        metrics = compute_detailed_metrics(outputs['room'], room_labels, config.NUM_ROOM_CLASSES)
                        val_metrics.append(metrics)
                    
                    val_pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
            
            val_pbar.close()
            
            # Average validation metrics
            avg_val_loss = val_loss / len(val_loader)
            if val_metrics:
                avg_val_acc = sum(m[0] for m in val_metrics) / len(val_metrics)
                avg_val_f1 = sum(m[3] for m in val_metrics) / len(val_metrics)
            else:
                avg_val_acc, avg_val_f1 = 0, 0
                
        else:
            avg_val_loss = float('inf')
            avg_val_acc, avg_val_f1 = 0, 0
        
        # Training averages
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc_sum / max(1, num_batches // metrics_update_freq)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f'\nEpoch {epoch+1}/{config.NUM_EPOCHS} ({epoch_time:.1f}s):')
        print(f'  Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.3f}')
        if epoch % 2 == 0 or epoch == config.NUM_EPOCHS - 1:
            print(f'  Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.3f} | Val F1: {avg_val_f1:.3f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
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
        
        # Memory cleanup
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    print('\n🎉 Training completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')


def profile_training_speed():
    """Helper function to profile and identify bottlenecks"""
    config = Config()
    
    print("🔍 Profiling training speed...")
    
    # Test dataset loading speed
    train_dataset = FloorplanDataset('data/data/processed/train', config, is_train=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Time data loading
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        if i >= 10:  # Test first 10 batches
            break
    data_loading_time = (time.time() - start_time) / 10
    print(f"Average data loading time per batch: {data_loading_time:.3f}s")
    
    # Time model forward pass
    model = DeepFloorplanNet(config.NUM_BOUNDARY_CLASSES, config.NUM_ROOM_CLASSES)
    model.to(config.DEVICE)
    
    dummy_input = torch.randn(config.BATCH_SIZE, 3, config.INPUT_SIZE, config.INPUT_SIZE).to(config.DEVICE)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    forward_time = (time.time() - start_time) / 20
    print(f"Average forward pass time: {forward_time:.3f}s")
    
    if data_loading_time > forward_time * 2:
        print("⚠️  Data loading is the bottleneck! Increase num_workers or optimize preprocessing")
    elif forward_time > data_loading_time * 2:
        print("⚠️  Model forward pass is the bottleneck! Consider reducing model size or input resolution")
    else:
        print("✅ Balanced data loading and model computation")


if __name__ == "__main__":
    # Uncomment to profile first
    # profile_training_speed()
    
    train_model()