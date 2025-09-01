import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from models.multi_task_net import DeepFloorplanNet
from data.datasets import FloorplanDataset
from config import Config
import os
from tqdm import tqd
import time
import gc
import numpy as np
import warnings

# Suppress the torchvision warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

def compute_metrics(preds, labels):
    """Compute accuracy efficiently"""
    with torch.no_grad():
        preds = torch.argmax(preds, dim=1)
        correct = (preds == labels).float()
        acc = correct.mean().item()
        return acc

def compute_detailed_metrics(all_preds, all_labels):
    """Compute detailed metrics for validation"""
    try:
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return acc, precision, recall, f1
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return 0.0, 0.0, 0.0, 0.0

def get_scheduler(optimizer, config):
    """Get scheduler based on config"""
    if config.SCHEDULER_TYPE == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=config.COSINE_ETA_MIN
        )
    elif config.SCHEDULER_TYPE == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.STEP_SIZE,
            gamma=config.STEP_GAMMA
        )
    elif config.SCHEDULER_TYPE == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.PLATEAU_PATIENCE,
            factor=config.PLATEAU_FACTOR,
            verbose=True
        )
    else:
        return optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)  # No scheduling

def train_model():
    config = Config()

    # Setup device and optimizations
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    else:
        print("CUDA is not available. Using CPU.")

    # Load datasets
    print("Loading datasets...")
    train_dataset = FloorplanDataset('data/data/processed/train', config, is_train=True)
    val_dataset = FloorplanDataset('data/data/processed/val', config, is_train=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    # Model
    print("Initializing model...")
    model = DeepFloorplanNet(config.NUM_BOUNDARY_CLASSES, config.NUM_ROOM_CLASSES)
    model.to(config.DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        eps=1e-8
    )

    scheduler = get_scheduler(optimizer, config)

    # Loss functions
    boundary_criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    room_criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    # Mixed precision
    scaler = None
    if config.MIXED_PRECISION and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda',
                                    init_scale=2**10,  # Conservative initial scale
                                    growth_factor=1.05,
                                    backoff_factor=0.8,
                                    growth_interval=2000)
        print("Using Mixed Precision Training")

    # Training tracking
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    print(f"Effective batch size: {config.get_effective_batch_size()}")

    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()

        # =============== TRAINING ===============
        model.train()
        train_loss = 0.0
        train_acc_sum = 0.0
        num_batches = 0
        accumulated_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}', leave=False)

        for batch_idx, (images, boundary_labels, room_labels) in enumerate(train_pbar):
            try:
                images = images.to(config.DEVICE, non_blocking=True)
                boundary_labels = boundary_labels.to(config.DEVICE, non_blocking=True)
                room_labels = room_labels.to(config.DEVICE, non_blocking=True)

                # Forward pass
                if scaler is not None:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        outputs = model(images)
                        boundary_loss = boundary_criterion(outputs['boundary'], boundary_labels)
                        room_loss = room_criterion(outputs['room'], room_labels)
                        total_loss = (config.BOUNDARY_WEIGHT * boundary_loss +
                                     config.ROOM_WEIGHT * room_loss)

                        # Scale loss for gradient accumulation
                        total_loss = total_loss / config.ACCUMULATE_GRAD_BATCHES
                else:
                    outputs = model(images)
                    boundary_loss = boundary_criterion(outputs['boundary'], boundary_labels)
                    room_loss = room_criterion(outputs['room'], room_labels)
                    total_loss = (config.BOUNDARY_WEIGHT * boundary_loss +
                                 config.ROOM_WEIGHT * room_loss)

                    # Scale loss for gradient accumulation
                    total_loss = total_loss / config.ACCUMULATE_GRAD_BATCHES

                # Check for non-finite loss
                if not torch.isfinite(total_loss):
                    print(f"Non-finite loss detected at batch {batch_idx}, skipping...")
                    continue

                # Backward pass
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

                accumulated_loss += total_loss.item()

                # Optimizer step with gradient accumulation
                if (batch_idx + 1) % config.ACCUMULATE_GRAD_BATCHES == 0:
                    if scaler is not None:
                        # Gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

                    # Track metrics
                    train_loss += accumulated_loss * config.ACCUMULATE_GRAD_BATCHES
                    acc = compute_metrics(outputs['room'], room_labels)
                    train_acc_sum += acc
                    num_batches += 1
                    accumulated_loss = 0.0

                # Update progress bar
                if batch_idx % config.LOG_INTERVAL == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    avg_loss = train_loss / max(1, num_batches) if num_batches > 0 else 0
                    avg_acc = train_acc_sum / max(1, num_batches) if num_batches > 0 else 0

                    train_pbar.set_postfix({
                        'Loss': f'{avg_loss:.4f}',
                        'Acc': f'{avg_acc:.3f}',
                        'LR': f'{current_lr:.2e}'
                    })

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue

        train_pbar.close()

        # Step scheduler (except ReduceLROnPlateau)
        if config.SCHEDULER_TYPE != 'plateau':
            scheduler.step()

        # =============== VALIDATION ===============
        if epoch % config.VAL_FREQ == 0 or epoch == config.NUM_EPOCHS - 1:
            model.eval()
            val_loss = 0.0
            val_batch_count = 0
            all_preds, all_labels = [], []

            val_pbar = tqdm(val_loader, desc=f'Validation', leave=False)

            with torch.no_grad():
                for batch_idx, (images, boundary_labels, room_labels) in enumerate(val_pbar):
                    try:
                        images = images.to(config.DEVICE, non_blocking=True)
                        boundary_labels = boundary_labels.to(config.DEVICE, non_blocking=True)
                        room_labels = room_labels.to(config.DEVICE, non_blocking=True)

                        if scaler is not None:
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

                        if torch.isfinite(total_loss):
                            val_loss += total_loss.item()
                            val_batch_count += 1

                            # Collect predictions
                            preds = torch.argmax(outputs['room'], dim=1).view(-1).cpu().numpy()
                            labels = room_labels.view(-1).cpu().numpy()
                            all_preds.append(preds)
                            all_labels.append(labels)

                        val_pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})

                    except Exception as e:
                        print(f"Error in validation batch {batch_idx}: {e}")
                        continue

            val_pbar.close()

            # Calculate validation metrics
            if val_batch_count > 0 and all_preds:
                avg_val_loss = val_loss / val_batch_count
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                val_acc, val_prec, val_rec, val_f1 = compute_detailed_metrics(all_preds, all_labels)

                # Step ReduceLROnPlateau scheduler
                if config.SCHEDULER_TYPE == 'plateau':
                    scheduler.step(avg_val_loss)

            else:
                avg_val_loss = float('inf')
                val_acc = val_prec = val_rec = val_f1 = 0.0
        else:
            avg_val_loss = float('inf')
            val_acc = val_prec = val_rec = val_f1 = 0.0

        # Calculate training averages
        avg_train_loss = train_loss / max(1, num_batches)
        avg_train_acc = train_acc_sum / max(1, num_batches)

        epoch_time = time.time() - epoch_start_time

        # =============== LOGGING ===============
        print(f'\nEpoch {epoch+1}/{config.NUM_EPOCHS} ({epoch_time:.1f}s):')
        print(f'  Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.3f}')

        if epoch % config.VAL_FREQ == 0 or epoch == config.NUM_EPOCHS - 1:
            print(f'  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}')

        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')

        # =============== MODEL SAVING ===============
        # Save best model based on validation accuracy
        if val_acc > best_val_acc and avg_val_loss < float('inf'):
            best_val_acc = val_acc
            best_val_loss = avg_val_loss

            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'config': config.__dict__,
                'val_metrics': {
                    'accuracy': val_acc,
                    'precision': val_prec,
                    'recall': val_rec,
                    'f1': val_f1
                }
            }, 'best_model.pth')

            print(f'  ★ New best model saved! Val Acc: {best_val_acc:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint every N epochs
        if epoch % config.SAVE_FREQ == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config.__dict__,
            }, checkpoint_path)
            print(f'  💾 Checkpoint saved: {checkpoint_path}')

        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f'\n🛑 Early stopping triggered after {patience_counter} epochs without improvement')
            break

        # Memory cleanup
        if epoch % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Check for potential overfitting
        if epoch > 10 and avg_train_acc > 0.99 and val_acc < 0.8:
            print("⚠️  Warning: Potential overfitting detected (high train acc, low val acc)")

    print('\n🎉 Training completed!')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'Best validation loss: {best_val_loss:.4f}')

    return best_val_acc, best_val_loss

def quick_debug_run():
    """Run a quick debug session to test the training loop"""
    print("🐛 Running debug session...")

    config = Config()
    config.update_for_debugging()  # 5 epochs, frequent validation

    try:
        best_acc, best_loss = train_model()
        print(f"✅ Debug run completed successfully!")
        print(f"Final metrics: Acc={best_acc:.3f}, Loss={best_loss:.3f}")
        return True
    except Exception as e:
        print(f"❌ Debug run failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data_integrity():
    """Check if the data is properly loaded and formatted"""
    print("🔍 Checking data integrity...")

    config = Config()

    try:
        # Load a small sample
        train_dataset = FloorplanDataset('data/data/processed/train', config, is_train=True)
        val_dataset = FloorplanDataset('data/data/processed/val', config, is_train=False)

        print(f"✅ Datasets loaded successfully")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val: {len(val_dataset)} samples")

        # Check sample data
        for i, (image, boundary, room) in enumerate([train_dataset[0], val_dataset[0]]):
            dataset_name = "Train" if i == 0 else "Val"
            print(f"\n{dataset_name} sample 0:")
            print(f"   Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"   Boundary shape: {boundary.shape}, unique values: {torch.unique(boundary)}")
            print(f"   Room shape: {room.shape}, unique values: {torch.unique(room)}")

            # Check for invalid values
            if torch.isnan(image).any():
                print(f"   ❌ NaN values found in {dataset_name} image!")
            if torch.unique(boundary).max() >= config.NUM_BOUNDARY_CLASSES:
                print(f"   ❌ Invalid boundary labels in {dataset_name}!")
            if torch.unique(room).max() >= config.NUM_ROOM_CLASSES:
                print(f"   ❌ Invalid room labels in {dataset_name}!")

        # Test DataLoader
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(train_loader))
        print(f"\n✅ DataLoader test successful")
        print(f"   Batch shapes: Image={batch[0].shape}, Boundary={batch[1].shape}, Room={batch[2].shape}")

        return True

    except Exception as e:
        print(f"❌ Data integrity check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Step 1: Check data integrity
    if not check_data_integrity():
        print("Fix data issues before training!")
        exit(1)

    # Step 2: Run quick debug
    print("\n" + "="*50)
    if not quick_debug_run():
        print("Fix training issues before full training!")
        exit(1)

    # Step 3: Ask user for full training
    print("\n" + "="*50)
    response = input("Debug run successful! Run full training? (y/n): ")

    if response.lower() == 'y':
        config = Config()  # Reset to full config
        train_model()
    else:
        print("Training cancelled. You can run full training anytime with train_model()")
